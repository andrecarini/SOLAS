"""
Pipeline utilities: Environment detection, device management, and core helpers.

This module provides low-level utilities used across the pipeline:
- Warning suppression
- Environment detection (Colab vs local)
- Device and dtype management
- Performance metrics collection
- Text chunking utilities
"""

import os
import sys
import gc
import time
import warnings
from typing import Dict, Any, Tuple, Optional


# =============================================================================
# WARNING SUPPRESSION
# =============================================================================

def get_verbosity() -> bool:
    """
    Get verbosity setting from Colab secrets or environment variable.

    Checks in order:
    1. Colab secrets (VERBOSITY)
    2. Environment variable (VERBOSITY)

    Returns:
        True if verbosity is set to 'DEBUG', False otherwise
    """
    # Try Colab secrets first
    try:
        from google.colab import userdata
        verbosity = userdata.get('VERBOSITY')
        if verbosity:
            return verbosity.upper() == 'DEBUG'
    except (ImportError, Exception):
        pass

    # Fall back to environment variable
    verbosity = os.environ.get('VERBOSITY', '')
    return verbosity.upper() == 'DEBUG'


def suppress_torch_dtype_warnings():
    """Suppress torch_dtype deprecation warnings unless in debug mode."""
    if not get_verbosity():
        warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*", category=UserWarning)


def suppress_generation_flags_warnings():
    """Suppress generation flags warnings (invalid parameters) unless in debug mode."""
    if not get_verbosity():
        warnings.filterwarnings("ignore", message=".*generation flags are not valid.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*generation flags are not valid.*", category=FutureWarning)


def suppress_tts_warnings():
    """Suppress TTS library warnings unless in debug mode."""
    if not get_verbosity():
        import logging
        logging.getLogger("TTS").setLevel(logging.ERROR)


class SuppressOutput:
    """Context manager to temporarily suppress stdout and stderr."""

    def __init__(self, suppress=True):
        self.suppress = suppress
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        if self.suppress:
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr


# =============================================================================
# ENVIRONMENT DETECTION
# =============================================================================

def is_colab_environment() -> bool:
    """
    Check if running in Google Colab environment.

    Returns:
        True if running in Colab, False otherwise
    """
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def check_colab_environment(log_fn=None, load_template_fn=None) -> bool:
    """
    Check if running in Colab and display warning if not.

    Args:
        log_fn: Optional logging function
        load_template_fn: Optional template loading function

    Returns:
        True if running in Colab, False otherwise
    """
    verbose = get_verbosity()
    is_colab = is_colab_environment()

    if is_colab:
        if verbose and log_fn:
            log_fn("Running in Google Colab", 'info', verbose)
        return True
    else:
        if log_fn:
            log_fn("Not running in Google Colab", 'warning', verbose)
        try:
            from IPython.display import display, HTML
            if load_template_fn:
                warning_html = load_template_fn('warning_non_colab.html')
                display(HTML(warning_html))
        except (ImportError, NameError, FileNotFoundError, Exception) as e:
            if verbose and log_fn:
                log_fn(f"Could not display Colab warning: {e}", 'warning', verbose)
        return False


# =============================================================================
# DEVICE AND DTYPE MANAGEMENT
# =============================================================================

def get_device() -> str:
    """Get the appropriate device (cuda or cpu)."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_compute_dtype():
    """Get the appropriate compute dtype based on device availability."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    except ImportError:
        return None


def get_vram_usage() -> float:
    """Get current VRAM usage in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)
    except ImportError:
        pass
    return 0.0


def get_peak_vram() -> float:
    """Get peak VRAM usage in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except ImportError:
        pass
    return 0.0


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


# =============================================================================
# TEXT PROCESSING UTILITIES
# =============================================================================

def chunk_text(text: str, max_chars: int) -> list:
    """
    Split text into chunks at sentence boundaries, respecting max_chars limit.

    Args:
        text: Text to split
        max_chars: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    import re

    if not text:
        return []

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If sentence itself exceeds max_chars, split it
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Split long sentence at word boundaries
            words = sentence.split()
            for word in words:
                if len(current_chunk) + len(word) + 1 <= max_chars:
                    current_chunk += (" " if current_chunk else "") + word
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = word
        elif len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def collect_performance_metrics(
    stage_name: str,
    start_time: float,
    start_vram: float = 0.0
) -> Dict[str, Any]:
    """
    Collect performance metrics for a pipeline stage.

    Args:
        stage_name: Name of the stage
        start_time: Start time from time.time()
        start_vram: VRAM usage at start (GB)

    Returns:
        Dictionary with timing and memory metrics
    """
    elapsed = time.time() - start_time
    peak_vram = get_peak_vram()
    end_vram = get_vram_usage()

    # Try to get CPU usage
    avg_cpu_percent = 0.0
    try:
        import psutil
        avg_cpu_percent = psutil.cpu_percent(interval=0.1)
    except ImportError:
        pass

    return {
        'stage': stage_name,
        'time_seconds': round(elapsed, 2),
        'peak_vram_gb': round(peak_vram, 3),
        'end_vram_gb': round(end_vram, 3),
        'avg_cpu_percent': avg_cpu_percent,
    }


class StageMetrics:
    """Context manager for collecting stage performance metrics."""

    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.start_time = None
        self.start_vram = None
        self.metrics = None

    def __enter__(self):
        clear_gpu_memory()
        self.start_time = time.time()
        self.start_vram = get_vram_usage()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.metrics = collect_performance_metrics(
            self.stage_name,
            self.start_time,
            self.start_vram
        )


# =============================================================================
# STOP EXECUTION
# =============================================================================

class StopExecution(Exception):
    """Custom exception to halt notebook execution without showing traceback."""
    def _render_traceback_(self):
        pass


def show_restart_warning(log_fn=None, load_template_fn=None) -> None:
    """
    Display restart warning and halt notebook execution.

    Args:
        log_fn: Optional logging function
        load_template_fn: Optional template loading function

    Raises:
        StopExecution: Always raises this exception to halt execution
    """
    verbose = get_verbosity()
    if log_fn:
        log_fn("Runtime restart required after bitsandbytes update", 'warning', verbose)

    try:
        from IPython.display import display, HTML
        if load_template_fn:
            warning_html = load_template_fn('restart_warning.html')
            display(HTML(warning_html))
            if log_fn:
                log_fn("Restart warning displayed", 'info', verbose)
    except (ImportError, NameError, FileNotFoundError, Exception) as e:
        if log_fn:
            log_fn(f"Could not display restart warning: {e}", 'error', verbose)
            log_fn("Please restart the runtime manually", 'warning', verbose)

    if log_fn:
        log_fn("Halting execution - restart required", 'important', verbose)
    raise StopExecution
