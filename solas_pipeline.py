"""
SOLAS Pipeline: Collection of functions for SOLAS pipeline execution and notebook interaction.

This module provides:
1. Core pipeline functions for processing lecture audio into translated transcripts, summaries, and podcast audio
2. Interactive notebook helper functions for setup, widget creation, and results display
3. Template loading utilities for HTML display

Main Pipeline Entry Point:
    run_pipeline(config: dict) -> dict: Execute the full pipeline with given configuration.

Main Setup Entry Point:
    setup_environment_with_progress(verbose: bool) -> dict: Setup environment with progress tracking.

Input:
    SOLAS_CONFIG dictionary with all parameters (see implementation-plan.md for structure).

Output:
    Dictionary containing:
    - text_outputs: All generated text artifacts
    - file_paths: Paths to saved files
    - performance_metrics: Timing and resource usage for each stage
"""

"""
Standard library imports only - heavy dependencies imported lazily in functions.
This allows the module to be imported before dependencies are installed (for setup functions).
"""
import os
import sys
import time
import gc
import warnings
import subprocess

# Suppress torch_dtype deprecation warnings in non-debug mode
def _suppress_torch_dtype_warnings():
    """Suppress torch_dtype deprecation warnings unless in debug mode."""
    verbose = get_verbosity()
    if not verbose:
        warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*", category=UserWarning)

def _suppress_generation_flags_warnings():
    """Suppress generation flags warnings (invalid parameters) unless in debug mode."""
    verbose = get_verbosity()
    if not verbose:
        # Suppress warnings about invalid generation flags (temperature, top_p, top_k)
        warnings.filterwarnings("ignore", message=".*generation flags are not valid.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*generation flags are not valid.*", category=FutureWarning)

def _suppress_tts_warnings():
    """Suppress TTS library warnings unless in debug mode."""
    verbose = get_verbosity()
    if not verbose:
        # Set TTS logger to ERROR level to reduce noise
        import logging
        logging.getLogger("TTS").setLevel(logging.ERROR)

class _SuppressOutput:
    """Context manager to temporarily suppress stdout and stderr."""
    def __init__(self, suppress=True):
        self.suppress = suppress
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        if self.suppress:
            import sys
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress:
            import sys
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr

import shutil
import platform
import importlib.metadata
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Callable
from functools import lru_cache

# Heavy dependencies are imported lazily in functions that need them
# This allows importing this module before dependencies are installed


# ============================================================================
# Setup Completion Check
# ============================================================================

def show_config_warning() -> None:
    """
    Display configuration warning and halt notebook execution.
    
    This function loads the configuration warning template, displays it, logs debug
    messages, and raises a StopExecution exception to halt notebook execution.
    The user must run the Configuration cell (Cell 2) first.
    
    Raises:
        StopExecution: Always raises this exception to halt execution
    """
    verbose = get_verbosity()
    log_setup("Configuration not completed - user must run Configuration cell first", 'error', verbose)
    
    try:
        from IPython.display import display, HTML
        # Load and display configuration warning template
        warning_html = load_template('config_not_completed.html')
        display(HTML(warning_html))
        log_setup("Configuration warning displayed", 'info', verbose)
    except (ImportError, NameError, FileNotFoundError, Exception) as e:
        # If template loading fails, log error but still halt execution
        log_setup(f"Could not display configuration warning: {e}", 'error', verbose)
        log_setup("Please run the Configuration cell (Cell 2) first", 'error', verbose)
    
    # Halt execution with custom exception that suppresses traceback
    class StopExecution(Exception):
        """Custom exception to halt notebook execution without showing traceback."""
        def _render_traceback_(self):
            pass
    
    log_setup("Halting execution - configuration required", 'important', verbose)
    raise StopExecution


def show_pipeline_warning() -> None:
    """
    Display pipeline warning and halt notebook execution.
    
    This function loads the pipeline warning template, displays it, logs debug
    messages, and raises a StopExecution exception to halt notebook execution.
    The user must run the Run Pipeline cell (Cell 5) first.
    
    Raises:
        StopExecution: Always raises this exception to halt execution
    """
    verbose = get_verbosity()
    log_setup("Pipeline not completed - user must run Run Pipeline cell first", 'error', verbose)
    
    try:
        from IPython.display import display, HTML
        # Load and display pipeline warning template
        warning_html = load_template('no_results_warning.html')
        display(HTML(warning_html))
        log_setup("Pipeline warning displayed", 'info', verbose)
    except (ImportError, NameError, FileNotFoundError, Exception) as e:
        # If template loading fails, log error but still halt execution
        log_setup(f"Could not display pipeline warning: {e}", 'error', verbose)
        log_setup("Please run the Run Pipeline cell (Cell 5) first", 'error', verbose)
    
    # Halt execution with custom exception that suppresses traceback
    class StopExecution(Exception):
        """Custom exception to halt notebook execution without showing traceback."""
        def _render_traceback_(self):
            pass
    
    log_setup("Halting execution - pipeline execution required", 'important', verbose)
    raise StopExecution


# ============================================================================
# Environment Detection Functions
# ============================================================================

def get_verbosity() -> bool:
    """
    Get verbosity setting from Colab secrets or environment variable.
    
    Checks in order:
    1. Colab secrets (VERBOSITY)
    2. Environment variable (VERBOSITY)
    
    If verbosity is enabled, prints a debug message using log_setup.
    
    Returns:
        True if verbosity is set to 'DEBUG', False otherwise
    """
    import os
    
    # Try Colab secrets first
    try:
        from google.colab import userdata
        verbosity = userdata.get('VERBOSITY')
        if verbosity:
            is_verbose = verbosity.upper() == 'DEBUG'
        else:
            is_verbose = False
    except (ImportError, Exception):
        is_verbose = False
    
    # Fall back to environment variable if not set via Colab
    if not is_verbose:
        verbosity = os.environ.get('VERBOSITY', '')
        is_verbose = verbosity.upper() == 'DEBUG'
    
    return is_verbose


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


def check_colab_environment() -> bool:
    """
    Check if running in Colab and display warning if not.
    
    Returns:
        True if running in Colab, False otherwise
    """
    verbose = get_verbosity()
    is_colab = is_colab_environment()
    
    if is_colab:
        if verbose:
            log_setup("Running in Google Colab", 'info', verbose)
        return True
    else:
        log_setup("Not running in Google Colab", 'warning', verbose)
        try:
            from IPython.display import display, HTML
            # load_template is defined later in this module, but will be available at runtime
            warning_html = load_template('warning_non_colab.html')
            display(HTML(warning_html))
        except (ImportError, NameError, FileNotFoundError, Exception) as e:
            # If template loading fails, at least log the warning
            if verbose:
                log_setup(f"Could not display Colab warning: {e}", 'warning', verbose)
        return False


def show_restart_warning() -> None:
    """
    Display restart warning and halt notebook execution.
    
    This function loads the restart warning template, displays it, logs debug
    messages, and raises a StopExecution exception to halt notebook execution.
    The user must restart the runtime and re-run the setup cell.
    
    Raises:
        StopExecution: Always raises this exception to halt execution
    """
    verbose = get_verbosity()
    log_setup("Runtime restart required after bitsandbytes update", 'warning', verbose)
    
    try:
        from IPython.display import display, HTML
        # Load and display restart warning template
        warning_html = load_template('restart_warning.html')
        display(HTML(warning_html))
        log_setup("Restart warning displayed", 'info', verbose)
    except (ImportError, NameError, FileNotFoundError, Exception) as e:
        # If template loading fails, log error but still halt execution
        log_setup(f"Could not display restart warning: {e}", 'error', verbose)
        log_setup("Please restart the runtime manually", 'warning', verbose)
    
    # Halt execution with custom exception that suppresses traceback
    class StopExecution(Exception):
        """Custom exception to halt notebook execution without showing traceback."""
        def _render_traceback_(self):
            pass
    
    log_setup("Halting execution - restart required", 'important', verbose)
    raise StopExecution


# ============================================================================
# Helper Functions
# ============================================================================

def _get_device() -> str:
    """Get the appropriate device (cuda or cpu)."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_compute_dtype():
    """Get the appropriate compute dtype based on device availability."""
    try:
        import torch
        return torch.float16 if torch.cuda.is_available() else torch.float32
    except ImportError:
        return None  # Will be handled by calling code


def _create_quantization_config(quantization: Optional[str], compute_dtype) -> Optional[Any]:
    """
    Create quantization configuration if requested.
    
    Input:
        quantization: "4-bit" or None
        compute_dtype: torch dtype for computation
    
    Output:
        BitsAndBytesConfig or None
    """
    if quantization != "4-bit":
        return None
    
    try:
        import bitsandbytes  # noqa: F401
        import importlib.metadata
        from packaging import version as pkg_version
        from transformers import BitsAndBytesConfig
        
        # Verify bitsandbytes version is >= 0.43.1 (required for 4-bit quantization)
        try:
            bnb_version = importlib.metadata.version("bitsandbytes")
            if pkg_version.parse(bnb_version) < pkg_version.parse("0.43.1"):
                raise ImportError(
                    f"bitsandbytes version {bnb_version} is too old. "
                    f"Version >= 0.43.1 is required for 4-bit quantization. "
                    f"Please upgrade: pip install -U bitsandbytes"
                )
        except Exception as version_error:
            # If version check fails, still try to use bitsandbytes
            # (transformers will do its own validation)
            warnings.warn(
                f"Could not verify bitsandbytes version: {version_error}. "
                f"Proceeding anyway - transformers will validate compatibility."
            )
        
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    except ImportError as e:
        error_msg = str(e)
        if "bitsandbytes" in error_msg.lower() or "version" in error_msg.lower():
            if "requires the latest version" in error_msg or "0.43.1" in error_msg:
                warnings.warn(
                    f"\n{'='*60}\n"
                    f"bitsandbytes version validation failed: {error_msg}\n"
                    f"{'='*60}\n"
                    f"This may happen if transformers can't detect the installed bitsandbytes version.\n"
                    f"\nTo fix (try in order):\n"
                    f"  1. First, try reloading: !pip install --force-reinstall bitsandbytes\n"
                    f"  2. If that doesn't work, restart runtime (Runtime → Restart runtime)\n"
                    f"     NOTE: Restarting clears Python variables/models but keeps installed packages\n"
                    f"  3. After restart, re-run the 'Setup Environment' cell\n"
                    f"  4. If still failing, use quantization='None' to disable 4-bit quantization\n"
                    f"{'='*60}\n"
                    f"4-bit quantization will be disabled for now.\n"
                )
            else:
                warnings.warn(f"bitsandbytes issue: {error_msg}. 4-bit quantization disabled.")
        else:
            warnings.warn("bitsandbytes not available. 4-bit quantization disabled.")
        return None


@lru_cache(maxsize=3)
def ensure_llm(model_id: str, quantization: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Load and cache a language model with optional quantization.
    
    Input:
        model_id: Hugging Face model identifier
        quantization: "4-bit" or None
    
    Output:
        Tuple of (tokenizer, model)
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    compute_dtype = _get_compute_dtype()
    
    # If using 4-bit quantization, ensure bitsandbytes is imported and available
    # This helps transformers detect it properly
    if quantization == "4-bit":
        try:
            import bitsandbytes  # noqa: F401
            import importlib
            # Force reload to ensure transformers can detect it
            if 'bitsandbytes' in sys.modules:
                importlib.reload(bitsandbytes)
        except Exception as e:
            warnings.warn(f"Could not import bitsandbytes: {e}. 4-bit quantization may fail.")
    
    quant_cfg = _create_quantization_config(quantization, compute_dtype)
    device_map = "auto" if torch.cuda.is_available() else None
    
    verbose = get_verbosity()
    _suppress_torch_dtype_warnings()  # Suppress deprecation warning in non-debug mode
    log_setup(f"[LLM] Loading tokenizer for {model_id}...", 'info', verbose)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    log_setup("[LLM] Tokenizer loaded.", 'info', verbose)
    
    quant_str = f" ({quantization})" if quantization else ""
    log_setup(f"[LLM] Loading model{quant_str} (this may take a while for large models)...", 'info', verbose)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_cfg,
            device_map=device_map,
            dtype=compute_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        log_setup("[LLM] Model loaded.", 'success', verbose)
    except ImportError as e:
        if "bitsandbytes" in str(e) and "0.43.1" in str(e):
            # Transformers version check failed - try workaround
            log_setup("[LLM] WARNING: Transformers version check failed. Attempting workaround...", 'warning', verbose)
            log_setup("[LLM] This might be a false positive if bitsandbytes 0.48.2 is installed.", 'warning', verbose)
            log_setup("[LLM] Trying to load without explicit quantization config...", 'info', verbose)
            # Try loading without quantization config and let transformers handle it
            try:
                import bitsandbytes as bnb
                # Force import to ensure it's available
                _ = bnb.__version__
                # Retry with quantization config
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quant_cfg,
                    device_map=device_map,
                    dtype=compute_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                )
                log_setup("[LLM] Model loaded (workaround succeeded).", 'success', verbose)
            except Exception as e2:
                raise ImportError(
                    f"Failed to load model with 4-bit quantization: {e}\n"
                    f"Workaround also failed: {e2}\n"
                    f"Please try:\n"
                    f"  1. Verify bitsandbytes is installed: !pip show bitsandbytes\n"
                    f"  2. If version < 0.43.1, upgrade: !pip install -U bitsandbytes\n"
                    f"  3. If version is correct, you may need to restart the runtime\n"
                    f"     (Runtime → Restart runtime, then re-run setup cell)\n"
                    f"  4. Alternatively, use quantization='None' to disable 4-bit quantization"
                ) from e
        else:
            raise
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return tokenizer, model


def chunk_text(text: str, max_chars: int) -> List[str]:
    """
    Split text into chunks of maximum character length.
    
    Input:
        text: Text to chunk
        max_chars: Maximum characters per chunk
    
    Output:
        List of text chunks
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        start = end
    return chunks


def _collect_performance_metrics(stage_name: str, start_time: float, start_vram: float = 0.0) -> Dict[str, Any]:
    """
    Collect performance metrics for a stage.
    
    Input:
        stage_name: Name of the stage
        start_time: Start time (from time.time())
        start_vram: Starting VRAM usage in GB
    
    Output:
        Dictionary with performance metrics
    """
    end_time = time.time()
    time_seconds = end_time - start_time
    
    metrics = {"time_seconds": time_seconds}
    
    # VRAM usage
    try:
        import torch
        if torch.cuda.is_available():
            end_vram = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            metrics["peak_vram_gb"] = peak_vram
            metrics["end_vram_gb"] = end_vram
        else:
            metrics["peak_vram_gb"] = 0.0
    except ImportError:
        metrics["peak_vram_gb"] = 0.0
    
    # CPU usage (if psutil available)
    try:
        import psutil
        metrics["avg_cpu_percent"] = psutil.cpu_percent(interval=0.1)
    except ImportError:
        metrics["avg_cpu_percent"] = 0.0
    
    return metrics


# ============================================================================
# Pipeline Stage Functions
# ============================================================================

def load_and_preprocess_audio(audio_path: str) -> Tuple[Any, int]:
    """
    Load audio from file, resample to 16kHz, convert to mono, and normalize.
    
    Input:
        audio_path: Path to the audio file
    
    Output:
        Tuple of (processed_audio_tensor, sample_rate)
    """
    import torch
    import torchaudio
    import soundfile as sf
    
    try:
        audio_waveform, audio_sample_rate = sf.read(audio_path, dtype='float32')
    except Exception:
        import librosa
        audio_waveform, audio_sample_rate = librosa.load(audio_path, sr=None, mono=False)
    
    waveform = torch.as_tensor(audio_waveform)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [1, frames]
    elif waveform.dim() == 2:
        waveform = waveform.transpose(0, 1)  # [channels, frames]
    
    waveform = waveform.to(torch.float32)
    orig_sr = int(audio_sample_rate)
    
    # Downmix to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample to 16 kHz
    target_sr = 16000
    if orig_sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, target_sr)
    else:
        target_sr = orig_sr
    
    # Peak normalize to [-1, 1]
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    
    return waveform.contiguous(), target_sr


def transcribe_audio(audio_tensor: Any, sample_rate: int, config: Dict[str, Any], progress_callback: Optional[Callable[[int, str, int], None]] = None) -> str:
    """
    Transcribe audio using Whisper ASR model with native long-form transcription.

    Uses Whisper's built-in long-form transcription mechanism (Whisper paper, section 3.8)
    via return_timestamps=True, which activates the model's internal windowing and stitching.

    Input:
        audio_tensor: Preprocessed audio tensor [1, frames]
        sample_rate: Sample rate of the audio
        config: SOLAS_CONFIG dictionary

    Output:
        Transcribed text (str)
    """
    import torch
    import numpy as np
    from transformers import pipeline

    device = _get_device()
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    asr_model_id = config["asr_model_id"]
    source_language = config.get("source_language", "auto")

    # Convert language name to code if needed (Whisper expects codes)
    if source_language != "auto":
        lang_name_to_code = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Russian": "ru",
            "Chinese": "zh",
            "Japanese": "ja",
            "Korean": "ko",
        }
        source_language = lang_name_to_code.get(source_language, source_language)

    verbose = get_verbosity()
    _suppress_torch_dtype_warnings()  # Suppress deprecation warning in non-debug mode

    # Process audio
    audio_np = audio_tensor.squeeze(0).cpu().numpy().astype(np.float32)
    audio_duration_seconds = audio_np.shape[0] / sample_rate

    log_setup(f"[ASR] Loading model: {asr_model_id}...", 'info', verbose)
    log_setup(f"[ASR] Audio duration: {audio_duration_seconds:.1f}s", 'info', verbose)

    # Suppress transformers warnings in non-verbose mode
    import os
    original_verbosity = os.environ.get("TRANSFORMERS_VERBOSITY", "warning")
    if not verbose:
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    # Use pipeline with Whisper's native long-form transcription
    # return_timestamps=True activates Whisper's internal chunking mechanism
    pipe = pipeline(
        "automatic-speech-recognition",
        model=asr_model_id,
        dtype=dtype,  # Use 'dtype' instead of deprecated 'torch_dtype'
        device=device,
        model_kwargs={"use_safetensors": True}
    )

    # Restore original verbosity
    if not verbose:
        os.environ["TRANSFORMERS_VERBOSITY"] = original_verbosity

    log_setup(f"[ASR] Model loaded on {device}.", 'info', verbose)

    # Prepare generation kwargs
    generate_kwargs = {"task": "transcribe"}
    if source_language != "auto":
        generate_kwargs["language"] = source_language

    # For long audio, inform user we're using native long-form transcription
    if audio_duration_seconds > 30:
        log_setup(f"[ASR] Long audio detected, using Whisper's native long-form transcription...", 'info', verbose)

    # Create progress indicator
    estimated_processing_time = max(5.0, audio_duration_seconds / 20.0)
    start_time = time.time()

    # Setup progress tracking (works in both verbose and non-verbose modes)
    import threading
    stop_flag = threading.Event()
    progress_thread = None

    if progress_callback or verbose:
        if verbose:
            # Verbose mode: use tqdm
            try:
                from tqdm.std import tqdm
                pbar = tqdm(total=100, desc="Transcribing", unit="%", ncols=80)

                def update_progress_verbose():
                    while not stop_flag.is_set():
                        elapsed = time.time() - start_time
                        progress = min(95, int((elapsed / estimated_processing_time) * 100))
                        pbar.update(progress - pbar.n)
                        if progress_callback:
                            progress_callback(2, "Transcribing audio (ASR)", progress)
                        if elapsed >= estimated_processing_time * 1.5:
                            break
                        time.sleep(0.5)
                    pbar.update(100 - pbar.n)
                    pbar.close()

                progress_thread = threading.Thread(target=update_progress_verbose, daemon=True)
                progress_thread.start()
            except ImportError:
                pass
        elif progress_callback:
            # Non-verbose mode: update callback only
            def update_progress_callback():
                while not stop_flag.is_set():
                    elapsed = time.time() - start_time
                    progress = min(95, int((elapsed / estimated_processing_time) * 100))
                    progress_callback(2, "Transcribing audio (ASR)", progress)
                    if elapsed >= estimated_processing_time * 1.5:
                        break
                    time.sleep(0.3)

            progress_thread = threading.Thread(target=update_progress_callback, daemon=True)
            progress_thread.start()

    log_setup(f"[ASR] Transcribing audio ({audio_duration_seconds:.1f}s duration)...", 'info', verbose)

    # Run transcription using Whisper's native long-form mode
    # return_timestamps=True enables internal windowing and stitching (no chunk_length_s needed)
    result = pipe(
        audio_np,
        generate_kwargs=generate_kwargs,
        return_timestamps=True  # Activates Whisper's built-in long-form transcription
    )

    # Stop progress thread and report completion
    if progress_thread is not None:
        stop_flag.set()
        if progress_thread.is_alive():
            progress_thread.join(timeout=1.0)

    if progress_callback:
        progress_callback(2, "Transcribing audio (ASR)", 100)

    # Extract transcript (timestamps are included in chunks, we concatenate the text)
    if "chunks" in result:
        # Long-form transcription with timestamps returns chunks
        transcript = " ".join([chunk["text"] for chunk in result["chunks"]]).strip()
    else:
        # Fallback for short audio
        transcript = result["text"].strip()

    actual_time = time.time() - start_time
    speed_factor = audio_duration_seconds / actual_time if actual_time > 0 else 0
    log_setup(f"[ASR] Transcription complete ({actual_time:.1f}s, {speed_factor:.1f}x real-time)", 'success', verbose)
    log_setup(f"[ASR] Transcript length: {len(transcript)} characters", 'info', verbose)

    # Cleanup
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return transcript


def translate_transcript(transcript: str, config: Dict[str, Any], progress_callback: Optional[Callable[[int, str, int], None]] = None) -> str:
    """
    Translate transcript to target language using LLM.
    
    Input:
        transcript: Original transcript text
        config: SOLAS_CONFIG dictionary
    
    Output:
        Translated text (str)
    """
    llm_model_id = config["llm_model_id"]
    quantization = config.get("quantization")
    chunk_size_chars = config.get("chunk_size_chars", 2000)
    # Default to 1.2 only if key is absent; allow explicit None for no penalty
    repetition_penalty = config["repetition_penalty"] if "repetition_penalty" in config else 1.2
    target_language = config.get("target_language", "Portuguese")
    max_new_tokens = config.get("translation_max_new_tokens", 1024)
    
    verbose = get_verbosity()
    _suppress_generation_flags_warnings()  # Suppress invalid generation flags warnings
    log_setup(f"[Translation] Loading LLM: {llm_model_id}...", 'info', verbose)
    tokenizer, model = ensure_llm(llm_model_id, quantization)

    chunks = chunk_text(transcript, chunk_size_chars)
    verbose = get_verbosity()
    log_setup(f"Processing {len(chunks)} chunk(s)...", 'info', verbose)
    translated_parts = []

    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        if len(chunks) > 1:
            log_setup(f"Processing chunk {i}/{len(chunks)}...", 'info', verbose)

        # Report progress before processing chunk
        if progress_callback:
            progress = int((i - 1) / total_chunks * 100)
            progress_callback(3, "Translating transcript", progress)
        
        # Calculate dynamic max_new_tokens based on input length
        # Translation can be up to 1.5x longer than original, so ensure we have enough tokens
        chunk_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
        # Use at least max_new_tokens, but scale up if chunk is large
        dynamic_max_tokens = max(max_new_tokens, int(chunk_tokens * 1.5))
        
        gen_kwargs = {
            "max_new_tokens": dynamic_max_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty
        system_prompt = (
            f"You are a professional technical translator. Translate the provided transcript into {target_language}. "
            "Be faithful, precise, and complete. Preserve technical terms whenever possible. "
            "Do not summarize, omit, or add content. Do not include explanations. Output only the translation."
        )
        user_prompt = (
            f"Translate the following transcript segment into {target_language}. "
            "Only output the translation, nothing else.\n\n"
            f"{chunk}"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception:
            fallback_prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
            input_ids = tokenizer(fallback_prompt, return_tensors="pt").input_ids
        
        import torch
        model_device = next(model.parameters()).device
        input_ids = input_ids.to(model_device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model_device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )
        
        gen_only = output_ids[0, input_ids.shape[1]:]
        translated = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
        
        # Debug logging to check if translation is complete
        log_setup(f"Chunk {i}: Input {len(chunk)} chars ({chunk_tokens} tokens), Output {len(translated)} chars", 'info', verbose)
        translated_parts.append(translated)
    
    return "\n".join(translated_parts).strip()


def summarize_text(translated_text: str, config: Dict[str, Any], progress_callback: Optional[Callable[[int, str, int], None]] = None) -> str:
    """
    Generate key points summary from translated text.

    Input:
        translated_text: Translated transcript text
        config: SOLAS_CONFIG dictionary

    Output:
        Key points summary as markdown bullets (str)
    """
    llm_model_id = config["llm_model_id"]
    quantization = config.get("quantization")
    chunk_size_chars = config.get("chunk_size_chars", 2000)
    # Default to 1.2 only if key is absent; allow explicit None for no penalty
    repetition_penalty = config["repetition_penalty"] if "repetition_penalty" in config else 1.2
    summary_mode = config.get("summary_mode", "greedy")
    max_new_tokens = config.get("summary_max_new_tokens", 512)
    target_language = config.get("target_language", "English")
    
    verbose = get_verbosity()
    log_setup(f"[Summary] Loading LLM: {llm_model_id}...", 'info', verbose)
    tokenizer, model = ensure_llm(llm_model_id, quantization)
    
    chunks = chunk_text(translated_text, chunk_size_chars)
    partial_bullets = []
    
    # Suppress generation flags warnings in non-debug mode
    _suppress_generation_flags_warnings()
    
    # Chunk-level generation kwargs
    from transformers import GenerationConfig

    # Build base config dict, conditionally adding repetition_penalty
    base_config = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if repetition_penalty is not None:
        base_config["repetition_penalty"] = repetition_penalty

    if summary_mode == "sampled":
        # Use GenerationConfig for all generation parameters to avoid warnings
        gen_config = GenerationConfig(
            **base_config,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
        )
        gen_chunk_kwargs = {
            "generation_config": gen_config
        }
    else:
        # Use GenerationConfig for greedy mode too to avoid model default config issues
        gen_config = GenerationConfig(
            **base_config,
            do_sample=False,
        )
        gen_chunk_kwargs = {
            "generation_config": gen_config
        }

    # Aggregation generation kwargs
    if summary_mode == "hybrid":
        gen_config_agg = GenerationConfig(
            **base_config,
            do_sample=False,
        )
        gen_agg_kwargs = {
            "generation_config": gen_config_agg
        }
    else:
        # Make a copy to avoid modifying the original
        gen_agg_kwargs = gen_chunk_kwargs.copy()
    
    chunk_prompt_header = (
        f"Extract ONLY the most important key points from the transcript segment below in {target_language}. "
        "Focus on core concepts, definitions, equations, conclusions, and actionable takeaways. "
        f"Write a concise markdown bulleted list (use '-' bullets) in {target_language}. Do not include any preamble or epilogue."
    )

    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        messages = [
            {"role": "system", "content": f"You are an expert technical summarizer. Provide summaries in {target_language}."},
            {"role": "user", "content": f"{chunk_prompt_header}\n\nTranscript segment:\n\n{chunk}"},
        ]

        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception:
            fallback_prompt = (
                f"System: You are an expert technical summarizer.\n"
                f"User: {chunk_prompt_header}\n\nTranscript segment:\n\n{chunk}\nAssistant:"
            )
            input_ids = tokenizer(fallback_prompt, return_tensors="pt").input_ids

        import torch
        model_device = next(model.parameters()).device
        input_ids = input_ids.to(model_device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_chunk_kwargs
            )

        gen_only = output_ids[0, input_ids.shape[1]:]
        partial = tokenizer.decode(gen_only, skip_special_tokens=True).strip()

        # Normalize bullets
        norm_lines = []
        for line in partial.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("- "):
                s = s[2:]
            s = s.replace("**Key Point**:", "").replace("**Key Point**", "").lstrip("- ").strip()
            norm_lines.append(f"- {s}")
        partial_bullets.append("\n".join(norm_lines))

        # Report progress (chunks phase is ~70% of total, aggregation is ~30%)
        if progress_callback:
            chunk_progress = int((i + 1) / total_chunks * 70)
            progress_callback(4, "Generating key points summary", chunk_progress)
    
    # Aggregation
    if progress_callback:
        progress_callback(4, "Generating key points summary", 75)

    aggregation_prompt = (
        f"You will be given multiple markdown bullet lists produced from segments of the same lecture transcript in {target_language}. "
        "Merge and deduplicate them into a single, concise, well-structured set of bullets. "
        f"Prioritize clarity and completeness, keep only the most salient points, and maintain markdown '-' bullets in {target_language}."
    )

    merged_input = "\n\n".join(partial_bullets)
    messages = [
        {"role": "system", "content": f"You are an expert editor and summarizer. Work in {target_language}."},
        {"role": "user", "content": f"{aggregation_prompt}\n\nHere are the partial bullet lists:\n\n{merged_input}"},
    ]
    
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except Exception:
        fallback_prompt = (
            f"System: You are an expert editor and summarizer.\n"
            f"User: {aggregation_prompt}\n\nHere are the partial bullet lists:\n\n{merged_input}\nAssistant:"
        )
        input_ids = tokenizer(fallback_prompt, return_tensors="pt").input_ids
    
    import torch
    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_agg_kwargs
        )
    
    gen_only = output_ids[0, input_ids.shape[1]:]
    summary = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
    
    # Normalize final bullets
    final_lines = []
    for line in summary.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("- "):
            s = s[2:]
        s = s.replace("**Key Point**:", "").replace("**Key Point**", "").lstrip("- ").strip()
        final_lines.append(f"- {s}")
    
    return "\n".join(final_lines)


def generate_podcast_script(translated_text: str, summary: str, config: Dict[str, Any], progress_callback: Optional[Callable[[int, str, int], None]] = None) -> str:
    """
    Generate conversational podcast script from translated transcript and summary.
    Uses chunking for long transcripts to stay within context limits.

    Input:
        translated_text: Translated transcript
        summary: Key points summary
        config: SOLAS_CONFIG dictionary
        progress_callback: Optional callback for progress updates

    Output:
        Podcast script with Host A/B dialogue (str)
    """
    llm_model_id = config["llm_model_id"]
    quantization = config.get("quantization")
    podcast_creativity_temp = config.get("podcast_creativity_temp", 0.3)
    max_new_tokens = config.get("podcast_max_new_tokens", 1024)
    # Default to 1.2 only if key is absent; allow explicit None for no penalty
    repetition_penalty = config["repetition_penalty"] if "repetition_penalty" in config else 1.2
    target_language = config.get("target_language", "Portuguese")
    chunk_size_chars = config.get("chunk_size_chars", 2000)

    verbose = get_verbosity()
    _suppress_generation_flags_warnings()  # Suppress invalid generation flags warnings
    log_setup(f"[Podcast Script] Loading LLM: {llm_model_id}...", 'info', verbose)
    tokenizer, model = ensure_llm(llm_model_id, quantization)

    # Chunk the transcript if it's too long
    chunks = chunk_text(translated_text.strip(), chunk_size_chars)
    total_chunks = len(chunks)
    log_setup(f"[Podcast Script] Processing {total_chunks} segment(s)...", 'info', verbose)

    system_prompt = (
        f"You are a scriptwriter for an educational podcast. Create a two-host conversation in {target_language} that is engaging, "
        "technically accurate, and pedagogically effective. Use plain language and helpful analogies. "
        "The dialogue must include clear speaker labels at the start of each line: 'Host A:' and 'Host B:'. "
        "Avoid meta commentary, stage directions, or non-dialogue text. "
        f"All dialogue content must be written in {target_language}."
    )

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    import torch
    model_device = next(model.parameters()).device

    script_segments = []
    previous_context = ""

    for i, chunk in enumerate(chunks):
        if total_chunks > 1:
            log_setup(f"[Podcast Script] Processing segment {i+1}/{total_chunks}...", 'info', verbose)

        # Report progress
        if progress_callback:
            progress = int((i / total_chunks) * 85)  # 0-85% for generation, 85-100% for stitching
            progress_callback(5, "Generating podcast script", progress)

        # Build prompt for this chunk
        if i == 0:
            # First chunk: introduce the podcast
            user_prompt = (
                f"Write the BEGINNING of a podcast script in {target_language}.\n"
                "Start with a brief introduction where Host A welcomes listeners and introduces the topic.\n"
                f"Use the transcript segment and summary below to create the opening discussion.\n\n"
                f"Summary (for structure):\n{summary}\n\n"
                f"Transcript segment:\n{chunk}\n\n"
                "Requirements:\n"
                "- Host A explains; Host B asks concise, clarifying questions.\n"
                f"- All dialogue must be in {target_language}.\n"
                "- Strictly output alternating lines beginning with 'Host A:' or 'Host B:' and nothing else."
            )
        elif i == total_chunks - 1:
            # Last chunk: conclude the podcast
            user_prompt = (
                f"Continue and CONCLUDE the podcast script in {target_language}.\n"
                f"Previous context:\n{previous_context}\n\n"
                f"Transcript segment:\n{chunk}\n\n"
                "Requirements:\n"
                "- Continue naturally from the previous dialogue.\n"
                "- Cover the remaining content and wrap up with a brief conclusion.\n"
                "- Host A thanks listeners; Host B adds a final thought.\n"
                f"- All dialogue must be in {target_language}.\n"
                "- Strictly output alternating lines beginning with 'Host A:' or 'Host B:' and nothing else."
            )
        else:
            # Middle chunks: continue the conversation
            user_prompt = (
                f"Continue the podcast script in {target_language}.\n"
                f"Previous context:\n{previous_context}\n\n"
                f"Transcript segment:\n{chunk}\n\n"
                "Requirements:\n"
                "- Continue naturally from the previous dialogue.\n"
                "- Cover the new content in this segment.\n"
                f"- All dialogue must be in {target_language}.\n"
                "- Strictly output alternating lines beginning with 'Host A:' or 'Host B:' and nothing else."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception:
            fallback_prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
            input_ids = tokenizer(fallback_prompt, return_tensors="pt").input_ids

        input_ids = input_ids.to(model_device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model_device)

        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

        gen_only = output_ids[0, input_ids.shape[1]:]
        segment_script = tokenizer.decode(gen_only, skip_special_tokens=True).strip()

        # Enforce label format for this segment
        lines = [line.strip() for line in segment_script.splitlines() if line.strip()]
        fixed_lines = []
        expected = "Host A:"
        for line in lines:
            if line.startswith("Host A:") or line.startswith("Host B:"):
                fixed_lines.append(line)
                expected = "Host B:" if line.startswith("Host A:") else "Host A:"
            else:
                fixed_lines.append(f"{expected} {line}")
                expected = "Host B:" if expected == "Host A:" else "Host A:"

        segment_text = "\n".join(fixed_lines)
        script_segments.append(segment_text)

        # Keep last 2-3 lines as context for next chunk
        context_lines = fixed_lines[-3:] if len(fixed_lines) >= 3 else fixed_lines
        previous_context = "\n".join(context_lines)

    # Combine all segments
    if progress_callback:
        progress_callback(5, "Generating podcast script", 90)

    full_script = "\n".join(script_segments)

    # Final cleanup pass: ensure alternating speakers throughout
    if progress_callback:
        progress_callback(5, "Generating podcast script", 95)

    all_lines = [line.strip() for line in full_script.splitlines() if line.strip()]
    final_lines = []
    expected = "Host A:"

    for line in all_lines:
        if line.startswith("Host A:") or line.startswith("Host B:"):
            final_lines.append(line)
            expected = "Host B:" if line.startswith("Host A:") else "Host A:"
        else:
            final_lines.append(f"{expected} {line}")
            expected = "Host B:" if expected == "Host A:" else "Host A:"

    return "\n".join(final_lines)


def synthesize_podcast(script: str, config: Dict[str, Any], progress_callback: Optional[Callable[[int, str, int], None]] = None) -> str:
    """
    Synthesize podcast audio from script using Coqui XTTS.
    
    Input:
        script: Podcast script with Host A/B labels
        config: SOLAS_CONFIG dictionary
    
    Output:
        Path to generated podcast audio file (str)
    """
    tts_model_id = config.get("tts_model_id", "tts_models/multilingual/multi-dataset/xtts_v2")
    host_a_wav_path = config.get("host_a_wav_path")
    host_b_wav_path = config.get("host_b_wav_path")
    target_language = config.get("target_language", "Portuguese")
    output_directory = config.get("output_directory", "/content/solas_outputs")
    
    # Language code mapping
    lang_mapping = {
        "English": "en",
        "Portuguese": "pt",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
    }
    lang_code = lang_mapping.get(target_language, "en")
    
    device = _get_device()
    # Automatically accept Coqui TTS terms of service to avoid interactive prompt
    os.environ["COQUI_TOS_AGREED"] = "1"
    
    # Suppress jieba SyntaxWarnings BEFORE importing TTS (which imports jieba)
    verbose = get_verbosity()
    if not verbose:
        # Suppress all SyntaxWarnings (jieba has invalid escape sequences in regex patterns)
        warnings.filterwarnings("ignore", category=SyntaxWarning)

    # Suppress TTS warnings
    _suppress_tts_warnings()

    # Install torchcodec if not available (required by torchaudio for some audio loading operations)

    # torchcodec is needed when torchaudio tries to use it as a backend
    try:
        import torchcodec
    except ImportError:
        try:
            import subprocess
            import sys
            log_setup("[TTS] Installing torchcodec (required for audio loading)...", 'info', verbose)
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "torchcodec", "-q"],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                import torchcodec
                log_setup("[TTS] ✓ torchcodec installed successfully", 'success', verbose)
            else:
                log_setup(f"[TTS] ⚠ Could not install torchcodec: {result.stderr}", 'warning', verbose)
                log_setup("[TTS] Attempting to continue with soundfile backend...", 'info', verbose)
        except subprocess.TimeoutExpired:
            log_setup("[TTS] ⚠ torchcodec installation timed out", 'warning', verbose)
        except Exception as e:
            log_setup(f"[TTS] ⚠ Could not install torchcodec: {e}", 'warning', verbose)
            log_setup("[TTS] Attempting to continue with soundfile backend...", 'info', verbose)

    # Lazy import TTS - only import when this function is called
    # Suppress TTS initialization output ("Device set to use cuda") in non-verbose mode
    with _SuppressOutput(suppress=not verbose):
        from TTS.api import TTS as COQUI_TTS
        tts_engine = COQUI_TTS(model_name=tts_model_id)
        try:
            tts_engine = tts_engine.to(device)
        except Exception:
            pass
    
    # Determine output sample rate
    output_sample_rate = getattr(tts_engine, "output_sample_rate", 24000)

    # Helper function to chunk text into segments <= 250 characters
    def chunk_text(text, max_length=250):
        """Split text into chunks at sentence boundaries, respecting max_length."""
        if len(text) <= max_length:
            return [text]

        chunks = []
        import re
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""
        for sentence in sentences:
            # If a single sentence is longer than max_length, split it at commas or spaces
            if len(sentence) > max_length:
                # First, add any accumulated chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Split long sentence at commas
                parts = re.split(r',\s+', sentence)
                for part in parts:
                    if len(part) > max_length:
                        # Split at spaces as last resort
                        words = part.split()
                        temp = ""
                        for word in words:
                            if len(temp) + len(word) + 1 <= max_length:
                                temp = f"{temp} {word}".strip()
                            else:
                                if temp:
                                    chunks.append(temp)
                                temp = word
                        if temp:
                            chunks.append(temp)
                    else:
                        if len(current_chunk) + len(part) + 2 <= max_length:
                            current_chunk = f"{current_chunk}, {part}".strip(", ")
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = part
            else:
                # Check if adding this sentence would exceed max_length
                if len(current_chunk) + len(sentence) + 1 <= max_length:
                    current_chunk = f"{current_chunk} {sentence}".strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    import numpy as np
    lines = [line.strip() for line in script.splitlines() if line.strip()]
    segments = []

    # Calculate total chunks for progress tracking
    total_chunks = 0
    for line in lines:
        text = line.split(":", 1)[1].strip() if ":" in line else line
        if text:
            total_chunks += len(chunk_text(text, max_length=250))

    processed_chunks = 0
    for line in lines:
        speaker = "A" if line.startswith("Host A:") else ("B" if line.startswith("Host B:") else None)
        text = line.split(":", 1)[1].strip() if ":" in line else line
        if not text:
            continue

        speaker_wav = None
        if speaker == "A" and host_a_wav_path and Path(host_a_wav_path).exists():
            speaker_wav = host_a_wav_path
        elif speaker == "B" and host_b_wav_path and Path(host_b_wav_path).exists():
            speaker_wav = host_b_wav_path

        # Chunk text to respect XTTS 250-character limit
        text_chunks = chunk_text(text, max_length=250)

        for chunk in text_chunks:
            try:
                wav = tts_engine.tts(text=chunk, language=lang_code, speaker_wav=speaker_wav)
            except TypeError:
                wav = tts_engine.tts(text=chunk, speaker_wav=speaker_wav, language=lang_code)

            wav = np.asarray(wav, dtype=np.float32)
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            segments.append(wav)

            # Report progress
            processed_chunks += 1
            if progress_callback and total_chunks > 0:
                progress = int((processed_chunks / total_chunks) * 100)
                progress_callback(6, "Synthesizing podcast audio (TTS)", progress)
    
    if not segments:
        raise RuntimeError("No audio segments were generated.")
    
    # Concatenate and save
    import soundfile as sf
    final_audio = np.concatenate(segments)
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = str(out_dir / "podcast.wav")
    sf.write(final_path, final_audio, output_sample_rate, subtype="PCM_16")
    
    return final_path


# ============================================================================
# Main Pipeline Function
# ============================================================================

def run_pipeline(config: Dict[str, Any], progress_callback: Optional[Callable[[int, str, int], None]] = None) -> Dict[str, Any]:
    """
    Execute the full SOLAS pipeline.
    
    Input:
        config: SOLAS_CONFIG dictionary with all parameters
        progress_callback: Optional callback function(stage_num, stage_name, progress_pct) to update progress
    
    Output:
        Dictionary containing:
        - text_outputs: All generated text artifacts
        - file_paths: Paths to saved files
        - performance_metrics: Timing and resource usage for each stage
    """
    verbose = get_verbosity()

    # Suppress warnings and progress bars early, before any model loading
    if not verbose:
        # Suppress all SyntaxWarnings (usually from third-party libs like jieba with invalid escape sequences)
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        # Suppress Hugging Face Hub download progress bars
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        # Suppress transformers verbosity
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        # Disable tqdm globally to suppress all progress bars
        os.environ["TQDM_DISABLE"] = "1"

    # Suppress generation flags warnings
    _suppress_generation_flags_warnings()
    
    pipeline_start_time = time.time()
    performance_metrics = {}
    
    # Stage 1: Load and preprocess audio
    log_setup("Stage 1/6: Loading and preprocessing audio...", 'info', verbose)
    if progress_callback:
        progress_callback(1, "Loading and preprocessing audio", 0)
    stage_start = time.time()
    audio_tensor, sample_rate = load_and_preprocess_audio(config["input_audio_path"])
    performance_metrics["audio_preprocessing"] = _collect_performance_metrics("audio_preprocessing", stage_start)
    if progress_callback:
        progress_callback(1, "Loading and preprocessing audio", 100)
    log_setup(f"✓ Stage 1 complete ({performance_metrics['audio_preprocessing']['time_seconds']:.2f}s)", 'success', verbose)
    
    # Stage 2: ASR transcription
    log_setup("Stage 2/6: Transcribing audio (ASR)...", 'info', verbose)
    if progress_callback:
        progress_callback(2, "Transcribing audio (ASR)", 0)
    stage_start = time.time()
    original_transcript = transcribe_audio(audio_tensor, sample_rate, config, progress_callback)
    performance_metrics["asr"] = _collect_performance_metrics("asr", stage_start)
    log_setup(f"✓ Stage 2 complete ({performance_metrics['asr']['time_seconds']:.2f}s)", 'success', verbose)
    log_setup(f"Transcript length: {len(original_transcript)} characters", 'info', verbose)
    
    # Stage 3: Translation
    log_setup("Stage 3/6: Translating transcript...", 'info', verbose)
    if progress_callback:
        progress_callback(3, "Translating transcript", 0)
    stage_start = time.time()
    translated_transcript = translate_transcript(original_transcript, config, progress_callback)
    performance_metrics["translation"] = _collect_performance_metrics("translation", stage_start)
    if progress_callback:
        progress_callback(3, "Translating transcript", 100)
    log_setup(f"✓ Stage 3 complete ({performance_metrics['translation']['time_seconds']:.2f}s)", 'success', verbose)
    
    # Stage 4: Summarization
    log_setup("Stage 4/6: Generating key points summary...", 'info', verbose)
    if progress_callback:
        progress_callback(4, "Generating key points summary", 0)
    stage_start = time.time()
    key_points_summary = summarize_text(translated_transcript, config, progress_callback)
    performance_metrics["summary"] = _collect_performance_metrics("summary", stage_start)
    if progress_callback:
        progress_callback(4, "Generating key points summary", 100)
    log_setup(f"✓ Stage 4 complete ({performance_metrics['summary']['time_seconds']:.2f}s)", 'success', verbose)
    
    # Stage 5: Podcast script generation
    log_setup("Stage 5/6: Generating podcast script...", 'info', verbose)
    if progress_callback:
        progress_callback(5, "Generating podcast script", 0)
    stage_start = time.time()
    podcast_script = generate_podcast_script(translated_transcript, key_points_summary, config, progress_callback)
    performance_metrics["podcast_script"] = _collect_performance_metrics("podcast_script", stage_start)
    if progress_callback:
        progress_callback(5, "Generating podcast script", 100)
    log_setup(f"✓ Stage 5 complete ({performance_metrics['podcast_script']['time_seconds']:.2f}s)", 'success', verbose)
    
    # Stage 6: TTS synthesis
    log_setup("Stage 6/6: Synthesizing podcast audio (TTS)...", 'info', verbose)
    if progress_callback:
        progress_callback(6, "Synthesizing podcast audio (TTS)", 0)
    stage_start = time.time()
    podcast_audio_path = synthesize_podcast(podcast_script, config, progress_callback)
    tts_metrics = _collect_performance_metrics("tts", stage_start)
    if progress_callback:
        progress_callback(6, "Synthesizing podcast audio (TTS)", 100)
    
    # Calculate audio duration for RTF
    try:
        import soundfile as sf
        audio_info = sf.info(podcast_audio_path)
        audio_duration = audio_info.duration
        tts_metrics["audio_duration_seconds"] = audio_duration
        tts_metrics["real_time_factor"] = tts_metrics["time_seconds"] / audio_duration if audio_duration > 0 else 0.0
    except Exception:
        tts_metrics["audio_duration_seconds"] = 0.0
        tts_metrics["real_time_factor"] = 0.0
    
    performance_metrics["tts"] = tts_metrics
    log_setup(f"✓ Stage 6 complete ({tts_metrics['time_seconds']:.2f}s)", 'success', verbose)
    
    # Total runtime
    performance_metrics["total_runtime_seconds"] = time.time() - pipeline_start_time
    
    # Save text artifacts to files
    log_setup("Saving output files...", 'info', verbose)
    output_dir = Path(config.get("output_directory", "/content/solas_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    original_path = str(output_dir / "original_transcript.txt")
    translated_path = str(output_dir / "translated_transcript.txt")
    summary_path = str(output_dir / "key_points.md")
    script_path = str(output_dir / "podcast_script.txt")
    
    with open(original_path, "w", encoding="utf-8") as f:
        f.write(original_transcript)
    
    with open(translated_path, "w", encoding="utf-8") as f:
        f.write(translated_transcript)
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(key_points_summary)
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(podcast_script)
    
    log_setup(f"✓ Pipeline complete! Total time: {performance_metrics['total_runtime_seconds']:.2f}s", 'success', verbose)
    log_setup(f"Output directory: {output_dir}", 'info', verbose)
    
    return {
        "text_outputs": {
            "original_transcript": original_transcript,
            "translated_transcript": translated_transcript,
            "key_points_summary": key_points_summary,
            "podcast_script": podcast_script,
        },
        "file_paths": {
            "original_transcript": original_path,
            "translated_transcript": translated_path,
            "key_points_summary": summary_path,
            "podcast_script": script_path,
            "final_podcast_audio": podcast_audio_path,
            "output_directory": str(output_dir),
        },
        "performance_metrics": performance_metrics,
    }


# ============================================================================
# Template Loading Functions
# ============================================================================

def load_template(template_name: str, **kwargs) -> str:
    """
    Load and format an HTML template from the templates directory.
    
    Args:
        template_name: Name of the template file (e.g., 'warning_non_colab.html')
        **kwargs: Variables for template substitution
    
    Returns:
        Formatted HTML string with variables substituted
    
    Raises:
        FileNotFoundError: If template file doesn't exist
        KeyError: If required template variable is missing
    """
    template_path = Path(__file__).parent / 'templates' / template_name
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_name}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()
    
    # Remove HTML comments (template documentation)
    # Keep only the actual HTML content, BUT preserve placeholder comments
    lines = template_content.split('\n')
    html_lines = []
    in_comment = False

    for line in lines:
        stripped = line.strip()
        # Preserve placeholder comments (used for widget insertion)
        if 'PLACEHOLDER' in stripped and stripped.startswith('<!--') and stripped.endswith('-->'):
            html_lines.append(line)
            continue
        if stripped.startswith('<!--'):
            in_comment = True
            continue
        if in_comment and stripped.endswith('-->'):
            in_comment = False
            continue
        if not in_comment:
            html_lines.append(line)

    template_content = '\n'.join(html_lines)
    
    # Perform variable substitution using regex to avoid conflicts with CSS variables
    # Only replace {variable_name} patterns, not CSS variables like {--color-primary}
    import re
    for key, value in kwargs.items():
        # Escape special regex characters in the key
        escaped_key = re.escape(key)
        # Match {key} but not {--something} (CSS variables)
        pattern = r'\{' + escaped_key + r'\}'
        template_content = re.sub(pattern, str(value), template_content)
    
    # Check if any required variables are still missing (unreplaced {variable} patterns)
    remaining_vars = re.findall(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}', template_content)
    # Filter out CSS variables (they start with --)
    remaining_vars = [v for v in remaining_vars if not v.startswith('-')]
    if remaining_vars and kwargs:
        # Only warn if there are actual template variables (not CSS) that weren't replaced
        missing = [v for v in remaining_vars if v not in kwargs]
        if missing:
            raise KeyError(f"Missing required template variable(s): {', '.join(missing)}")
    
    return template_content


# ============================================================================
# Setup & Environment Functions
# ============================================================================

def log_setup(message: str, level: str = 'info', verbose: bool = False) -> None:
    """
    Log message based on verbosity - only prints if verbose is True.
    Uses ANSI color codes for better readability in terminals.
    
    Args:
        message: Message to log
        level: Log level ('info', 'success', 'warning', 'error', 'important')
        verbose: If True, print message. If False, silent (widgets/HTML handle display)
    """
    # Only show if verbose - no exceptions for error/warning
    if not verbose:
        return
    
    # ANSI color codes
    RESET = '\033[0m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    CYAN = '\033[36m'
    BOLD = '\033[1m'
    
    if level == 'error':
        print(f"{RED}[SOLAS] ✗ {message}{RESET}")
    elif level == 'warning':
        print(f"{YELLOW}[SOLAS] ⚠ {message}{RESET}")
    elif level == 'success':
        print(f"{GREEN}[SOLAS] ✓ {message}{RESET}")
    elif level == 'important':
        print(f"{BOLD}{CYAN}[SOLAS] ⚠ {message}{RESET}")
    else:
        print(f"[SOLAS] {message}")


def _is_package_installed(package_spec: str) -> Tuple[str, bool, Optional[str]]:
    """
    Check if a package is installed with the exact required version.
    
    Args:
        package_spec: Package specification like "transformers==4.57.1"
    
    Returns:
        (package_name, is_installed_correctly, installed_version)
    """
    # Parse package name and version from "package==version" format
    if '==' not in package_spec:
        pkg_name = package_spec.strip()
        return (pkg_name, False, None)
    
    pkg_name, required_version = package_spec.split('==', 1)
    pkg_name = pkg_name.strip()
    required_version = required_version.strip()
    
    # Check if installed
    try:
        installed_version = importlib.metadata.version(pkg_name)
        is_correct = (installed_version == required_version)
        return (pkg_name, is_correct, installed_version)
    except importlib.metadata.PackageNotFoundError:
        return (pkg_name, False, None)


def _get_system_package_version(package_name: str) -> Optional[str]:
    """Get installed version of a system package using dpkg"""
    try:
        result = subprocess.run(
            ["dpkg", "-l", package_name],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0 and 'ii' in result.stdout:
            # Parse version from dpkg output: "ii  package  version  arch  description"
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.startswith('ii') and package_name in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        return parts[2]  # Version is the 3rd field
        return None
    except Exception:
        return None


def _compare_debian_versions(version1: str, version2: str) -> bool:
    """Compare two Debian package versions using dpkg --compare-versions"""
    try:
        # dpkg --compare-versions returns 0 if version1 >= version2
        result = subprocess.run(
            ["dpkg", "--compare-versions", version1, "ge", version2],
            capture_output=True,
            check=False
        )
        return result.returncode == 0
    except Exception:
        return False


def _check_system_package_satisfies_constraint(package_name: str, min_version: str) -> Tuple[bool, Optional[str]]:
    """Check if installed system package version satisfies minimum version constraint"""
    installed_version = _get_system_package_version(package_name)
    if installed_version is None:
        return False, None
    
    # Compare versions using dpkg
    satisfies = _compare_debian_versions(installed_version, min_version)
    return satisfies, installed_version


# ============================================================================
# Setup Progress Widget Functions
# ============================================================================

def _create_progress_widgets():
    """
    Create progress widgets for setup display.
    
    Returns:
        Dict with 'container', 'step_labels', 'step_bars', 'step_rows', 'substeps_container'
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, HTML
    except ImportError:
        # If ipywidgets not available, return None widgets
        return None
    
    # Use Output widgets instead of HTML widgets to avoid CDN notice
    # Each Output widget will contain HTML that we display into it
    def create_label_output(html_content):
        """Create an Output widget with initial HTML content."""
        output = widgets.Output(layout=widgets.Layout(width='420px'))
        with output:
            display(HTML(html_content))
        return output
    
    step_labels = {
        1: create_label_output('<div style="color: var(--color-text-secondary);">⏺ Step 1/5: Checking system dependencies...</div>'),
        2: create_label_output('<div style="color: var(--color-text-secondary);">⏺ Step 2/5: Installing system dependencies...</div>'),
        3: create_label_output('<div style="color: var(--color-text-secondary);">⏺ Step 3/5: Checking Python dependencies...</div>'),
        4: create_label_output('<div style="color: var(--color-text-secondary);">⏺ Step 4/5: Installing Python dependencies...</div>'),
        '4a': create_label_output('<div style="color: var(--color-text-secondary); margin-left: 20px;">⏺ Collecting packages...</div>'),
        '4b': create_label_output('<div style="color: var(--color-text-secondary); margin-left: 20px;">⏺ Downloading packages...</div>'),
        '4c': create_label_output('<div style="color: var(--color-text-secondary); margin-left: 20px;">⏺ Building wheels...</div>'),
        '4d': create_label_output('<div style="color: var(--color-text-secondary); margin-left: 20px;">⏺ Uninstalling old versions...</div>'),
        '4e': create_label_output('<div style="color: var(--color-text-secondary); margin-left: 20px;">⏺ Installing packages...</div>'),
        5: create_label_output('<div style="color: var(--color-text-secondary);">⏺ Step 5/5: Finalizing setup...</div>'),
    }
    
    step_bars = {
        1: widgets.IntProgress(value=0, min=0, max=100, bar_style='', layout=widgets.Layout(width='200px', height='20px', margin='0 0 0 10px')),
        2: widgets.IntProgress(value=0, min=0, max=100, bar_style='', layout=widgets.Layout(width='200px', height='20px', margin='0 0 0 10px')),
        3: widgets.IntProgress(value=0, min=0, max=100, bar_style='', layout=widgets.Layout(width='200px', height='20px', margin='0 0 0 10px')),
        4: widgets.IntProgress(value=0, min=0, max=100, bar_style='', layout=widgets.Layout(width='200px', height='20px', margin='0 0 0 10px')),
        '4a': widgets.IntProgress(value=0, min=0, max=100, bar_style='', layout=widgets.Layout(width='200px', height='20px', margin='0 0 0 10px')),
        '4b': widgets.IntProgress(value=0, min=0, max=100, bar_style='', layout=widgets.Layout(width='200px', height='20px', margin='0 0 0 10px')),
        '4c': widgets.IntProgress(value=0, min=0, max=100, bar_style='', layout=widgets.Layout(width='200px', height='20px', margin='0 0 0 10px')),
        '4d': widgets.IntProgress(value=0, min=0, max=100, bar_style='', layout=widgets.Layout(width='200px', height='20px', margin='0 0 0 10px')),
        '4e': widgets.IntProgress(value=0, min=0, max=100, bar_style='', layout=widgets.Layout(width='200px', height='20px', margin='0 0 0 10px')),
        5: widgets.IntProgress(value=0, min=0, max=100, bar_style='', layout=widgets.Layout(width='200px', height='20px', margin='0 0 0 10px')),
    }
    
    # Create HBox for each step (label + bar side by side)
    step_rows = {
        1: widgets.HBox([step_labels[1], step_bars[1]], layout=widgets.Layout(margin='4px 0')),
        2: widgets.HBox([step_labels[2], step_bars[2]], layout=widgets.Layout(margin='4px 0')),
        3: widgets.HBox([step_labels[3], step_bars[3]], layout=widgets.Layout(margin='4px 0')),
        4: widgets.HBox([step_labels[4], step_bars[4]], layout=widgets.Layout(margin='4px 0')),
        '4a': widgets.HBox([step_labels['4a'], step_bars['4a']], layout=widgets.Layout(margin='2px 0')),
        '4b': widgets.HBox([step_labels['4b'], step_bars['4b']], layout=widgets.Layout(margin='2px 0')),
        '4c': widgets.HBox([step_labels['4c'], step_bars['4c']], layout=widgets.Layout(margin='2px 0')),
        '4d': widgets.HBox([step_labels['4d'], step_bars['4d']], layout=widgets.Layout(margin='2px 0')),
        '4e': widgets.HBox([step_labels['4e'], step_bars['4e']], layout=widgets.Layout(margin='2px 0')),
        5: widgets.HBox([step_labels[5], step_bars[5]], layout=widgets.Layout(margin='4px 0')),
    }
    
    # Container for sub-steps 4a-4e (initially hidden)
    substeps_container = widgets.VBox([
        step_rows['4a'],
        step_rows['4b'],
        step_rows['4c'],
        step_rows['4d'],
        step_rows['4e'],
    ], layout=widgets.Layout(display='none'))
    
    # Create header using Output widget to avoid CDN notice
    header_output = widgets.Output()
    with header_output:
        display(HTML('<h3 style="color: var(--color-primary); margin-bottom: 10px;">🔧 Setting up SOLAS Environment</h3>'))
    
    progress_container = widgets.VBox([
        header_output,
        step_rows[1],
        step_rows[2],
        step_rows[3],
        step_rows[4],
        substeps_container,
        step_rows[5],
    ], layout=widgets.Layout(padding='10px'))
    
    return {
        'container': progress_container,
        'step_labels': step_labels,
        'step_bars': step_bars,
        'step_rows': step_rows,
        'substeps_container': substeps_container
    }


def _update_progress_widget(message: str, step, total: int, status: str, progress: Optional[int],
                           step_labels: dict, step_bars: dict) -> None:
    """Update progress widget display (uses Output widgets for labels)"""
    if step is None:
        return
    
    try:
        from IPython.display import display, HTML, clear_output
    except ImportError:
        return
    
    # Update the specific step indicator
    if status == 'active':
        icon = '⏳'
        color = 'var(--color-primary)'
        weight = 'bold'
        bar_style = 'info'
    elif status == 'complete':
        icon = '✓'
        color = 'var(--color-success)'
        weight = 'normal'
        bar_style = 'success'
    else:  # pending
        icon = '⏺'
        color = 'var(--color-text-secondary)'
        weight = 'normal'
        bar_style = ''
    
    # Handle substeps (string keys) vs main steps (numeric keys)
    if isinstance(step, str):
        # Substep - no "Step X/Y" prefix
        margin_left = 'margin-left: 20px;' if step.startswith('4') else ''
        step_html = f'<div style="color: {color}; font-weight: {weight}; {margin_left}">{icon} {message}</div>'
    else:
        # Main step - requires total
        if total is not None:
            step_html = f'<div style="color: {color}; font-weight: {weight};">{icon} Step {step}/{total}: {message}</div>'
        else:
            step_html = f'<div style="color: {color}; font-weight: {weight};">{icon} {message}</div>'
    
    # Update Output widget by clearing and redisplaying HTML
    if step in step_labels:
        label_output = step_labels[step]
        with label_output:
            clear_output(wait=True)
            display(HTML(step_html))
    
    # Update progress bar
    if step in step_bars:
        if progress is not None:
            step_bars[step].value = progress
        elif status == 'complete':
            step_bars[step].value = 100
        elif status == 'active':
            # Set to indeterminate (just show as active, no specific progress)
            step_bars[step].value = 50
        
        step_bars[step].bar_style = bar_style


def _update_progress_bar_only(step, progress: int, step_bars: dict) -> None:
    """Update only the progress bar percentage without changing the step label"""
    if step in step_bars:
        step_bars[step].value = progress


def _load_requirements() -> List[str]:
    """
    Load package requirements from requirements.txt file.
    
    Returns:
        List of package specifications (e.g., ["torch==2.9.0", "transformers==4.57.1"])
    """
    requirements_path = Path(__file__).parent / 'requirements.txt'
    
    if not requirements_path.exists():
        raise FileNotFoundError(
            f"requirements.txt not found at {requirements_path}. "
            "Please create it with the required package versions."
        )
    
    packages = []
    with open(requirements_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                packages.append(line)
    
    return packages


# Load package list from requirements.txt
_SETUP_PYTHON_PACKAGES = _load_requirements()

# System packages for Colab
_SETUP_SYSTEM_PACKAGES = [
    ("espeak-ng", "1.50+dfsg-10ubuntu0.1"),
    ("libsndfile1", "1.0.31-2ubuntu0.2"),
    ("ffmpeg", "7:4.4.2-0ubuntu0.22.04.1"),
]


def _pip_install_packages(pkgs: List[str], check_first: bool = True, progress_step: Optional[int] = None,
                         step_labels: Optional[dict] = None, step_bars: Optional[dict] = None,
                         substeps_container: Optional[Any] = None) -> Tuple[bool, bool]:
    """
    Install packages, checking if they're already installed with exact versions first.
    
    Returns:
        Tuple of (packages_installed, bnb_updated)
    """
    verbose = get_verbosity()
    
    # Quick check: are all packages already installed with correct versions?
    all_correct = True
    if check_first:
        log_setup(f"Checking {len(pkgs)} package(s)...", 'important', verbose)
        total_pkgs = len(pkgs)
        for idx, pkg in enumerate(pkgs):
            # Update progress bar during checking
            if progress_step is not None and step_bars is not None:
                progress_pct = int((idx / total_pkgs) * 100)
                _update_progress_bar_only(progress_step, progress_pct, step_bars)
            
            _, is_correct, _ = _is_package_installed(pkg)
            if not is_correct:
                all_correct = False
                break
        
        # Ensure progress bar is at 100% after checking
        if progress_step is not None and step_bars is not None:
            _update_progress_bar_only(progress_step, 100, step_bars)
        
        if all_correct:
            log_setup("All packages are installed with correct versions", 'success', verbose)
            # No installation happened, so no restart needed
            # Progress widget will be updated to 'complete' by caller
            return False, False
        else:
            log_setup("Some packages need installation/upgrade, installing all packages...", 'important', verbose)
    
    # Install all packages - pip will skip/upgrade/downgrade as needed
    # Only check for bitsandbytes if we're actually installing (not all were correct)
    if not all_correct or not check_first:
        # Check if bitsandbytes is in the package list - if we install, it might need restart
        bnb_updated = any('bitsandbytes' in pkg.lower() for pkg in pkgs)
        
        try:
            # Reveal substeps container
            if substeps_container is not None:
                substeps_container.layout.display = 'block'
            
            # Update main progress bar to 0% (starting installation)
            if progress_step is not None and step_bars is not None:
                _update_progress_bar_only(progress_step, 0, step_bars)
            
            log_setup(f"Installing {len(pkgs)} package(s)...", 'important', verbose)
            # Extract package names for display
            pkg_names = []
            for p in pkgs:
                if '==' in p:
                    pkg_names.append(p.split('==')[0].strip())
                else:
                    pkg_names.append(p.strip())
            if verbose:
                log_setup(f"Packages: {', '.join(pkg_names)}", 'important', verbose)
            
            # Build pip command - install all packages, pip will handle what's needed
            if verbose:
                cmd = [sys.executable, "-m", "pip", "install", "-v"] + pkgs
            else:
                cmd = [sys.executable, "-m", "pip", "install"] + pkgs
            
            # Update progress bar to 10% (running pip)
            if progress_step is not None and step_bars is not None:
                _update_progress_bar_only(progress_step, 10, step_bars)
            
            log_setup(f"Executing: {' '.join(cmd)}", 'info', verbose)
            
            # Run pip with output capture for progress tracking
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1  # Line buffered
            )
            
            # Track progress through different pip phases
            seen_collecting = False
            seen_downloading = False
            seen_building = False
            seen_uninstalling = False
            seen_installing = False
            
            log_setup("VERBOSE MODE: Streaming pip output in real-time...", 'info', verbose)
            
            # Stream output line by line and track progress
            for line in process.stdout:
                if verbose:
                    print(line, end='', flush=True)
                
                # Update progress based on pip phases
                if progress_step is not None and step_labels is not None and step_bars is not None:
                    line_lower = line.lower()
                    
                    if not seen_collecting and line_lower.startswith('collecting'):
                        seen_collecting = True
                        _update_progress_widget('Collecting packages...', '4a', 5, 'active', 50, step_labels, step_bars)
                        _update_progress_bar_only(progress_step, 15, step_bars)
                    elif not seen_downloading and (line_lower.startswith('downloading') or line_lower.startswith('obtaining')):
                        seen_downloading = True
                        _update_progress_widget('Collecting packages...', '4a', 5, 'complete', 100, step_labels, step_bars)
                        _update_progress_widget('Downloading packages...', '4b', 5, 'active', 50, step_labels, step_bars)
                        _update_progress_bar_only(progress_step, 30, step_bars)
                    elif not seen_building and (line_lower.startswith('building wheels') or 'preparing metadata (pyproject.toml)' in line_lower):
                        seen_building = True
                        _update_progress_widget('Collecting packages...', '4a', 5, 'complete', 100, step_labels, step_bars)
                        _update_progress_widget('Downloading packages...', '4b', 5, 'complete', 100, step_labels, step_bars)
                        _update_progress_widget('Building wheels...', '4c', 5, 'active', 50, step_labels, step_bars)
                        _update_progress_bar_only(progress_step, 55, step_bars)
                    elif not seen_uninstalling and line_lower.startswith('attempting uninstall'):
                        seen_uninstalling = True
                        _update_progress_widget('Collecting packages...', '4a', 5, 'complete', 100, step_labels, step_bars)
                        _update_progress_widget('Downloading packages...', '4b', 5, 'complete', 100, step_labels, step_bars)
                        _update_progress_widget('Building wheels...', '4c', 5, 'complete', 100, step_labels, step_bars)
                        _update_progress_widget('Uninstalling old versions...', '4d', 5, 'active', 50, step_labels, step_bars)
                        _update_progress_bar_only(progress_step, 65, step_bars)
                    elif not seen_installing and line_lower.startswith('installing collected packages'):
                        seen_installing = True
                        _update_progress_widget('Collecting packages...', '4a', 5, 'complete', 100, step_labels, step_bars)
                        _update_progress_widget('Downloading packages...', '4b', 5, 'complete', 100, step_labels, step_bars)
                        _update_progress_widget('Building wheels...', '4c', 5, 'complete', 100, step_labels, step_bars)
                        _update_progress_widget('Uninstalling old versions...', '4d', 5, 'complete', 100, step_labels, step_bars)
                        _update_progress_widget('Installing packages...', '4e', 5, 'active', 50, step_labels, step_bars)
                        _update_progress_bar_only(progress_step, 85, step_bars)
                    elif line_lower.startswith('successfully installed'):
                        _update_progress_widget('Installing packages...', '4e', 5, 'complete', 100, step_labels, step_bars)
                        _update_progress_bar_only(progress_step, 98, step_bars)
            
            process.wait()
            returncode = process.returncode
            
            # Ensure we hit 100% at the end
            if progress_step is not None and step_bars is not None:
                _update_progress_bar_only(progress_step, 100, step_bars)
            
            log_setup(f"Pip command completed with exit code: {returncode}", 'info', verbose)
            
            if returncode == 0:
                log_setup(f"Successfully installed packages", 'success', verbose)
                return True, bnb_updated
            else:
                log_setup(f"Failed to install packages (exit code: {returncode})", 'error', verbose)
                log_setup("You may need to manually install failed packages", 'warning', verbose)
                return False, bnb_updated
        except Exception as e:
            log_setup(f"pip install failed: {e}", 'error', verbose)
            return False, bnb_updated


def _apt_check_packages(pkgs_with_constraints: List[Tuple[str, Optional[str]]], progress_step: Optional[int] = None,
                       step_bars: Optional[dict] = None) -> Tuple[List[str], List[str]]:
    """
    Check system packages against version constraints.
    
    Returns:
        (to_install, to_upgrade) lists of package names
    """
    verbose = get_verbosity()
    to_install = []
    to_upgrade = []
    
    try:
        if not shutil.which("apt-get"):
            log_setup("apt-get not available (not on Debian/Ubuntu system)", 'warning', verbose)
            return [], []
        
        # Normalize input: convert strings to (name, None) tuples
        normalized_pkgs = []
        for pkg in pkgs_with_constraints:
            if isinstance(pkg, tuple):
                normalized_pkgs.append(pkg)
            else:
                normalized_pkgs.append((pkg, None))
        
        # Update package list first (needed for version checks)
        if progress_step is not None and step_bars is not None:
            _update_progress_bar_only(progress_step, 10, step_bars)
        log_setup("Updating package list (this may take a moment)...", 'important', verbose)
        if verbose:
            log_setup("Running: apt-get update -y...", 'info', verbose)
            subprocess.run(["apt-get", "update", "-y"], check=True)
        else:
            subprocess.run(
                ["apt-get", "update", "-y"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        log_setup("Package list updated", 'success', verbose)
        
        total_pkgs = len(normalized_pkgs)
        checked = 0
        
        for pkg_name, min_version in normalized_pkgs:
            checked += 1
            if progress_step is not None and step_bars is not None:
                progress_pct = 10 + int((checked / total_pkgs) * 90)  # 10-100% for checking
                _update_progress_bar_only(progress_step, progress_pct, step_bars)
            installed_version = _get_system_package_version(pkg_name)
            
            if installed_version is not None:
                # Package is installed - check if it satisfies constraint
                if min_version:
                    satisfies, _ = _check_system_package_satisfies_constraint(pkg_name, min_version)
                    if satisfies:
                        log_setup(f"{pkg_name}=={installed_version} already installed (satisfies >= {min_version})", 'success', verbose)
                    else:
                        to_upgrade.append(pkg_name)
                        log_setup(f"{pkg_name}=={installed_version} installed but doesn't satisfy constraint (need >= {min_version})", 'warning', verbose)
                else:
                    # No version constraint - just check if installed
                    log_setup(f"{pkg_name}=={installed_version} already installed", 'success', verbose)
            else:
                # Package not installed
                to_install.append(pkg_name)
                if min_version:
                    log_setup(f"{pkg_name} not installed (will install >= {min_version})", 'info', verbose)
                else:
                    log_setup(f"{pkg_name} not installed, will install", 'info', verbose)
        
        return to_install, to_upgrade
        
    except Exception as e:
        log_setup(f"apt checking failed: {e}", 'error', verbose)
        return [], []


def _apt_install_packages(to_install: List[str], to_upgrade: List[str], progress_step: Optional[int] = None,
                         step_bars: Optional[dict] = None) -> None:
    """Install or upgrade system packages."""
    verbose = get_verbosity()
    try:
        # Install missing packages and upgrade outdated ones
        packages_to_process = to_install + to_upgrade
        if packages_to_process:
            action = "Installing" if to_install and not to_upgrade else "Upgrading" if to_upgrade and not to_install else "Installing/upgrading"
            log_setup(f"{action} {len(packages_to_process)} system package(s): {', '.join(packages_to_process)}", 'important', verbose)
            cmd = ["apt-get", "install", "-y"]
            # Only add --only-upgrade if we're ONLY upgrading (no new installs)
            if to_upgrade and not to_install:
                cmd.append("--only-upgrade")
            cmd.extend(packages_to_process)
            
            if progress_step is not None and step_bars is not None:
                _update_progress_bar_only(progress_step, 10, step_bars)
            log_setup(f"Running apt-get install (this may take a few minutes)...", 'important', verbose)
            if verbose:
                log_setup(f"Running: {' '.join(cmd[:5])}...", 'info', verbose)
                subprocess.run(cmd, check=True)
            else:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            if progress_step is not None and step_bars is not None:
                _update_progress_bar_only(progress_step, 100, step_bars)
            log_setup("apt-get install completed", 'success', verbose)
            if to_install:
                log_setup(f"Successfully installed {len(to_install)} system package(s)", 'success', verbose)
            if to_upgrade:
                log_setup(f"Successfully upgraded {len(to_upgrade)} system package(s)", 'success', verbose)
        else:
            log_setup("No packages to install/upgrade", 'success', verbose)
            if progress_step is not None and step_bars is not None:
                _update_progress_bar_only(progress_step, 100, step_bars)
                
    except Exception as e:
        log_setup(f"apt-get failed: {e}", 'error', verbose)


def _finalize_setup(package_list: List[str], bnb_updated: bool, progress_step: Optional[int] = None,
                   step_labels: Optional[dict] = None, step_bars: Optional[dict] = None,
                   progress_container: Optional[Any] = None) -> Dict[str, Any]:
    """
    Finalize setup: GPU check, dependency table generation, restart detection.
    
    IMPORTANT: Restart check happens at the END, after all HTML is generated and displayed.
    
    Returns:
        Dict with: restart_needed, dependency_data, gpu_available, device_name, progress_container
    """
    verbose = get_verbosity()
    
    # Only import widgets if we have a progress container to update
    widgets = None
    HTML = None
    if progress_container is not None:
        try:
            import ipywidgets as widgets
            from IPython.display import HTML
        except ImportError:
            pass
    
    # Always need HTML for display
    if HTML is None:
        try:
            from IPython.display import HTML
        except ImportError:
            pass
    
    # Step 1: GPU check
    if progress_step is not None and step_labels is not None and step_bars is not None:
        _update_progress_widget("Finalizing setup...", progress_step, 5, 'active', 10, step_labels, step_bars)
    
    gpu_available = False
    device_name = None
    try:
        import torch
        if progress_step is not None and step_labels is not None and step_bars is not None:
            _update_progress_widget("Finalizing setup...", progress_step, 5, 'active', 20, step_labels, step_bars)
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            device_name = torch.cuda.get_device_name(0)
            log_setup(f'GPU detected: {device_name}', 'success', verbose)
        else:
            log_setup('WARNING: No GPU detected. This notebook will run extremely slowly without a GPU.', 'warning', verbose)
    except ImportError:
        log_setup('WARNING: torch not available, skipping GPU check', 'warning', verbose)
    except Exception as e:
        log_setup(f'WARNING: Error checking GPU: {e}', 'warning', verbose)
    
    # Step 2: Create dependency status table
    if progress_step is not None and step_labels is not None and step_bars is not None:
        _update_progress_widget("Finalizing setup...", progress_step, 5, 'active', 30, step_labels, step_bars)
    
    dependency_data = []
    total_pkgs = len(package_list)
    for idx, pkg_spec in enumerate(package_list):
        try:
            # Extract required version from spec
            if '==' in pkg_spec:
                pkg_name, required_ver = pkg_spec.split('==', 1)
                pkg_name = pkg_name.strip()
                required_ver = required_ver.strip()
            else:
                pkg_name = pkg_spec.strip()
                required_ver = "Any"
            
            # Check if installed with correct version
            _, is_correct, installed_ver = _is_package_installed(pkg_spec)
            
            status = "✓" if is_correct else "✗" if installed_ver else "—"
            dependency_data.append({
                'name': pkg_name,
                'required': required_ver,
                'installed': installed_ver or "Not installed",
                'status': status
            })
        except Exception as e:
            log_setup(f'WARNING: Error checking package {pkg_spec}: {e}', 'warning', verbose)
            # Extract package name from spec
            if '==' in pkg_spec:
                pkg_name = pkg_spec.split('==')[0].strip()
            else:
                pkg_name = pkg_spec.strip()
            dependency_data.append({
                'name': pkg_name,
                'required': 'Unknown',
                'installed': 'Error',
                'status': '✗'
            })
        # Update progress: 30% + (idx/total_pkgs) * 40% = 30-70%
        if progress_step is not None and step_bars is not None:
            if (idx + 1) % max(1, total_pkgs // 3) == 0 or (idx + 1) == total_pkgs:
                progress = 30 + int((idx + 1) / total_pkgs * 40)
                _update_progress_bar_only(progress_step, progress, step_bars)
    
    # Step 3: Generate HTML table using template
    if progress_step is not None and step_labels is not None and step_bars is not None:
        _update_progress_widget("Finalizing setup...", progress_step, 5, 'active', 70, step_labels, step_bars)
    
    # Build dependency table rows
    dependency_rows_html = ""
    for dep in dependency_data:
        status_class = "status-ok" if dep['status'] == "✓" else "status-fail" if dep['status'] == "✗" else "status-missing"
        dependency_rows_html += f"""
        <tr>
            <td><strong>{dep['name']}</strong></td>
            <td>{dep['required']}</td>
            <td>{dep['installed']}</td>
            <td class="{status_class}">{dep['status']}</td>
        </tr>
        """
    
    # Load dependency table template
    table_html = load_template('dependency_table.html', dependency_rows=dependency_rows_html)
    
    # Check if all packages match required versions
    if progress_step is not None and step_labels is not None and step_bars is not None:
        _update_progress_widget("Finalizing setup...", progress_step, 5, 'active', 80, step_labels, step_bars)
    
    all_satisfied = all(dep['status'] == '✓' for dep in dependency_data) if dependency_data else False
    failed_packages = [dep for dep in dependency_data if dep['status'] == '✗']
    missing_packages = [dep for dep in dependency_data if dep['status'] == '—']
    
    # Step 4: Generate completion messages
    if all_satisfied:
        completion_title = '<h3 style="color: var(--color-success); margin-bottom: 15px;">✓ Setup Complete!</h3>'
        completion_items = '<p style="color: var(--color-success); margin: 8px 0;">✓ All dependencies have been checked and installed.</p>'
    else:
        if failed_packages:
            failed_names = ', '.join([dep['name'] for dep in failed_packages])
            completion_title = '<h3 style="color: var(--color-error); margin-bottom: 15px;">⚠ Setup Incomplete</h3>'
            completion_items = f'<p style="color: var(--color-error); margin: 8px 0;">⚠ {len(failed_packages)} package(s) do not match required versions: {failed_names}</p>'
        elif missing_packages:
            missing_names = ', '.join([dep['name'] for dep in missing_packages])
            completion_title = '<h3 style="color: var(--color-warning); margin-bottom: 15px;">⚠ Setup Incomplete</h3>'
            completion_items = f'<p style="color: var(--color-warning); margin: 8px 0;">⚠ {len(missing_packages)} package(s) are missing: {missing_names}</p>'
        else:
            completion_title = '<h3 style="color: var(--color-error); margin-bottom: 15px;">⚠ Setup Incomplete</h3>'
            completion_items = '<p style="color: var(--color-error); margin: 8px 0;">⚠ Some dependencies are not satisfied.</p>'
    
    # GPU status
    if gpu_available:
        gpu_item = f'<p style="color: var(--color-success); margin: 8px 0;">✓ GPU Available: {device_name}</p>'
    else:
        gpu_item = '<p style="color: var(--color-error); margin: 8px 0;">⚠ No GPU Detected - Performance will be limited</p>'
    
    # Success message and restart warning
    success_msg = ""
    restart_warning_html = ""
    
    if bnb_updated:
        restart_warning_html = load_template('restart_warning.html')
    
    if all_satisfied and not bnb_updated:
        success_msg = load_template('success_message.html')
    
    # Error HTML if needed
    error_html = ""
    if not all_satisfied:
        error_details = []
        if failed_packages:
            for dep in failed_packages:
                error_details.append(f"<li><strong>{dep['name']}</strong>: Installed {dep['installed']} (required: {dep['required']})</li>")
        if missing_packages:
            for dep in missing_packages:
                error_details.append(f"<li><strong>{dep['name']}</strong>: Not installed (required: {dep['required']})</li>")
        
        if error_details:
            error_html = f"""
            <div style="background: color-mix(in srgb, var(--color-error) 20%, var(--color-bg-primary)); 
                        border-left: 4px solid var(--color-error); 
                        padding: 20px; 
                        margin-top: 20px; 
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="color: var(--color-error); margin: 0 0 15px 0; font-size: 16px; font-weight: 600;">
                    ⚠ Some dependencies are not satisfied. Please fix the following:
                </p>
                <ul style="color: var(--color-error); margin: 10px 0; padding-left: 20px;">
                    {''.join(error_details)}
                </ul>
                <p style="color: var(--color-error); margin-top: 15px; font-size: 14px;">
                    <strong>To fix:</strong> Run the setup cell again, or manually install the packages with:<br>
                    <code style="background: #f8f9fa; color: #202124; padding: 2px 6px; border-radius: 3px; border: 1px solid #dadce0;">!pip install PACKAGE_NAME==VERSION</code>
                </p>
            </div>
            """
    
    # Load setup_complete template
    setup_html = load_template('setup_complete.html',
        completion_title=completion_title,
        completion_items=completion_items,
        gpu_item=gpu_item,
        success_msg=success_msg,
        restart_warning_html=restart_warning_html,
        table_html=table_html,
        error_html=error_html
    )
    
    # Clear the progress container (hide the progress widgets)
    if progress_container is not None and widgets is not None:
        try:
            # Hide the container by clearing its children
            progress_container.children = []
            progress_container.layout.display = 'none'
            log_setup("Progress container cleared", 'info', verbose)
        except Exception as e:
            log_setup(f"Failed to clear progress container: {e}", 'warning', verbose)
    
    # Always display final HTML directly (more reliable than updating container)
    if HTML is not None:
        try:
            from IPython.display import display
            display(HTML(setup_html))
            log_setup("Setup completion HTML displayed", 'info', verbose)
        except (ImportError, Exception) as e:
            log_setup(f"Could not display setup HTML: {e}", 'warning', verbose)
    
    # Mark step as complete
    if progress_step is not None and step_labels is not None and step_bars is not None:
        _update_progress_widget("Finalizing setup...", progress_step, 5, 'complete', 100, step_labels, step_bars)
    
    # IMPORTANT: Restart check happens at the END, after HTML is displayed
    restart_needed = bnb_updated
    
    # Note: SETUP_COMPLETE is set in the notebook after this function returns
    # (only if restart is not needed)
    
    return {
        'restart_needed': restart_needed,
        'dependency_data': dependency_data,
        'gpu_available': gpu_available,
        'device_name': device_name,
        'progress_container': progress_container,
        'all_satisfied': all_satisfied
    }


def setup_environment_with_progress() -> Dict[str, Any]:
    """
    Main setup orchestrator with progress tracking.
    
    This function:
    1. Creates progress widgets
    2. Checks and installs system dependencies (Colab only)
    3. Checks and installs Python dependencies (including torch/torchvision/torchaudio)
    4. Finalizes setup with GPU check, dependency table, and restart detection
    
    IMPORTANT: Restart check happens in finalize_setup AFTER HTML is displayed.
    
    Returns:
        Dict with:
        - restart_needed: bool - Whether runtime restart is required
        - dependency_data: list - Dependency status data
        - gpu_available: bool - Whether GPU is available
        - device_name: str - GPU device name (if available)
        - progress_container: widget - Progress container widget
        - all_satisfied: bool - Whether all dependencies are satisfied
    """
    verbose = get_verbosity()
    
    try:
        from IPython.display import display
    except ImportError:
        display = None
    
    # Create progress widgets
    widgets_data = _create_progress_widgets()
    if widgets_data is None:
        # Fallback if widgets not available
        return {
            'restart_needed': False,
            'dependency_data': [],
            'gpu_available': False,
            'device_name': None,
            'progress_container': None,
            'all_satisfied': False
        }
    
    progress_container = widgets_data['container']
    step_labels = widgets_data['step_labels']
    step_bars = widgets_data['step_bars']
    substeps_container = widgets_data['substeps_container']
    
    # Display progress container
    if display is not None:
        display(progress_container)
    
    # Step 1: Check system dependencies (Colab only)
    _update_progress_widget("Checking system dependencies...", 1, 5, 'active', 0, step_labels, step_bars)
    sys_to_install = []
    sys_to_upgrade = []
    
    if 'google.colab' in sys.modules:
        sys_to_install, sys_to_upgrade = _apt_check_packages(
            _SETUP_SYSTEM_PACKAGES,
            progress_step=1,
            step_bars=step_bars
        )
    
    _update_progress_widget("Checking system dependencies...", 1, 5, 'complete', 100, step_labels, step_bars)
    
    # Step 2: Install system dependencies
    _update_progress_widget("Installing system dependencies...", 2, 5, 'active', 0, step_labels, step_bars)
    if sys_to_install or sys_to_upgrade:
        _apt_install_packages(sys_to_install, sys_to_upgrade, progress_step=2, step_bars=step_bars)
    else:
        log_setup("All system packages are up to date", 'success', verbose)
        _update_progress_bar_only(2, 100, step_bars)
    
    _update_progress_widget("Installing system dependencies...", 2, 5, 'complete', 100, step_labels, step_bars)
    
    # Step 3: Check Python dependencies (includes torch/torchvision/torchaudio)
    _update_progress_widget("Checking Python dependencies...", 3, 5, 'active', 0, step_labels, step_bars)
    log_setup(f"Checking {len(_SETUP_PYTHON_PACKAGES)} Python packages...", 'important', verbose)
    _update_progress_widget("Checking Python dependencies...", 3, 5, 'complete', 100, step_labels, step_bars)
    
    # Step 4: Install Python dependencies (includes torch/torchvision/torchaudio)
    _update_progress_widget("Installing Python dependencies...", 4, 5, 'active', 0, step_labels, step_bars)
    packages_installed, bnb_updated = _pip_install_packages(
        _SETUP_PYTHON_PACKAGES,
        check_first=True,
        progress_step=4,
        step_labels=step_labels,
        step_bars=step_bars,
        substeps_container=substeps_container
    )
    if packages_installed:
        log_setup("Python package installation/update completed", 'success', verbose)
    else:
        log_setup("Python package check completed (no changes needed)", 'success', verbose)
    _update_progress_widget("Installing Python dependencies...", 4, 5, 'complete', 100, step_labels, step_bars)
    
    # Step 5: Finalize setup (GPU check, dependency table, restart detection)
    # IMPORTANT: Restart check happens INSIDE finalize_setup, AFTER HTML is generated
    result = _finalize_setup(
        _SETUP_PYTHON_PACKAGES,
        bnb_updated,
        progress_step=5,
        step_labels=step_labels,
        step_bars=step_bars,
        progress_container=progress_container
    )
    
    return result


# ============================================================================
# Interactive Notebook Helper Functions
# ============================================================================
def create_config_widgets():
    """
    Create and return all configuration widgets for the interactive interface.
    
    Returns:
        Dictionary containing all widgets and the config box widget
    """
    import ipywidgets as widgets
    
    # Model options
    ASR_MODELS = [
        "openai/whisper-tiny",
        "openai/whisper-small",
        "openai/whisper-large-v3"
    ]
    
    LLM_MODELS = [
        "Qwen/Qwen2-0.5B-Instruct",
        "Qwen/Qwen2-1.5B-Instruct",
        "microsoft/phi-3-mini-4k-instruct",
        "mistralai/Mistral-7B-Instruct-v0.3"
    ]
    
    # Create widgets
    widgets_dict = {
        "asr_dropdown": widgets.Dropdown(
            options=ASR_MODELS,
            value=ASR_MODELS[0],
            description='ASR Model:'
        ),
        "llm_dropdown": widgets.Dropdown(
            options=LLM_MODELS,
            value=LLM_MODELS[0],
            description='LLM Model:'
        ),
        "quantization_dropdown": widgets.Dropdown(
            options=["None", "4-bit"],
            value="None",
            description='Quantization:'
        ),
        "chunk_size_dropdown": widgets.Dropdown(
            options=[2000, 8000],
            value=2000,
            description='Chunk Size:'
        ),
        "repetition_penalty_dropdown": widgets.Dropdown(
            options=[("None (no penalty)", "none"), ("1.2 (default)", 1.2), ("1.8 (aggressive)", 1.8)],
            value=1.2,
            description='Repetition Penalty:'
        ),
        "summary_mode_dropdown": widgets.Dropdown(
            options=["greedy", "sampled", "hybrid"],
            value="greedy",
            description='Summary Mode:'
        ),
        "podcast_temp_slider": widgets.FloatSlider(
            value=0.3,
            min=0.1,
            max=1.0,
            step=0.1,
            description='Podcast Temp:'
        ),
        "source_lang_dropdown": widgets.Dropdown(
            options=["Portuguese", "English", "Spanish", "French", "German", "Italian", "Russian", "Chinese", "Japanese", "Korean"],
            value="Portuguese",
            description='Source Lang:'
        ),
        "target_lang_dropdown": widgets.Dropdown(
            options=["English", "Portuguese", "Spanish", "French", "German", "Italian", "Russian", "Chinese", "Japanese", "Korean"],
            value="English",
            description='Target Lang:'
        ),
        "input_audio_text": widgets.Text(
            value="",  # Will be set dynamically below
            description='Audio Path:'
        ),
        "output_dir_text": widgets.Text(
            value="",  # Will be set dynamically below
            description='Output Dir:'
        ),
        "host_a_text": widgets.Text(
            value="",  # Will be set dynamically below
            description='Host A Voice:'
        ),
        "host_b_text": widgets.Text(
            value="",  # Will be set dynamically below
            description='Host B Voice:'
        ),
    }
    
    # Set default paths dynamically based on environment
    from pathlib import Path
    BASE_DIR = Path('/content') if Path('/content').exists() else Path.cwd()
    SOLAS_DIR = BASE_DIR / 'SOLAS'
    
    # Set default input audio to short sample
    widgets_dict["input_audio_text"].value = str(SOLAS_DIR / 'input_audio_samples' / 'short.ogg')
    
    # Set default output directory
    widgets_dict["output_dir_text"].value = str(BASE_DIR / 'solas_outputs')
    
    # Set default host voices to sample voices
    widgets_dict["host_a_text"].value = str(SOLAS_DIR / 'TTS_voice_samples' / 'male.wav')
    widgets_dict["host_b_text"].value = str(SOLAS_DIR / 'TTS_voice_samples' / 'female.wav')
    
    # Create config box with proper layout
    # Section labels will be displayed separately using IPython.display.HTML
    config_box = widgets.VBox([
        widgets_dict["asr_dropdown"],
        widgets_dict["llm_dropdown"],
        widgets_dict["quantization_dropdown"],
        widgets_dict["chunk_size_dropdown"],
        widgets_dict["repetition_penalty_dropdown"],
        widgets_dict["summary_mode_dropdown"],
        widgets_dict["podcast_temp_slider"],
        widgets_dict["source_lang_dropdown"],
        widgets_dict["target_lang_dropdown"],
    ], layout=widgets.Layout(
        border='1px solid #dadce0',
        padding='10px',
        margin='10px 0',
        width='auto'
    ))
    
    widgets_dict["config_box"] = config_box
    
    return widgets_dict


def create_audio_upload_widget(widgets_dict):
    """
    Create audio upload widget (Colab or local Jupyter).
    
    Args:
        widgets_dict: Dictionary from create_config_widgets() containing input_audio_text widget
    
    Returns:
        Widget box for audio upload
    """
    import ipywidgets as widgets
    from pathlib import Path
    
    _BASE_DIR = Path('/content') if Path('/content').exists() else Path.cwd()
    input_audio_text = widgets_dict["input_audio_text"]
    
    try:
        from google.colab import files as _colab_files
        _is_colab = True
    except Exception:
        _is_colab = False
    
    if _is_colab:
        # Colab file upload - button directly triggers upload (required for security)
        upload_btn = widgets.Button(
            description='📁 Upload Audio File',
            button_style='primary',
            layout=widgets.Layout(width='auto', height='40px')
        )
        
        def on_colab_upload(_):
            print('Select an audio file to upload...')
            _uploaded = _colab_files.upload()
            if _uploaded:
                _name = next(iter(_uploaded.keys()))
                _data = _uploaded[_name]
                save_path = _BASE_DIR / _name
                with open(save_path, 'wb') as f:
                    f.write(_data)
                input_audio_text.value = str(save_path)
                print(f'✓ Uploaded and set audio path to: {save_path}')
            else:
                print('No file selected.')
        
        upload_btn.on_click(on_colab_upload)
        label = widgets.Label(value='Upload input audio')
        label.style.font_weight = 'bold'
        return widgets.VBox([label, upload_btn])
    else:
        # Local Jupyter file upload - auto-save when file is selected
        uploader = widgets.FileUpload(accept='.wav,.mp3,.m4a,.flac,.ogg', multiple=False)
        status = widgets.Label()
        
        def _on_file_selected(change):
            if not uploader.value:
                return
            (fname, meta), = uploader.value.items()
            data = meta['content']
            save_path = _BASE_DIR / fname
            with open(save_path, 'wb') as f:
                f.write(data)
            input_audio_text.value = str(save_path)
            status.value = f'✓ Saved and set audio path to: {save_path}'
        
        uploader.observe(_on_file_selected, names='value')
        label = widgets.Label(value='Upload input audio')
        label.style.font_weight = 'bold'
        return widgets.VBox([label, uploader, status])


def create_host_voice_upload_widget(widgets_dict, host_key):
    """
    Create audio upload widget for Host A or Host B voice cloning.
    
    Args:
        widgets_dict: Dictionary from create_config_widgets() containing host text widgets
        host_key: Either "host_a_text" or "host_b_text"
    
    Returns:
        Widget box for host voice upload
    """
    import ipywidgets as widgets
    from pathlib import Path
    
    _BASE_DIR = Path('/content') if Path('/content').exists() else Path.cwd()
    host_text = widgets_dict[host_key]
    host_name = "Host A" if host_key == "host_a_text" else "Host B"
    
    try:
        from google.colab import files as _colab_files
        _is_colab = True
    except Exception:
        _is_colab = False
    
    if _is_colab:
        # Colab file upload - button directly triggers upload (required for security)
        upload_btn = widgets.Button(
            description=f'📁 Upload {host_name} Voice',
            button_style='primary',
            layout=widgets.Layout(width='auto', height='40px')
        )
        
        def on_colab_upload(_):
            print(f'Select {host_name} voice audio file to upload...')
            _uploaded = _colab_files.upload()
            if _uploaded:
                _name = next(iter(_uploaded.keys()))
                _data = _uploaded[_name]
                save_path = _BASE_DIR / _name
                with open(save_path, 'wb') as f:
                    f.write(_data)
                host_text.value = str(save_path)
                print(f'✓ Uploaded and set {host_name} voice path to: {save_path}')
            else:
                print('No file selected.')
        
        upload_btn.on_click(on_colab_upload)
        label = widgets.Label(value=f'Upload {host_name} voice (for TTS cloning)')
        label.style.font_weight = 'bold'
        return widgets.VBox([label, upload_btn])
    else:
        # Local Jupyter file upload - auto-save when file is selected
        uploader = widgets.FileUpload(accept='.wav,.mp3,.m4a,.flac,.ogg', multiple=False)
        status = widgets.Label()
        
        def _on_file_selected(change):
            if not uploader.value:
                return
            (fname, meta), = uploader.value.items()
            data = meta['content']
            save_path = _BASE_DIR / fname
            with open(save_path, 'wb') as f:
                f.write(data)
            host_text.value = str(save_path)
            status.value = f'✓ Saved and set {host_name} voice path to: {save_path}'
        
        uploader.observe(_on_file_selected, names='value')
        label = widgets.Label(value=f'Upload {host_name} voice (for TTS cloning)')
        label.style.font_weight = 'bold'
        return widgets.VBox([label, uploader, status])


def build_config_from_widgets(widgets_dict):
    """
    Build SOLAS_CONFIG dictionary from widget values.
    
    Args:
        widgets_dict: Dictionary from create_config_widgets()
    
    Returns:
        SOLAS_CONFIG dictionary
    """
    # Convert quantization string to None if "None"
    quantization_value = widgets_dict["quantization_dropdown"].value
    if quantization_value == "None":
        quantization_value = None

    # Convert repetition_penalty "none" string to Python None
    repetition_penalty_value = widgets_dict["repetition_penalty_dropdown"].value
    if repetition_penalty_value == "none":
        repetition_penalty_value = None

    return {
        # Variable inputs
        "asr_model_id": widgets_dict["asr_dropdown"].value,
        "llm_model_id": widgets_dict["llm_dropdown"].value,
        "quantization": quantization_value,
        "chunk_size_chars": widgets_dict["chunk_size_dropdown"].value,
        "repetition_penalty": repetition_penalty_value,
        "summary_mode": widgets_dict["summary_mode_dropdown"].value,
        "podcast_creativity_temp": widgets_dict["podcast_temp_slider"].value,
        
        # Fixed parameters
        "input_audio_path": widgets_dict["input_audio_text"].value,
        "output_directory": widgets_dict["output_dir_text"].value,
        "source_language": widgets_dict["source_lang_dropdown"].value,
        "target_language": widgets_dict["target_lang_dropdown"].value,
        "host_a_wav_path": widgets_dict["host_a_text"].value,
        "host_b_wav_path": widgets_dict["host_b_text"].value,
        "translation_max_new_tokens": 1024,
        "summary_max_new_tokens": 512,
        "podcast_max_new_tokens": 1024,
        "tts_model_id": "tts_models/multilingual/multi-dataset/xtts_v2",
    }


def display_results(results):
    """
    Display pipeline results in a user-friendly format using HTML.
    
    In non-debug mode, shows only a success banner. In debug mode, shows full details.
    
    Args:
        results: Dictionary returned from run_pipeline()
    """
    from IPython.display import Audio, display, HTML
    
    verbose = get_verbosity()
    
    # Log to console if verbose
    log_setup("Displaying pipeline results...", 'info', verbose)
    
    # In non-debug mode, show only success banner
    if not verbose:
        try:
            html = load_template('pipeline_complete.html')
            display(HTML(html))
        except Exception as e:
            log_setup(f"Error displaying success banner: {e}", 'error', True)  # Always log errors
            log_setup("Pipeline completed successfully. Run the View Results cell to see details.", 'success', True)
        return
    
    # Debug mode: Display full results using HTML template
    try:
        metrics = results["performance_metrics"]
        
        # Build HTML display
        html_parts = []
        
        # Translated Transcript section
        html_parts.append(f'''
        <div style="margin: 20px 0; padding: 15px; background: var(--color-bg-secondary); border-radius: 8px;">
            <h4 style="color: var(--color-primary); margin-top: 0;">📝 Translated Transcript</h4>
            <pre style="white-space: pre-wrap; word-wrap: break-word; font-size: 13px; max-height: 200px; overflow-y: auto;">{results["text_outputs"]["translated_transcript"][:1000]}...</pre>
        </div>
        ''')
        
        # Key Points section
        html_parts.append(f'''
        <div style="margin: 20px 0; padding: 15px; background: var(--color-bg-secondary); border-radius: 8px;">
            <h4 style="color: var(--color-primary); margin-top: 0;">📌 Key Points Summary</h4>
            <pre style="white-space: pre-wrap; word-wrap: break-word; font-size: 13px;">{results["text_outputs"]["key_points_summary"]}</pre>
        </div>
        ''')
        
        # Performance Metrics section
        html_parts.append(f'''
        <div style="margin: 20px 0; padding: 15px; background: var(--color-bg-secondary); border-radius: 8px;">
            <h4 style="color: var(--color-primary); margin-top: 0;">⚡ Performance Metrics</h4>
            <p><strong>Total Runtime:</strong> {metrics['total_runtime_seconds']:.2f} seconds</p>
            <ul style="margin: 0; padding-left: 20px; font-size: 13px;">
                <li>ASR: {metrics['asr']['time_seconds']:.2f}s, VRAM: {metrics['asr']['peak_vram_gb']:.2f}GB</li>
                <li>Translation: {metrics['translation']['time_seconds']:.2f}s, VRAM: {metrics['translation']['peak_vram_gb']:.2f}GB</li>
                <li>Summary: {metrics['summary']['time_seconds']:.2f}s, VRAM: {metrics['summary']['peak_vram_gb']:.2f}GB</li>
                <li>Podcast Script: {metrics['podcast_script']['time_seconds']:.2f}s, VRAM: {metrics['podcast_script']['peak_vram_gb']:.2f}GB</li>
                <li>TTS: {metrics['tts']['time_seconds']:.2f}s, RTF: {metrics['tts']['real_time_factor']:.2f}, VRAM: {metrics['tts']['peak_vram_gb']:.2f}GB</li>
            </ul>
        </div>
        ''')
        
        # Display HTML
        display(HTML(''.join(html_parts)))
        
        # Display audio player
        audio_path = results["file_paths"]["final_podcast_audio"]
        display(HTML(f'''
        <div style="margin: 20px 0; padding: 15px; background: var(--color-bg-secondary); border-radius: 8px;">
            <h4 style="color: var(--color-primary); margin-top: 0;">🎧 Podcast Audio</h4>
            <p style="font-size: 13px;">Audio saved to: <code>{audio_path}</code></p>
        </div>
        '''))
        display(Audio(filename=audio_path))
        
    except Exception as e:
        log_setup(f"Error displaying results: {e}", 'error', verbose)
        # Fallback to simple text display
        log_setup("Results generated successfully (see output files)", 'success', verbose)
