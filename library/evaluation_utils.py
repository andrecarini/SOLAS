"""
Utility functions for SOLAS evaluation notebook.
Extracted from notebook to reduce size and improve maintainability.
"""

import os
import sys
import json
import time
import hashlib
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import torch
import psutil


# =============================================================================
# HARDWARE INFORMATION
# =============================================================================
def get_hardware_info() -> Dict[str, Any]:
    """Collect comprehensive hardware information."""
    info = {
        'platform': sys.platform,
        'python_version': sys.version,
        'cpu_count': psutil.cpu_count(logical=True),
        'cpu_count_physical': psutil.cpu_count(logical=False),
        'ram_total_gb': psutil.virtual_memory().total / (1024**3),
        'timestamp': datetime.now().isoformat(),
    }

    # GPU info
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info['cuda_version'] = torch.version.cuda
        # Simplified GPU identifier for caching (e.g., "T4", "A100", "V100")
        gpu_name = info['gpu_name']
        if 'T4' in gpu_name:
            info['gpu_type'] = 'T4'
        elif 'A100' in gpu_name:
            info['gpu_type'] = 'A100'
        elif 'V100' in gpu_name:
            info['gpu_type'] = 'V100'
        elif 'L4' in gpu_name:
            info['gpu_type'] = 'L4'
        elif 'P100' in gpu_name:
            info['gpu_type'] = 'P100'
        else:
            info['gpu_type'] = gpu_name.split()[0]  # First word as identifier
    else:
        info['gpu_name'] = 'No GPU'
        info['gpu_memory_total_gb'] = 0
        info['cuda_version'] = None
        info['gpu_type'] = 'CPU'

    # Disk info
    try:
        disk = psutil.disk_usage('/')
        info['disk_total_gb'] = disk.total / (1024**3)
        info['disk_free_gb'] = disk.free / (1024**3)
    except:
        info['disk_total_gb'] = 0
        info['disk_free_gb'] = 0

    # Library versions
    try:
        import transformers
        info['transformers_version'] = transformers.__version__
    except:
        pass

    info['torch_version'] = torch.__version__

    return info


# =============================================================================
# MEMORY MEASUREMENT UTILITIES
# =============================================================================
def get_memory_snapshot() -> Dict[str, float]:
    """Get current RAM and VRAM usage snapshot."""
    # RAM usage
    process = psutil.Process()
    ram_used_gb = process.memory_info().rss / (1024**3)
    ram_available_gb = psutil.virtual_memory().available / (1024**3)
    ram_total_gb = psutil.virtual_memory().total / (1024**3)

    # VRAM usage
    if torch.cuda.is_available():
        vram_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        vram_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        vram_allocated_gb = vram_reserved_gb = vram_total_gb = 0

    # Disk usage
    try:
        disk = psutil.disk_usage('/')
        disk_used_gb = disk.used / (1024**3)
        disk_free_gb = disk.free / (1024**3)
    except:
        disk_used_gb = disk_free_gb = 0

    return {
        'ram_used_gb': ram_used_gb,
        'ram_available_gb': ram_available_gb,
        'ram_total_gb': ram_total_gb,
        'vram_allocated_gb': vram_allocated_gb,
        'vram_reserved_gb': vram_reserved_gb,
        'vram_total_gb': vram_total_gb,
        'disk_used_gb': disk_used_gb,
        'disk_free_gb': disk_free_gb,
    }


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force garbage collection again after CUDA cleanup
        gc.collect()


class StageMetricsCollector:
    """
    Context manager to collect comprehensive metrics for a stage.
    Records time, memory usage (before/after/peak), and hardware state.
    """

    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.start_time = None
        self.start_snapshot = None
        self.metrics = {}

    def __enter__(self):
        # Clear memory before starting
        clear_gpu_memory()

        # Record start state
        self.start_time = time.time()
        self.start_snapshot = get_memory_snapshot()

        # Reset peak memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        return self

    def __exit__(self, *args):
        # Record end state
        end_time = time.time()
        end_snapshot = get_memory_snapshot()

        # Collect peak VRAM
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        else:
            peak_vram_gb = 0

        # Build metrics dict
        self.metrics = {
            'stage_name': self.stage_name,
            'time_seconds': end_time - self.start_time,
            'timestamp_start': datetime.fromtimestamp(self.start_time).isoformat(),
            'timestamp_end': datetime.fromtimestamp(end_time).isoformat(),

            # Memory before
            'ram_before_gb': self.start_snapshot['ram_used_gb'],
            'vram_before_gb': self.start_snapshot['vram_allocated_gb'],

            # Memory after
            'ram_after_gb': end_snapshot['ram_used_gb'],
            'vram_after_gb': end_snapshot['vram_allocated_gb'],

            # Peak VRAM during execution
            'vram_peak_gb': peak_vram_gb,

            # Memory delta
            'ram_delta_gb': end_snapshot['ram_used_gb'] - self.start_snapshot['ram_used_gb'],
            'vram_delta_gb': end_snapshot['vram_allocated_gb'] - self.start_snapshot['vram_allocated_gb'],

            # System state
            'ram_available_gb': end_snapshot['ram_available_gb'],
            'disk_free_gb': end_snapshot['disk_free_gb'],
        }


# =============================================================================
# MODEL LIFECYCLE MANAGEMENT
# =============================================================================
# We manage models explicitly (not using LRU cache) for accurate memory tracking

_current_llm = {
    'model_id': None,
    'quantization': None,
    'tokenizer': None,
    'model': None,
    'load_time_seconds': None,
}

_current_asr = {
    'model_id': None,
    'pipeline': None,
    'load_time_seconds': None,
}


def unload_llm():
    """Explicitly unload current LLM from memory."""
    global _current_llm
    if _current_llm['model'] is not None:
        del _current_llm['model']
        del _current_llm['tokenizer']
        _current_llm = {'model_id': None, 'quantization': None, 'tokenizer': None, 'model': None, 'load_time_seconds': None}
        clear_gpu_memory()


def unload_asr():
    """Explicitly unload current ASR model from memory."""
    global _current_asr
    if _current_asr['pipeline'] is not None:
        del _current_asr['pipeline']
        _current_asr = {'model_id': None, 'pipeline': None, 'load_time_seconds': None}
        clear_gpu_memory()


def unload_all_models():
    """Unload all models from memory."""
    unload_llm()
    unload_asr()


def load_llm(model_id: str, quantization: Optional[str], log_fn=None) -> Tuple[Any, Any, float]:
    """
    Load LLM, unloading current one if different.
    Returns (tokenizer, model, load_time_seconds).
    Matches solas_pipeline.py implementation.
    """
    global _current_llm

    # Check if already loaded
    if _current_llm['model_id'] == model_id and _current_llm['quantization'] == quantization:
        if log_fn:
            log_fn(f"LLM already loaded: {model_id}", 'detail')
        return _current_llm['tokenizer'], _current_llm['model'], 0.0

    # Unload current model first
    unload_llm()

    if log_fn:
        log_fn(f"Loading LLM: {model_id} (quantization={quantization})", 'progress')

    # Import and load
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    start_time = time.time()

    # Quantization config - matches solas_pipeline.py
    compute_dtype = _get_compute_dtype()

    if quantization == '4-bit':
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        quant_config = None

    device_map = 'auto' if torch.cuda.is_available() else None

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map=device_map,
        dtype=compute_dtype,  # FIX: Use 'dtype' not deprecated 'torch_dtype'
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    load_time = time.time() - start_time

    _current_llm = {
        'model_id': model_id,
        'quantization': quantization,
        'tokenizer': tokenizer,
        'model': model,
        'load_time_seconds': load_time,
    }

    if log_fn:
        log_fn(f"LLM loaded in {load_time:.1f}s", 'success')
    return tokenizer, model, load_time


def load_asr(model_id: str, log_fn=None) -> Tuple[Any, float]:
    """
    Load ASR pipeline, unloading current one if different.
    Returns (pipeline, load_time_seconds).
    Matches solas_pipeline.py implementation.
    """
    global _current_asr

    # Check if already loaded
    if _current_asr['model_id'] == model_id:
        if log_fn:
            log_fn(f"ASR already loaded: {model_id}", 'detail')
        return _current_asr['pipeline'], 0.0

    # Unload current model first
    unload_asr()

    if log_fn:
        log_fn(f"Loading ASR: {model_id}", 'progress')

    from transformers import pipeline

    start_time = time.time()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = pipeline(
        'automatic-speech-recognition',
        model=model_id,
        dtype=dtype,  # FIX: Use 'dtype' not deprecated 'torch_dtype'
        device=device,
        model_kwargs={'use_safetensors': True}
    )

    load_time = time.time() - start_time

    _current_asr = {
        'model_id': model_id,
        'pipeline': pipe,
        'load_time_seconds': load_time,
    }

    if log_fn:
        log_fn(f"ASR loaded in {load_time:.1f}s", 'success')
    return pipe, load_time


def _get_compute_dtype():
    """
    Get the appropriate compute dtype based on device availability.
    Matches solas_pipeline.py:297-303
    """
    return torch.float16 if torch.cuda.is_available() else torch.float32


# =============================================================================
# CONFIG HASHING
# =============================================================================
def config_hash(config: Dict[str, Any], hardware_info: Dict[str, Any], include_hardware: bool = True) -> str:
    """
    Generate a deterministic hash of a configuration dict.
    Includes GPU type to ensure runs on different hardware are not mixed.
    """
    # Create a copy to avoid modifying the original
    hash_config = config.copy()

    # Include hardware identifier in hash
    if include_hardware:
        hash_config['_gpu_type'] = hardware_info.get('gpu_type', 'Unknown')

    # Sort keys for consistent ordering
    config_str = json.dumps(hash_config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def is_config_cached(results: Dict, config: Dict, stages: List[str], hardware_info: Dict[str, Any]) -> Optional[str]:
    """
    Check if this exact config+stages+hardware combination has been run.
    Returns experiment ID if found, None otherwise.
    """
    target_hash = config_hash(config, hardware_info, include_hardware=True)
    target_stages = set(stages)

    for exp_id, exp_data in results.get('experiments', {}).items():
        if 'error' not in exp_data and 'config_hash' in exp_data:
            if exp_data['config_hash'] == target_hash:
                # Also check stages match
                exp_stages = set(exp_data.get('stages_run', []))
                if exp_stages == target_stages:
                    return exp_id
    return None


# =============================================================================
# RESULTS MANAGEMENT
# =============================================================================
def load_results(results_file: Path) -> Dict[str, Any]:
    """Load existing results from Drive."""
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return {
        'experiments': {},
        'metadata': {
            'started': datetime.now().isoformat(),
        }
    }


def save_results(results: Dict[str, Any], results_file: Path):
    """Save results to Drive immediately."""
    results['metadata']['updated'] = datetime.now().isoformat()
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def save_experiment_output(exp_id: str, stage: str, output_type: str, content: str, outputs_dir: Path):
    """Save an experiment's output to a separate file on Drive."""
    exp_dir = outputs_dir / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    filename = f'{stage}_{output_type}.txt'
    with open(exp_dir / filename, 'w', encoding='utf-8') as f:
        f.write(content)


def is_experiment_complete(results: Dict, exp_id: str) -> bool:
    """Check if an experiment has already been completed successfully."""
    exp_data = results.get('experiments', {}).get(exp_id)
    if exp_data is None:
        return False
    return 'error' not in exp_data


# =============================================================================
# TRANSCRIPT CACHE
# =============================================================================
def load_cached_transcript(cache_file: Path) -> Optional[Dict[str, Any]]:
    """Load cached transcript from best ASR model."""
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_cached_transcript(transcript: str, asr_model: str, audio_path: str, metrics: Dict, cache_file: Path):
    """Save transcript to cache."""
    cache = {
        'transcript': transcript,
        'asr_model': asr_model,
        'audio_path': audio_path,
        'metrics': metrics,
        'created': datetime.now().isoformat(),
    }
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)
