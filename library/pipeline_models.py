"""
Pipeline models: Model loading and management for LLM and ASR.

This module provides:
- LLM loading with optional quantization
- Quantization configuration
- Model caching and unloading
"""

import sys
import warnings
from typing import Any, Tuple, Optional
from functools import lru_cache

from .pipeline_utils import (
    get_verbosity,
    get_device,
    get_compute_dtype,
    suppress_torch_dtype_warnings,
)


# =============================================================================
# QUANTIZATION CONFIG
# =============================================================================

def create_quantization_config(quantization: Optional[str], compute_dtype) -> Optional[Any]:
    """
    Create quantization configuration if requested.

    Args:
        quantization: "4-bit" or None
        compute_dtype: torch dtype for computation

    Returns:
        BitsAndBytesConfig or None
    """
    if quantization != "4-bit":
        return None

    try:
        import bitsandbytes  # noqa: F401
        import importlib.metadata
        from packaging import version as pkg_version
        from transformers import BitsAndBytesConfig

        # Verify bitsandbytes version is >= 0.43.1
        try:
            bnb_version = importlib.metadata.version("bitsandbytes")
            if pkg_version.parse(bnb_version) < pkg_version.parse("0.43.1"):
                raise ImportError(
                    f"bitsandbytes version {bnb_version} is too old. "
                    f"Version >= 0.43.1 is required for 4-bit quantization."
                )
        except Exception as version_error:
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
                    f"4-bit quantization will be disabled for now.\n"
                )
            else:
                warnings.warn(f"bitsandbytes issue: {error_msg}. 4-bit quantization disabled.")
        else:
            warnings.warn("bitsandbytes not available. 4-bit quantization disabled.")
        return None


# =============================================================================
# LLM LOADING
# =============================================================================

# Cache for loaded models
_llm_cache = {}


def ensure_llm(
    model_id: str,
    quantization: Optional[str] = None,
    log_fn=None
) -> Tuple[Any, Any]:
    """
    Load and cache a language model with optional quantization.

    Args:
        model_id: Hugging Face model identifier
        quantization: "4-bit" or None
        log_fn: Optional logging function

    Returns:
        Tuple of (tokenizer, model)
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    # Check cache first
    cache_key = (model_id, quantization)
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    verbose = get_verbosity()

    def log(msg, level='info'):
        if log_fn:
            log_fn(msg, level, verbose)

    # Get compute dtype
    compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # If using 4-bit quantization, ensure bitsandbytes is imported
    if quantization == "4-bit":
        try:
            import bitsandbytes  # noqa: F401
            import importlib
            if 'bitsandbytes' in sys.modules:
                importlib.reload(sys.modules['bitsandbytes'])
        except Exception as e:
            warnings.warn(f"Could not import bitsandbytes: {e}. 4-bit quantization may fail.")

    quant_cfg = create_quantization_config(quantization, compute_dtype)
    device_map = "auto" if torch.cuda.is_available() else None

    suppress_torch_dtype_warnings()

    log(f"[LLM] Loading tokenizer for {model_id}...", 'info')
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    log("[LLM] Tokenizer loaded.", 'info')

    quant_str = f" ({quantization})" if quantization else ""
    log(f"[LLM] Loading model{quant_str} (this may take a while for large models)...", 'info')

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_cfg,
            device_map=device_map,
            dtype=compute_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        log("[LLM] Model loaded.", 'success')
    except ImportError as e:
        if "bitsandbytes" in str(e) and "0.43.1" in str(e):
            log("[LLM] WARNING: Transformers version check failed. Attempting workaround...", 'warning')
            try:
                import bitsandbytes as bnb
                _ = bnb.__version__
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quant_cfg,
                    device_map=device_map,
                    dtype=compute_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                )
                log("[LLM] Model loaded (workaround succeeded).", 'success')
            except Exception as e2:
                raise ImportError(
                    f"Failed to load model with 4-bit quantization: {e}\n"
                    f"Workaround also failed: {e2}\n"
                    f"Please try:\n"
                    f"  1. Verify bitsandbytes is installed: !pip show bitsandbytes\n"
                    f"  2. If version < 0.43.1, upgrade: !pip install -U bitsandbytes\n"
                    f"  3. If version is correct, restart the runtime\n"
                    f"  4. Use quantization='None' to disable 4-bit quantization"
                ) from e
        else:
            raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Cache the model
    _llm_cache[cache_key] = (tokenizer, model)

    return tokenizer, model


def unload_llm(model_id: str = None, quantization: str = None):
    """
    Unload a cached LLM to free memory.

    Args:
        model_id: Model to unload. If None, unloads all.
        quantization: Quantization setting of model to unload.
    """
    import gc

    if model_id is None:
        # Unload all
        _llm_cache.clear()
    else:
        cache_key = (model_id, quantization)
        if cache_key in _llm_cache:
            del _llm_cache[cache_key]

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def unload_all_models():
    """Unload all cached models to free memory."""
    unload_llm()

    # Also clear any ASR models
    try:
        from .pipeline_stages import unload_asr
        unload_asr()
    except (ImportError, Exception):
        pass
