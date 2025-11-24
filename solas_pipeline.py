"""
SOLAS Pipeline: Core processing logic for the Self-hosted Open-source Lecture Assistant System.

This module provides a non-interactive, parameter-driven pipeline that processes lecture audio
into translated transcripts, summaries, and podcast audio.

Main Entry Point:
    run_pipeline(config: dict) -> dict: Execute the full pipeline with given configuration.

Input:
    SOLAS_CONFIG dictionary with all parameters (see implementation-plan.md for structure).

Output:
    Dictionary containing:
    - text_outputs: All generated text artifacts
    - file_paths: Paths to saved files
    - performance_metrics: Timing and resource usage for each stage
"""

import os
import sys
import time
import gc
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Callable
from functools import lru_cache

import torch
import torchaudio
import numpy as np
import soundfile as sf
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
# TTS import is lazy - only imported when synthesize_podcast is called

# Progress bar support
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable=None, *args, **kwargs):
        if iterable is None:
            return iterable
        return iterable

# Silence benign pydub regex SyntaxWarnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"pydub\.utils")

# Performance monitoring
try:
    import psutil
except ImportError:
    psutil = None


# ============================================================================
# Helper Functions
# ============================================================================

def _get_device() -> str:
    """Get the appropriate device (cuda or cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_compute_dtype() -> torch.dtype:
    """Get the appropriate compute dtype based on device availability."""
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _create_quantization_config(quantization: Optional[str], compute_dtype: torch.dtype) -> Optional[BitsAndBytesConfig]:
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
def ensure_llm(model_id: str, quantization: Optional[str] = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load and cache a language model with optional quantization.
    
    Input:
        model_id: Hugging Face model identifier
        quantization: "4-bit" or None
    
    Output:
        Tuple of (tokenizer, model)
    """
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
    
    print(f"[LLM] Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    print("[LLM] Tokenizer loaded.")
    
    quant_str = f" ({quantization})" if quantization else ""
    print(f"[LLM] Loading model{quant_str} (this may take a while for large models)...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_cfg,
            device_map=device_map,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        print("[LLM] Model loaded.")
    except ImportError as e:
        if "bitsandbytes" in str(e) and "0.43.1" in str(e):
            # Transformers version check failed - try workaround
            print("[LLM] WARNING: Transformers version check failed. Attempting workaround...")
            print("[LLM] This might be a false positive if bitsandbytes 0.48.2 is installed.")
            print("[LLM] Trying to load without explicit quantization config...")
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
                    torch_dtype=compute_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                )
                print("[LLM] Model loaded (workaround succeeded).")
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
    if torch.cuda.is_available():
        end_vram = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        metrics["peak_vram_gb"] = peak_vram
        metrics["end_vram_gb"] = end_vram
    else:
        metrics["peak_vram_gb"] = 0.0
    
    # CPU usage (if psutil available)
    if psutil is not None:
        metrics["avg_cpu_percent"] = psutil.cpu_percent(interval=0.1)
    else:
        metrics["avg_cpu_percent"] = 0.0
    
    return metrics


# ============================================================================
# Pipeline Stage Functions
# ============================================================================

def load_and_preprocess_audio(audio_path: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio from file, resample to 16kHz, convert to mono, and normalize.
    
    Input:
        audio_path: Path to the audio file
    
    Output:
        Tuple of (processed_audio_tensor, sample_rate)
    """
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


def transcribe_audio(audio_tensor: torch.Tensor, sample_rate: int, config: Dict[str, Any]) -> str:
    """
    Transcribe audio using Whisper ASR model.
    
    Input:
        audio_tensor: Preprocessed audio tensor [1, frames]
        sample_rate: Sample rate of the audio
        config: SOLAS_CONFIG dictionary
    
    Output:
        Transcribed text (str)
    """
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
    
    print(f"[ASR] Loading model: {asr_model_id}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        asr_model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    processor = AutoProcessor.from_pretrained(asr_model_id)
    
    # Move model to device
    model_device = torch.device(device)
    model = model.to(model_device)
    print(f"[ASR] Model loaded on {device}.")
    
    # Process audio
    audio_np = audio_tensor.squeeze(0).cpu().numpy().astype(np.float32)
    input_audio = {"array": audio_np, "sampling_rate": int(sample_rate)}
    
    # Extract features
    inputs = processor(
        input_audio["array"],
        sampling_rate=input_audio["sampling_rate"],
        return_tensors="pt",
    )
    # Move to device and convert to model's dtype
    inputs = {k: v.to(model_device).to(dtype) if v.dtype.is_floating_point else v.to(model_device) 
              for k, v in inputs.items()}
    
    # Create attention mask to fix warning about pad/eos token conflict
    # For Whisper, all input features should be attended to (no padding in mel spectrograms)
    input_features = inputs["input_features"]
    attention_mask = torch.ones(
        input_features.shape[:2], 
        dtype=torch.long, 
        device=model_device
    )
    
    # Prepare generation kwargs - Use task and language parameters (preferred over forced_decoder_ids)
    gen_kwargs = {
        "task": "transcribe",
        "attention_mask": attention_mask
    }
    if source_language != "auto":
        gen_kwargs["language"] = source_language
    # If "auto", let Whisper detect language automatically (language=None or omit)
    
    # Calculate audio duration for progress estimation
    audio_duration_seconds = audio_np.shape[0] / sample_rate
    
    # Generate transcription using model's own chunking mechanism
    print(f"[ASR] Transcribing audio ({audio_duration_seconds:.1f}s duration)...")
    
    # Create progress indicator
    # Whisper typically processes at ~10-30x real-time, so estimate total time
    estimated_processing_time = max(5.0, audio_duration_seconds / 20.0)  # Conservative estimate
    start_time = time.time()
    
    if TQDM_AVAILABLE:
        # Use tqdm progress bar
        pbar = tqdm(total=100, desc="Transcribing", unit="%", ncols=80)
        
        def update_progress():
            elapsed = time.time() - start_time
            progress = min(95, int((elapsed / estimated_processing_time) * 100))
            pbar.update(progress - pbar.n)
            return elapsed < estimated_processing_time * 1.5  # Stop updating after 1.5x estimate
        
        # Start a simple timer-based progress update
        import threading
        stop_flag = threading.Event()
        
        def progress_updater():
            while not stop_flag.is_set() and update_progress():
                time.sleep(0.5)
            pbar.update(100 - pbar.n)  # Complete the bar
            pbar.close()
        
        progress_thread = threading.Thread(target=progress_updater, daemon=True)
        progress_thread.start()
    else:
        progress_thread = None
        stop_flag = None
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            **gen_kwargs
        )
    
    if TQDM_AVAILABLE and progress_thread:
        stop_flag.set()
        if progress_thread.is_alive():
            progress_thread.join(timeout=1.0)
    
    # Decode transcription
    transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    actual_time = time.time() - start_time
    speed_factor = audio_duration_seconds / actual_time if actual_time > 0 else 0
    print(f"[ASR] Transcription complete. ({actual_time:.1f}s, {speed_factor:.1f}x real-time)")
    
    # Cleanup
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return transcript


def translate_transcript(transcript: str, config: Dict[str, Any]) -> str:
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
    chunk_size_chars = config.get("chunk_size_chars", 4000)
    target_language = config.get("target_language", "Portuguese")
    max_new_tokens = config.get("translation_max_new_tokens", 1024)
    
    print(f"[Translation] Loading LLM: {llm_model_id}...")
    tokenizer, model = ensure_llm(llm_model_id, quantization)
    
    chunks = chunk_text(transcript, chunk_size_chars)
    print(f"[Translation] Processing {len(chunks)} chunk(s)...")
    translated_parts = []
    
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    for i, chunk in enumerate(chunks, 1):
        if len(chunks) > 1:
            print(f"[Translation] Processing chunk {i}/{len(chunks)}...")
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
        translated_parts.append(translated)
    
    return "\n".join(translated_parts).strip()


def summarize_text(translated_text: str, config: Dict[str, Any]) -> str:
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
    chunk_size_chars = config.get("chunk_size_chars", 4000)
    summary_mode = config.get("summary_mode", "greedy")
    max_new_tokens = config.get("summary_max_new_tokens", 512)
    
    print(f"[Summary] Loading LLM: {llm_model_id}...")
    tokenizer, model = ensure_llm(llm_model_id, quantization)
    
    chunks = chunk_text(translated_text, chunk_size_chars)
    partial_bullets = []
    
    # Chunk-level generation kwargs
    if summary_mode == "sampled":
        gen_chunk_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.2,
            "top_p": 0.9,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    else:
        gen_chunk_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    
    # Aggregation generation kwargs
    if summary_mode == "hybrid":
        gen_agg_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    else:
        gen_agg_kwargs = gen_chunk_kwargs
    
    chunk_prompt_header = (
        "Extract ONLY the most important key points from the transcript segment below. "
        "Focus on core concepts, definitions, equations, conclusions, and actionable takeaways. "
        "Write a concise markdown bulleted list (use '-' bullets). Do not include any preamble or epilogue."
    )
    
    for chunk in chunks:
        messages = [
            {"role": "system", "content": "You are an expert technical summarizer."},
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
    
    # Aggregation
    aggregation_prompt = (
        "You will be given multiple markdown bullet lists produced from segments of the same lecture transcript. "
        "Merge and deduplicate them into a single, concise, well-structured set of bullets. "
        "Prioritize clarity and completeness, keep only the most salient points, and maintain markdown '-' bullets."
    )
    
    merged_input = "\n\n".join(partial_bullets)
    messages = [
        {"role": "system", "content": "You are an expert editor and summarizer."},
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


def generate_podcast_script(translated_text: str, summary: str, config: Dict[str, Any]) -> str:
    """
    Generate conversational podcast script from translated transcript and summary.
    
    Input:
        translated_text: Translated transcript
        summary: Key points summary
        config: SOLAS_CONFIG dictionary
    
    Output:
        Podcast script with Host A/B dialogue (str)
    """
    llm_model_id = config["llm_model_id"]
    quantization = config.get("quantization")
    podcast_creativity_temp = config.get("podcast_creativity_temp", 0.3)
    max_new_tokens = config.get("podcast_max_new_tokens", 1024)
    target_language = config.get("target_language", "Portuguese")
    
    print(f"[Podcast Script] Loading LLM: {llm_model_id}...")
    tokenizer, model = ensure_llm(llm_model_id, quantization)
    
    combined = translated_text.strip()
    if summary:
        combined += "\n\n---\nKey Points Summary (for structure):\n" + summary
    
    system_prompt = (
        f"You are a scriptwriter for an educational podcast. Create a two-host conversation in {target_language} that is engaging, "
        "technically accurate, and pedagogically effective. Use plain language and helpful analogies. "
        "The dialogue must include clear speaker labels at the start of each line: 'Host A:' and 'Host B:'. "
        "Avoid meta commentary, stage directions, or non-dialogue text. "
        f"All dialogue content must be written in {target_language}."
    )
    
    user_prompt = (
        f"Write a podcast script in {target_language} using BOTH inputs below:\n"
        "1) Translated transcript: treat this as the primary factual source.\n"
        "2) Key points summary: use this to structure the flow and ensure coverage of salient topics.\n"
        "Requirements:\n"
        "- Host A explains; Host B asks concise, clarifying questions.\n"
        "- Keep it focused, progressive, and practical; prefer short turns.\n"
        f"- All dialogue must be in {target_language}.\n"
        "- Strictly output alternating lines beginning with 'Host A:' or 'Host B:' and nothing else.\n\n"
        f"INPUTS:\n\n{combined}"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except Exception:
        fallback_prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        input_ids = tokenizer(fallback_prompt, return_tensors="pt").input_ids
    
    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model_device)
    
    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
    
    gen_only = output_ids[0, input_ids.shape[1]:]
    script = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
    
    # Enforce label format
    lines = [line.strip() for line in script.splitlines() if line.strip()]
    fixed_lines = []
    expected = "Host A:"
    for line in lines:
        if line.startswith("Host A:") or line.startswith("Host B:"):
            fixed_lines.append(line)
            expected = "Host B:" if line.startswith("Host A:") else "Host A:"
        else:
            fixed_lines.append(f"{expected} {line}")
            expected = "Host B:" if expected == "Host A:" else "Host A:"
    
    return "\n".join(fixed_lines)


def synthesize_podcast(script: str, config: Dict[str, Any]) -> str:
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
    
    # Suppress jieba SyntaxWarnings (invalid escape sequences in regex patterns)
    import warnings
    warnings.filterwarnings("ignore", category=SyntaxWarning, module="jieba")
    
    # Install torchcodec if not available (required by torchaudio for some audio loading operations)
    # torchcodec is needed when torchaudio tries to use it as a backend
    try:
        import torchcodec
    except ImportError:
        try:
            import subprocess
            import sys
            print("[TTS] Installing torchcodec (required for audio loading)...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "torchcodec", "-q"],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                import torchcodec
                print("[TTS] ✓ torchcodec installed successfully")
            else:
                print(f"[TTS] ⚠ Could not install torchcodec: {result.stderr}")
                print("[TTS] Attempting to continue with soundfile backend...")
        except subprocess.TimeoutExpired:
            print("[TTS] ⚠ torchcodec installation timed out")
        except Exception as e:
            print(f"[TTS] ⚠ Could not install torchcodec: {e}")
            print("[TTS] Attempting to continue with soundfile backend...")
    
    # Lazy import TTS - only import when this function is called
    from TTS.api import TTS as COQUI_TTS
    tts_engine = COQUI_TTS(model_name=tts_model_id)
    try:
        tts_engine = tts_engine.to(device)
    except Exception:
        pass
    
    # Determine output sample rate
    output_sample_rate = getattr(tts_engine, "output_sample_rate", 24000)
    
    lines = [line.strip() for line in script.splitlines() if line.strip()]
    segments = []
    
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
        
        try:
            wav = tts_engine.tts(text=text, language=lang_code, speaker_wav=speaker_wav)
        except TypeError:
            wav = tts_engine.tts(text=text, speaker_wav=speaker_wav, language=lang_code)
        
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        segments.append(wav)
    
    if not segments:
        raise RuntimeError("No audio segments were generated.")
    
    # Concatenate and save
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
    pipeline_start_time = time.time()
    performance_metrics = {}
    
    # Stage 1: Load and preprocess audio
    print("\n[Pipeline] Stage 1/6: Loading and preprocessing audio...")
    if progress_callback:
        progress_callback(1, "Loading and preprocessing audio", 0)
    stage_start = time.time()
    audio_tensor, sample_rate = load_and_preprocess_audio(config["input_audio_path"])
    performance_metrics["audio_preprocessing"] = _collect_performance_metrics("audio_preprocessing", stage_start)
    if progress_callback:
        progress_callback(1, "Loading and preprocessing audio", 100)
    print(f"[Pipeline] ✓ Stage 1 complete ({performance_metrics['audio_preprocessing']['time_seconds']:.2f}s)")
    
    # Stage 2: ASR transcription
    print("[Pipeline] Stage 2/6: Transcribing audio (ASR)...")
    if progress_callback:
        progress_callback(2, "Transcribing audio (ASR)", 0)
    stage_start = time.time()
    original_transcript = transcribe_audio(audio_tensor, sample_rate, config)
    performance_metrics["asr"] = _collect_performance_metrics("asr", stage_start)
    if progress_callback:
        progress_callback(2, "Transcribing audio (ASR)", 100)
    print(f"[Pipeline] ✓ Stage 2 complete ({performance_metrics['asr']['time_seconds']:.2f}s)")
    print(f"[Pipeline] Transcript length: {len(original_transcript)} characters")
    
    # Stage 3: Translation
    print("[Pipeline] Stage 3/6: Translating transcript...")
    if progress_callback:
        progress_callback(3, "Translating transcript", 0)
    stage_start = time.time()
    translated_transcript = translate_transcript(original_transcript, config)
    performance_metrics["translation"] = _collect_performance_metrics("translation", stage_start)
    if progress_callback:
        progress_callback(3, "Translating transcript", 100)
    print(f"[Pipeline] ✓ Stage 3 complete ({performance_metrics['translation']['time_seconds']:.2f}s)")
    
    # Stage 4: Summarization
    print("[Pipeline] Stage 4/6: Generating key points summary...")
    if progress_callback:
        progress_callback(4, "Generating key points summary", 0)
    stage_start = time.time()
    key_points_summary = summarize_text(translated_transcript, config)
    performance_metrics["summary"] = _collect_performance_metrics("summary", stage_start)
    if progress_callback:
        progress_callback(4, "Generating key points summary", 100)
    print(f"[Pipeline] ✓ Stage 4 complete ({performance_metrics['summary']['time_seconds']:.2f}s)")
    
    # Stage 5: Podcast script generation
    print("[Pipeline] Stage 5/6: Generating podcast script...")
    if progress_callback:
        progress_callback(5, "Generating podcast script", 0)
    stage_start = time.time()
    podcast_script = generate_podcast_script(translated_transcript, key_points_summary, config)
    performance_metrics["podcast_script"] = _collect_performance_metrics("podcast_script", stage_start)
    if progress_callback:
        progress_callback(5, "Generating podcast script", 100)
    print(f"[Pipeline] ✓ Stage 5 complete ({performance_metrics['podcast_script']['time_seconds']:.2f}s)")
    
    # Stage 6: TTS synthesis
    print("[Pipeline] Stage 6/6: Synthesizing podcast audio (TTS)...")
    if progress_callback:
        progress_callback(6, "Synthesizing podcast audio (TTS)", 0)
    stage_start = time.time()
    podcast_audio_path = synthesize_podcast(podcast_script, config)
    tts_metrics = _collect_performance_metrics("tts", stage_start)
    if progress_callback:
        progress_callback(6, "Synthesizing podcast audio (TTS)", 100)
    
    # Calculate audio duration for RTF
    try:
        audio_info = sf.info(podcast_audio_path)
        audio_duration = audio_info.duration
        tts_metrics["audio_duration_seconds"] = audio_duration
        tts_metrics["real_time_factor"] = tts_metrics["time_seconds"] / audio_duration if audio_duration > 0 else 0.0
    except Exception:
        tts_metrics["audio_duration_seconds"] = 0.0
        tts_metrics["real_time_factor"] = 0.0
    
    performance_metrics["tts"] = tts_metrics
    print(f"[Pipeline] ✓ Stage 6 complete ({tts_metrics['time_seconds']:.2f}s)")
    
    # Total runtime
    performance_metrics["total_runtime_seconds"] = time.time() - pipeline_start_time
    
    # Save text artifacts to files
    print("[Pipeline] Saving output files...")
    output_dir = Path(config.get("output_directory", "/content/solas_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    translated_path = str(output_dir / "translated_transcript.txt")
    summary_path = str(output_dir / "key_points.md")
    script_path = str(output_dir / "podcast_script.txt")
    
    with open(translated_path, "w", encoding="utf-8") as f:
        f.write(translated_transcript)
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(key_points_summary)
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(podcast_script)
    
    print(f"[Pipeline] ✓ Pipeline complete! Total time: {performance_metrics['total_runtime_seconds']:.2f}s")
    print(f"[Pipeline] Output directory: {output_dir}")
    
    return {
        "text_outputs": {
            "original_transcript": original_transcript,
            "translated_transcript": translated_transcript,
            "key_points_summary": key_points_summary,
            "podcast_script": podcast_script,
        },
        "file_paths": {
            "translated_transcript": translated_path,
            "key_points_summary": summary_path,
            "podcast_script": script_path,
            "final_podcast_audio": podcast_audio_path,
        },
        "performance_metrics": performance_metrics,
    }


# ============================================================================
# Interactive Notebook Helper Functions
# ============================================================================

def setup_environment():
    """
    Install dependencies and check GPU availability.
    Call this once at the start of the interactive notebook.
    """
    import sys
    import subprocess
    import shutil
    import platform
    import os
    
    print(f"[SOLAS] Python {sys.version.split()[0]} | Platform: {platform.platform()}")
    print(f"[SOLAS] CWD: {os.getcwd()}")
    
    def pip_install(pkgs):
        try:
            print(f"[setup] pip install: {pkgs}")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
            print("[setup] pip install done")
        except Exception as e:
            print(f"[setup] pip install failed: {e}")
    
    def apt_install(pkgs):
        try:
            if shutil.which("apt-get"):
                print(f"[setup] apt-get install: {pkgs}")
                subprocess.run(["apt-get", "update", "-y"], check=True, capture_output=True)
                subprocess.run(["apt-get", "install", "-y"] + pkgs, check=True, capture_output=True)
                print("[setup] apt-get install done")
        except Exception as e:
            print(f"[setup] apt-get failed: {e}")
    
    # System deps needed by audio/TTS (Colab)
    if 'google.colab' in sys.modules:
        apt_install(["espeak-ng", "libsndfile1", "ffmpeg"])
    
    # Install torch/torchaudio appropriately
    try:
        import torch, torchaudio  # noqa: F401
    except Exception:
        if 'google.colab' in sys.modules:
            pip_install(["--index-url", "https://download.pytorch.org/whl/cu121", "torch", "torchaudio"])
        else:
            pip_install(["torch", "torchaudio"])
    
    all_packages = [
        "transformers==4.56.2",
        "accelerate==1.10.1",
        "librosa==0.11.0",
        "pydub==0.25.1",
        "SoundFile==0.13.1",
        "datasets==4.0.0",
        "sentencepiece==0.2.1",
        "ipywidgets==7.7.1",
        "widgetsnbextension==3.6.10",
        "bitsandbytes>=0.43.1",  # Required for 4-bit quantization
        "coqui-tts",
        "tqdm",
    ]
    
    pip_install(all_packages)
    
    # Authenticate with Hugging Face if token was provided
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            print("[SOLAS] ✓ Authenticated with Hugging Face Hub")
        except ImportError:
            print("[SOLAS] ℹ huggingface_hub not available yet, token will be used when needed")
        except Exception as e:
            print(f"[SOLAS] ⚠ Could not authenticate with Hugging Face: {e}")
            print("[SOLAS] Token is set as environment variable and will be used automatically")
    else:
        print("[SOLAS] ℹ No Hugging Face token found (using public access)")
    
    # Verify bitsandbytes installation and version
    bnb_just_installed = False
    
    # Check if bitsandbytes was already available before we installed packages
    bnb_was_available = 'bitsandbytes' in sys.modules
    
    try:
        import bitsandbytes
        import importlib.metadata
        from packaging import version as pkg_version
        bnb_version = importlib.metadata.version("bitsandbytes")
        print(f"[SOLAS] bitsandbytes version: {bnb_version}")
        
        if pkg_version.parse(bnb_version) < pkg_version.parse("0.43.1"):
            print(f"[SOLAS] WARNING: bitsandbytes version {bnb_version} is too old. Version >= 0.43.1 required.")
            print("[SOLAS] Attempting to upgrade bitsandbytes...")
            pip_install(["-U", "bitsandbytes"])
            bnb_just_installed = True  # Mark as just installed/upgraded
            # Re-check version after upgrade
            try:
                importlib.reload(bitsandbytes)
                bnb_version = importlib.metadata.version("bitsandbytes")
                print(f"[SOLAS] bitsandbytes upgraded to version: {bnb_version}")
            except Exception:
                print("[SOLAS] WARNING: Could not verify bitsandbytes version after upgrade.")
        else:
            print(f"[SOLAS] ✓ bitsandbytes version {bnb_version} is compatible for 4-bit quantization")
            
    except ImportError:
        # bitsandbytes wasn't available, so it was just installed
        bnb_just_installed = True
        try:
            import bitsandbytes
            import importlib.metadata
            bnb_version = importlib.metadata.version("bitsandbytes")
            print(f"[SOLAS] bitsandbytes installed, version: {bnb_version}")
        except Exception as e:
            print(f"[SOLAS] WARNING: bitsandbytes installed but could not verify version: {e}")
    
    except Exception as e:
        print(f"[SOLAS] WARNING: Could not verify bitsandbytes installation: {e}")
        print("[SOLAS] 4-bit quantization may not work. If you need it:")
        print("[SOLAS]   1. Try: pip install -U bitsandbytes")
        print("[SOLAS]   2. Restart runtime: Runtime → Restart runtime")
        print("[SOLAS]   3. Re-run the 'Setup Environment' cell")

    # Important: bitsandbytes requires runtime restart after installation to work with CUDA
    if bnb_just_installed and not bnb_was_available:
        print()
        print("=" * 60)
        print("⚠️  RUNTIME RESTART REQUIRED")
        print("=" * 60)
        print("bitsandbytes was just installed. It requires a runtime restart to properly")
        print("initialize CUDA bindings. Without restarting, 4-bit quantization will fail.")
        print()
        print("Next steps:")
        print("  1. Click: Runtime → Restart runtime")
        print("  2. After restart, re-run this 'Setup Environment' cell")
        print("  3. Then continue with the rest of the notebook cells")
        print()
        print("NOTE: Restarting clears Python variables but keeps installed packages.")
        print("=" * 60)
        print()
    
    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[SOLAS] GPU detected: {device_name}")
    else:
        print('WARNING: No GPU detected. This notebook will run extremely slowly without a GPU.')


def create_config_widgets():
    """
    Create and return all configuration widgets for the interactive interface.
    
    Returns:
        Dictionary containing all widgets and the config box widget
    """
    try:
        import ipywidgets as widgets
        from google.colab import output as _colab_output
        _colab_output.enable_custom_widget_manager()
    except Exception:
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
            options=["English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian", "Chinese", "Japanese", "Korean"],
            value="English",
            description='Source Lang:'
        ),
        "target_lang_dropdown": widgets.Dropdown(
            options=["Portuguese", "English", "Spanish", "French", "German", "Italian", "Russian", "Chinese", "Japanese", "Korean"],
            value="Portuguese",
            description='Target Lang:'
        ),
        "input_audio_text": widgets.Text(
            value="/content/SOLAS/input_audio_samples/short.wav",
            description='Audio Path:'
        ),
        "output_dir_text": widgets.Text(
            value="/content/solas_outputs",
            description='Output Dir:'
        ),
        "host_a_text": widgets.Text(
            value="/content/SOLAS/TTS_voice_samples/male.wav",
            description='Host A Voice:'
        ),
        "host_b_text": widgets.Text(
            value="/content/SOLAS/TTS_voice_samples/female.wav",
            description='Host B Voice:'
        ),
    }
    
    # Create config box
    config_box = widgets.VBox([
        widgets.HTML('<h3>SOLAS Configuration</h3>'),
        widgets.HTML('<b>Models</b>'),
        widgets_dict["asr_dropdown"],
        widgets_dict["llm_dropdown"],
        widgets_dict["quantization_dropdown"],
        widgets.HTML('<b>Processing Parameters</b>'),
        widgets_dict["chunk_size_dropdown"],
        widgets_dict["summary_mode_dropdown"],
        widgets_dict["podcast_temp_slider"],
        widgets.HTML('<b>Languages</b>'),
        widgets_dict["source_lang_dropdown"],
        widgets_dict["target_lang_dropdown"],
    ])
    
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
        return widgets.VBox([
            widgets.HTML('<b>Upload input audio</b>'),
            upload_btn
        ])
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
        return widgets.VBox([
            widgets.HTML('<b>Upload input audio</b>'),
            uploader,
            status
        ])


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
        return widgets.VBox([
            widgets.HTML(f'<b>Upload {host_name} voice (for TTS cloning)</b>'),
            upload_btn
        ])
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
        return widgets.VBox([
            widgets.HTML(f'<b>Upload {host_name} voice (for TTS cloning)</b>'),
            uploader,
            status
        ])


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
    
    return {
        # Variable inputs
        "asr_model_id": widgets_dict["asr_dropdown"].value,
        "llm_model_id": widgets_dict["llm_dropdown"].value,
        "quantization": quantization_value,
        "chunk_size_chars": widgets_dict["chunk_size_dropdown"].value,
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
    Display pipeline results in a user-friendly format.
    
    Args:
        results: Dictionary returned from run_pipeline()
    """
    from IPython.display import Audio, display
    
    # Display text outputs
    print("=" * 60)
    print("TRANSLATED TRANSCRIPT")
    print("=" * 60)
    print(results["text_outputs"]["translated_transcript"][:500] + "...")
    print("\n")
    
    print("=" * 60)
    print("KEY POINTS SUMMARY")
    print("=" * 60)
    print(results["text_outputs"]["key_points_summary"])
    print("\n")
    
    print("=" * 60)
    print("PODCAST SCRIPT")
    print("=" * 60)
    print(results["text_outputs"]["podcast_script"][:500] + "...")
    print("\n")
    
    # Display performance metrics
    print("=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    metrics = results["performance_metrics"]
    print(f"Total Runtime: {metrics['total_runtime_seconds']:.2f} seconds")
    print(f"ASR: {metrics['asr']['time_seconds']:.2f}s, VRAM: {metrics['asr']['peak_vram_gb']:.2f}GB")
    print(f"Translation: {metrics['translation']['time_seconds']:.2f}s, VRAM: {metrics['translation']['peak_vram_gb']:.2f}GB")
    print(f"Summary: {metrics['summary']['time_seconds']:.2f}s, VRAM: {metrics['summary']['peak_vram_gb']:.2f}GB")
    print(f"Podcast Script: {metrics['podcast_script']['time_seconds']:.2f}s, VRAM: {metrics['podcast_script']['peak_vram_gb']:.2f}GB")
    print(f"TTS: {metrics['tts']['time_seconds']:.2f}s, RTF: {metrics['tts']['real_time_factor']:.2f}, VRAM: {metrics['tts']['peak_vram_gb']:.2f}GB")
    
    # Display audio
    print("\n" + "=" * 60)
    print("PODCAST AUDIO")
    print("=" * 60)
    audio_path = results["file_paths"]["final_podcast_audio"]
    display(Audio(filename=audio_path))
    print(f"Audio saved to: {audio_path}")


def run_pipeline_from_widgets(widgets_dict):
    """
    Run the pipeline directly using the current widget values.
    
    Args:
        widgets_dict: Dictionary from create_config_widgets()
    
    Returns:
        Results dictionary from run_pipeline()
    """
    import json
    
    print("Building configuration from widgets...")
    config = build_config_from_widgets(widgets_dict)
    
    print("Configuration:")
    print(json.dumps({k: v for k, v in config.items() if k not in ['host_a_wav_path', 'host_b_wav_path']}, indent=2))
    print("\nStarting pipeline execution...\n")
    
    results = run_pipeline(config)
    display_results(results)
    
    return results

