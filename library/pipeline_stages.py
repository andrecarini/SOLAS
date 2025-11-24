"""
Pipeline stages: Core functions for each SOLAS pipeline stage.

This module provides the main pipeline stage functions:
- load_and_preprocess_audio: Audio loading and preprocessing
- transcribe_audio: ASR transcription using Whisper
- translate_transcript: Translation using LLM
- summarize_text: Key points summarization using LLM
- generate_podcast_script: Podcast script generation using LLM
- synthesize_podcast: TTS synthesis using Coqui XTTS
"""

import os
import gc
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Callable

from .pipeline_utils import (
    get_verbosity,
    get_device,
    suppress_torch_dtype_warnings,
    suppress_generation_flags_warnings,
    suppress_tts_warnings,
    SuppressOutput,
    chunk_text,
    collect_performance_metrics,
)
from .pipeline_models import ensure_llm


# Lazy import for log_setup to avoid circular imports
def _get_log_setup():
    """Get log_setup function, importing lazily to avoid circular imports."""
    try:
        from .pipeline_setup import log_setup
        return log_setup
    except ImportError:
        # Fallback if pipeline_setup not yet available
        def log_setup(msg, level='info', verbose=False):
            if verbose:
                print(f"[{level.upper()}] {msg}")
        return log_setup


# =============================================================================
# STAGE 1: AUDIO LOADING
# =============================================================================

def load_and_preprocess_audio(audio_path: str) -> Tuple[Any, int]:
    """
    Load audio from file, resample to 16kHz, convert to mono, and normalize.

    Args:
        audio_path: Path to the audio file

    Returns:
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
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.transpose(0, 1)

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

    # Peak normalize
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    return waveform.contiguous(), target_sr


# =============================================================================
# STAGE 2: ASR TRANSCRIPTION
# =============================================================================

def transcribe_audio(
    audio_tensor: Any,
    sample_rate: int,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[int, str, int], None]] = None
) -> str:
    """
    Transcribe audio using Whisper ASR model with native long-form transcription.

    Args:
        audio_tensor: Preprocessed audio tensor [1, frames]
        sample_rate: Sample rate of the audio
        config: SOLAS_CONFIG dictionary
        progress_callback: Optional callback for progress updates

    Returns:
        Transcribed text
    """
    import torch
    import numpy as np
    from transformers import pipeline
    import threading

    log_setup = _get_log_setup()
    verbose = get_verbosity()
    device = get_device()
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    asr_model_id = config["asr_model_id"]
    source_language = config.get("source_language", "auto")

    # Convert language name to code
    if source_language != "auto":
        lang_name_to_code = {
            "English": "en", "Spanish": "es", "French": "fr",
            "German": "de", "Italian": "it", "Portuguese": "pt",
            "Russian": "ru", "Chinese": "zh", "Japanese": "ja", "Korean": "ko",
        }
        source_language = lang_name_to_code.get(source_language, source_language)

    suppress_torch_dtype_warnings()

    # Process audio
    audio_np = audio_tensor.squeeze(0).cpu().numpy().astype(np.float32)
    audio_duration_seconds = audio_np.shape[0] / sample_rate

    log_setup(f"[ASR] Loading model: {asr_model_id}...", 'info', verbose)
    log_setup(f"[ASR] Audio duration: {audio_duration_seconds:.1f}s", 'info', verbose)

    # Suppress transformers warnings
    original_verbosity = os.environ.get("TRANSFORMERS_VERBOSITY", "warning")
    if not verbose:
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    pipe = pipeline(
        "automatic-speech-recognition",
        model=asr_model_id,
        dtype=dtype,
        device=device,
        model_kwargs={"use_safetensors": True}
    )

    if not verbose:
        os.environ["TRANSFORMERS_VERBOSITY"] = original_verbosity

    log_setup(f"[ASR] Model loaded on {device}.", 'info', verbose)

    generate_kwargs = {"task": "transcribe"}
    if source_language != "auto":
        generate_kwargs["language"] = source_language

    if audio_duration_seconds > 30:
        log_setup(f"[ASR] Long audio detected, using Whisper's native long-form transcription...", 'info', verbose)

    # Progress tracking
    estimated_processing_time = max(5.0, audio_duration_seconds / 20.0)
    start_time = time.time()
    stop_flag = threading.Event()
    progress_thread = None

    if progress_callback:
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

    result = pipe(
        audio_np,
        generate_kwargs=generate_kwargs,
        return_timestamps=True
    )

    if progress_thread is not None:
        stop_flag.set()
        if progress_thread.is_alive():
            progress_thread.join(timeout=1.0)

    if progress_callback:
        progress_callback(2, "Transcribing audio (ASR)", 100)

    if "chunks" in result:
        transcript = " ".join([chunk["text"] for chunk in result["chunks"]]).strip()
    else:
        transcript = result["text"].strip()

    actual_time = time.time() - start_time
    speed_factor = audio_duration_seconds / actual_time if actual_time > 0 else 0
    log_setup(f"[ASR] Transcription complete ({actual_time:.1f}s, {speed_factor:.1f}x real-time)", 'success', verbose)

    # Cleanup
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return transcript


# =============================================================================
# STAGE 3: TRANSLATION
# =============================================================================

def translate_transcript(
    transcript: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[int, str, int], None]] = None
) -> str:
    """
    Translate transcript to target language using LLM.

    Args:
        transcript: Original transcript text
        config: SOLAS_CONFIG dictionary
        progress_callback: Optional callback for progress updates

    Returns:
        Translated text
    """
    import torch

    log_setup = _get_log_setup()
    verbose = get_verbosity()

    llm_model_id = config["llm_model_id"]
    quantization = config.get("quantization")
    chunk_size_chars = config.get("chunk_size_chars", 2000)
    repetition_penalty = config["repetition_penalty"] if "repetition_penalty" in config else 1.2
    target_language = config.get("target_language", "Portuguese")
    max_new_tokens = config.get("translation_max_new_tokens", 1024)

    suppress_generation_flags_warnings()
    log_setup(f"[Translation] Loading LLM: {llm_model_id}...", 'info', verbose)
    tokenizer, model = ensure_llm(llm_model_id, quantization, log_fn=log_setup)

    chunks = chunk_text(transcript, chunk_size_chars)
    log_setup(f"Processing {len(chunks)} chunk(s)...", 'info', verbose)
    translated_parts = []

    total_chunks = len(chunks)
    model_device = next(model.parameters()).device

    for i, chunk in enumerate(chunks, 1):
        if len(chunks) > 1:
            log_setup(f"Processing chunk {i}/{len(chunks)}...", 'info', verbose)

        if progress_callback:
            progress = int((i - 1) / total_chunks * 100)
            progress_callback(3, "Translating transcript", progress)

        chunk_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
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
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        except Exception:
            fallback_prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
            input_ids = tokenizer(fallback_prompt, return_tensors="pt").input_ids

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

        log_setup(f"Chunk {i}: Input {len(chunk)} chars ({chunk_tokens} tokens), Output {len(translated)} chars", 'info', verbose)
        translated_parts.append(translated)

    return "\n".join(translated_parts).strip()


# =============================================================================
# STAGE 4: SUMMARIZATION
# =============================================================================

def summarize_text(
    translated_text: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[int, str, int], None]] = None
) -> str:
    """
    Generate key points summary from translated text.

    Args:
        translated_text: Translated transcript text
        config: SOLAS_CONFIG dictionary
        progress_callback: Optional callback for progress updates

    Returns:
        Key points summary as markdown bullets
    """
    import torch
    from transformers import GenerationConfig

    log_setup = _get_log_setup()
    verbose = get_verbosity()

    llm_model_id = config["llm_model_id"]
    quantization = config.get("quantization")
    chunk_size_chars = config.get("chunk_size_chars", 2000)
    repetition_penalty = config["repetition_penalty"] if "repetition_penalty" in config else 1.2
    summary_mode = config.get("summary_mode", "greedy")
    max_new_tokens = config.get("summary_max_new_tokens", 512)
    target_language = config.get("target_language", "English")

    suppress_generation_flags_warnings()
    log_setup(f"[Summary] Loading LLM: {llm_model_id}...", 'info', verbose)
    tokenizer, model = ensure_llm(llm_model_id, quantization, log_fn=log_setup)

    chunks = chunk_text(translated_text, chunk_size_chars)
    partial_bullets = []

    # Base config without max_new_tokens - we'll set it dynamically
    base_config = {
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if repetition_penalty is not None:
        base_config["repetition_penalty"] = repetition_penalty

    # Sampling settings based on mode
    if summary_mode == "sampled" or summary_mode == "hybrid":
        chunk_sampling_config = {"do_sample": True, "temperature": 0.2, "top_p": 0.9}
    else:
        chunk_sampling_config = {"do_sample": False}

    # Aggregation uses greedy for hybrid, same as chunks otherwise
    if summary_mode == "hybrid":
        agg_sampling_config = {"do_sample": False}
    else:
        agg_sampling_config = chunk_sampling_config.copy()

    chunk_prompt_header = (
        f"Extract ONLY the most important key points from the transcript segment below in {target_language}. "
        "Focus on core concepts, definitions, equations, conclusions, and actionable takeaways. "
        f"Write a concise markdown bulleted list (use '-' bullets) in {target_language}. Do not include any preamble or epilogue."
    )

    model_device = next(model.parameters()).device
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        messages = [
            {"role": "system", "content": f"You are an expert technical summarizer. Provide summaries in {target_language}."},
            {"role": "user", "content": f"{chunk_prompt_header}\n\nTranscript segment:\n\n{chunk}"},
        ]

        try:
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        except Exception:
            fallback_prompt = f"System: You are an expert technical summarizer.\nUser: {chunk_prompt_header}\n\nTranscript segment:\n\n{chunk}\nAssistant:"
            input_ids = tokenizer(fallback_prompt, return_tensors="pt").input_ids

        input_ids = input_ids.to(model_device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        # Dynamic token limit based on input chunk size
        chunk_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
        dynamic_max_tokens = max(max_new_tokens, int(chunk_tokens * 0.5))  # Summary is shorter than input
        gen_config = GenerationConfig(**base_config, max_new_tokens=dynamic_max_tokens, **chunk_sampling_config)

        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=gen_config)

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
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    except Exception:
        fallback_prompt = f"System: You are an expert editor and summarizer.\nUser: {aggregation_prompt}\n\nHere are the partial bullet lists:\n\n{merged_input}\nAssistant:"
        input_ids = tokenizer(fallback_prompt, return_tensors="pt").input_ids

    input_ids = input_ids.to(model_device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

    # Dynamic token limit for aggregation - needs enough space to merge all bullets
    merged_tokens = len(tokenizer.encode(merged_input, add_special_tokens=False))
    agg_max_tokens = max(max_new_tokens, int(merged_tokens * 0.75))  # Merged summary can be up to 75% of input
    gen_config_agg = GenerationConfig(**base_config, max_new_tokens=agg_max_tokens, **agg_sampling_config)

    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=gen_config_agg)

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


# =============================================================================
# STAGE 5: PODCAST SCRIPT
# =============================================================================

def generate_podcast_script(
    translated_text: str,
    summary: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[int, str, int], None]] = None
) -> str:
    """
    Generate conversational podcast script from translated transcript and summary.

    Args:
        translated_text: Translated transcript
        summary: Key points summary
        config: SOLAS_CONFIG dictionary
        progress_callback: Optional callback for progress updates

    Returns:
        Podcast script with Host A/B dialogue
    """
    import torch

    log_setup = _get_log_setup()
    verbose = get_verbosity()

    llm_model_id = config["llm_model_id"]
    quantization = config.get("quantization")
    podcast_creativity_temp = config.get("podcast_creativity_temp", 0.3)
    max_new_tokens = config.get("podcast_max_new_tokens", 1024)
    repetition_penalty = config["repetition_penalty"] if "repetition_penalty" in config else 1.2
    target_language = config.get("target_language", "Portuguese")
    chunk_size_chars = config.get("chunk_size_chars", 2000)

    suppress_generation_flags_warnings()
    log_setup(f"[Podcast Script] Loading LLM: {llm_model_id}...", 'info', verbose)
    tokenizer, model = ensure_llm(llm_model_id, quantization, log_fn=log_setup)

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
        "do_sample": True,
        "temperature": podcast_creativity_temp,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    model_device = next(model.parameters()).device
    script_segments = []
    previous_context = ""

    for i, chunk in enumerate(chunks):
        if total_chunks > 1:
            log_setup(f"[Podcast Script] Processing segment {i+1}/{total_chunks}...", 'info', verbose)

        if progress_callback:
            progress = int((i / total_chunks) * 85)
            progress_callback(5, "Generating podcast script", progress)

        # Build prompt based on position
        if i == 0:
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
                messages, add_generation_prompt=True, return_tensors="pt"
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

        # Enforce label format
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

        # Keep context for next chunk
        context_lines = fixed_lines[-3:] if len(fixed_lines) >= 3 else fixed_lines
        previous_context = "\n".join(context_lines)

    if progress_callback:
        progress_callback(5, "Generating podcast script", 90)

    full_script = "\n".join(script_segments)

    # Final cleanup
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


# =============================================================================
# STAGE 6: TTS SYNTHESIS
# =============================================================================

def synthesize_podcast(
    script: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[int, str, int], None]] = None
) -> str:
    """
    Synthesize podcast audio from script using Coqui XTTS.

    Args:
        script: Podcast script with Host A/B labels
        config: SOLAS_CONFIG dictionary
        progress_callback: Optional callback for progress updates

    Returns:
        Path to generated podcast audio file
    """
    import re
    import numpy as np
    import soundfile as sf

    log_setup = _get_log_setup()
    verbose = get_verbosity()

    tts_model_id = config.get("tts_model_id", "tts_models/multilingual/multi-dataset/xtts_v2")
    host_a_wav_path = config.get("host_a_wav_path")
    host_b_wav_path = config.get("host_b_wav_path")
    target_language = config.get("target_language", "Portuguese")
    output_directory = config.get("output_directory", "/content/solas_outputs")

    lang_mapping = {
        "English": "en", "Portuguese": "pt", "Spanish": "es",
        "French": "fr", "German": "de", "Italian": "it",
    }
    lang_code = lang_mapping.get(target_language, "en")

    device = get_device()
    os.environ["COQUI_TOS_AGREED"] = "1"

    if not verbose:
        warnings.filterwarnings("ignore", category=SyntaxWarning)
    suppress_tts_warnings()

    # Install torchcodec if needed
    try:
        import torchcodec
    except ImportError:
        try:
            import subprocess
            import sys
            log_setup("[TTS] Installing torchcodec (required for audio loading)...", 'info', verbose)
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "torchcodec", "-q"],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                log_setup("[TTS] ✓ torchcodec installed successfully", 'success', verbose)
        except Exception as e:
            log_setup(f"[TTS] ⚠ Could not install torchcodec: {e}", 'warning', verbose)

    # Load TTS engine
    with SuppressOutput(suppress=not verbose):
        from TTS.api import TTS as COQUI_TTS
        tts_engine = COQUI_TTS(model_name=tts_model_id)
        try:
            tts_engine = tts_engine.to(device)
        except Exception:
            pass

    output_sample_rate = getattr(tts_engine, "output_sample_rate", 24000)

    def chunk_tts_text(text, max_length=250):
        """Split text into chunks for TTS."""
        if len(text) <= max_length:
            return [text]

        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""

        for sentence in sentences:
            if len(sentence) > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                parts = re.split(r',\s+', sentence)
                for part in parts:
                    if len(part) > max_length:
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
                if len(current_chunk) + len(sentence) + 1 <= max_length:
                    current_chunk = f"{current_chunk} {sentence}".strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    lines = [line.strip() for line in script.splitlines() if line.strip()]
    segments = []

    # Calculate total chunks
    total_chunks = 0
    for line in lines:
        text = line.split(":", 1)[1].strip() if ":" in line else line
        if text:
            total_chunks += len(chunk_tts_text(text, max_length=250))

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

        text_chunks = chunk_tts_text(text, max_length=250)

        for chunk in text_chunks:
            try:
                wav = tts_engine.tts(text=chunk, language=lang_code, speaker_wav=speaker_wav)
            except TypeError:
                wav = tts_engine.tts(text=chunk, speaker_wav=speaker_wav, language=lang_code)

            wav = np.asarray(wav, dtype=np.float32)
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            segments.append(wav)

            processed_chunks += 1
            if progress_callback and total_chunks > 0:
                progress = int((processed_chunks / total_chunks) * 100)
                progress_callback(6, "Synthesizing podcast audio (TTS)", progress)

    if not segments:
        raise RuntimeError("No audio segments were generated.")

    final_audio = np.concatenate(segments)
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = str(out_dir / "podcast.wav")
    sf.write(final_path, final_audio, output_sample_rate, subtype="PCM_16")

    return final_path


# =============================================================================
# CLEANUP
# =============================================================================

def unload_asr():
    """Unload ASR models to free memory."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
