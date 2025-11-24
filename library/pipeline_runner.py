"""
Pipeline runner: Main orchestrator for the SOLAS pipeline.

This module provides the main entry point for running the complete pipeline.
"""

import os
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from .pipeline_utils import (
    get_verbosity,
    suppress_generation_flags_warnings,
    collect_performance_metrics,
)
from .pipeline_stages import (
    load_and_preprocess_audio,
    transcribe_audio,
    translate_transcript,
    summarize_text,
    generate_podcast_script,
    synthesize_podcast,
)


def _get_log_setup():
    """Get log_setup function, importing lazily to avoid circular imports."""
    try:
        from .pipeline_setup import log_setup
        return log_setup
    except ImportError:
        def log_setup(msg, level='info', verbose=False):
            if verbose:
                print(f"[{level.upper()}] {msg}")
        return log_setup


def run_pipeline(
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[int, str, int], None]] = None
) -> Dict[str, Any]:
    """
    Execute the full SOLAS pipeline.

    Args:
        config: SOLAS_CONFIG dictionary with all parameters
        progress_callback: Optional callback function(stage_num, stage_name, progress_pct)

    Returns:
        Dictionary containing:
        - text_outputs: All generated text artifacts
        - file_paths: Paths to saved files
        - performance_metrics: Timing and resource usage for each stage
    """
    log_setup = _get_log_setup()
    verbose = get_verbosity()

    # Suppress warnings early
    if not verbose:
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TQDM_DISABLE"] = "1"

    suppress_generation_flags_warnings()

    pipeline_start_time = time.time()
    performance_metrics = {}

    # Stage 1: Load and preprocess audio
    log_setup("Stage 1/6: Loading and preprocessing audio...", 'info', verbose)
    if progress_callback:
        progress_callback(1, "Loading and preprocessing audio", 0)
    stage_start = time.time()
    audio_tensor, sample_rate = load_and_preprocess_audio(config["input_audio_path"])
    performance_metrics["audio_preprocessing"] = collect_performance_metrics("audio_preprocessing", stage_start)
    if progress_callback:
        progress_callback(1, "Loading and preprocessing audio", 100)
    log_setup(f"✓ Stage 1 complete ({performance_metrics['audio_preprocessing']['time_seconds']:.2f}s)", 'success', verbose)

    # Stage 2: ASR transcription
    log_setup("Stage 2/6: Transcribing audio (ASR)...", 'info', verbose)
    if progress_callback:
        progress_callback(2, "Transcribing audio (ASR)", 0)
    stage_start = time.time()
    original_transcript = transcribe_audio(audio_tensor, sample_rate, config, progress_callback)
    performance_metrics["asr"] = collect_performance_metrics("asr", stage_start)
    log_setup(f"✓ Stage 2 complete ({performance_metrics['asr']['time_seconds']:.2f}s)", 'success', verbose)
    log_setup(f"Transcript length: {len(original_transcript)} characters", 'info', verbose)

    # Stage 3: Translation
    log_setup("Stage 3/6: Translating transcript...", 'info', verbose)
    if progress_callback:
        progress_callback(3, "Translating transcript", 0)
    stage_start = time.time()
    translated_transcript = translate_transcript(original_transcript, config, progress_callback)
    performance_metrics["translation"] = collect_performance_metrics("translation", stage_start)
    if progress_callback:
        progress_callback(3, "Translating transcript", 100)
    log_setup(f"✓ Stage 3 complete ({performance_metrics['translation']['time_seconds']:.2f}s)", 'success', verbose)

    # Stage 4: Summarization
    log_setup("Stage 4/6: Generating key points summary...", 'info', verbose)
    if progress_callback:
        progress_callback(4, "Generating key points summary", 0)
    stage_start = time.time()
    key_points_summary = summarize_text(translated_transcript, config, progress_callback)
    performance_metrics["summary"] = collect_performance_metrics("summary", stage_start)
    if progress_callback:
        progress_callback(4, "Generating key points summary", 100)
    log_setup(f"✓ Stage 4 complete ({performance_metrics['summary']['time_seconds']:.2f}s)", 'success', verbose)

    # Stage 5: Podcast script generation
    log_setup("Stage 5/6: Generating podcast script...", 'info', verbose)
    if progress_callback:
        progress_callback(5, "Generating podcast script", 0)
    stage_start = time.time()
    podcast_script = generate_podcast_script(translated_transcript, key_points_summary, config, progress_callback)
    performance_metrics["podcast_script"] = collect_performance_metrics("podcast_script", stage_start)
    if progress_callback:
        progress_callback(5, "Generating podcast script", 100)
    log_setup(f"✓ Stage 5 complete ({performance_metrics['podcast_script']['time_seconds']:.2f}s)", 'success', verbose)

    # Stage 6: TTS synthesis
    log_setup("Stage 6/6: Synthesizing podcast audio (TTS)...", 'info', verbose)
    if progress_callback:
        progress_callback(6, "Synthesizing podcast audio (TTS)", 0)
    stage_start = time.time()
    podcast_audio_path = synthesize_podcast(podcast_script, config, progress_callback)
    tts_metrics = collect_performance_metrics("tts", stage_start)
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

    # Save text artifacts
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
