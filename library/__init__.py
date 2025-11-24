"""
SOLAS Library

Provides modules for:
- Pipeline execution (pipeline_*)
- Evaluation framework (evaluation_*)

Note: Heavy dependencies (torch, transformers, etc.) are imported lazily
to allow testing and basic usage without GPU/ML dependencies.
"""

# Pipeline utilities (no heavy deps)
from .pipeline_utils import (
    get_verbosity,
    is_colab_environment,
    check_colab_environment,
    get_device,
    get_compute_dtype,
    get_vram_usage,
    get_peak_vram,
    clear_gpu_memory,
    chunk_text,
    collect_performance_metrics,
    StageMetrics,
    SuppressOutput,
    StopExecution,
    suppress_torch_dtype_warnings,
    suppress_generation_flags_warnings,
    suppress_tts_warnings,
)

# Pipeline templates (no heavy deps)
from .pipeline_templates import (
    load_template,
    get_template_path,
    template_exists,
)

# Pipeline setup (no heavy deps)
from .pipeline_setup import (
    log_setup,
    setup_environment_with_progress,
    show_config_warning,
    show_pipeline_warning,
    show_restart_warning,
    is_package_installed,
    load_requirements,
    SETUP_SYSTEM_PACKAGES,
)

# Pipeline widgets (requires ipywidgets but not torch)
from .pipeline_widgets import (
    create_config_widgets,
    create_audio_upload_widget,
    create_host_voice_upload_widget,
    build_config_from_widgets,
    display_results,
    ASR_MODELS,
    LLM_MODELS,
)

# Lazy imports for modules requiring heavy dependencies (torch, transformers, etc.)
# These are imported on first use to avoid ImportError when torch is not installed
def __getattr__(name):
    """Lazy import for heavy-dependency modules."""
    if name == 'ensure_llm':
        from .pipeline_models import ensure_llm
        return ensure_llm
    elif name == 'create_quantization_config':
        from .pipeline_models import create_quantization_config
        return create_quantization_config
    elif name == 'unload_llm':
        from .pipeline_models import unload_llm
        return unload_llm
    elif name == 'unload_all_models':
        from .pipeline_models import unload_all_models
        return unload_all_models
    elif name == 'load_and_preprocess_audio':
        from .pipeline_stages import load_and_preprocess_audio
        return load_and_preprocess_audio
    elif name == 'transcribe_audio':
        from .pipeline_stages import transcribe_audio
        return transcribe_audio
    elif name == 'translate_transcript':
        from .pipeline_stages import translate_transcript
        return translate_transcript
    elif name == 'summarize_text':
        from .pipeline_stages import summarize_text
        return summarize_text
    elif name == 'generate_podcast_script':
        from .pipeline_stages import generate_podcast_script
        return generate_podcast_script
    elif name == 'synthesize_podcast':
        from .pipeline_stages import synthesize_podcast
        return synthesize_podcast
    elif name == 'run_pipeline':
        from .pipeline_runner import run_pipeline
        return run_pipeline
    elif name == 'EvaluationNotebook':
        from .evaluation_notebook import EvaluationNotebook
        return EvaluationNotebook
    raise AttributeError(f"module 'library' has no attribute '{name}'")
