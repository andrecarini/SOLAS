"""
Pipeline widgets: Interactive notebook widgets for configuration and display.

This module provides widget functions for the SOLAS Interactive notebook:
- create_config_widgets: Create configuration widgets
- create_audio_upload_widget: Create audio file upload widget
- create_host_voice_upload_widget: Create voice sample upload widget
- build_config_from_widgets: Build configuration dictionary from widgets
- display_results: Display pipeline results
"""

from pathlib import Path
from typing import Dict, Any

from .pipeline_utils import get_verbosity
from .pipeline_setup import log_setup
from .pipeline_templates import load_template


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


def create_config_widgets() -> Dict[str, Any]:
    """
    Create and return all configuration widgets for the interactive interface.

    Returns:
        Dictionary containing all widgets and the config box widget
    """
    import ipywidgets as widgets

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
            value="",
            description='Audio Path:'
        ),
        "output_dir_text": widgets.Text(
            value="",
            description='Output Dir:'
        ),
        "host_a_text": widgets.Text(
            value="",
            description='Host A Voice:'
        ),
        "host_b_text": widgets.Text(
            value="",
            description='Host B Voice:'
        ),
    }

    # Set default paths dynamically based on environment
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
        border='1px solid var(--color-border)',
        padding='10px',
        margin='10px 0',
        width='auto'
    ))

    widgets_dict["config_box"] = config_box

    return widgets_dict


def create_audio_upload_widget(widgets_dict: Dict[str, Any]) -> Any:
    """
    Create audio upload widget (Colab or local Jupyter).

    Args:
        widgets_dict: Dictionary from create_config_widgets() containing input_audio_text widget

    Returns:
        Widget box for audio upload
    """
    import ipywidgets as widgets

    BASE_DIR = Path('/content') if Path('/content').exists() else Path.cwd()
    input_audio_text = widgets_dict["input_audio_text"]

    try:
        from google.colab import files as colab_files
        is_colab = True
    except ImportError:
        is_colab = False

    if is_colab:
        # Colab file upload - button directly triggers upload (required for security)
        upload_btn = widgets.Button(
            description='Upload Audio File',
            button_style='primary',
            layout=widgets.Layout(width='auto', height='40px')
        )

        def on_colab_upload(_):
            print('Select an audio file to upload...')
            uploaded = colab_files.upload()
            if uploaded:
                name = next(iter(uploaded.keys()))
                data = uploaded[name]
                save_path = BASE_DIR / name
                with open(save_path, 'wb') as f:
                    f.write(data)
                input_audio_text.value = str(save_path)
                print(f'Uploaded and set audio path to: {save_path}')
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

        def on_file_selected(change):
            if not uploader.value:
                return
            (fname, meta), = uploader.value.items()
            data = meta['content']
            save_path = BASE_DIR / fname
            with open(save_path, 'wb') as f:
                f.write(data)
            input_audio_text.value = str(save_path)
            status.value = f'Saved and set audio path to: {save_path}'

        uploader.observe(on_file_selected, names='value')
        label = widgets.Label(value='Upload input audio')
        label.style.font_weight = 'bold'
        return widgets.VBox([label, uploader, status])


def create_host_voice_upload_widget(widgets_dict: Dict[str, Any], host_key: str) -> Any:
    """
    Create audio upload widget for Host A or Host B voice cloning.

    Args:
        widgets_dict: Dictionary from create_config_widgets() containing host text widgets
        host_key: Either "host_a_text" or "host_b_text"

    Returns:
        Widget box for host voice upload
    """
    import ipywidgets as widgets

    BASE_DIR = Path('/content') if Path('/content').exists() else Path.cwd()
    host_text = widgets_dict[host_key]
    host_name = "Host A" if host_key == "host_a_text" else "Host B"

    try:
        from google.colab import files as colab_files
        is_colab = True
    except ImportError:
        is_colab = False

    if is_colab:
        # Colab file upload - button directly triggers upload (required for security)
        upload_btn = widgets.Button(
            description=f'Upload {host_name} Voice',
            button_style='primary',
            layout=widgets.Layout(width='auto', height='40px')
        )

        def on_colab_upload(_):
            print(f'Select {host_name} voice audio file to upload...')
            uploaded = colab_files.upload()
            if uploaded:
                name = next(iter(uploaded.keys()))
                data = uploaded[name]
                save_path = BASE_DIR / name
                with open(save_path, 'wb') as f:
                    f.write(data)
                host_text.value = str(save_path)
                print(f'Uploaded and set {host_name} voice path to: {save_path}')
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

        def on_file_selected(change):
            if not uploader.value:
                return
            (fname, meta), = uploader.value.items()
            data = meta['content']
            save_path = BASE_DIR / fname
            with open(save_path, 'wb') as f:
                f.write(data)
            host_text.value = str(save_path)
            status.value = f'Saved and set {host_name} voice path to: {save_path}'

        uploader.observe(on_file_selected, names='value')
        label = widgets.Label(value=f'Upload {host_name} voice (for TTS cloning)')
        label.style.font_weight = 'bold'
        return widgets.VBox([label, uploader, status])


def build_config_from_widgets(widgets_dict: Dict[str, Any]) -> Dict[str, Any]:
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


def display_results(results: Dict[str, Any]) -> None:
    """
    Display pipeline results in a user-friendly format using HTML.

    In non-debug mode, shows only a success banner. In debug mode, shows full details.

    Args:
        results: Dictionary returned from run_pipeline()
    """
    from IPython.display import Audio, display, HTML

    verbose = get_verbosity()

    log_setup("Displaying pipeline results...", 'info', verbose)

    # In non-debug mode, show only success banner
    if not verbose:
        try:
            html = load_template('pipeline_complete.html')
            display(HTML(html))
        except Exception as e:
            log_setup(f"Error displaying success banner: {e}", 'error', True)
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
            <h4 style="color: var(--color-primary); margin-top: 0;">Translated Transcript</h4>
            <pre style="white-space: pre-wrap; word-wrap: break-word; font-size: 13px; max-height: 200px; overflow-y: auto;">{results["text_outputs"]["translated_transcript"][:1000]}...</pre>
        </div>
        ''')

        # Key Points section
        html_parts.append(f'''
        <div style="margin: 20px 0; padding: 15px; background: var(--color-bg-secondary); border-radius: 8px;">
            <h4 style="color: var(--color-primary); margin-top: 0;">Key Points Summary</h4>
            <pre style="white-space: pre-wrap; word-wrap: break-word; font-size: 13px;">{results["text_outputs"]["key_points_summary"]}</pre>
        </div>
        ''')

        # Performance Metrics section
        html_parts.append(f'''
        <div style="margin: 20px 0; padding: 15px; background: var(--color-bg-secondary); border-radius: 8px;">
            <h4 style="color: var(--color-primary); margin-top: 0;">Performance Metrics</h4>
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
            <h4 style="color: var(--color-primary); margin-top: 0;">Podcast Audio</h4>
            <p style="font-size: 13px;">Audio saved to: <code>{audio_path}</code></p>
        </div>
        '''))
        display(Audio(filename=audio_path))

    except Exception as e:
        log_setup(f"Error displaying results: {e}", 'error', verbose)
        log_setup("Results generated successfully (see output files)", 'success', verbose)
