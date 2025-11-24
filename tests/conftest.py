"""
Shared pytest fixtures for SOLAS tests.

These fixtures provide mock objects and test data for testing without
requiring actual GPU, model downloads, or external dependencies.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Add the SOLAS directory to path for imports
SOLAS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SOLAS_DIR))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_torch():
    """Mock torch module to avoid GPU requirements."""
    mock = MagicMock()
    mock.cuda.is_available.return_value = False
    mock.cuda.is_bf16_supported.return_value = False
    mock.cuda.memory_allocated.return_value = 0
    mock.cuda.max_memory_allocated.return_value = 0
    mock.float16 = "float16"
    mock.bfloat16 = "bfloat16"
    return mock


@pytest.fixture
def mock_transformers():
    """Mock transformers module to avoid model downloads."""
    mock = MagicMock()

    # Mock AutoModelForCausalLM
    mock_model = MagicMock()
    mock_model.generate.return_value = MagicMock()
    mock.AutoModelForCausalLM.from_pretrained.return_value = mock_model

    # Mock AutoTokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    mock_tokenizer.decode.return_value = "Mocked output text"
    mock_tokenizer.apply_chat_template.return_value = "Mocked chat template"
    mock.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

    # Mock WhisperProcessor and WhisperForConditionalGeneration
    mock_processor = MagicMock()
    mock_processor.return_value = {"input_features": MagicMock()}
    mock.WhisperProcessor.from_pretrained.return_value = mock_processor

    mock_whisper = MagicMock()
    mock_whisper.generate.return_value = MagicMock()
    mock.WhisperForConditionalGeneration.from_pretrained.return_value = mock_whisper

    # Mock BitsAndBytesConfig
    mock.BitsAndBytesConfig.return_value = MagicMock()

    return mock


@pytest.fixture
def sample_config():
    """Create a sample SOLAS configuration dictionary."""
    return {
        "asr_model_id": "openai/whisper-tiny",
        "llm_model_id": "Qwen/Qwen2-0.5B-Instruct",
        "quantization": None,
        "chunk_size_chars": 2000,
        "repetition_penalty": 1.2,
        "summary_mode": "greedy",
        "podcast_creativity_temp": 0.3,
        "input_audio_path": "/tmp/test_audio.wav",
        "output_directory": "/tmp/solas_outputs",
        "source_language": "Portuguese",
        "target_language": "English",
        "host_a_wav_path": "/tmp/host_a.wav",
        "host_b_wav_path": "/tmp/host_b.wav",
        "translation_max_new_tokens": 1024,
        "summary_max_new_tokens": 512,
        "podcast_max_new_tokens": 1024,
        "tts_model_id": "tts_models/multilingual/multi-dataset/xtts_v2",
    }


@pytest.fixture
def sample_transcript():
    """Sample transcript text for testing."""
    return """This is a sample lecture transcript.
    It contains multiple sentences for testing purposes.
    The content discusses various topics related to machine learning.
    Natural language processing is a fascinating field.
    Deep learning has revolutionized many areas of AI.
    """


@pytest.fixture
def sample_translation():
    """Sample translated text for testing."""
    return """Esta e uma transcricao de aula de exemplo.
    Contem varias frases para fins de teste.
    O conteudo discute varios topicos relacionados a aprendizado de maquina.
    O processamento de linguagem natural e um campo fascinante.
    O aprendizado profundo revolucionou muitas areas da IA.
    """


@pytest.fixture
def sample_summary():
    """Sample summary text for testing."""
    return """Key Points:
    - Machine learning is discussed
    - NLP is a fascinating field
    - Deep learning has revolutionized AI
    """


@pytest.fixture
def sample_podcast_script():
    """Sample podcast script for testing."""
    return """[Host A]: Welcome to today's episode!
    [Host B]: Today we're discussing machine learning.
    [Host A]: That's right! It's a fascinating topic.
    [Host B]: Let's dive into the details.
    """


@pytest.fixture
def mock_colab_environment():
    """Patch to simulate Colab environment."""
    with patch.dict(sys.modules, {'google.colab': MagicMock()}):
        yield


@pytest.fixture
def mock_non_colab_environment():
    """Patch to simulate non-Colab environment."""
    # Ensure google.colab is not in sys.modules
    modules_to_remove = [k for k in sys.modules if 'google.colab' in k]
    for mod in modules_to_remove:
        del sys.modules[mod]
    yield


@pytest.fixture
def mock_ipython():
    """Mock IPython display functions."""
    mock = MagicMock()
    mock.display.display = MagicMock()
    mock.display.HTML = MagicMock()
    mock.display.Audio = MagicMock()
    return mock


@pytest.fixture
def sample_requirements_content():
    """Sample requirements.txt content."""
    return """torch==2.0.0
transformers==4.30.0
accelerate==0.20.0
bitsandbytes==0.40.0
TTS==0.17.0
"""


@pytest.fixture
def templates_dir():
    """Get the templates directory path."""
    return SOLAS_DIR / "library" / "templates"


@pytest.fixture
def mock_performance_metrics():
    """Sample performance metrics for testing."""
    return {
        "audio_preprocessing": {
            "stage": "audio_preprocessing",
            "time_seconds": 1.5,
            "peak_vram_gb": 0.0,
            "end_vram_gb": 0.0,
            "avg_cpu_percent": 25.0,
        },
        "asr": {
            "stage": "asr",
            "time_seconds": 30.0,
            "peak_vram_gb": 2.5,
            "end_vram_gb": 2.0,
            "avg_cpu_percent": 15.0,
        },
        "translation": {
            "stage": "translation",
            "time_seconds": 45.0,
            "peak_vram_gb": 4.0,
            "end_vram_gb": 3.5,
            "avg_cpu_percent": 20.0,
        },
        "summary": {
            "stage": "summary",
            "time_seconds": 20.0,
            "peak_vram_gb": 4.0,
            "end_vram_gb": 3.5,
            "avg_cpu_percent": 18.0,
        },
        "podcast_script": {
            "stage": "podcast_script",
            "time_seconds": 60.0,
            "peak_vram_gb": 4.5,
            "end_vram_gb": 4.0,
            "avg_cpu_percent": 22.0,
        },
        "tts": {
            "stage": "tts",
            "time_seconds": 120.0,
            "peak_vram_gb": 6.0,
            "end_vram_gb": 5.5,
            "avg_cpu_percent": 30.0,
            "audio_duration_seconds": 180.0,
            "real_time_factor": 0.67,
        },
        "total_runtime_seconds": 276.5,
    }


@pytest.fixture
def mock_pipeline_results(sample_transcript, sample_translation, sample_summary,
                          sample_podcast_script, mock_performance_metrics, temp_dir):
    """Create mock pipeline results."""
    # Create output files
    (temp_dir / "original_transcript.txt").write_text(sample_transcript)
    (temp_dir / "translated_transcript.txt").write_text(sample_translation)
    (temp_dir / "key_points.md").write_text(sample_summary)
    (temp_dir / "podcast_script.txt").write_text(sample_podcast_script)

    # Create a dummy audio file
    audio_path = temp_dir / "podcast.wav"
    audio_path.write_bytes(b"RIFF" + b"\x00" * 100)  # Minimal WAV header

    return {
        "text_outputs": {
            "original_transcript": sample_transcript,
            "translated_transcript": sample_translation,
            "key_points_summary": sample_summary,
            "podcast_script": sample_podcast_script,
        },
        "file_paths": {
            "original_transcript": str(temp_dir / "original_transcript.txt"),
            "translated_transcript": str(temp_dir / "translated_transcript.txt"),
            "key_points_summary": str(temp_dir / "key_points.md"),
            "podcast_script": str(temp_dir / "podcast_script.txt"),
            "final_podcast_audio": str(audio_path),
            "output_directory": str(temp_dir),
        },
        "performance_metrics": mock_performance_metrics,
    }
