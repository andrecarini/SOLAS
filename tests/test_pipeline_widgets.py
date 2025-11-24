"""
Tests for library/pipeline_widgets.py

Tests widget creation and configuration building.
Note: These tests use mocked ipywidgets to avoid GUI dependencies.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


# Mock ipywidgets before importing the module
@pytest.fixture(autouse=True)
def mock_ipywidgets():
    """Mock ipywidgets for all tests in this module."""
    mock_widgets = MagicMock()

    # Mock widget classes
    mock_widgets.Dropdown = MagicMock(return_value=MagicMock(value="default"))
    mock_widgets.FloatSlider = MagicMock(return_value=MagicMock(value=0.3))
    mock_widgets.Text = MagicMock(return_value=MagicMock(value=""))
    mock_widgets.VBox = MagicMock(return_value=MagicMock())
    mock_widgets.HBox = MagicMock(return_value=MagicMock())
    mock_widgets.Button = MagicMock(return_value=MagicMock())
    mock_widgets.Label = MagicMock(return_value=MagicMock(style=MagicMock()))
    mock_widgets.Output = MagicMock(return_value=MagicMock())
    mock_widgets.FileUpload = MagicMock(return_value=MagicMock(value={}))
    mock_widgets.Layout = MagicMock(return_value=MagicMock())
    mock_widgets.RadioButtons = MagicMock(return_value=MagicMock())

    with patch.dict(sys.modules, {'ipywidgets': mock_widgets}):
        yield mock_widgets


class TestModelOptions:
    """Tests for model option constants."""

    def test_asr_models_defined(self):
        """ASR_MODELS should be a non-empty list."""
        from library.pipeline_widgets import ASR_MODELS

        assert isinstance(ASR_MODELS, list)
        assert len(ASR_MODELS) > 0

    def test_asr_models_contain_whisper(self):
        """ASR_MODELS should contain Whisper models."""
        from library.pipeline_widgets import ASR_MODELS

        assert any("whisper" in model.lower() for model in ASR_MODELS)

    def test_llm_models_defined(self):
        """LLM_MODELS should be a non-empty list."""
        from library.pipeline_widgets import LLM_MODELS

        assert isinstance(LLM_MODELS, list)
        assert len(LLM_MODELS) > 0

    def test_llm_models_contain_expected(self):
        """LLM_MODELS should contain expected models."""
        from library.pipeline_widgets import LLM_MODELS

        model_names_lower = [m.lower() for m in LLM_MODELS]
        # Check for at least one known model
        has_qwen = any("qwen" in m for m in model_names_lower)
        has_phi = any("phi" in m for m in model_names_lower)
        has_mistral = any("mistral" in m for m in model_names_lower)

        assert has_qwen or has_phi or has_mistral


class TestCreateConfigWidgets:
    """Tests for create_config_widgets function."""

    def test_create_config_widgets_returns_dict(self, mock_ipywidgets):
        """Should return a dictionary."""
        from library.pipeline_widgets import create_config_widgets

        result = create_config_widgets()
        assert isinstance(result, dict)

    def test_create_config_widgets_has_required_keys(self, mock_ipywidgets):
        """Should have all required widget keys."""
        from library.pipeline_widgets import create_config_widgets

        result = create_config_widgets()

        required_keys = [
            "asr_dropdown",
            "llm_dropdown",
            "quantization_dropdown",
            "chunk_size_dropdown",
            "repetition_penalty_dropdown",
            "summary_mode_dropdown",
            "podcast_temp_slider",
            "source_lang_dropdown",
            "target_lang_dropdown",
            "input_audio_text",
            "output_dir_text",
            "host_a_text",
            "host_b_text",
            "config_box",
        ]

        for key in required_keys:
            assert key in result, f"Missing required key: {key}"


class TestBuildConfigFromWidgets:
    """Tests for build_config_from_widgets function."""

    def test_build_config_returns_dict(self, mock_ipywidgets, sample_config):
        """Should return a dictionary."""
        from library.pipeline_widgets import build_config_from_widgets

        # Create mock widgets dict
        mock_widgets_dict = {
            "asr_dropdown": MagicMock(value="openai/whisper-tiny"),
            "llm_dropdown": MagicMock(value="Qwen/Qwen2-0.5B-Instruct"),
            "quantization_dropdown": MagicMock(value="None"),
            "chunk_size_dropdown": MagicMock(value=2000),
            "repetition_penalty_dropdown": MagicMock(value=1.2),
            "summary_mode_dropdown": MagicMock(value="greedy"),
            "podcast_temp_slider": MagicMock(value=0.3),
            "source_lang_dropdown": MagicMock(value="Portuguese"),
            "target_lang_dropdown": MagicMock(value="English"),
            "input_audio_text": MagicMock(value="/tmp/audio.wav"),
            "output_dir_text": MagicMock(value="/tmp/output"),
            "host_a_text": MagicMock(value="/tmp/host_a.wav"),
            "host_b_text": MagicMock(value="/tmp/host_b.wav"),
        }

        config = build_config_from_widgets(mock_widgets_dict)
        assert isinstance(config, dict)

    def test_build_config_has_required_keys(self, mock_ipywidgets):
        """Config should have all required keys."""
        from library.pipeline_widgets import build_config_from_widgets

        mock_widgets_dict = {
            "asr_dropdown": MagicMock(value="openai/whisper-tiny"),
            "llm_dropdown": MagicMock(value="Qwen/Qwen2-0.5B-Instruct"),
            "quantization_dropdown": MagicMock(value="None"),
            "chunk_size_dropdown": MagicMock(value=2000),
            "repetition_penalty_dropdown": MagicMock(value=1.2),
            "summary_mode_dropdown": MagicMock(value="greedy"),
            "podcast_temp_slider": MagicMock(value=0.3),
            "source_lang_dropdown": MagicMock(value="Portuguese"),
            "target_lang_dropdown": MagicMock(value="English"),
            "input_audio_text": MagicMock(value="/tmp/audio.wav"),
            "output_dir_text": MagicMock(value="/tmp/output"),
            "host_a_text": MagicMock(value="/tmp/host_a.wav"),
            "host_b_text": MagicMock(value="/tmp/host_b.wav"),
        }

        config = build_config_from_widgets(mock_widgets_dict)

        required_keys = [
            "asr_model_id",
            "llm_model_id",
            "quantization",
            "chunk_size_chars",
            "repetition_penalty",
            "summary_mode",
            "podcast_creativity_temp",
            "input_audio_path",
            "output_directory",
            "source_language",
            "target_language",
            "host_a_wav_path",
            "host_b_wav_path",
            "translation_max_new_tokens",
            "summary_max_new_tokens",
            "podcast_max_new_tokens",
            "tts_model_id",
        ]

        for key in required_keys:
            assert key in config, f"Missing required config key: {key}"

    def test_build_config_converts_none_quantization(self, mock_ipywidgets):
        """Should convert 'None' string to Python None for quantization."""
        from library.pipeline_widgets import build_config_from_widgets

        mock_widgets_dict = {
            "asr_dropdown": MagicMock(value="openai/whisper-tiny"),
            "llm_dropdown": MagicMock(value="Qwen/Qwen2-0.5B-Instruct"),
            "quantization_dropdown": MagicMock(value="None"),  # String "None"
            "chunk_size_dropdown": MagicMock(value=2000),
            "repetition_penalty_dropdown": MagicMock(value=1.2),
            "summary_mode_dropdown": MagicMock(value="greedy"),
            "podcast_temp_slider": MagicMock(value=0.3),
            "source_lang_dropdown": MagicMock(value="Portuguese"),
            "target_lang_dropdown": MagicMock(value="English"),
            "input_audio_text": MagicMock(value="/tmp/audio.wav"),
            "output_dir_text": MagicMock(value="/tmp/output"),
            "host_a_text": MagicMock(value="/tmp/host_a.wav"),
            "host_b_text": MagicMock(value="/tmp/host_b.wav"),
        }

        config = build_config_from_widgets(mock_widgets_dict)
        assert config["quantization"] is None

    def test_build_config_keeps_4bit_quantization(self, mock_ipywidgets):
        """Should keep '4-bit' string for quantization."""
        from library.pipeline_widgets import build_config_from_widgets

        mock_widgets_dict = {
            "asr_dropdown": MagicMock(value="openai/whisper-tiny"),
            "llm_dropdown": MagicMock(value="Qwen/Qwen2-0.5B-Instruct"),
            "quantization_dropdown": MagicMock(value="4-bit"),
            "chunk_size_dropdown": MagicMock(value=2000),
            "repetition_penalty_dropdown": MagicMock(value=1.2),
            "summary_mode_dropdown": MagicMock(value="greedy"),
            "podcast_temp_slider": MagicMock(value=0.3),
            "source_lang_dropdown": MagicMock(value="Portuguese"),
            "target_lang_dropdown": MagicMock(value="English"),
            "input_audio_text": MagicMock(value="/tmp/audio.wav"),
            "output_dir_text": MagicMock(value="/tmp/output"),
            "host_a_text": MagicMock(value="/tmp/host_a.wav"),
            "host_b_text": MagicMock(value="/tmp/host_b.wav"),
        }

        config = build_config_from_widgets(mock_widgets_dict)
        assert config["quantization"] == "4-bit"

    def test_build_config_converts_none_repetition_penalty(self, mock_ipywidgets):
        """Should convert 'none' string to Python None for repetition_penalty."""
        from library.pipeline_widgets import build_config_from_widgets

        mock_widgets_dict = {
            "asr_dropdown": MagicMock(value="openai/whisper-tiny"),
            "llm_dropdown": MagicMock(value="Qwen/Qwen2-0.5B-Instruct"),
            "quantization_dropdown": MagicMock(value="None"),
            "chunk_size_dropdown": MagicMock(value=2000),
            "repetition_penalty_dropdown": MagicMock(value="none"),  # String "none"
            "summary_mode_dropdown": MagicMock(value="greedy"),
            "podcast_temp_slider": MagicMock(value=0.3),
            "source_lang_dropdown": MagicMock(value="Portuguese"),
            "target_lang_dropdown": MagicMock(value="English"),
            "input_audio_text": MagicMock(value="/tmp/audio.wav"),
            "output_dir_text": MagicMock(value="/tmp/output"),
            "host_a_text": MagicMock(value="/tmp/host_a.wav"),
            "host_b_text": MagicMock(value="/tmp/host_b.wav"),
        }

        config = build_config_from_widgets(mock_widgets_dict)
        assert config["repetition_penalty"] is None
