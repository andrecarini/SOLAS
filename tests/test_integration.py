"""
Integration tests for SOLAS pipeline with mocked models.

These tests verify that pipeline components work together correctly
without actually downloading or running LLM/ASR/TTS models.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


class TestPipelineRunner:
    """Integration tests for pipeline_runner module."""

    @pytest.mark.skipif(True, reason="Requires torch - skip in lightweight test mode")
    def test_run_pipeline_function_exists(self):
        """run_pipeline function should be importable."""
        from library.pipeline_runner import run_pipeline
        assert callable(run_pipeline)

    @pytest.mark.skipif(True, reason="Requires torch - skip in lightweight test mode")
    def test_run_pipeline_accepts_config(self, sample_config, mock_torch):
        """run_pipeline should accept a config dictionary."""
        # This test verifies the function signature, not actual execution
        from library.pipeline_runner import run_pipeline
        import inspect

        sig = inspect.signature(run_pipeline)
        params = list(sig.parameters.keys())

        assert 'config' in params
        assert 'progress_callback' in params


class TestPipelineStages:
    """Integration tests for pipeline stage functions."""

    @pytest.mark.skipif(True, reason="Requires torch - skip in lightweight test mode")
    def test_stage_functions_importable(self):
        """All stage functions should be importable."""
        from library.pipeline_stages import (
            load_and_preprocess_audio,
            transcribe_audio,
            translate_transcript,
            summarize_text,
            generate_podcast_script,
            synthesize_podcast,
        )

        assert callable(load_and_preprocess_audio)
        assert callable(transcribe_audio)
        assert callable(translate_transcript)
        assert callable(summarize_text)
        assert callable(generate_podcast_script)
        assert callable(synthesize_podcast)


class TestPipelineModels:
    """Integration tests for pipeline_models module."""

    @pytest.mark.skipif(True, reason="Requires torch - skip in lightweight test mode")
    def test_model_functions_importable(self):
        """Model management functions should be importable."""
        from library.pipeline_models import (
            ensure_llm,
            create_quantization_config,
            unload_llm,
            unload_all_models,
        )

        assert callable(ensure_llm)
        assert callable(create_quantization_config)
        assert callable(unload_llm)
        assert callable(unload_all_models)

    @pytest.mark.skipif(True, reason="Requires torch - skip in lightweight test mode")
    def test_unload_functions_safe_without_models(self, mock_torch):
        """Unload functions should not raise errors when no models loaded."""
        with patch.dict(sys.modules, {'torch': mock_torch}):
            from library import pipeline_models
            import importlib
            importlib.reload(pipeline_models)

            # These should not raise exceptions
            pipeline_models.unload_llm()
            pipeline_models.unload_all_models()


class TestLibraryExports:
    """Tests for library module exports."""

    def test_lightweight_exports(self):
        """Lightweight functions should be exported from library without torch."""
        from library import (
            # Utils
            get_verbosity,
            is_colab_environment,
            chunk_text,
            # Templates
            load_template,
            template_exists,
            # Setup
            log_setup,
        )

        assert callable(get_verbosity)
        assert callable(is_colab_environment)
        assert callable(chunk_text)
        assert callable(load_template)
        assert callable(template_exists)
        assert callable(log_setup)

    def test_widget_exports(self):
        """Widget functions should be exported from library."""
        # Need to mock ipywidgets first
        mock_widgets = MagicMock()
        with patch.dict(sys.modules, {'ipywidgets': mock_widgets}):
            from library import (
                create_config_widgets,
                build_config_from_widgets,
                ASR_MODELS,
                LLM_MODELS,
            )

            assert callable(create_config_widgets)
            assert callable(build_config_from_widgets)
            assert isinstance(ASR_MODELS, list)
            assert isinstance(LLM_MODELS, list)


class TestEndToEndMocked:
    """End-to-end tests with fully mocked pipeline."""

    @pytest.fixture
    def mock_all_heavy_deps(self):
        """Mock all heavy dependencies (torch, transformers, TTS)."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = "float16"
        mock_torch.bfloat16 = "bfloat16"

        mock_transformers = MagicMock()
        mock_tts = MagicMock()
        mock_whisper = MagicMock()
        mock_torchaudio = MagicMock()
        mock_soundfile = MagicMock()

        patches = {
            'torch': mock_torch,
            'transformers': mock_transformers,
            'TTS': mock_tts,
            'TTS.api': mock_tts,
            'whisper': mock_whisper,
            'torchaudio': mock_torchaudio,
            'soundfile': mock_soundfile,
        }

        with patch.dict(sys.modules, patches):
            yield patches

    def test_config_to_widgets_round_trip(self, mock_all_heavy_deps):
        """Config should survive widget creation and extraction."""
        mock_ipywidgets = MagicMock()
        mock_ipywidgets.Dropdown = MagicMock(return_value=MagicMock(value="default"))
        mock_ipywidgets.FloatSlider = MagicMock(return_value=MagicMock(value=0.3))
        mock_ipywidgets.Text = MagicMock(return_value=MagicMock(value=""))
        mock_ipywidgets.VBox = MagicMock(return_value=MagicMock())
        mock_ipywidgets.Layout = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {'ipywidgets': mock_ipywidgets}):
            from library.pipeline_widgets import build_config_from_widgets

            # Create mock widgets with expected values
            mock_widgets = {
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

            config = build_config_from_widgets(mock_widgets)

            # Verify key values survived
            assert config["asr_model_id"] == "openai/whisper-tiny"
            assert config["llm_model_id"] == "Qwen/Qwen2-0.5B-Instruct"
            assert config["quantization"] == "4-bit"
            assert config["chunk_size_chars"] == 2000
            assert config["repetition_penalty"] == 1.2
            assert config["summary_mode"] == "greedy"
            assert config["podcast_creativity_temp"] == 0.3


class TestTextProcessingPipeline:
    """Tests for text processing without models."""

    def test_chunk_text_pipeline(self, sample_transcript):
        """chunk_text should handle transcript-like text."""
        from library.pipeline_utils import chunk_text

        chunks = chunk_text(sample_transcript, 100)

        # Should produce multiple chunks
        assert len(chunks) >= 1

        # All chunks should be non-empty
        for chunk in chunks:
            assert len(chunk) > 0

        # Joined chunks should contain original content
        joined = " ".join(chunks)
        # Check key phrases are preserved
        assert "machine learning" in joined.lower() or "learning" in joined.lower()


class TestTemplateIntegration:
    """Tests for template integration with setup."""

    def test_templates_work_with_setup(self, templates_dir):
        """Templates should load correctly for setup display."""
        from library.pipeline_templates import load_template, template_exists

        setup_templates = [
            "warning_non_colab.html",
            "config_ready.html",
        ]

        for template_name in setup_templates:
            if template_exists(template_name):
                html = load_template(template_name)
                # Should be valid HTML
                assert "<" in html
                assert ">" in html


class TestErrorHandling:
    """Tests for error handling in pipeline."""

    def test_stop_execution_halts_gracefully(self):
        """StopExecution should provide clean halt."""
        from library.pipeline_utils import StopExecution

        with pytest.raises(StopExecution):
            raise StopExecution("Test halt")

    def test_missing_template_raises_error(self):
        """Missing template should raise FileNotFoundError."""
        from library.pipeline_templates import load_template

        with pytest.raises(FileNotFoundError):
            load_template("this_template_does_not_exist.html")
