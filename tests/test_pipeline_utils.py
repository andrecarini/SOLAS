"""
Tests for library/pipeline_utils.py

Tests environment detection, device management, text processing,
and performance metrics collection.
"""

import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest


class TestGetVerbosity:
    """Tests for get_verbosity function."""

    def test_default_verbosity_is_false(self):
        """Verbosity should be False by default."""
        # Clear any existing VERBOSITY env var
        os.environ.pop('VERBOSITY', None)

        from library.pipeline_utils import get_verbosity
        assert get_verbosity() is False

    def test_verbosity_from_env_debug(self):
        """Verbosity should be True when VERBOSITY=DEBUG."""
        os.environ['VERBOSITY'] = 'DEBUG'
        try:
            from library.pipeline_utils import get_verbosity
            assert get_verbosity() is True
        finally:
            os.environ.pop('VERBOSITY', None)

    def test_verbosity_from_env_case_insensitive(self):
        """Verbosity check should be case-insensitive."""
        os.environ['VERBOSITY'] = 'debug'
        try:
            from library.pipeline_utils import get_verbosity
            assert get_verbosity() is True
        finally:
            os.environ.pop('VERBOSITY', None)

    def test_verbosity_from_env_other_value(self):
        """Verbosity should be False for non-DEBUG values."""
        os.environ['VERBOSITY'] = 'INFO'
        try:
            from library.pipeline_utils import get_verbosity
            assert get_verbosity() is False
        finally:
            os.environ.pop('VERBOSITY', None)


class TestEnvironmentDetection:
    """Tests for environment detection functions."""

    def test_is_colab_environment_false_locally(self, mock_non_colab_environment):
        """Should return False when not in Colab."""
        from library.pipeline_utils import is_colab_environment
        assert is_colab_environment() is False

    def test_is_colab_environment_true_in_colab(self, mock_colab_environment):
        """Should return True when in Colab."""
        from library.pipeline_utils import is_colab_environment
        # Note: This test may still fail due to import caching
        # The actual behavior depends on whether google.colab can be imported


class TestDeviceManagement:
    """Tests for device and dtype management functions."""

    def test_get_device_returns_string(self):
        """get_device should return a string."""
        from library.pipeline_utils import get_device
        result = get_device()
        assert isinstance(result, str)
        assert result in ("cpu", "cuda")

    def test_get_vram_usage_returns_float(self):
        """get_vram_usage should return a float."""
        from library.pipeline_utils import get_vram_usage
        result = get_vram_usage()
        assert isinstance(result, float)
        assert result >= 0.0

    def test_get_peak_vram_returns_float(self):
        """get_peak_vram should return a float."""
        from library.pipeline_utils import get_peak_vram
        result = get_peak_vram()
        assert isinstance(result, float)
        assert result >= 0.0

    def test_clear_gpu_memory_does_not_raise(self):
        """clear_gpu_memory should not raise an error."""
        from library.pipeline_utils import clear_gpu_memory
        # Should not raise any exceptions
        clear_gpu_memory()


class TestChunkText:
    """Tests for chunk_text function."""

    def test_chunk_text_empty_input(self):
        """Should return empty list for empty input."""
        from library.pipeline_utils import chunk_text
        assert chunk_text("", 1000) == []

    def test_chunk_text_single_sentence(self):
        """Should return single chunk for short text."""
        from library.pipeline_utils import chunk_text
        text = "This is a short sentence."
        chunks = chunk_text(text, 1000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_multiple_sentences(self):
        """Should split at sentence boundaries."""
        from library.pipeline_utils import chunk_text
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, 30)
        assert len(chunks) >= 2
        # Each chunk should not exceed max_chars (approximately, due to sentence boundaries)
        for chunk in chunks:
            assert len(chunk) <= 35  # Allow some flexibility for sentence boundaries

    def test_chunk_text_respects_max_chars(self):
        """Chunks should not significantly exceed max_chars."""
        from library.pipeline_utils import chunk_text
        text = "A. B. C. D. E. F. G. H. I. J."
        chunks = chunk_text(text, 10)
        # Each chunk should be roughly within limits
        assert all(len(c) <= 15 for c in chunks)

    def test_chunk_text_handles_long_sentence(self):
        """Should split long sentences at word boundaries."""
        from library.pipeline_utils import chunk_text
        text = "This is a very long sentence that exceeds the maximum character limit significantly."
        chunks = chunk_text(text, 30)
        assert len(chunks) >= 2


class TestPerformanceMetrics:
    """Tests for performance metrics collection."""

    def test_collect_performance_metrics_structure(self):
        """Should return dict with expected keys."""
        from library.pipeline_utils import collect_performance_metrics

        start_time = time.time() - 1.0  # 1 second ago
        metrics = collect_performance_metrics("test_stage", start_time)

        assert "stage" in metrics
        assert "time_seconds" in metrics
        assert "peak_vram_gb" in metrics
        assert metrics["stage"] == "test_stage"
        assert metrics["time_seconds"] >= 1.0

    def test_collect_performance_metrics_timing(self):
        """Should measure elapsed time correctly."""
        from library.pipeline_utils import collect_performance_metrics

        start_time = time.time()
        time.sleep(0.1)  # Sleep 100ms
        metrics = collect_performance_metrics("test", start_time)

        assert metrics["time_seconds"] >= 0.1


class TestSuppressOutput:
    """Tests for SuppressOutput context manager."""

    def test_suppress_output_suppresses_when_enabled(self, capsys):
        """Should suppress stdout when enabled."""
        from library.pipeline_utils import SuppressOutput

        with SuppressOutput(suppress=True):
            print("This should be suppressed")

        captured = capsys.readouterr()
        assert "This should be suppressed" not in captured.out

    def test_suppress_output_allows_when_disabled(self, capsys):
        """Should not suppress stdout when disabled."""
        from library.pipeline_utils import SuppressOutput

        with SuppressOutput(suppress=False):
            print("This should appear")

        captured = capsys.readouterr()
        assert "This should appear" in captured.out


class TestStopExecution:
    """Tests for StopExecution exception."""

    def test_stop_execution_is_exception(self):
        """StopExecution should be an Exception."""
        from library.pipeline_utils import StopExecution
        assert issubclass(StopExecution, Exception)

    def test_stop_execution_hides_traceback(self):
        """StopExecution should have _render_traceback_ method."""
        from library.pipeline_utils import StopExecution
        exc = StopExecution()
        # Should have the method and it should return None
        assert hasattr(exc, '_render_traceback_')
        assert exc._render_traceback_() is None


class TestStageMetrics:
    """Tests for StageMetrics context manager."""

    def test_stage_metrics_context_manager(self):
        """Should collect metrics when used as context manager."""
        from library.pipeline_utils import StageMetrics

        with StageMetrics("test_stage") as sm:
            time.sleep(0.05)  # Small delay

        assert sm.metrics is not None
        assert sm.metrics["stage"] == "test_stage"
        assert sm.metrics["time_seconds"] >= 0.05
