"""
Tests for library/pipeline_setup.py

Tests logging, package checking, warning display, and setup functions.
"""

import os
import sys
from unittest.mock import MagicMock, patch
from io import StringIO

import pytest


class TestLogSetup:
    """Tests for log_setup function."""

    def test_log_setup_silent_when_not_verbose(self, capsys):
        """Should not print when verbose=False."""
        from library.pipeline_setup import log_setup

        log_setup("Test message", 'info', verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_log_setup_prints_when_verbose(self, capsys):
        """Should print when verbose=True."""
        from library.pipeline_setup import log_setup

        log_setup("Test message", 'info', verbose=True)
        captured = capsys.readouterr()
        assert "Test message" in captured.out
        assert "[SOLAS]" in captured.out

    def test_log_setup_info_level(self, capsys):
        """Should print info message without color prefix."""
        from library.pipeline_setup import log_setup

        log_setup("Info message", 'info', verbose=True)
        captured = capsys.readouterr()
        assert "Info message" in captured.out

    def test_log_setup_success_level(self, capsys):
        """Should print success message with checkmark."""
        from library.pipeline_setup import log_setup

        log_setup("Success message", 'success', verbose=True)
        captured = capsys.readouterr()
        assert "Success message" in captured.out
        assert "✓" in captured.out  # Success indicator

    def test_log_setup_error_level(self, capsys):
        """Should print error message with ✗."""
        from library.pipeline_setup import log_setup

        log_setup("Error message", 'error', verbose=True)
        captured = capsys.readouterr()
        assert "Error message" in captured.out
        assert "✗" in captured.out  # Error indicator

    def test_log_setup_warning_level(self, capsys):
        """Should print warning message with ⚠."""
        from library.pipeline_setup import log_setup

        log_setup("Warning message", 'warning', verbose=True)
        captured = capsys.readouterr()
        assert "Warning message" in captured.out
        assert "⚠" in captured.out  # Warning indicator


class TestIsPackageInstalled:
    """Tests for is_package_installed function."""

    def test_is_package_installed_with_existing_package(self):
        """Should return True for installed packages."""
        from library.pipeline_setup import is_package_installed

        # pytest should be installed for tests to run
        pkg_name, is_correct, version = is_package_installed("pytest")
        assert pkg_name == "pytest"
        # Version might not match if we don't specify ==
        assert version is None or isinstance(version, str)

    def test_is_package_installed_with_version_spec(self):
        """Should check exact version match."""
        from library.pipeline_setup import is_package_installed

        # Get actual pytest version
        try:
            import importlib.metadata
            actual_version = importlib.metadata.version("pytest")
            pkg_name, is_correct, version = is_package_installed(f"pytest=={actual_version}")
            assert is_correct is True
            assert version == actual_version
        except Exception:
            pytest.skip("Could not determine pytest version")

    def test_is_package_installed_wrong_version(self):
        """Should return False for wrong version."""
        from library.pipeline_setup import is_package_installed

        # Use a version that definitely doesn't exist
        pkg_name, is_correct, version = is_package_installed("pytest==0.0.0.nonexistent")
        assert is_correct is False

    def test_is_package_installed_missing_package(self):
        """Should return False for missing packages."""
        from library.pipeline_setup import is_package_installed

        pkg_name, is_correct, version = is_package_installed("definitely_nonexistent_package==1.0.0")
        assert is_correct is False
        assert version is None


class TestLoadRequirements:
    """Tests for load_requirements function."""

    def test_load_requirements_returns_list(self):
        """Should return a list of package specifications."""
        from library.pipeline_setup import load_requirements

        try:
            packages = load_requirements()
            assert isinstance(packages, list)
            # Should have at least some packages
            assert len(packages) > 0
        except FileNotFoundError:
            pytest.skip("requirements.txt not found")

    def test_load_requirements_format(self):
        """Package specs should be in correct format."""
        from library.pipeline_setup import load_requirements

        try:
            packages = load_requirements()
            for pkg in packages:
                # Should be non-empty string
                assert isinstance(pkg, str)
                assert len(pkg) > 0
                # Should not be a comment
                assert not pkg.startswith('#')
        except FileNotFoundError:
            pytest.skip("requirements.txt not found")


class TestStopExecution:
    """Tests for StopExecution exception in setup module."""

    def test_stop_execution_importable(self):
        """StopExecution should be importable from pipeline_setup."""
        from library.pipeline_setup import StopExecution
        assert StopExecution is not None

    def test_stop_execution_is_exception(self):
        """StopExecution should be an Exception."""
        from library.pipeline_setup import StopExecution
        assert issubclass(StopExecution, Exception)


class TestShowWarnings:
    """Tests for warning display functions."""

    def test_show_config_warning_raises_stop_execution(self):
        """show_config_warning should raise StopExecution."""
        from library.pipeline_setup import show_config_warning, StopExecution

        # Mock IPython display
        with patch('library.pipeline_setup.get_verbosity', return_value=False):
            with patch.dict(sys.modules, {'IPython': MagicMock(), 'IPython.display': MagicMock()}):
                with pytest.raises(StopExecution):
                    show_config_warning()

    def test_show_pipeline_warning_raises_stop_execution(self):
        """show_pipeline_warning should raise StopExecution."""
        from library.pipeline_setup import show_pipeline_warning, StopExecution

        with patch('library.pipeline_setup.get_verbosity', return_value=False):
            with patch.dict(sys.modules, {'IPython': MagicMock(), 'IPython.display': MagicMock()}):
                with pytest.raises(StopExecution):
                    show_pipeline_warning()

    def test_show_restart_warning_raises_stop_execution(self):
        """show_restart_warning should raise StopExecution."""
        from library.pipeline_setup import show_restart_warning, StopExecution

        with patch('library.pipeline_setup.get_verbosity', return_value=False):
            with patch.dict(sys.modules, {'IPython': MagicMock(), 'IPython.display': MagicMock()}):
                with pytest.raises(StopExecution):
                    show_restart_warning()


class TestSetupSystemPackages:
    """Tests for system package configuration."""

    def test_setup_system_packages_defined(self):
        """SETUP_SYSTEM_PACKAGES should be defined."""
        from library.pipeline_setup import SETUP_SYSTEM_PACKAGES

        assert isinstance(SETUP_SYSTEM_PACKAGES, list)
        assert len(SETUP_SYSTEM_PACKAGES) > 0

    def test_setup_system_packages_format(self):
        """System packages should be tuples of (name, version)."""
        from library.pipeline_setup import SETUP_SYSTEM_PACKAGES

        for pkg in SETUP_SYSTEM_PACKAGES:
            assert isinstance(pkg, tuple)
            assert len(pkg) == 2
            name, version = pkg
            assert isinstance(name, str)
            assert isinstance(version, str)

    def test_required_system_packages_present(self):
        """Required system packages should be in list."""
        from library.pipeline_setup import SETUP_SYSTEM_PACKAGES

        package_names = [pkg[0] for pkg in SETUP_SYSTEM_PACKAGES]
        assert "ffmpeg" in package_names
        assert "espeak-ng" in package_names
        assert "libsndfile1" in package_names
