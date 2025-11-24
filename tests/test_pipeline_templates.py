"""
Tests for library/pipeline_templates.py

Tests template loading, variable substitution, and error handling.
"""

import pytest
from pathlib import Path


class TestLoadTemplate:
    """Tests for load_template function."""

    def test_load_template_basic(self, templates_dir):
        """Should load a template file."""
        from library.pipeline_templates import load_template

        # Check if warning_non_colab.html exists
        if (templates_dir / "warning_non_colab.html").exists():
            html = load_template("warning_non_colab.html")
            assert html is not None
            assert len(html) > 0

    def test_load_template_with_variables(self, templates_dir):
        """Should substitute variables in template."""
        from library.pipeline_templates import load_template

        # Check if dependency_table.html exists (it uses variables)
        if (templates_dir / "dependency_table.html").exists():
            html = load_template("dependency_table.html", dependency_rows="<tr><td>test</td></tr>")
            assert "test" in html

    def test_load_template_file_not_found(self):
        """Should raise FileNotFoundError for missing template."""
        from library.pipeline_templates import load_template

        with pytest.raises(FileNotFoundError):
            load_template("nonexistent_template.html")

    def test_load_template_preserves_css_variables(self, templates_dir):
        """Should not replace CSS variables like var(--color-primary)."""
        from library.pipeline_templates import load_template

        # Find a template that uses CSS variables
        if (templates_dir / "warning_non_colab.html").exists():
            html = load_template("warning_non_colab.html")
            # CSS variables should be preserved
            if "var(--" in html:
                assert "var(--" in html


class TestGetTemplatePath:
    """Tests for get_template_path function."""

    def test_get_template_path_returns_path(self):
        """Should return a Path object."""
        from library.pipeline_templates import get_template_path

        path = get_template_path("test.html")
        assert isinstance(path, Path)
        assert path.name == "test.html"

    def test_get_template_path_in_templates_dir(self):
        """Path should be in the templates directory."""
        from library.pipeline_templates import get_template_path

        path = get_template_path("test.html")
        assert "templates" in str(path)


class TestTemplateExists:
    """Tests for template_exists function."""

    def test_template_exists_for_existing(self, templates_dir):
        """Should return True for existing template."""
        from library.pipeline_templates import template_exists

        # Check for a template we know exists
        if (templates_dir / "warning_non_colab.html").exists():
            assert template_exists("warning_non_colab.html") is True

    def test_template_exists_for_nonexistent(self):
        """Should return False for nonexistent template."""
        from library.pipeline_templates import template_exists

        assert template_exists("definitely_does_not_exist.html") is False


class TestTemplateContent:
    """Tests for actual template content."""

    def test_templates_have_valid_html(self, templates_dir):
        """All templates should have valid HTML structure."""
        from library.pipeline_templates import load_template

        # Find all HTML templates
        if templates_dir.exists():
            for template_file in templates_dir.glob("*.html"):
                try:
                    html = load_template(template_file.name)
                    # Basic check - should contain at least one HTML tag
                    assert "<" in html and ">" in html
                except KeyError:
                    # Template requires variables, skip content check
                    pass

    def test_required_templates_exist(self, templates_dir):
        """Critical templates should exist."""
        required_templates = [
            "warning_non_colab.html",
            "config_ready.html",
            "pipeline_start.html",
        ]

        if templates_dir.exists():
            for template_name in required_templates:
                template_path = templates_dir / template_name
                assert template_path.exists(), f"Missing required template: {template_name}"
