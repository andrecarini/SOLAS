"""
Pipeline templates: Template loading utilities for HTML display.

This module provides functions to load and format HTML templates
used for displaying warnings, progress, and results in the notebook.
"""

import re
from pathlib import Path

# Cached common CSS content
_common_css_cache: str | None = None


def _get_common_css() -> str:
    """Load and cache the common CSS file content."""
    global _common_css_cache
    if _common_css_cache is None:
        css_path = Path(__file__).parent.parent / 'templates' / '_common.css'
        if css_path.exists():
            with open(css_path, 'r', encoding='utf-8') as f:
                _common_css_cache = f'<style>\n{f.read()}\n</style>\n'
        else:
            _common_css_cache = ''
    return _common_css_cache


def load_template(template_name: str, include_common_css: bool = True, **kwargs) -> str:
    """
    Load and format an HTML template from the templates directory.

    Args:
        template_name: Name of the template file (e.g., 'warning_non_colab.html')
        include_common_css: Whether to prepend common CSS variables (default: True)
        **kwargs: Variables for template substitution

    Returns:
        Formatted HTML string with variables substituted

    Raises:
        FileNotFoundError: If template file doesn't exist
        KeyError: If required template variable is missing
    """
    template_path = Path(__file__).parent.parent / 'templates' / template_name

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_name}")

    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()

    # Remove HTML comments (template documentation)
    # Keep only the actual HTML content, BUT preserve placeholder comments
    lines = template_content.split('\n')
    html_lines = []
    in_comment = False

    for line in lines:
        stripped = line.strip()
        # Preserve placeholder comments (used for widget insertion)
        if 'PLACEHOLDER' in stripped and stripped.startswith('<!--') and stripped.endswith('-->'):
            html_lines.append(line)
            continue
        if stripped.startswith('<!--'):
            in_comment = True
            continue
        if in_comment and stripped.endswith('-->'):
            in_comment = False
            continue
        if not in_comment:
            html_lines.append(line)

    template_content = '\n'.join(html_lines)

    # Perform variable substitution using regex to avoid conflicts with CSS variables
    # Only replace {variable_name} patterns, not CSS variables like {--color-primary}
    for key, value in kwargs.items():
        escaped_key = re.escape(key)
        pattern = r'\{' + escaped_key + r'\}'
        template_content = re.sub(pattern, str(value), template_content)

    # Check if any required variables are still missing
    remaining_vars = re.findall(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}', template_content)
    # Filter out CSS variables (they start with --)
    remaining_vars = [v for v in remaining_vars if not v.startswith('-')]
    if remaining_vars and kwargs:
        missing = [v for v in remaining_vars if v not in kwargs]
        if missing:
            raise KeyError(f"Missing required template variable(s): {', '.join(missing)}")

    # Prepend common CSS if requested (skip for CSS-only files)
    if include_common_css and not template_name.endswith('.css'):
        template_content = _get_common_css() + template_content

    return template_content


def get_template_path(template_name: str) -> Path:
    """
    Get the full path to a template file.

    Args:
        template_name: Name of the template file

    Returns:
        Path to the template file
    """
    return Path(__file__).parent.parent / 'templates' / template_name


def template_exists(template_name: str) -> bool:
    """
    Check if a template file exists.

    Args:
        template_name: Name of the template file

    Returns:
        True if template exists, False otherwise
    """
    return get_template_path(template_name).exists()
