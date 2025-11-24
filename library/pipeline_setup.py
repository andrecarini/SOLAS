"""
Pipeline setup: Environment setup, logging, and package installation.

This module provides functions for setting up the SOLAS environment:
- log_setup: Verbose logging function
- setup_environment_with_progress: Main setup entry point
- Package installation helpers
- Warning display functions
"""

import os
import sys
import shutil
import subprocess
import importlib.metadata
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

from .pipeline_utils import get_verbosity
from .pipeline_templates import load_template


# =============================================================================
# LOGGING
# =============================================================================

def log_setup(message: str, level: str = 'info', verbose: bool = False) -> None:
    """
    Log message based on verbosity - only prints if verbose is True.
    Uses ANSI color codes for better readability in terminals.

    Args:
        message: Message to log
        level: Log level ('info', 'success', 'warning', 'error', 'important')
        verbose: If True, print message. If False, silent
    """
    if not verbose:
        return

    RESET = '\033[0m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    CYAN = '\033[36m'
    BOLD = '\033[1m'

    if level == 'error':
        print(f"{RED}[SOLAS] ✗ {message}{RESET}")
    elif level == 'warning':
        print(f"{YELLOW}[SOLAS] ⚠ {message}{RESET}")
    elif level == 'success':
        print(f"{GREEN}[SOLAS] ✓ {message}{RESET}")
    elif level == 'important':
        print(f"{BOLD}{CYAN}[SOLAS] ℹ {message}{RESET}")
    else:
        print(f"[SOLAS] {message}")


# =============================================================================
# WARNING DISPLAY FUNCTIONS
# =============================================================================

class StopExecution(Exception):
    """Custom exception to halt notebook execution without showing traceback."""
    def _render_traceback_(self):
        pass


def show_config_warning() -> None:
    """
    Display configuration warning and halt notebook execution.
    The user must run the Configuration cell (Cell 2) first.

    Raises:
        StopExecution: Always raises this exception to halt execution
    """
    verbose = get_verbosity()
    log_setup("Configuration not completed - user must run Configuration cell first", 'error', verbose)

    try:
        from IPython.display import display, HTML
        warning_html = load_template('config_not_completed.html')
        display(HTML(warning_html))
        log_setup("Configuration warning displayed", 'info', verbose)
    except (ImportError, NameError, FileNotFoundError, Exception) as e:
        log_setup(f"Could not display configuration warning: {e}", 'error', verbose)
        log_setup("Please run the Configuration cell (Cell 2) first", 'error', verbose)

    log_setup("Halting execution - configuration required", 'important', verbose)
    raise StopExecution


def show_pipeline_warning() -> None:
    """
    Display pipeline warning and halt notebook execution.
    The user must run the Run Pipeline cell (Cell 5) first.

    Raises:
        StopExecution: Always raises this exception to halt execution
    """
    verbose = get_verbosity()
    log_setup("Pipeline not completed - user must run Run Pipeline cell first", 'error', verbose)

    try:
        from IPython.display import display, HTML
        warning_html = load_template('no_results_warning.html')
        display(HTML(warning_html))
        log_setup("Pipeline warning displayed", 'info', verbose)
    except (ImportError, NameError, FileNotFoundError, Exception) as e:
        log_setup(f"Could not display pipeline warning: {e}", 'error', verbose)
        log_setup("Please run the Run Pipeline cell (Cell 5) first", 'error', verbose)

    log_setup("Halting execution - pipeline execution required", 'important', verbose)
    raise StopExecution


def show_restart_warning() -> None:
    """
    Display restart warning and halt notebook execution.
    The user must restart the runtime and re-run the setup cell.

    Raises:
        StopExecution: Always raises this exception to halt execution
    """
    verbose = get_verbosity()
    log_setup("Runtime restart required after bitsandbytes update", 'warning', verbose)

    try:
        from IPython.display import display, HTML
        warning_html = load_template('restart_warning.html')
        display(HTML(warning_html))
        log_setup("Restart warning displayed", 'info', verbose)
    except (ImportError, NameError, FileNotFoundError, Exception) as e:
        log_setup(f"Could not display restart warning: {e}", 'error', verbose)
        log_setup("Please restart the runtime manually", 'warning', verbose)

    log_setup("Halting execution - restart required", 'important', verbose)
    raise StopExecution


# =============================================================================
# PACKAGE CHECKING
# =============================================================================

def is_package_installed(package_spec: str) -> Tuple[str, bool, Optional[str]]:
    """
    Check if a package is installed with the exact required version.

    Args:
        package_spec: Package specification like "transformers==4.57.1"

    Returns:
        (package_name, is_installed_correctly, installed_version)
    """
    if '==' not in package_spec:
        pkg_name = package_spec.strip()
        return (pkg_name, False, None)

    pkg_name, required_version = package_spec.split('==', 1)
    pkg_name = pkg_name.strip()
    required_version = required_version.strip()

    try:
        installed_version = importlib.metadata.version(pkg_name)
        is_correct = (installed_version == required_version)
        return (pkg_name, is_correct, installed_version)
    except importlib.metadata.PackageNotFoundError:
        return (pkg_name, False, None)


def get_system_package_version(package_name: str) -> Optional[str]:
    """Get installed version of a system package using dpkg."""
    try:
        result = subprocess.run(
            ["dpkg", "-l", package_name],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0 and 'ii' in result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.startswith('ii') and package_name in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        return parts[2]
        return None
    except Exception:
        return None


def compare_debian_versions(version1: str, version2: str) -> bool:
    """Compare two Debian package versions using dpkg --compare-versions."""
    try:
        result = subprocess.run(
            ["dpkg", "--compare-versions", version1, "ge", version2],
            capture_output=True,
            check=False
        )
        return result.returncode == 0
    except Exception:
        return False


def check_system_package_satisfies_constraint(
    package_name: str,
    min_version: str
) -> Tuple[bool, Optional[str]]:
    """Check if installed system package version satisfies minimum version constraint."""
    installed_version = get_system_package_version(package_name)
    if installed_version is None:
        return False, None

    satisfies = compare_debian_versions(installed_version, min_version)
    return satisfies, installed_version


# =============================================================================
# PROGRESS WIDGETS
# =============================================================================

def create_progress_widgets() -> Optional[Dict[str, Any]]:
    """
    Create progress widgets for setup display.

    Returns:
        Dict with 'container', 'step_labels', 'step_bars', 'step_rows', 'substeps_container'
        or None if ipywidgets not available
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, HTML
    except ImportError:
        return None

    def create_label_output(html_content):
        """Create an Output widget with initial HTML content."""
        output = widgets.Output(layout=widgets.Layout(width='420px'))
        with output:
            display(HTML(html_content))
        return output

    step_labels = {
        1: create_label_output('<div style="color: var(--color-text-secondary);">○ Step 1/5: Checking system dependencies...</div>'),
        2: create_label_output('<div style="color: var(--color-text-secondary);">○ Step 2/5: Installing system dependencies...</div>'),
        3: create_label_output('<div style="color: var(--color-text-secondary);">○ Step 3/5: Checking Python dependencies...</div>'),
        4: create_label_output('<div style="color: var(--color-text-secondary);">○ Step 4/5: Installing Python dependencies...</div>'),
        '4a': create_label_output('<div style="color: var(--color-text-secondary); margin-left: 20px;">○ Collecting packages...</div>'),
        '4b': create_label_output('<div style="color: var(--color-text-secondary); margin-left: 20px;">○ Downloading packages...</div>'),
        '4c': create_label_output('<div style="color: var(--color-text-secondary); margin-left: 20px;">○ Building wheels...</div>'),
        '4d': create_label_output('<div style="color: var(--color-text-secondary); margin-left: 20px;">○ Uninstalling old versions...</div>'),
        '4e': create_label_output('<div style="color: var(--color-text-secondary); margin-left: 20px;">○ Installing packages...</div>'),
        5: create_label_output('<div style="color: var(--color-text-secondary);">○ Step 5/5: Finalizing setup...</div>'),
    }

    step_bars = {
        1: widgets.IntProgress(value=0, min=0, max=100, bar_style='', layout=widgets.Layout(width='200px', height='20px', margin='0 0 0 10px')),
        2: widgets.IntProgress(value=0, min=0, max=100, bar_style='', layout=widgets.Layout(width='200px', height='20px', margin='0 0 0 10px')),
        3: widgets.IntProgress(value=0, min=0, max=100, bar_style='', layout=widgets.Layout(width='200px', height='20px', margin='0 0 0 10px')),
        4: widgets.IntProgress(value=0, min=0, max=100, bar_style='', layout=widgets.Layout(width='200px', height='20px', margin='0 0 0 10px')),
        '4a': widgets.IntProgress(value=0, min=0, max=100, bar_style='', layout=widgets.Layout(width='200px', height='20px', margin='0 0 0 10px')),
        '4b': widgets.IntProgress(value=0, min=0, max=100, bar_style='', layout=widgets.Layout(width='200px', height='20px', margin='0 0 0 10px')),
        '4c': widgets.IntProgress(value=0, min=0, max=100, bar_style='', layout=widgets.Layout(width='200px', height='20px', margin='0 0 0 10px')),
        '4d': widgets.IntProgress(value=0, min=0, max=100, bar_style='', layout=widgets.Layout(width='200px', height='20px', margin='0 0 0 10px')),
        '4e': widgets.IntProgress(value=0, min=0, max=100, bar_style='', layout=widgets.Layout(width='200px', height='20px', margin='0 0 0 10px')),
        5: widgets.IntProgress(value=0, min=0, max=100, bar_style='', layout=widgets.Layout(width='200px', height='20px', margin='0 0 0 10px')),
    }

    step_rows = {
        1: widgets.HBox([step_labels[1], step_bars[1]], layout=widgets.Layout(margin='4px 0')),
        2: widgets.HBox([step_labels[2], step_bars[2]], layout=widgets.Layout(margin='4px 0')),
        3: widgets.HBox([step_labels[3], step_bars[3]], layout=widgets.Layout(margin='4px 0')),
        4: widgets.HBox([step_labels[4], step_bars[4]], layout=widgets.Layout(margin='4px 0')),
        '4a': widgets.HBox([step_labels['4a'], step_bars['4a']], layout=widgets.Layout(margin='2px 0')),
        '4b': widgets.HBox([step_labels['4b'], step_bars['4b']], layout=widgets.Layout(margin='2px 0')),
        '4c': widgets.HBox([step_labels['4c'], step_bars['4c']], layout=widgets.Layout(margin='2px 0')),
        '4d': widgets.HBox([step_labels['4d'], step_bars['4d']], layout=widgets.Layout(margin='2px 0')),
        '4e': widgets.HBox([step_labels['4e'], step_bars['4e']], layout=widgets.Layout(margin='2px 0')),
        5: widgets.HBox([step_labels[5], step_bars[5]], layout=widgets.Layout(margin='4px 0')),
    }

    substeps_container = widgets.VBox([
        step_rows['4a'],
        step_rows['4b'],
        step_rows['4c'],
        step_rows['4d'],
        step_rows['4e'],
    ], layout=widgets.Layout(display='none'))

    header_output = widgets.Output()
    with header_output:
        display(HTML(load_template('setup_title.html', title="Setting up SOLAS Environment")))

    progress_container = widgets.VBox([
        header_output,
        step_rows[1],
        step_rows[2],
        step_rows[3],
        step_rows[4],
        substeps_container,
        step_rows[5],
    ], layout=widgets.Layout(padding='10px'))

    return {
        'container': progress_container,
        'step_labels': step_labels,
        'step_bars': step_bars,
        'step_rows': step_rows,
        'substeps_container': substeps_container
    }


def update_progress_widget(message: str, step, total: int, status: str, progress: Optional[int],
                          step_labels: dict, step_bars: dict) -> None:
    """Update progress widget display (uses Output widgets for labels)"""
    if step is None:
        return

    try:
        from IPython.display import display, HTML, clear_output
    except ImportError:
        return

    if status == 'active':
        icon = '⏳'
        color = 'var(--color-primary)'
        weight = 'bold'
        bar_style = 'info'
    elif status == 'complete':
        icon = '✓'
        color = 'var(--color-success)'
        weight = 'normal'
        bar_style = 'success'
    else:  # pending
        icon = '○'
        color = 'var(--color-text-secondary)'
        weight = 'normal'
        bar_style = ''

    if isinstance(step, str):
        margin_left = 'margin-left: 20px;' if step.startswith('4') else ''
        step_html = f'<div style="color: {color}; font-weight: {weight}; {margin_left}">{icon} {message}</div>'
    else:
        if total is not None:
            step_html = f'<div style="color: {color}; font-weight: {weight};">{icon} Step {step}/{total}: {message}</div>'
        else:
            step_html = f'<div style="color: {color}; font-weight: {weight};">{icon} {message}</div>'

    if step in step_labels:
        label_output = step_labels[step]
        with label_output:
            clear_output(wait=True)
            display(HTML(step_html))

    if step in step_bars:
        if progress is not None:
            step_bars[step].value = progress
        elif status == 'complete':
            step_bars[step].value = 100
        elif status == 'active':
            step_bars[step].value = 50

        step_bars[step].bar_style = bar_style


def update_progress_bar_only(step, progress: int, step_bars: dict) -> None:
    """Update only the progress bar percentage without changing the step label"""
    if step in step_bars:
        step_bars[step].value = progress


# =============================================================================
# REQUIREMENTS LOADING
# =============================================================================

def load_requirements() -> List[str]:
    """
    Load package requirements from requirements.txt file.

    Returns:
        List of package specifications (e.g., ["torch==2.9.0", "transformers==4.57.1"])
    """
    # Look for requirements.txt in the library directory (same as this module)
    requirements_path = Path(__file__).parent / 'requirements.txt'

    if not requirements_path.exists():
        raise FileNotFoundError(
            f"requirements.txt not found at {requirements_path}. "
            "Please create it with the required package versions."
        )

    packages = []
    with open(requirements_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                packages.append(line)

    return packages


# System packages for Colab
SETUP_SYSTEM_PACKAGES = [
    ("espeak-ng", "1.50+dfsg-10ubuntu0.1"),
    ("libsndfile1", "1.0.31-2ubuntu0.2"),
    ("ffmpeg", "7:4.4.2-0ubuntu0.22.04.1"),
]


# =============================================================================
# PACKAGE INSTALLATION
# =============================================================================

def pip_install_packages(pkgs: List[str], check_first: bool = True, progress_step: Optional[int] = None,
                        step_labels: Optional[dict] = None, step_bars: Optional[dict] = None,
                        substeps_container: Optional[Any] = None) -> Tuple[bool, bool]:
    """
    Install packages, checking if they're already installed with exact versions first.

    Returns:
        Tuple of (packages_installed, bnb_updated)
    """
    verbose = get_verbosity()

    all_correct = True
    if check_first:
        log_setup(f"Checking {len(pkgs)} package(s)...", 'important', verbose)
        total_pkgs = len(pkgs)
        for idx, pkg in enumerate(pkgs):
            if progress_step is not None and step_bars is not None:
                progress_pct = int((idx / total_pkgs) * 100)
                update_progress_bar_only(progress_step, progress_pct, step_bars)

            _, is_correct, _ = is_package_installed(pkg)
            if not is_correct:
                all_correct = False
                break

        if progress_step is not None and step_bars is not None:
            update_progress_bar_only(progress_step, 100, step_bars)

        if all_correct:
            log_setup("All packages are installed with correct versions", 'success', verbose)
            return False, False
        else:
            log_setup("Some packages need installation/upgrade, installing all packages...", 'important', verbose)

    if not all_correct or not check_first:
        bnb_updated = any('bitsandbytes' in pkg.lower() for pkg in pkgs)

        try:
            if substeps_container is not None:
                substeps_container.layout.display = 'block'

            if progress_step is not None and step_bars is not None:
                update_progress_bar_only(progress_step, 0, step_bars)

            log_setup(f"Installing {len(pkgs)} package(s)...", 'important', verbose)
            pkg_names = []
            for p in pkgs:
                if '==' in p:
                    pkg_names.append(p.split('==')[0].strip())
                else:
                    pkg_names.append(p.strip())
            if verbose:
                log_setup(f"Packages: {', '.join(pkg_names)}", 'important', verbose)

            if verbose:
                cmd = [sys.executable, "-m", "pip", "install", "-v"] + pkgs
            else:
                cmd = [sys.executable, "-m", "pip", "install"] + pkgs

            if progress_step is not None and step_bars is not None:
                update_progress_bar_only(progress_step, 10, step_bars)

            log_setup(f"Executing: {' '.join(cmd)}", 'info', verbose)

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            seen_collecting = False
            seen_downloading = False
            seen_building = False
            seen_uninstalling = False
            seen_installing = False

            log_setup("VERBOSE MODE: Streaming pip output in real-time...", 'info', verbose)

            for line in process.stdout:
                if verbose:
                    print(line, end='', flush=True)

                if progress_step is not None and step_labels is not None and step_bars is not None:
                    line_lower = line.lower()

                    if not seen_collecting and line_lower.startswith('collecting'):
                        seen_collecting = True
                        update_progress_widget('Collecting packages...', '4a', 5, 'active', 50, step_labels, step_bars)
                        update_progress_bar_only(progress_step, 15, step_bars)
                    elif not seen_downloading and (line_lower.startswith('downloading') or line_lower.startswith('obtaining')):
                        seen_downloading = True
                        update_progress_widget('Collecting packages...', '4a', 5, 'complete', 100, step_labels, step_bars)
                        update_progress_widget('Downloading packages...', '4b', 5, 'active', 50, step_labels, step_bars)
                        update_progress_bar_only(progress_step, 30, step_bars)
                    elif not seen_building and (line_lower.startswith('building wheels') or 'preparing metadata (pyproject.toml)' in line_lower):
                        seen_building = True
                        update_progress_widget('Collecting packages...', '4a', 5, 'complete', 100, step_labels, step_bars)
                        update_progress_widget('Downloading packages...', '4b', 5, 'complete', 100, step_labels, step_bars)
                        update_progress_widget('Building wheels...', '4c', 5, 'active', 50, step_labels, step_bars)
                        update_progress_bar_only(progress_step, 55, step_bars)
                    elif not seen_uninstalling and line_lower.startswith('attempting uninstall'):
                        seen_uninstalling = True
                        update_progress_widget('Collecting packages...', '4a', 5, 'complete', 100, step_labels, step_bars)
                        update_progress_widget('Downloading packages...', '4b', 5, 'complete', 100, step_labels, step_bars)
                        update_progress_widget('Building wheels...', '4c', 5, 'complete', 100, step_labels, step_bars)
                        update_progress_widget('Uninstalling old versions...', '4d', 5, 'active', 50, step_labels, step_bars)
                        update_progress_bar_only(progress_step, 65, step_bars)
                    elif not seen_installing and line_lower.startswith('installing collected packages'):
                        seen_installing = True
                        update_progress_widget('Collecting packages...', '4a', 5, 'complete', 100, step_labels, step_bars)
                        update_progress_widget('Downloading packages...', '4b', 5, 'complete', 100, step_labels, step_bars)
                        update_progress_widget('Building wheels...', '4c', 5, 'complete', 100, step_labels, step_bars)
                        update_progress_widget('Uninstalling old versions...', '4d', 5, 'complete', 100, step_labels, step_bars)
                        update_progress_widget('Installing packages...', '4e', 5, 'active', 50, step_labels, step_bars)
                        update_progress_bar_only(progress_step, 85, step_bars)
                    elif line_lower.startswith('successfully installed'):
                        update_progress_widget('Installing packages...', '4e', 5, 'complete', 100, step_labels, step_bars)
                        update_progress_bar_only(progress_step, 98, step_bars)

            process.wait()
            returncode = process.returncode

            if progress_step is not None and step_bars is not None:
                update_progress_bar_only(progress_step, 100, step_bars)

            log_setup(f"Pip command completed with exit code: {returncode}", 'info', verbose)

            if returncode == 0:
                log_setup(f"Successfully installed packages", 'success', verbose)
                return True, bnb_updated
            else:
                log_setup(f"Failed to install packages (exit code: {returncode})", 'error', verbose)
                log_setup("You may need to manually install failed packages", 'warning', verbose)
                return False, bnb_updated
        except Exception as e:
            log_setup(f"pip install failed: {e}", 'error', verbose)
            return False, bnb_updated

    return False, False


def apt_check_packages(pkgs_with_constraints: List[Tuple[str, Optional[str]]], progress_step: Optional[int] = None,
                      step_bars: Optional[dict] = None) -> Tuple[List[str], List[str]]:
    """
    Check system packages against version constraints.

    Returns:
        (to_install, to_upgrade) lists of package names
    """
    verbose = get_verbosity()
    to_install = []
    to_upgrade = []

    try:
        if not shutil.which("apt-get"):
            log_setup("apt-get not available (not on Debian/Ubuntu system)", 'warning', verbose)
            return [], []

        normalized_pkgs = []
        for pkg in pkgs_with_constraints:
            if isinstance(pkg, tuple):
                normalized_pkgs.append(pkg)
            else:
                normalized_pkgs.append((pkg, None))

        if progress_step is not None and step_bars is not None:
            update_progress_bar_only(progress_step, 10, step_bars)
        log_setup("Updating package list (this may take a moment)...", 'important', verbose)
        if verbose:
            log_setup("Running: apt-get update -y...", 'info', verbose)
            subprocess.run(["apt-get", "update", "-y"], check=True)
        else:
            subprocess.run(
                ["apt-get", "update", "-y"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        log_setup("Package list updated", 'success', verbose)

        total_pkgs = len(normalized_pkgs)
        checked = 0

        for pkg_name, min_version in normalized_pkgs:
            checked += 1
            if progress_step is not None and step_bars is not None:
                progress_pct = 10 + int((checked / total_pkgs) * 90)
                update_progress_bar_only(progress_step, progress_pct, step_bars)
            installed_version = get_system_package_version(pkg_name)

            if installed_version is not None:
                if min_version:
                    satisfies, _ = check_system_package_satisfies_constraint(pkg_name, min_version)
                    if satisfies:
                        log_setup(f"{pkg_name}=={installed_version} already installed (satisfies >= {min_version})", 'success', verbose)
                    else:
                        to_upgrade.append(pkg_name)
                        log_setup(f"{pkg_name}=={installed_version} installed but doesn't satisfy constraint (need >= {min_version})", 'warning', verbose)
                else:
                    log_setup(f"{pkg_name}=={installed_version} already installed", 'success', verbose)
            else:
                to_install.append(pkg_name)
                if min_version:
                    log_setup(f"{pkg_name} not installed (will install >= {min_version})", 'info', verbose)
                else:
                    log_setup(f"{pkg_name} not installed, will install", 'info', verbose)

        return to_install, to_upgrade

    except Exception as e:
        log_setup(f"apt checking failed: {e}", 'error', verbose)
        return [], []


def apt_install_packages(to_install: List[str], to_upgrade: List[str], progress_step: Optional[int] = None,
                        step_bars: Optional[dict] = None) -> None:
    """Install or upgrade system packages."""
    verbose = get_verbosity()
    try:
        packages_to_process = to_install + to_upgrade
        if packages_to_process:
            action = "Installing" if to_install and not to_upgrade else "Upgrading" if to_upgrade and not to_install else "Installing/upgrading"
            log_setup(f"{action} {len(packages_to_process)} system package(s): {', '.join(packages_to_process)}", 'important', verbose)
            cmd = ["apt-get", "install", "-y"]
            if to_upgrade and not to_install:
                cmd.append("--only-upgrade")
            cmd.extend(packages_to_process)

            if progress_step is not None and step_bars is not None:
                update_progress_bar_only(progress_step, 10, step_bars)
            log_setup(f"Running apt-get install (this may take a few minutes)...", 'important', verbose)
            if verbose:
                log_setup(f"Running: {' '.join(cmd[:5])}...", 'info', verbose)
                subprocess.run(cmd, check=True)
            else:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            if progress_step is not None and step_bars is not None:
                update_progress_bar_only(progress_step, 100, step_bars)
            log_setup("apt-get install completed", 'success', verbose)
            if to_install:
                log_setup(f"Successfully installed {len(to_install)} system package(s)", 'success', verbose)
            if to_upgrade:
                log_setup(f"Successfully upgraded {len(to_upgrade)} system package(s)", 'success', verbose)
        else:
            log_setup("No packages to install/upgrade", 'success', verbose)
            if progress_step is not None and step_bars is not None:
                update_progress_bar_only(progress_step, 100, step_bars)

    except Exception as e:
        log_setup(f"apt-get failed: {e}", 'error', verbose)


def finalize_setup(package_list: List[str], bnb_updated: bool, progress_step: Optional[int] = None,
                  step_labels: Optional[dict] = None, step_bars: Optional[dict] = None,
                  progress_container: Optional[Any] = None) -> Dict[str, Any]:
    """
    Finalize setup: GPU check, dependency table generation, restart detection.

    Returns:
        Dict with: restart_needed, dependency_data, gpu_available, device_name, progress_container
    """
    verbose = get_verbosity()

    widgets = None
    HTML = None
    if progress_container is not None:
        try:
            import ipywidgets as widgets
            from IPython.display import HTML
        except ImportError:
            pass

    if HTML is None:
        try:
            from IPython.display import HTML
        except ImportError:
            pass

    if progress_step is not None and step_labels is not None and step_bars is not None:
        update_progress_widget("Finalizing setup...", progress_step, 5, 'active', 10, step_labels, step_bars)

    gpu_available = False
    device_name = None
    try:
        import torch
        if progress_step is not None and step_labels is not None and step_bars is not None:
            update_progress_widget("Finalizing setup...", progress_step, 5, 'active', 20, step_labels, step_bars)
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            device_name = torch.cuda.get_device_name(0)
            log_setup(f'GPU detected: {device_name}', 'success', verbose)
        else:
            log_setup('WARNING: No GPU detected. This notebook will run extremely slowly without a GPU.', 'warning', verbose)
    except ImportError:
        log_setup('WARNING: torch not available, skipping GPU check', 'warning', verbose)
    except Exception as e:
        log_setup(f'WARNING: Error checking GPU: {e}', 'warning', verbose)

    if progress_step is not None and step_labels is not None and step_bars is not None:
        update_progress_widget("Finalizing setup...", progress_step, 5, 'active', 30, step_labels, step_bars)

    dependency_data = []
    total_pkgs = len(package_list)
    for idx, pkg_spec in enumerate(package_list):
        try:
            if '==' in pkg_spec:
                pkg_name, required_ver = pkg_spec.split('==', 1)
                pkg_name = pkg_name.strip()
                required_ver = required_ver.strip()
            else:
                pkg_name = pkg_spec.strip()
                required_ver = "Any"

            _, is_correct, installed_ver = is_package_installed(pkg_spec)

            status = "v" if is_correct else "x" if installed_ver else "-"
            dependency_data.append({
                'name': pkg_name,
                'required': required_ver,
                'installed': installed_ver or "Not installed",
                'status': status
            })
        except Exception as e:
            log_setup(f'WARNING: Error checking package {pkg_spec}: {e}', 'warning', verbose)
            if '==' in pkg_spec:
                pkg_name = pkg_spec.split('==')[0].strip()
            else:
                pkg_name = pkg_spec.strip()
            dependency_data.append({
                'name': pkg_name,
                'required': 'Unknown',
                'installed': 'Error',
                'status': 'x'
            })
        if progress_step is not None and step_bars is not None:
            if (idx + 1) % max(1, total_pkgs // 3) == 0 or (idx + 1) == total_pkgs:
                progress = 30 + int((idx + 1) / total_pkgs * 40)
                update_progress_bar_only(progress_step, progress, step_bars)

    if progress_step is not None and step_labels is not None and step_bars is not None:
        update_progress_widget("Finalizing setup...", progress_step, 5, 'active', 70, step_labels, step_bars)

    dependency_rows_html = ""
    for dep in dependency_data:
        status_class = "status-ok" if dep['status'] == "v" else "status-fail" if dep['status'] == "x" else "status-missing"
        dependency_rows_html += f"""
        <tr>
            <td><strong>{dep['name']}</strong></td>
            <td>{dep['required']}</td>
            <td>{dep['installed']}</td>
            <td class="{status_class}">{dep['status']}</td>
        </tr>
        """

    table_html = load_template('dependency_table.html', dependency_rows=dependency_rows_html)

    if progress_step is not None and step_labels is not None and step_bars is not None:
        update_progress_widget("Finalizing setup...", progress_step, 5, 'active', 80, step_labels, step_bars)

    all_satisfied = all(dep['status'] == 'v' for dep in dependency_data) if dependency_data else False
    failed_packages = [dep for dep in dependency_data if dep['status'] == 'x']
    missing_packages = [dep for dep in dependency_data if dep['status'] == '-']

    if all_satisfied:
        completion_title = '<h3 style="color: var(--color-success); margin-bottom: 15px;">v Setup Complete!</h3>'
        completion_items = '<p style="color: var(--color-success); margin: 8px 0;">v All dependencies have been checked and installed.</p>'
    else:
        if failed_packages:
            failed_names = ', '.join([dep['name'] for dep in failed_packages])
            completion_title = '<h3 style="color: var(--color-error); margin-bottom: 15px;">! Setup Incomplete</h3>'
            completion_items = f'<p style="color: var(--color-error); margin: 8px 0;">! {len(failed_packages)} package(s) do not match required versions: {failed_names}</p>'
        elif missing_packages:
            missing_names = ', '.join([dep['name'] for dep in missing_packages])
            completion_title = '<h3 style="color: var(--color-warning); margin-bottom: 15px;">! Setup Incomplete</h3>'
            completion_items = f'<p style="color: var(--color-warning); margin: 8px 0;">! {len(missing_packages)} package(s) are missing: {missing_names}</p>'
        else:
            completion_title = '<h3 style="color: var(--color-error); margin-bottom: 15px;">! Setup Incomplete</h3>'
            completion_items = '<p style="color: var(--color-error); margin: 8px 0;">! Some dependencies are not satisfied.</p>'

    if gpu_available:
        gpu_item = f'<p style="color: var(--color-success); margin: 8px 0;">v GPU Available: {device_name}</p>'
    else:
        gpu_item = '<p style="color: var(--color-error); margin: 8px 0;">! No GPU Detected - Performance will be limited</p>'

    success_msg = ""
    restart_warning_html = ""

    if bnb_updated:
        restart_warning_html = load_template('restart_warning.html')

    if all_satisfied and not bnb_updated:
        success_msg = load_template('success_message.html')

    error_html = ""
    if not all_satisfied:
        error_details = []
        if failed_packages:
            for dep in failed_packages:
                error_details.append(f"<li><strong>{dep['name']}</strong>: Installed {dep['installed']} (required: {dep['required']})</li>")
        if missing_packages:
            for dep in missing_packages:
                error_details.append(f"<li><strong>{dep['name']}</strong>: Not installed (required: {dep['required']})</li>")

        if error_details:
            error_html = f"""
            <div style="background: color-mix(in srgb, var(--color-error) 20%, var(--color-bg-primary));
                        border-left: 4px solid var(--color-error);
                        padding: 20px;
                        margin-top: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="color: var(--color-error); margin: 0 0 15px 0; font-size: 16px; font-weight: 600;">
                    ! Some dependencies are not satisfied. Please fix the following:
                </p>
                <ul style="color: var(--color-error); margin: 10px 0; padding-left: 20px;">
                    {''.join(error_details)}
                </ul>
                <p style="color: var(--color-error); margin-top: 15px; font-size: 14px;">
                    <strong>To fix:</strong> Run the setup cell again, or manually install the packages with:<br>
                    <code style="background: var(--color-bg-secondary); color: var(--color-text-primary); padding: 2px 6px; border-radius: 3px; border: 1px solid var(--color-border);">!pip install PACKAGE_NAME==VERSION</code>
                </p>
            </div>
            """

    setup_html = load_template('setup_complete.html',
        completion_title=completion_title,
        completion_items=completion_items,
        gpu_item=gpu_item,
        success_msg=success_msg,
        restart_warning_html=restart_warning_html,
        table_html=table_html,
        error_html=error_html
    )

    if progress_container is not None and widgets is not None:
        try:
            progress_container.children = []
            progress_container.layout.display = 'none'
            log_setup("Progress container cleared", 'info', verbose)
        except Exception as e:
            log_setup(f"Failed to clear progress container: {e}", 'warning', verbose)

    if HTML is not None:
        try:
            from IPython.display import display
            display(HTML(setup_html))
            log_setup("Setup completion HTML displayed", 'info', verbose)
        except (ImportError, Exception) as e:
            log_setup(f"Could not display setup HTML: {e}", 'warning', verbose)

    if progress_step is not None and step_labels is not None and step_bars is not None:
        update_progress_widget("Finalizing setup...", progress_step, 5, 'complete', 100, step_labels, step_bars)

    restart_needed = bnb_updated

    return {
        'restart_needed': restart_needed,
        'dependency_data': dependency_data,
        'gpu_available': gpu_available,
        'device_name': device_name,
        'progress_container': progress_container,
        'all_satisfied': all_satisfied
    }


# =============================================================================
# MAIN SETUP FUNCTION
# =============================================================================

def setup_environment_with_progress() -> Dict[str, Any]:
    """
    Main setup orchestrator with progress tracking.

    This function:
    1. Creates progress widgets
    2. Checks and installs system dependencies (Colab only)
    3. Checks and installs Python dependencies
    4. Finalizes setup with GPU check, dependency table, and restart detection

    Returns:
        Dict with:
        - restart_needed: bool - Whether runtime restart is required
        - dependency_data: list - Dependency status data
        - gpu_available: bool - Whether GPU is available
        - device_name: str - GPU device name (if available)
        - progress_container: widget - Progress container widget
        - all_satisfied: bool - Whether all dependencies are satisfied
    """
    verbose = get_verbosity()

    try:
        from IPython.display import display
    except ImportError:
        display = None

    # Load requirements
    try:
        python_packages = load_requirements()
    except FileNotFoundError as e:
        log_setup(f"Error loading requirements: {e}", 'error', verbose)
        return {
            'restart_needed': False,
            'dependency_data': [],
            'gpu_available': False,
            'device_name': None,
            'progress_container': None,
            'all_satisfied': False
        }

    widgets_data = create_progress_widgets()
    if widgets_data is None:
        return {
            'restart_needed': False,
            'dependency_data': [],
            'gpu_available': False,
            'device_name': None,
            'progress_container': None,
            'all_satisfied': False
        }

    progress_container = widgets_data['container']
    step_labels = widgets_data['step_labels']
    step_bars = widgets_data['step_bars']
    substeps_container = widgets_data['substeps_container']

    if display is not None:
        display(progress_container)

    # Step 1: Check system dependencies (Colab only)
    update_progress_widget("Checking system dependencies...", 1, 5, 'active', 0, step_labels, step_bars)
    sys_to_install = []
    sys_to_upgrade = []

    if 'google.colab' in sys.modules:
        sys_to_install, sys_to_upgrade = apt_check_packages(
            SETUP_SYSTEM_PACKAGES,
            progress_step=1,
            step_bars=step_bars
        )

    update_progress_widget("Checking system dependencies...", 1, 5, 'complete', 100, step_labels, step_bars)

    # Step 2: Install system dependencies
    update_progress_widget("Installing system dependencies...", 2, 5, 'active', 0, step_labels, step_bars)
    if sys_to_install or sys_to_upgrade:
        apt_install_packages(sys_to_install, sys_to_upgrade, progress_step=2, step_bars=step_bars)
    else:
        log_setup("All system packages are up to date", 'success', verbose)
        update_progress_bar_only(2, 100, step_bars)

    update_progress_widget("Installing system dependencies...", 2, 5, 'complete', 100, step_labels, step_bars)

    # Step 3: Check Python dependencies
    update_progress_widget("Checking Python dependencies...", 3, 5, 'active', 0, step_labels, step_bars)
    log_setup(f"Checking {len(python_packages)} Python packages...", 'important', verbose)
    update_progress_widget("Checking Python dependencies...", 3, 5, 'complete', 100, step_labels, step_bars)

    # Step 4: Install Python dependencies
    update_progress_widget("Installing Python dependencies...", 4, 5, 'active', 0, step_labels, step_bars)
    packages_installed, bnb_updated = pip_install_packages(
        python_packages,
        check_first=True,
        progress_step=4,
        step_labels=step_labels,
        step_bars=step_bars,
        substeps_container=substeps_container
    )
    if packages_installed:
        log_setup("Python package installation/update completed", 'success', verbose)
    else:
        log_setup("Python package check completed (no changes needed)", 'success', verbose)
    update_progress_widget("Installing Python dependencies...", 4, 5, 'complete', 100, step_labels, step_bars)

    # Step 5: Finalize setup
    result = finalize_setup(
        python_packages,
        bnb_updated,
        progress_step=5,
        step_labels=step_labels,
        step_bars=step_bars,
        progress_container=progress_container
    )

    return result
