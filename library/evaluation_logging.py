"""
Logging utilities for SOLAS evaluation.
"""

from datetime import datetime

# ANSI color codes for terminal output
RESET = '\033[0m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
BOLD = '\033[1m'
DIM = '\033[2m'


def log(message: str, level: str = 'info') -> None:
    """
    Log message with color-coded output.

    Args:
        message: Message to log
        level: Log level ('info', 'success', 'warning', 'error', 'stage', 'detail', 'header')
    """
    timestamp = datetime.now().strftime('%H:%M:%S')

    if level == 'error':
        print(f"{DIM}[{timestamp}]{RESET} {RED}{BOLD}[ERROR]{RESET} {RED}{message}{RESET}")
    elif level == 'warning':
        print(f"{DIM}[{timestamp}]{RESET} {YELLOW}[WARN]{RESET} {YELLOW}{message}{RESET}")
    elif level == 'success':
        print(f"{DIM}[{timestamp}]{RESET} {GREEN}[OK]{RESET} {GREEN}{message}{RESET}")
    elif level == 'stage':
        print(f"{DIM}[{timestamp}]{RESET} {CYAN}{BOLD}[EXP]{RESET} {CYAN}{message}{RESET}")
    elif level == 'detail':
        print(f"{DIM}[{timestamp}]{RESET} {DIM}       {message}{RESET}")
    elif level == 'header':
        print(f"\n{BOLD}{MAGENTA}{'='*70}")
        print(f" {message}")
        print(f"{'='*70}{RESET}\n")
    elif level == 'progress':
        print(f"{DIM}[{timestamp}]{RESET} {BLUE}[...]{RESET} {message}")
    else:
        print(f"{DIM}[{timestamp}]{RESET} {DIM}[INFO]{RESET} {message}")
