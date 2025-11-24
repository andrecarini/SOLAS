# SOLAS Development Guide

This document covers development setup, testing, and common issues when working with SOLAS.

## Project Structure

```
SOLAS/
├── library/                    # Core library modules
│   ├── __init__.py            # Module exports with lazy loading
│   ├── pipeline_utils.py      # Utilities (device, chunking, metrics)
│   ├── pipeline_models.py     # Model loading/management
│   ├── pipeline_stages.py     # Pipeline stage implementations
│   ├── pipeline_runner.py     # Pipeline orchestration
│   ├── pipeline_setup.py      # Environment setup
│   ├── pipeline_widgets.py    # Jupyter widgets for configuration
│   ├── pipeline_templates.py  # HTML template loading
│   ├── evaluation_notebook.py # Evaluation framework
│   ├── evaluation_display.py  # Results visualization
│   ├── evaluation_analysis.py # Quality analysis functions
│   ├── requirements.txt       # Python dependencies
│   └── pytest.ini             # Pytest configuration
├── templates/                  # HTML templates for displays
├── tests/                      # Test suite
│   ├── conftest.py            # Shared pytest fixtures
│   ├── test_pipeline_utils.py
│   ├── test_pipeline_setup.py
│   ├── test_pipeline_templates.py
│   ├── test_pipeline_widgets.py
│   └── test_integration.py
├── docs/                       # Documentation
├── SOLAS_Interactive.ipynb    # Interactive pipeline notebook
└── SOLAS_Evaluation.ipynb     # Evaluation framework notebook
```

## Testing

The test suite is designed to run WITHOUT requiring heavy ML dependencies (torch, transformers, TTS, etc.). This allows quick validation of the library logic without GPU access.

### Running Tests

```bash
# Run all tests (from project root)
pytest -c library/pytest.ini

# Run with verbose output
pytest -c library/pytest.ini -v

# Run specific test file
pytest -c library/pytest.ini tests/test_pipeline_utils.py

# Run tests matching a pattern
pytest -k "chunk"

# Skip slow/integration tests
pytest -m "not slow"
```

### Test Categories

- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test module interactions (mocked heavy deps)
- **Skipped tests**: Tests requiring torch are marked with `@pytest.mark.skipif`

### Writing New Tests

Tests should mock heavy dependencies rather than importing them:

```python
import sys
from unittest.mock import MagicMock, patch

def test_with_mocked_torch():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch.dict(sys.modules, {'torch': mock_torch}):
        # Test code that imports torch
        pass
```

### Fixtures

Common fixtures are defined in `tests/conftest.py`:

- `mock_torch` - Mocked torch module
- `mock_transformers` - Mocked transformers module
- `sample_config` - Sample pipeline configuration
- `sample_transcript` - Sample transcript text
- `temp_dir` - Temporary directory for test outputs
- `templates_dir` - Path to templates directory

## Architecture

### Lazy Loading

Heavy dependencies are lazily imported to allow the library to be used without GPU/ML packages:

```python
# library/__init__.py uses __getattr__ for lazy loading
def __getattr__(name):
    if name == 'ensure_llm':
        from .pipeline_models import ensure_llm
        return ensure_llm
    raise AttributeError(f"module 'library' has no attribute '{name}'")
```

This means:
- `from library import get_verbosity` works without torch
- `from library import ensure_llm` only imports torch when called

### Pipeline Stages

The pipeline runs sequentially through these stages:

1. **Audio Loading** - Load and preprocess audio file
2. **ASR** - Transcribe with Whisper model
3. **Translation** - Translate transcript using LLM
4. **Summarization** - Generate bullet-point summary
5. **Podcast Script** - Create two-host dialogue
6. **TTS** - Synthesize audio with Coqui XTTS

Each stage is implemented in `pipeline_stages.py` and returns metrics and outputs.

## Common Development Gotchas

### Colab Widget CDN Notice

**Problem**: Using `widgets.HTML()` or transformers model loading progress bars triggers Colab's "Third-party Jupyter widgets" CDN notice panel.

**Root Causes**:
1. Using `widgets.HTML()` directly
2. Transformers library showing progress bars when loading models (uses tqdm with ipywidgets)

**Solutions**:
- For static labels: Use `widgets.Label()` with `label.style.font_weight = 'bold'`
- For styled content: Use `widgets.Output()` + `display(HTML(...))`
- For transformers progress bars: Set `os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'` before any transformers imports

```python
# Triggers CDN notice
label = widgets.HTML('<b>My Label</b>')

# No CDN notice - simple label
label = widgets.Label(value='My Label')
label.style.font_weight = 'bold'

# No CDN notice - styled content with updates
output = widgets.Output()
with output:
    display(HTML('<b style="color: green;">My Styled Content</b>'))

# No CDN notice - disable transformers progress bars
import os
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
from transformers import AutoModel  # Now won't show notebook-style progress bars
```

### File Encoding on Windows

**Problem**: Python on Windows defaults to `cp1252` encoding, causing `UnicodeDecodeError` when reading notebooks with emojis/special characters.

**Solution**: Always specify UTF-8 encoding explicitly.

```python
# Fails on Windows with Unicode content
data = json.load(open('notebook.ipynb'))

# Works everywhere
data = json.load(open('notebook.ipynb', encoding='utf-8'))
```

### LLM Generation Parameters

**Problem**: Passing `temperature`, `top_p`, `top_k` directly to `model.generate()` triggers deprecation warnings.

**Solution**: Use `GenerationConfig` for sampling parameters.

```python
from transformers import GenerationConfig

# Deprecated
output = model.generate(input_ids, temperature=0.7, top_p=0.9)

# Recommended
gen_config = GenerationConfig(
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
output = model.generate(input_ids, generation_config=gen_config)
```

### Display vs Return in Notebook Cells

**Problem**: Functions that return widgets may not display them automatically in all contexts.

**Solution**: Explicitly call `display()` for widgets, especially in callbacks.

```python
from IPython.display import display, HTML

# In callback functions, always use display()
def on_button_click(b):
    display(HTML('<b>Button clicked!</b>'))  # Explicit display
```
