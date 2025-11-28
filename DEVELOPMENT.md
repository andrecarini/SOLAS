## Development Gotchas

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
# ❌ Triggers CDN notice
label = widgets.HTML('<b>My Label</b>')

# ✅ No CDN notice - simple label
label = widgets.Label(value='My Label')
label.style.font_weight = 'bold'

# ✅ No CDN notice - styled content with updates
output = widgets.Output()
with output:
    display(HTML('<b style="color: green;">My Styled Content</b>'))

# ✅ No CDN notice - disable transformers progress bars
import os
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
from transformers import AutoModel  # Now won't show notebook-style progress bars
```

### File Encoding on Windows

**Problem**: Python on Windows defaults to `cp1252` encoding, causing `UnicodeDecodeError` when reading notebooks with emojis/special characters.

**Solution**: Always specify UTF-8 encoding explicitly.

```python
# ❌ Fails on Windows with Unicode content
data = json.load(open('notebook.ipynb'))

# ✅ Works everywhere
data = json.load(open('notebook.ipynb', encoding='utf-8'))
```

### LLM Generation Parameters

**Problem**: Passing `temperature`, `top_p`, `top_k` directly to `model.generate()` triggers deprecation warnings.

**Solution**: Use `GenerationConfig` for sampling parameters.

```python
from transformers import GenerationConfig

# ❌ Deprecated
output = model.generate(input_ids, temperature=0.7, top_p=0.9)

# ✅ Recommended
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
