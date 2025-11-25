# SOLAS HTML Templates

This directory contains reusable HTML templates for the SOLAS interactive notebook.

## Usage

Templates are loaded using the `load_template()` function from `solas_pipeline.py`:

```python
from solas_pipeline import load_template
from IPython.display import HTML, display

# Load a simple template
html = load_template('warning_non_colab.html')
display(HTML(html))

# Load a template with variables
html = load_template('setup_complete.html',
    completion_title='<h3>✓ Setup Complete!</h3>',
    completion_items='<p>All dependencies installed.</p>',
    gpu_item='<p>✓ GPU Available</p>',
    success_msg='',
    restart_warning_html='',
    table_html='<table>...</table>',
    error_html=''
)
display(HTML(html))
```

## Template Files

### `warning_non_colab.html`
- **Purpose**: Warning message for non-Colab environments
- **Variables**: None (static template)
- **Usage**: Display when notebook is run outside Google Colab

### `dependency_table.html`
- **Purpose**: Dependency status table with embedded styles
- **Variables**: 
  - `dependency_rows`: HTML string with `<tr>` elements for each dependency
- **Usage**: Generate dependency status table (includes all necessary CSS styles)

### `setup_complete.html`
- **Purpose**: Complete setup status display
- **Variables**:
  - `completion_title`: HTML for completion title
  - `completion_items`: HTML for completion messages
  - `gpu_item`: HTML for GPU status
  - `success_msg`: HTML for success message (empty if restart needed)
  - `restart_warning_html`: HTML for restart warning (empty if not needed)
  - `table_html`: HTML for dependency table (includes styles)
  - `error_html`: HTML for error details (empty if no errors)
- **Usage**: Display final setup status with dependency table

### `restart_warning.html`
- **Purpose**: Runtime restart warning message
- **Variables**: None (static template)
- **Usage**: Display when bitsandbytes update requires restart

### `setup_warning.html`
- **Purpose**: Warning message when setup has not been completed
- **Variables**: None (static template)
- **Usage**: Display when user tries to run cells before completing setup

### `config_not_completed.html`
- **Purpose**: Warning message when Configuration cell has not been run
- **Variables**: None (static template)
- **Usage**: Display when user tries to run cells that require configuration

### `audio_selection_ready.html`
- **Purpose**: Info message shown before audio selection interface
- **Variables**: None (static template)
- **Usage**: Display at the start of the Upload Input Audio cell

### `success_message.html`
- **Purpose**: Success message when setup completes
- **Variables**: None (static template)
- **Usage**: Display when setup completes successfully

### `config_ready.html`
- **Purpose**: Information message shown before configuration widgets
- **Variables**: None (static template)
- **Usage**: Display before configuration widgets in Configuration cell

### `config_confirmed.html`
- **Purpose**: Success message shown when configuration is confirmed
- **Variables**: None (static template)
- **Usage**: Display when user confirms configuration

### `results_display.html`
- **Purpose**: Rich HTML display for pipeline results
- **Variables**:
  - `total_runtime`: Total pipeline runtime in seconds (float)
  - `audio_duration`: Podcast audio duration in seconds (float)
  - `real_time_factor`: TTS real-time factor (float)
  - `stage_metrics_html`: HTML string with stage-by-stage metrics
  - `translated_transcript`: Translated transcript text
  - `key_points_summary`: Key points summary text
  - `podcast_script`: Podcast script text
  - `audio_path`: Path to podcast audio file
  - `output_directory`: Output directory path
  - `file_links_html`: HTML string with file download links
- **Usage**: Display comprehensive pipeline results

## Variable Substitution

Templates use Python's `.format()` method for variable substitution. Variables are specified using `{variable_name}` syntax in the template files.

Example:
```python
# Template contains: <p>Hello {name}!</p>
html = load_template('example.html', name='World')
# Result: <p>Hello World!</p>
```

## Adding New Templates

1. Create a new `.html` file in this directory
2. Add a comment header with:
   - Template name
   - Purpose
   - Usage instructions
   - Variables documentation
3. Use `{variable_name}` for variable substitution
4. Update this README with the new template information

## CSS Variables

All templates use CSS custom properties (variables) for theming:
- `--color-primary`: Primary accent color
- `--color-success`: Success/green color
- `--color-warning`: Warning/yellow color
- `--color-error`: Error/red color
- `--color-text-primary`: Primary text color
- `--color-text-secondary`: Secondary text color
- `--color-bg-primary`: Primary background color
- `--color-bg-secondary`: Secondary background color
- `--color-border`: Border color

These automatically adapt to Colab's dark/light theme.

