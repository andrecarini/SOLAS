# SOLAS HTML Templates

This directory contains reusable HTML templates for the SOLAS interactive notebook.

## Architecture

### CSS Theming System

All templates use CSS custom properties (variables) defined in `_common.css`. This file is **automatically injected** at the start of every template by `load_template()`, providing consistent theming across all displays.

The theme system supports:
- **Colab light mode**: `html[theme="light"]`
- **Colab dark mode**: `html[theme="dark"]`
- **Non-Colab fallback**: `@media (prefers-color-scheme: dark)` for environments without the `theme` attribute

### File Structure

```
templates/
├── _common.css           # Shared CSS variables and utility classes (auto-injected)
├── README.md             # This file
├── section_header.html   # Reusable section header
├── setup_title.html      # Setup progress title
├── download_button.html  # CSV download button
├── scrollable_box.html   # Scrollable text container
├── status_badge.html     # Inline status badge
├── metrics_card.html     # Card container for metrics
├── evaluation_styles.html # (Legacy) Evaluation CSS - now empty, uses _common.css
└── ... other templates
```

## Usage

Templates are loaded using the `load_template()` function from the library:

```python
from library import load_template
from IPython.display import HTML, display

# Load a simple template
html = load_template('warning_non_colab.html')
display(HTML(html))

# Load a template with variables
html = load_template('section_header.html',
    title='My Section Title',
    color_class='primary'
)
display(HTML(html))
```

## Template Files

### Core UI Components

#### `_common.css`
- **Purpose**: Shared CSS variables and utility classes for all templates
- **Usage**: Automatically injected by `load_template()` - do not load manually
- **Contains**:
  - Color variables for light/dark themes
  - Utility classes (`.status-ok`, `.download-btn`, `.scrollable-box`, etc.)
  - Theme detection for Colab and non-Colab environments

#### `section_header.html`
- **Purpose**: Section header with colored bottom border
- **Variables**:
  - `title`: The header text
  - `color_class`: One of `primary`, `success`, `warning`, `purple`, `teal`, `indigo`
- **Usage**: Section dividers in evaluation displays

#### `setup_title.html`
- **Purpose**: Title header for setup progress display
- **Variables**:
  - `title`: The header text
- **Usage**: Setup progress header

#### `download_button.html`
- **Purpose**: Download button for CSV exports
- **Variables**:
  - `data_uri`: Base64-encoded data URI
  - `filename`: Download filename
- **Usage**: CSV export buttons in evaluation displays

#### `scrollable_box.html`
- **Purpose**: Scrollable text container with optional status badge
- **Variables**:
  - `content`: Text content to display
  - `height`: Box height (e.g., "250px")
  - `status_html`: Optional status badge HTML
- **Usage**: Text output displays

#### `status_badge.html`
- **Purpose**: Inline status badge
- **Variables**:
  - `status_class`: One of `ok`, `warn`, `fail`, `muted`
  - `text`: Badge text
- **Usage**: Status indicators

#### `metrics_card.html`
- **Purpose**: Card container for metrics display
- **Variables**:
  - `title`: Card title
  - `content`: HTML content inside the card
- **Usage**: Metrics grouping

### Setup & Configuration

#### `warning_non_colab.html`
- **Purpose**: Warning message for non-Colab environments
- **Variables**: None (static template)

#### `dependency_table.html`
- **Purpose**: Dependency status table with embedded styles
- **Variables**:
  - `dependency_rows`: HTML string with `<tr>` elements

#### `setup_complete.html`
- **Purpose**: Complete setup status display
- **Variables**:
  - `completion_title`: HTML for completion title
  - `completion_items`: HTML for completion messages
  - `gpu_item`: HTML for GPU status
  - `success_msg`: HTML for success message
  - `restart_warning_html`: HTML for restart warning
  - `table_html`: HTML for dependency table
  - `error_html`: HTML for error details

#### `restart_warning.html`
- **Purpose**: Runtime restart warning message
- **Variables**: None (static template)

#### `setup_warning.html`
- **Purpose**: Warning when setup not completed
- **Variables**: None (static template)

#### `config_not_completed.html`
- **Purpose**: Warning when Configuration cell not run
- **Variables**: None (static template)

#### `config_ready.html`
- **Purpose**: Info message before configuration widgets
- **Variables**: None (static template)

#### `config_confirmed.html`
- **Purpose**: Success message when configuration confirmed
- **Variables**: None (static template)

#### `success_message.html`
- **Purpose**: Success message when setup completes
- **Variables**: None (static template)

### Pipeline Execution

#### `audio_selection_ready.html`
- **Purpose**: Info message before audio selection
- **Variables**: None (static template)

#### `host_voice_selection_ready.html`
- **Purpose**: Info message before host voice selection
- **Variables**: None (static template)

#### `pipeline_start.html`
- **Purpose**: Info message before pipeline execution
- **Variables**: None (static template)

#### `pipeline_config.html`
- **Purpose**: Display pipeline configuration
- **Variables**:
  - `config_html`: Formatted configuration details

#### `pipeline_complete.html`
- **Purpose**: Success message after pipeline completes
- **Variables**: None (static template)

#### `no_results_warning.html`
- **Purpose**: Warning when results not available
- **Variables**: None (static template)

#### `results_display.html`
- **Purpose**: Rich HTML display for pipeline results
- **Variables**:
  - `total_runtime`: Total runtime in seconds
  - `audio_duration`: Audio duration in seconds
  - `real_time_factor`: TTS speed factor
  - `stage_metrics_html`: Stage metrics HTML
  - `original_transcript`: Original transcript text
  - `translated_transcript`: Translated transcript text
  - `key_points_summary`: Summary text
  - `podcast_script`: Podcast script text
  - `output_directory`: Output path

### Evaluation Display

#### `evaluation_styles.html`
- **Purpose**: (Legacy) Shared CSS for evaluation display
- **Note**: Now empty - all styles defined in `_common.css` and auto-injected

## CSS Variables Reference

### Core Colors
| Variable | Light Mode | Dark Mode | Description |
|----------|------------|-----------|-------------|
| `--color-primary` | `#1a73e8` | `#8ab4f8` | Primary accent |
| `--color-primary-dark` | `#1565c0` | `#5d9eff` | Primary dark variant |
| `--color-success` | `#34a853` | `#81c995` | Success/green |
| `--color-success-dark` | `#2e7d32` | `#66bb6a` | Success dark |
| `--color-warning` | `#f9ab00` | `#fdd663` | Warning/yellow |
| `--color-warning-dark` | `#f57c00` | `#ffb74d` | Warning dark |
| `--color-error` | `#ea4335` | `#f28b82` | Error/red |
| `--color-error-dark` | `#c62828` | `#ef5350` | Error dark |
| `--color-info` | `#1a73e8` | `#8ab4f8` | Info/blue |
| `--color-info-dark` | `#2196f3` | `#64b5f6` | Info dark |
| `--color-muted` | `#9e9e9e` | `#9aa0a6` | Muted/gray |

### Text Colors
| Variable | Light Mode | Dark Mode |
|----------|------------|-----------|
| `--color-text-primary` | `#202124` | `#e8eaed` |
| `--color-text-secondary` | `#5f6368` | `#9aa0a6` |

### Background Colors
| Variable | Light Mode | Dark Mode |
|----------|------------|-----------|
| `--color-bg-primary` | `#ffffff` | `#202124` |
| `--color-bg-secondary` | `#f8f9fa` | `#292a2d` |
| `--color-bg-highlight` | `rgba(33,150,243,0.1)` | `rgba(100,181,246,0.15)` |

### Border Colors
| Variable | Light Mode | Dark Mode |
|----------|------------|-----------|
| `--color-border` | `#dadce0` | `#5f6368` |
| `--color-border-light` | `rgba(128,128,128,0.2)` | `rgba(255,255,255,0.1)` |
| `--color-border-primary` | `rgba(21,101,192,0.3)` | `rgba(138,180,248,0.3)` |

### Accent Colors (for section headers)
| Variable | Light Mode | Dark Mode |
|----------|------------|-----------|
| `--color-accent-purple` | `#7b1fa2` | `#ce93d8` |
| `--color-accent-teal` | `#00897b` | `#4db6ac` |
| `--color-accent-indigo` | `#5c6bc0` | `#9fa8da` |

### Status Colors (inline text)
| Variable | Light Mode | Dark Mode |
|----------|------------|-----------|
| `--color-status-ok` | `#4caf50` | `#81c995` |
| `--color-status-warn` | `#ff9800` | `#ffb74d` |
| `--color-status-fail` | `#ef5350` | `#ef5350` |

### Verdict Cell Colors
| Variable | Light Mode | Dark Mode |
|----------|------------|-----------|
| `--color-verdict-ok-bg` | `rgba(76,175,80,0.2)` | `#1b3d1b` |
| `--color-verdict-ok-text` | `#2e7d32` | `#66bb6a` |
| `--color-verdict-warn-bg` | `rgba(255,152,0,0.25)` | `#3d2e0f` |
| `--color-verdict-warn-text` | `#e65100` | `#ffa726` |
| `--color-verdict-fail-bg` | `rgba(239,83,80,0.2)` | `#3d1515` |
| `--color-verdict-fail-text` | `#c62828` | `#ef5350` |

## CSS Utility Classes

Defined in `_common.css` and available in all templates:

```css
/* Status text */
.status-ok { color: var(--color-status-ok); font-weight: bold; }
.status-warn { color: var(--color-status-warn); font-weight: bold; }
.status-fail { color: var(--color-status-fail); font-weight: bold; }
.status-muted { color: var(--color-muted); font-style: italic; }

/* Verdict cells */
.verdict-ok { background: var(--color-verdict-ok-bg); color: var(--color-verdict-ok-text); }
.verdict-warn { background: var(--color-verdict-warn-bg); color: var(--color-verdict-warn-text); }
.verdict-fail { background: var(--color-verdict-fail-bg); color: var(--color-verdict-fail-text); }

/* Section headers */
.section-header { border-bottom: 2px solid; padding-bottom: 10px; margin-top: 30px; }
.section-header-primary { border-color: var(--color-primary-dark); }
.section-header-success { border-color: var(--color-success-dark); }
.section-header-warning { border-color: var(--color-warning-dark); }
.section-header-purple { border-color: var(--color-accent-purple); }
.section-header-teal { border-color: var(--color-accent-teal); }
.section-header-indigo { border-color: var(--color-accent-indigo); }

/* Components */
.download-btn { /* Styled download button */ }
.scrollable-box { /* Scrollable content container */ }
.metrics-table { /* Data table styling */ }
.card { /* Card container */ }
.info-box { /* Info message box */ }
```

## Adding New Templates

1. Create a new `.html` file in this directory
2. Add a comment header with:
   - Template name
   - Purpose
   - Usage instructions
   - Variables documentation
3. Use `{variable_name}` for variable substitution
4. Use CSS variables from `_common.css` for colors
5. Update this README with the new template information

## Variable Substitution

Templates use Python's `.format()` method for variable substitution:

```python
# Template: <h2 class="section-header section-header-{color_class}">{title}</h2>
html = load_template('section_header.html', title='Results', color_class='primary')
# Result: <h2 class="section-header section-header-primary">Results</h2>
```
