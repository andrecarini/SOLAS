"""
Results display and export functions for SOLAS evaluation.
Includes HTML generation for Jupyter notebook displays.
"""

import csv
import json
import base64
import io
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict

from .pipeline_templates import load_template

_css_injected = False


def inject_evaluation_css():
    """Inject evaluation CSS styles once per session."""
    global _css_injected
    if _css_injected:
        return
    from IPython.display import display, HTML
    from .pipeline_templates import load_template
    css_html = load_template('evaluation_styles.html')
    display(HTML(css_html))
    _css_injected = True


def get_verdict_class(verdict: str) -> str:
    """Return CSS class for verdict cell."""
    if verdict == 'OK':
        return 'verdict-ok'
    elif verdict == 'WARNING' or verdict == 'WARN':
        return 'verdict-warn'
    elif verdict == 'FAIL':
        return 'verdict-fail'
    return ''

# Import analysis module (optional - gracefully handles missing dependencies)
try:
    from .evaluation_analysis import (
        run_full_analysis,
        load_results_as_dataframe,
        compare_outputs,
        create_blind_evaluation_template,
        generate_all_latex_tables,
        PANDAS_AVAILABLE,
        MATPLOTLIB_AVAILABLE,
    )
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False
    PANDAS_AVAILABLE = False
    MATPLOTLIB_AVAILABLE = False


def display_results_summary(results: Dict[str, Any], results_file: Path, outputs_dir: Path, log_fn):
    """Display a summary of evaluation results."""
    if not results.get('experiments'):
        log_fn("No results yet. Run the evaluation first.", 'warning')
        return

    log_fn("Evaluation Results Summary", 'header')

    hw = results['metadata'].get('hardware', {})
    log_fn(f"Hardware: {hw.get('gpu_name', 'Unknown')} ({hw.get('gpu_memory_total_gb', 0):.1f} GB VRAM)", 'info')
    log_fn(f"RAM: {hw.get('ram_total_gb', 0):.1f} GB", 'info')

    by_type = defaultdict(list)
    errors = []
    for exp_id, exp in results['experiments'].items():
        if 'error' in exp:
            errors.append(exp)
        else:
            by_type[exp.get('experiment_type', 'unknown')].append(exp)

    if 'asr_model' in by_type:
        log_fn("ASR Model Comparison", 'header')
        print(f"{'Model':<25} {'Time(s)':<10} {'RTF':<8} {'Words':<10} {'VRAM Peak(GB)':<15}")
        print("-" * 70)
        for exp in sorted(by_type['asr_model'], key=lambda e: e['id']):
            asr_data = exp.get('stages_data', {}).get('asr', {})
            m = asr_data.get('metrics', {})
            o = asr_data.get('output', {})
            model = exp['config']['asr_model_id'].split('/')[-1]
            print(f"{model:<25} {m.get('time_seconds', 0):<10.1f} {m.get('realtime_factor', 0):<8.2f} {o.get('transcript_words', 0):<10} {m.get('vram_peak_gb', 0):<15.2f}")

    if 'quantization' in by_type:
        log_fn("Quantization Impact", 'header')
        print(f"{'Model':<30} {'Quant':<8} {'Trans(s)':<10} {'Sum(s)':<10} {'Pod(s)':<10} {'VRAM Peak(GB)':<15}")
        print("-" * 85)
        for exp in sorted(by_type['quantization'], key=lambda e: (e['config']['llm_model_id'], str(e['config']['quantization']))):
            model = exp['config']['llm_model_id'].split('/')[-1][:25]
            quant = exp['config'].get('quantization') or 'none'
            trans_m = exp.get('stages_data', {}).get('translation', {}).get('metrics', {})
            sum_m = exp.get('stages_data', {}).get('summary', {}).get('metrics', {})
            pod_m = exp.get('stages_data', {}).get('podcast', {}).get('metrics', {})
            max_vram = max(trans_m.get('vram_peak_gb', 0), sum_m.get('vram_peak_gb', 0), pod_m.get('vram_peak_gb', 0))
            print(f"{model:<30} {quant:<8} {trans_m.get('time_seconds', 0):<10.1f} {sum_m.get('time_seconds', 0):<10.1f} {pod_m.get('time_seconds', 0):<10.1f} {max_vram:<15.2f}")

    if 'repetition_penalty' in by_type:
        log_fn("Repetition Penalty Impact", 'header')
        print(f"{'Model':<30} {'Penalty':<10} {'Pod Lines':<12} {'Time(s)':<10}")
        print("-" * 65)
        for exp in sorted(by_type['repetition_penalty'], key=lambda e: (e['config']['llm_model_id'], str(e['config']['repetition_penalty']))):
            model = exp['config']['llm_model_id'].split('/')[-1][:25]
            penalty = exp['config'].get('repetition_penalty')
            penalty_str = 'none' if penalty is None else str(penalty)
            pod_data = exp.get('stages_data', {}).get('podcast', {})
            pod_m = pod_data.get('metrics', {})
            pod_o = pod_data.get('output', {})
            print(f"{model:<30} {penalty_str:<10} {pod_o.get('total_lines', 0):<12} {pod_m.get('time_seconds', 0):<10.1f}")

    if 'summary_mode' in by_type:
        log_fn("Summary Mode Impact", 'header')
        print(f"{'Mode':<15} {'Bullets':<10} {'Sum(s)':<10} {'VRAM Peak(GB)':<15}")
        print("-" * 55)
        for exp in sorted(by_type['summary_mode'], key=lambda e: e['config']['summary_mode']):
            mode = exp['config']['summary_mode']
            sum_data = exp.get('stages_data', {}).get('summary', {})
            sum_m = sum_data.get('metrics', {})
            sum_o = sum_data.get('output', {})
            print(f"{mode:<15} {sum_o.get('bullet_count', 0):<10} {sum_m.get('time_seconds', 0):<10.1f} {sum_m.get('vram_peak_gb', 0):<15.2f}")

    if 'chunk_size' in by_type:
        log_fn("Chunk Size Impact", 'header')
        print(f"{'Chunk Size':<15} {'Trans(s)':<10} {'Sum(s)':<10} {'Pod(s)':<10} {'Total(s)':<10}")
        print("-" * 60)
        for exp in sorted(by_type['chunk_size'], key=lambda e: e['config']['chunk_size_chars']):
            chunk = exp['config']['chunk_size_chars']
            trans_m = exp.get('stages_data', {}).get('translation', {}).get('metrics', {})
            sum_m = exp.get('stages_data', {}).get('summary', {}).get('metrics', {})
            pod_m = exp.get('stages_data', {}).get('podcast', {}).get('metrics', {})
            total = trans_m.get('time_seconds', 0) + sum_m.get('time_seconds', 0) + pod_m.get('time_seconds', 0)
            print(f"{chunk:<15} {trans_m.get('time_seconds', 0):<10.1f} {sum_m.get('time_seconds', 0):<10.1f} {pod_m.get('time_seconds', 0):<10.1f} {total:<10.1f}")

    if 'temperature' in by_type:
        log_fn("Temperature Impact (Mistral-7B Podcast Generation)", 'header')
        print(f"{'Temp':<10} {'Pod Lines':<12} {'Host A':<10} {'Host B':<10} {'Time(s)':<10}")
        print("-" * 55)
        for exp in sorted(by_type['temperature'], key=lambda e: e['config']['podcast_creativity_temp']):
            temp = exp['config']['podcast_creativity_temp']
            pod_data = exp.get('stages_data', {}).get('podcast', {})
            pod_m = pod_data.get('metrics', {})
            pod_o = pod_data.get('output', {})
            print(f"{temp:<10} {pod_o.get('total_lines', 0):<12} {pod_o.get('host_a_lines', 0):<10} {pod_o.get('host_b_lines', 0):<10} {pod_m.get('time_seconds', 0):<10.1f}")

    if 'best_params' in by_type:
        log_fn("Best Params Comparison (4 LLMs with optimal settings)", 'header')
        print(f"{'Model':<30} {'Quant':<8} {'Trans(s)':<10} {'Sum(s)':<10} {'Pod(s)':<10} {'Total(s)':<10} {'VRAM(GB)':<10}")
        print("-" * 95)
        for exp in sorted(by_type['best_params'], key=lambda e: e['config']['llm_model_id']):
            model = exp['config']['llm_model_id'].split('/')[-1][:25]
            quant = exp['config'].get('quantization') or 'none'
            trans_m = exp.get('stages_data', {}).get('translation', {}).get('metrics', {})
            sum_m = exp.get('stages_data', {}).get('summary', {}).get('metrics', {})
            pod_m = exp.get('stages_data', {}).get('podcast', {}).get('metrics', {})
            trans_t = trans_m.get('time_seconds', 0)
            sum_t = sum_m.get('time_seconds', 0)
            pod_t = pod_m.get('time_seconds', 0)
            total = trans_t + sum_t + pod_t
            max_vram = max(trans_m.get('vram_peak_gb', 0), sum_m.get('vram_peak_gb', 0), pod_m.get('vram_peak_gb', 0))
            print(f"{model:<30} {quant:<8} {trans_t:<10.1f} {sum_t:<10.1f} {pod_t:<10.1f} {total:<10.1f} {max_vram:<10.2f}")

    if errors:
        log_fn(f"Errors ({len(errors)} experiments failed)", 'header')
        for exp in errors:
            log_fn(f"  - {exp.get('id', 'unknown')}: {exp['error'][:60]}...", 'error')

    log_fn(f"\nResults saved to: {results_file}", 'info')
    log_fn(f"Outputs saved to: {outputs_dir}", 'info')


def export_for_analysis(results: Dict[str, Any], drive_base: Path, log_fn):
    """Export results as CSV files for analysis."""
    if not results.get('experiments'):
        log_fn("No results to export.", 'warning')
        return

    export_dir = drive_base / 'csv_exports'
    export_dir.mkdir(exist_ok=True)

    all_exps = [e for e in results['experiments'].values() if 'error' not in e]
    if all_exps:
        with open(export_dir / 'all_experiments.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Experiment_Type', 'Experiment_ID', 'Description', 'Config_Hash', 'GPU_Type',
                'ASR_Model', 'LLM_Model', 'Quantization', 'Chunk_Size', 'Repetition_Penalty',
                'Summary_Mode', 'Temperature',
                'ASR_Time_s', 'ASR_VRAM_Peak_GB', 'ASR_RTF',
                'Trans_Time_s', 'Trans_VRAM_Peak_GB', 'Trans_Words',
                'Sum_Time_s', 'Sum_VRAM_Peak_GB', 'Sum_Bullets',
                'Pod_Time_s', 'Pod_VRAM_Peak_GB', 'Pod_Lines', 'Pod_HostA', 'Pod_HostB',
                'Model_Load_Time_s'
            ])
            for exp in all_exps:
                cfg = exp['config']
                sd = exp.get('stages_data', {})
                hw = exp.get('hardware_info', {})
                asr_m = sd.get('asr', {}).get('metrics', {})
                asr_o = sd.get('asr', {}).get('output', {})
                trans_m = sd.get('translation', {}).get('metrics', {})
                trans_o = sd.get('translation', {}).get('output', {})
                sum_m = sd.get('summary', {}).get('metrics', {})
                sum_o = sd.get('summary', {}).get('output', {})
                pod_m = sd.get('podcast', {}).get('metrics', {})
                pod_o = sd.get('podcast', {}).get('output', {})
                model_load_time = sum([
                    sd.get('asr', {}).get('model_load_time_seconds', 0),
                    sd.get('translation', {}).get('model_load_time_seconds', 0),
                    sd.get('summary', {}).get('model_load_time_seconds', 0),
                    sd.get('podcast', {}).get('model_load_time_seconds', 0)
                ])
                writer.writerow([
                    exp.get('experiment_type', ''),
                    exp['id'],
                    exp.get('description', ''),
                    exp.get('config_hash', ''),
                    hw.get('gpu_type', ''),
                    cfg.get('asr_model_id', '').split('/')[-1],
                    cfg.get('llm_model_id', '').split('/')[-1],
                    cfg.get('quantization') or 'none',
                    cfg.get('chunk_size_chars', ''),
                    cfg.get('repetition_penalty') if cfg.get('repetition_penalty') is not None else 'none',
                    cfg.get('summary_mode', 'greedy'),
                    cfg.get('podcast_creativity_temp', ''),
                    asr_m.get('time_seconds', ''),
                    asr_m.get('vram_peak_gb', ''),
                    asr_m.get('realtime_factor', ''),
                    trans_m.get('time_seconds', ''),
                    trans_m.get('vram_peak_gb', ''),
                    trans_o.get('translated_words', ''),
                    sum_m.get('time_seconds', ''),
                    sum_m.get('vram_peak_gb', ''),
                    sum_o.get('bullet_count', ''),
                    pod_m.get('time_seconds', ''),
                    pod_m.get('vram_peak_gb', ''),
                    pod_o.get('total_lines', ''),
                    pod_o.get('host_a_lines', ''),
                    pod_o.get('host_b_lines', ''),
                    model_load_time
                ])
        log_fn(f"Exported: {export_dir / 'all_experiments.csv'}", 'success')

    log_fn(f"\nAll exports saved to: {export_dir}", 'info')


def run_thesis_analysis(
    results: Dict[str, Any],
    drive_base: Path,
    log_fn,
    solas_podcast_path: Optional[Path] = None,
    notebooklm_podcast_path: Optional[Path] = None,
):
    """
    Run comprehensive analysis for thesis inclusion.
    Generates plots, LaTeX tables, and optional NotebookLM comparison.

    Args:
        results: Evaluation results dictionary
        drive_base: Base directory for output files
        log_fn: Logging function
        solas_podcast_path: Optional path to SOLAS podcast script
        notebooklm_podcast_path: Optional path to NotebookLM podcast transcript
    """
    if not ANALYSIS_AVAILABLE:
        log_fn("Analysis module not available. Install pandas and matplotlib:", 'warning')
        log_fn("  pip install pandas matplotlib seaborn", 'info')
        return

    if not results.get('experiments'):
        log_fn("No results to analyze.", 'warning')
        return

    # Load optional comparison texts
    solas_podcast = None
    notebooklm_podcast = None

    if solas_podcast_path and solas_podcast_path.exists():
        with open(solas_podcast_path, 'r', encoding='utf-8') as f:
            solas_podcast = f.read()
        log_fn(f"Loaded SOLAS podcast from: {solas_podcast_path}", 'info')

    if notebooklm_podcast_path and notebooklm_podcast_path.exists():
        with open(notebooklm_podcast_path, 'r', encoding='utf-8') as f:
            notebooklm_podcast = f.read()
        log_fn(f"Loaded NotebookLM podcast from: {notebooklm_podcast_path}", 'info')

    # Run full analysis
    analysis_dir = drive_base / 'thesis_analysis'
    generated = run_full_analysis(
        results=results,
        output_dir=analysis_dir,
        solas_podcast=solas_podcast,
        notebooklm_podcast=notebooklm_podcast,
        log_fn=log_fn,
    )

    # Report generated files
    log_fn("Thesis Analysis Complete", 'header')

    if generated.get('plots'):
        log_fn("Generated plots:", 'info')
        for name, path in generated['plots'].items():
            log_fn(f"  - {name}: {path}", 'detail')

    if generated.get('tables'):
        log_fn("Generated LaTeX tables:", 'info')
        for name, path in generated['tables'].items():
            log_fn(f"  - {name}: {path}", 'detail')

    if generated.get('comparison'):
        log_fn("Generated comparison files:", 'info')
        for name, path in generated['comparison'].items():
            log_fn(f"  - {name}: {path}", 'detail')

    log_fn(f"\nAll thesis materials saved to: {analysis_dir}", 'success')
    return generated


def setup_notebooklm_comparison(
    results: Dict[str, Any],
    drive_base: Path,
    log_fn,
):
    """
    Set up directories and instructions for NotebookLM comparison.
    Creates the folder structure and README with instructions.
    """
    comparison_dir = drive_base / 'notebooklm_comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (comparison_dir / 'solas_outputs').mkdir(exist_ok=True)
    (comparison_dir / 'notebooklm_outputs').mkdir(exist_ok=True)

    # Find best experiment output to use for comparison
    best_exp = None
    if results.get('experiments'):
        # Prefer best_params with phi-3-mini or mistral
        for exp_id, exp in results['experiments'].items():
            if 'error' not in exp and exp.get('experiment_type') == 'best_params':
                if 'phi-3' in exp.get('config', {}).get('llm_model_id', '').lower():
                    best_exp = exp
                    break
                if 'mistral' in exp.get('config', {}).get('llm_model_id', '').lower():
                    best_exp = exp

    # Copy SOLAS outputs if available
    if best_exp:
        stages_data = best_exp.get('stages_data', {})

        # Save translation
        trans_text = stages_data.get('translation', {}).get('output', {}).get('translated_text', '')
        if trans_text:
            with open(comparison_dir / 'solas_outputs' / 'translation.txt', 'w', encoding='utf-8') as f:
                f.write(trans_text)

        # Save summary
        sum_text = stages_data.get('summary', {}).get('output', {}).get('summary', '')
        if sum_text:
            with open(comparison_dir / 'solas_outputs' / 'summary.txt', 'w', encoding='utf-8') as f:
                f.write(sum_text)

        # Save podcast
        pod_text = stages_data.get('podcast', {}).get('output', {}).get('script', '')
        if pod_text:
            with open(comparison_dir / 'solas_outputs' / 'podcast_script.txt', 'w', encoding='utf-8') as f:
                f.write(pod_text)

        log_fn(f"Copied SOLAS outputs from experiment: {best_exp['id']}", 'success')

    # Create README with instructions
    readme = """# NotebookLM Comparison Setup

## Overview
This directory contains materials for comparing SOLAS outputs with Google NotebookLM.

## Directory Structure
```
notebooklm_comparison/
├── solas_outputs/
│   ├── translation.txt      # SOLAS translated text
│   ├── summary.txt          # SOLAS summary
│   └── podcast_script.txt   # SOLAS podcast script
├── notebooklm_outputs/
│   ├── briefing_doc.txt     # NotebookLM briefing document (copy here)
│   └── podcast_transcript.txt  # NotebookLM audio overview transcript (copy here)
└── README.md                # This file
```

## Instructions

### Step 1: Upload to NotebookLM
1. Go to https://notebooklm.google.com/
2. Create a new notebook
3. Upload `solas_outputs/translation.txt` as a source

### Step 2: Generate NotebookLM Outputs

#### For Briefing Document:
1. In NotebookLM chat, ask: "Create a bulleted summary of the most important concepts, definitions, and takeaways from this lecture transcript. Limit to 10-15 key points."
2. Copy the response to `notebooklm_outputs/briefing_doc.txt`

#### For Podcast/Audio Overview:
1. Click "Generate" -> "Audio Overview"
2. Optional: Use customization prompt: "Focus on the key technical concepts and make the discussion educational. Explain any complex terms in simple language. Target audience: university students learning the subject."
3. Wait for generation to complete
4. If transcript is available, copy it to `notebooklm_outputs/podcast_transcript.txt`
5. If no transcript, download the audio and transcribe it using Whisper

### Step 3: Run Comparison Analysis
After placing NotebookLM outputs in the correct locations, run:

```python
evaluation.run_thesis_analysis(
    solas_podcast_path=Path('notebooklm_comparison/solas_outputs/podcast_script.txt'),
    notebooklm_podcast_path=Path('notebooklm_comparison/notebooklm_outputs/podcast_transcript.txt'),
)
```

## Evaluation Rubric

When manually evaluating outputs, use this rubric:

| Criterion | 1 (Poor) | 3 (Adequate) | 5 (Excellent) |
|-----------|----------|--------------|---------------|
| Accuracy | Major factual errors | Minor errors | Factually correct |
| Completeness | Missing key points | Most points covered | All key points |
| Fluency | Awkward/robotic | Readable | Natural/engaging |
| Structure | Disorganized | Logical flow | Clear progression |
| Engagement | Boring/dry | Informative | Engaging dialogue |
"""

    with open(comparison_dir / 'README.md', 'w', encoding='utf-8') as f:
        f.write(readme)

    log_fn("NotebookLM Comparison Setup", 'header')
    log_fn(f"Created comparison directory: {comparison_dir}", 'success')
    log_fn("Next steps:", 'info')
    log_fn("  1. Upload translation.txt to NotebookLM", 'detail')
    log_fn("  2. Generate briefing document and audio overview", 'detail')
    log_fn("  3. Save outputs to notebooklm_outputs/ folder", 'detail')
    log_fn("  4. Run evaluation.run_thesis_analysis() with paths", 'detail')
    log_fn(f"\nSee {comparison_dir / 'README.md'} for detailed instructions", 'info')

    return comparison_dir


# =============================================================================
# HTML DISPLAY FUNCTIONS FOR JUPYTER NOTEBOOKS
# =============================================================================

# Model ordering and display names
MODEL_ORDER = ['Qwen2-0.5B-Instruct', 'Qwen2-1.5B-Instruct', 'phi-3-mini-4k-instruct', 'Mistral-7B-Instruct-v0.3']
MODEL_SHORT = {
    'Qwen2-0.5B-Instruct': 'Qwen2-0.5B',
    'Qwen2-1.5B-Instruct': 'Qwen2-1.5B',
    'phi-3-mini-4k-instruct': 'Phi-3-mini',
    'Mistral-7B-Instruct-v0.3': 'Mistral-7B'
}


def create_download_button(csv_content: str, filename: str) -> str:
    """Create HTML download button for CSV data."""
    csv_b64 = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')
    return f'''
    <div style="margin: 15px 0;">
        <a href="data:text/csv;base64,{csv_b64}" download="{filename}" class="download-btn">
            📥 Download CSV
        </a>
        <span style="margin-left: 10px; font-size: 12px; color: var(--color-text-secondary);">{filename}</span>
    </div>
    '''


def create_scrollable_box(
    title: str,
    content: str,
    height: str = "250px",
    status: str = None,
    fail_reason: str = None
) -> str:
    """Create HTML for a scrollable text box (dark mode compatible)."""
    escaped_content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')

    status_html = ''
    if status:
        if status == 'FAIL':
            reason = f" - {fail_reason}" if fail_reason else ""
            status_html = f'<div class="status-fail" style="font-size: 11px; margin-bottom: 5px;">FAIL{reason}</div>'
        elif status == 'WARNING':
            reason = f" - {fail_reason}" if fail_reason else ""
            status_html = f'<div class="status-warn" style="font-size: 11px; margin-bottom: 5px;">WARNING{reason}</div>'
        elif status == 'OK':
            status_html = '<div class="status-ok" style="font-size: 11px; margin-bottom: 5px;">OK</div>'
        elif status == 'NO DATA':
            status_html = '<div class="status-muted" style="font-size: 11px; margin-bottom: 5px;">No data available</div>'

    return f'''
    <div style="flex: 1; min-width: 0; overflow: hidden;">
        <h5 style="margin: 0 0 5px 0; font-size: 13px; color: var(--color-text-primary);">{title}</h5>
        {status_html}
        <div class="scrollable-box" style="height: {height};">{escaped_content if content else "(No output)"}</div>
    </div>
    '''


def create_text_boxes_row(
    stage_name: str,
    data_by_model: dict,
    quant_type: str,
    stage_key: str
) -> str:
    """Create a row of text boxes for a stage across all models."""
    html = f'<h4 style="margin: 15px 0 10px 0; color: var(--color-text-primary);">{stage_name}</h4>'
    html += '<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">'

    for model in MODEL_ORDER:
        model_short = MODEL_SHORT.get(model, model)
        if model in data_by_model and quant_type in data_by_model[model]:
            exp_data = data_by_model[model][quant_type]
            text = exp_data.get(f'{stage_key}_text', '')
            verdict = exp_data.get(f'{stage_key}_verdict', 'NO DATA')
            fail_reason = exp_data.get(f'{stage_key}_fail_reason', '')
            word_count = exp_data.get(f'{stage_key}_words', 0)
            title = f"{model_short} ({word_count:,} words)" if word_count else model_short
            html += create_scrollable_box(title, text, "250px", verdict, fail_reason)
        else:
            html += create_scrollable_box(model_short, "(Experiment not found)", "250px", "NO DATA")

    html += '</div>'
    return html


def create_stage_metrics_table(
    data_by_model: dict,
    quant_type: str,
    stage_name: str,
    stage_prefix: str,
    extra_metrics: List[tuple] = None
) -> str:
    """
    Create a metrics table for a single pipeline stage.
    Better separation with dedicated table per stage.

    Args:
        data_by_model: Dict of model -> quant_type -> metrics
        quant_type: 'none' or '4-bit'
        stage_name: Display name (e.g., "Translation")
        stage_prefix: Key prefix (e.g., "trans")
        extra_metrics: Optional list of (key, label, format_fn, lower_better) tuples
    """
    models = [m for m in MODEL_ORDER if m in data_by_model and quant_type in data_by_model[m]]
    if not models:
        return f'<p style="opacity: 0.6;">No data available for {stage_name}</p>'

    def get_val(model, key):
        return data_by_model.get(model, {}).get(quant_type, {}).get(key)

    def get_best_worst(key, lower_better=True):
        vals = [(i, get_val(models[i], key)) for i in range(len(models))]
        vals = [(i, v) for i, v in vals if v is not None and v != 0]
        if not vals:
            return -1, -1
        if lower_better:
            return min(vals, key=lambda x: x[1])[0], max(vals, key=lambda x: x[1])[0]
        return max(vals, key=lambda x: x[1])[0], min(vals, key=lambda x: x[1])[0]

    # Common cell border style for vertical bars
    cell_border = "border-left: 1px solid var(--color-border-light);"
    header_cell_border = "border-left: 1px solid var(--color-border-primary);"

    html = f'''
    <div class="card">
    <h4 class="card-title">{stage_name} Stage</h4>
    <table class="metrics-table">
    <thead><tr style="background: color-mix(in srgb, var(--color-primary-dark) 10%, var(--color-bg-primary));">
        <th style="padding: 8px; text-align: left;">Metric</th>'''

    for model in models:
        html += f'<th style="padding: 8px; text-align: center; {header_cell_border}">{MODEL_SHORT.get(model, model)}</th>'
    html += '</tr></thead><tbody>'

    # Core metrics for all stages
    metrics = [
        (f'{stage_prefix}_time', 'Processing Time', lambda v: f"{v:.1f}s", True),
        (f'{stage_prefix}_vram', 'Peak VRAM', lambda v: f"{v:.2f} GB", True),
        (f'{stage_prefix}_words', 'Word Count', lambda v: f"{v:,}", False),
        (f'{stage_prefix}_lex_div', 'Lexical Diversity', lambda v: f"{v:.1%}", False),
    ]

    # Add extra metrics if provided
    if extra_metrics:
        metrics.extend(extra_metrics)

    for key, label, fmt, lower_better in metrics:
        best_idx, worst_idx = get_best_worst(key, lower_better)
        html += f'<tr><td style="padding: 6px;">{label}</td>'
        for i, model in enumerate(models):
            val = get_val(model, key)
            formatted = fmt(val) if val else "—"
            style = f"padding: 6px; text-align: center; {cell_border}"
            if formatted == "—":
                style += " color: var(--color-muted); font-style: italic;"
            elif i == best_idx and len(models) > 1:
                style += " color: var(--color-status-ok); font-weight: bold;"
            elif i == worst_idx and len(models) > 1:
                style += " color: var(--color-status-fail);"
            html += f'<td style="{style}">{formatted}</td>'
        html += '</tr>'

    # Peak N-gram section header
    html += f'<tr style="background: var(--color-bg-secondary);"><td colspan="{len(models)+1}" style="padding: 8px 6px; font-weight: bold; font-style: italic; color: var(--color-text-secondary);">Peak N-gram Repetition</td></tr>'

    # Single peak n-gram row (the highest repetition)
    html += f'<tr><td style="padding: 6px;">Peak N-gram</td>'
    for model in models:
        top_ngrams = get_val(model, f'{stage_prefix}_top_ngrams') or []
        if top_ngrams:
            ng = top_ngrams[0]  # Just the top one
            rate = ng.get('rate', 0)
            n = ng.get('n', 0)
            count = ng.get('repeat_count', 0)
            pattern = ng.get('most_repeated', '')
            # Escape HTML entities in pattern
            pattern = pattern.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            # Color code based on count (>3 warn, >6 fail)
            if count > 6:
                color = 'var(--color-status-fail)'
            elif count > 3:
                color = 'var(--color-status-warn)'
            else:
                color = 'inherit'
            html += f'<td style="padding: 6px; font-size: 11px; text-align: left; color: {color}; {cell_border}">'
            html += f'<b>{count}x</b> ({rate:.1%}) n={n}<br><code style="font-size: 10px; background: var(--color-bg-secondary); padding: 1px 4px; border-radius: 3px; word-break: break-all;">{pattern}</code></td>'
        else:
            html += f'<td style="padding: 6px; text-align: center; color: var(--color-muted); font-size: 11px; {cell_border}">—</td>'
    html += '</tr>'

    # Consecutive repetition metrics section header
    html += f'<tr style="background: var(--color-bg-secondary);"><td colspan="{len(models)+1}" style="padding: 8px 6px; font-weight: bold; font-style: italic; color: var(--color-text-secondary);">Consecutive Repetitions</td></tr>'

    # Helper to format count + pattern
    def format_count_pattern(count, pattern):
        if not count or count < 2:
            return "—"
        escaped = pattern.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;') if pattern else ''
        return f'{count}x<br><code style="font-size: 10px; background: var(--color-bg-secondary); padding: 1px 4px; border-radius: 3px;">{escaped}</code>'

    # Char repetition row (>2 warn, >5 fail)
    html += f'<tr><td style="padding: 6px;">Max Char Repeat</td>'
    char_key = f'{stage_prefix}_char_count'
    best_idx, worst_idx = get_best_worst(char_key, lower_better=True)
    for i, model in enumerate(models):
        count = get_val(model, char_key) or 0
        char = get_val(model, f'{stage_prefix}_char') or ''
        formatted = format_count_pattern(count, char)
        style = f"padding: 6px; text-align: center; {cell_border}"
        if formatted == "—":
            style += " color: var(--color-muted); font-style: italic;"
        elif count and count > 10:
            style += " color: var(--color-status-fail);"
        elif count and count > 5:
            style += " color: var(--color-status-warn);"
        html += f'<td style="{style}">{formatted}</td>'
    html += '</tr>'

    # Word repetition row (>3 warn, >5 fail)
    html += f'<tr><td style="padding: 6px;">Max Word Repeat</td>'
    word_key = f'{stage_prefix}_word_count'
    best_idx, worst_idx = get_best_worst(word_key, lower_better=True)
    for i, model in enumerate(models):
        count = get_val(model, word_key) or 0
        pattern = get_val(model, f'{stage_prefix}_word_pattern') or ''
        formatted = format_count_pattern(count, pattern)
        style = f"padding: 6px; text-align: center; {cell_border}"
        if formatted == "—":
            style += " color: var(--color-muted); font-style: italic;"
        elif count and count > 5:
            style += " color: var(--color-status-fail);"
        elif count and count > 3:
            style += " color: var(--color-status-warn);"
        html += f'<td style="{style}">{formatted}</td>'
    html += '</tr>'

    # 2-gram repetition row (>3 warn, >5 fail)
    html += f'<tr><td style="padding: 6px;">Max 2-gram Repeat</td>'
    gram2_key = f'{stage_prefix}_2gram_count'
    for i, model in enumerate(models):
        count = get_val(model, gram2_key) or 0
        pattern = get_val(model, f'{stage_prefix}_2gram_pattern') or ''
        formatted = format_count_pattern(count, pattern)
        style = f"padding: 6px; text-align: center; {cell_border}"
        if formatted == "—":
            style += " color: var(--color-muted); font-style: italic;"
        elif count and count > 5:
            style += " color: var(--color-status-fail);"
        elif count and count > 3:
            style += " color: var(--color-status-warn);"
        html += f'<td style="{style}">{formatted}</td>'
    html += '</tr>'

    # 3-gram repetition row (>3 warn, >5 fail)
    html += f'<tr><td style="padding: 6px;">Max 3-gram Repeat</td>'
    gram3_key = f'{stage_prefix}_3gram_count'
    for i, model in enumerate(models):
        count = get_val(model, gram3_key) or 0
        pattern = get_val(model, f'{stage_prefix}_3gram_pattern') or ''
        formatted = format_count_pattern(count, pattern)
        style = f"padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light); {cell_border}"
        if formatted == "—":
            style += " color: var(--color-muted); font-style: italic;"
        elif count and count > 5:
            style += " color: var(--color-status-fail);"
        elif count and count > 3:
            style += " color: var(--color-status-warn);"
        html += f'<td style="{style}">{formatted}</td>'
    html += '</tr>'

    # 4-gram repetition row (>3 warn, >5 fail)
    html += f'<tr><td style="padding: 6px; border-bottom: 1px solid var(--color-border-light);">Max 4-gram Repeat</td>'
    gram4_key = f'{stage_prefix}_4gram_count'
    for i, model in enumerate(models):
        count = get_val(model, gram4_key) or 0
        pattern = get_val(model, f'{stage_prefix}_4gram_pattern') or ''
        formatted = format_count_pattern(count, pattern)
        style = f"padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light); {cell_border}"
        if formatted == "—":
            style += " color: var(--color-muted); font-style: italic;"
        elif count and count > 5:
            style += " color: var(--color-status-fail);"
        elif count and count > 3:
            style += " color: var(--color-status-warn);"
        html += f'<td style="{style}">{formatted}</td>'
    html += '</tr>'

    # Verdict row with special styling
    html += f'<tr style="background: var(--color-bg-secondary);"><td style="padding: 8px; font-weight: bold;">Verdict</td>'
    for model in models:
        verdict = get_val(model, f'{stage_prefix}_verdict')
        reason = get_val(model, f'{stage_prefix}_fail_reason')
        if verdict == 'FAIL':
            color = 'var(--color-status-fail)'
            display = f'FAIL<br><small style="font-weight:normal;">({reason})</small>' if reason else 'FAIL'
        elif verdict == 'WARNING':
            color = 'var(--color-status-warn)'
            display = f'WARN<br><small style="font-weight:normal;">({reason})</small>' if reason else 'WARN'
        elif verdict == 'OK':
            color = 'var(--color-status-ok)'
            display = 'OK'
        else:
            color = 'var(--color-muted)'
            display = verdict if verdict else '—'
        html += f'<td style="padding: 8px; text-align: center; color: {color}; font-weight: bold; {cell_border}">{display}</td>'
    html += '</tr>'

    html += '</tbody></table></div>'
    return html


def create_totals_table(data_by_model: dict, quant_type: str) -> str:
    """Create a summary totals table."""
    models = [m for m in MODEL_ORDER if m in data_by_model and quant_type in data_by_model[m]]
    if not models:
        return ''

    def get_val(model, key):
        return data_by_model.get(model, {}).get(quant_type, {}).get(key)

    def get_best_worst(key, lower_better=True):
        vals = [(i, get_val(models[i], key)) for i in range(len(models))]
        vals = [(i, v) for i, v in vals if v is not None and v != 0]
        if not vals:
            return -1, -1
        if lower_better:
            return min(vals, key=lambda x: x[1])[0], max(vals, key=lambda x: x[1])[0]
        return max(vals, key=lambda x: x[1])[0], min(vals, key=lambda x: x[1])[0]

    # Common cell border style for vertical bars
    cell_border = "border-left: 1px solid color-mix(in srgb, var(--color-success-dark) 30%, transparent);"

    html = '''
    <div style="margin: 20px 0; padding: 15px; border: 2px solid color-mix(in srgb, var(--color-success-dark) 50%, transparent); border-radius: 8px; background: color-mix(in srgb, var(--color-success-dark) 5%, var(--color-bg-primary));">
    <h4 style="margin: 0 0 15px 0; color: var(--color-success-dark);">Pipeline Totals</h4>
    <table style="border-collapse: collapse; font-size: 12px;">
    <thead><tr style="background: color-mix(in srgb, var(--color-success-dark) 10%, var(--color-bg-primary));">
        <th style="padding: 8px; text-align: left; border-bottom: 2px solid var(--color-success-dark);">Metric</th>'''

    for model in models:
        html += f'<th style="padding: 8px; text-align: center; border-bottom: 2px solid var(--color-success-dark); {cell_border}">{MODEL_SHORT.get(model, model)}</th>'
    html += '</tr></thead><tbody>'

    metrics = [
        ('total_time', 'Total Time', lambda v: f"{v:.1f}s", True),
        ('vram_peak_gb', 'Peak VRAM', lambda v: f"{v:.2f} GB", True),
        ('total_words', 'Total Words', lambda v: f"{v:,}", False),
    ]

    for key, label, fmt, lower_better in metrics:
        best_idx, worst_idx = get_best_worst(key, lower_better)
        html += f'<tr><td style="padding: 6px;">{label}</td>'
        for i, model in enumerate(models):
            val = get_val(model, key)
            formatted = fmt(val) if val else "—"
            style = f"padding: 6px; text-align: center; {cell_border}"
            if i == best_idx and len(models) > 1:
                style += " color: var(--color-status-ok); font-weight: bold;"
            elif i == worst_idx and len(models) > 1:
                style += " color: var(--color-status-fail);"
            html += f'<td style="{style}">{formatted}</td>'
        html += '</tr>'

    # Overall verdict
    html += '<tr style="background: color-mix(in srgb, var(--color-success-dark) 10%, var(--color-bg-primary));"><td style="padding: 8px; font-weight: bold;">Overall Verdict</td>'
    for model in models:
        verdict = get_val(model, 'verdict')
        reason = get_val(model, 'overall_fail_reason')
        if verdict == 'FAIL':
            color = 'var(--color-status-fail)'
            display = f'FAIL<br><small style="font-weight:normal;">({reason})</small>' if reason else 'FAIL'
        elif verdict == 'WARNING':
            color = 'var(--color-status-warn)'
            display = f'WARN<br><small style="font-weight:normal;">({reason})</small>' if reason else 'WARN'
        elif verdict == 'OK':
            color = 'var(--color-status-ok)'
            display = 'OK'
        else:
            color = 'var(--color-muted)'
            display = verdict if verdict else '—'
        html += f'<td style="padding: 8px; text-align: center; color: {color}; font-weight: bold; {cell_border}">{display}</td>'
    html += '</tr>'

    html += '</tbody></table></div>'
    return html


def create_comparison_summary_table(data_by_model: dict) -> str:
    """Create a comparison table between no quant and 4-bit."""
    models = [m for m in MODEL_ORDER if m in data_by_model]
    if not models:
        return '<p style="opacity: 0.6;">No comparison data available</p>'

    def get_val(model, quant, key):
        return data_by_model.get(model, {}).get(quant, {}).get(key)

    # Common cell border style for vertical bars
    cell_border = "border-left: 1px solid rgba(21, 101, 192, 0.3);"

    html = '''
    <div style="margin: 30px 0; padding: 20px; border: 2px solid var(--color-primary-dark); border-radius: 8px;">
    <h3 style="margin: 0 0 20px 0; color: var(--color-primary-dark);">Quantization Impact Summary</h3>
    <table style="border-collapse: collapse; font-size: 12px;">
    <thead><tr style="background: color-mix(in srgb, var(--color-primary-dark) 10%, var(--color-bg-primary));">
        <th style="padding: 10px; text-align: left; border-bottom: 2px solid var(--color-primary-dark);">Metric</th>'''

    for model in models:
        html += f'<th style="padding: 10px; text-align: center; border-bottom: 2px solid var(--color-primary-dark); {cell_border}">{MODEL_SHORT.get(model, model)}</th>'
    html += '</tr></thead><tbody>'

    # Time comparison
    html += '<tr><td style="padding: 6px;">Total Time (None)</td>'
    for model in models:
        val = get_val(model, 'none', 'total_time')
        html += f'<td style="padding: 6px; text-align: center; {cell_border}">{f"{val:.1f}s" if val else "—"}</td>'
    html += '</tr>'

    html += '<tr><td style="padding: 6px;">Total Time (4-bit)</td>'
    for model in models:
        val = get_val(model, '4-bit', 'total_time')
        html += f'<td style="padding: 6px; text-align: center; {cell_border}">{f"{val:.1f}s" if val else "—"}</td>'
    html += '</tr>'

    html += '<tr style="background: var(--color-bg-highlight);"><td style="padding: 8px; font-weight: bold;">Time Change</td>'
    for model in models:
        none_val = get_val(model, 'none', 'total_time')
        quant_val = get_val(model, '4-bit', 'total_time')
        if none_val and quant_val and none_val > 0:
            change = ((quant_val / none_val) - 1) * 100
            sign = '+' if change > 0 else ''
            color = 'var(--color-status-fail)' if change > 20 else ('var(--color-status-ok)' if change < -10 else 'var(--color-info-dark)')
            html += f'<td style="padding: 8px; text-align: center; color: {color}; font-weight: bold; {cell_border}">{sign}{change:.0f}%</td>'
        else:
            html += f'<td style="padding: 8px; text-align: center; color: var(--color-muted); {cell_border}">—</td>'
    html += '</tr>'

    # VRAM comparison
    html += '<tr><td colspan="' + str(len(models)+1) + '" style="padding: 10px 0 5px 0;"></td></tr>'

    html += '<tr><td style="padding: 6px;">Peak VRAM (None)</td>'
    for model in models:
        val = get_val(model, 'none', 'vram_peak_gb')
        html += f'<td style="padding: 6px; text-align: center; {cell_border}">{f"{val:.2f} GB" if val else "—"}</td>'
    html += '</tr>'

    html += '<tr><td style="padding: 6px;">Peak VRAM (4-bit)</td>'
    for model in models:
        val = get_val(model, '4-bit', 'vram_peak_gb')
        html += f'<td style="padding: 6px; text-align: center; {cell_border}">{f"{val:.2f} GB" if val else "—"}</td>'
    html += '</tr>'

    html += '<tr style="background: var(--color-bg-highlight);"><td style="padding: 8px; font-weight: bold;">VRAM Reduction</td>'
    for model in models:
        none_val = get_val(model, 'none', 'vram_peak_gb')
        quant_val = get_val(model, '4-bit', 'vram_peak_gb')
        if none_val and quant_val and none_val > 0:
            reduction = (1 - quant_val / none_val) * 100
            html += f'<td style="padding: 8px; text-align: center; color: var(--color-info-dark); font-weight: bold; {cell_border}">{reduction:.0f}%</td>'
        else:
            html += f'<td style="padding: 8px; text-align: center; color: var(--color-muted); {cell_border}">—</td>'
    html += '</tr>'

    # Verdict comparison
    html += '<tr><td colspan="' + str(len(models)+1) + '" style="padding: 10px 0 5px 0;"></td></tr>'

    html += '<tr><td style="padding: 6px;">Verdict (None)</td>'
    for model in models:
        val = get_val(model, 'none', 'verdict')
        color = 'var(--color-status-fail)' if val == 'FAIL' else ('var(--color-status-warn)' if val == 'WARNING' else ('var(--color-status-ok)' if val == 'OK' else 'var(--color-muted)'))
        html += f'<td style="padding: 6px; text-align: center; color: {color}; font-weight: bold; {cell_border}">{val if val else "—"}</td>'
    html += '</tr>'

    html += '<tr><td style="padding: 6px;">Verdict (4-bit)</td>'
    for model in models:
        val = get_val(model, '4-bit', 'verdict')
        color = 'var(--color-status-fail)' if val == 'FAIL' else ('var(--color-status-warn)' if val == 'WARNING' else ('var(--color-status-ok)' if val == 'OK' else 'var(--color-muted)'))
        html += f'<td style="padding: 6px; text-align: center; color: {color}; font-weight: bold; {cell_border}">{val if val else "—"}</td>'
    html += '</tr>'

    html += '</tbody></table>'
    html += '''<p style="font-size: 11px; opacity: 0.7; margin-top: 15px;">
        <span style="color: var(--color-status-ok); font-weight: bold;">Green</span> = Best/OK |
        <span style="color: var(--color-status-fail);">Red</span> = Worst/FAIL |
        <span style="color: var(--color-status-warn); font-weight: bold;">Orange</span> = Warning |
        <span style="color: var(--color-info-dark); font-weight: bold;">Blue</span> = Savings/Change
    </p></div>'''
    return html


def export_quant_to_csv(quant_data: list, output_path: Path) -> str:
    """Export quantization comparison data to CSV with all metrics."""
    quant_data = sorted(quant_data, key=lambda x: (
        MODEL_ORDER.index(x['model']) if x['model'] in MODEL_ORDER else 99,
        x['quantization'] != 'none'
    ))

    columns = [
        ('model', 'Model'), ('quantization', 'Quantization'),
        ('trans_time', 'Trans Time (s)'), ('trans_vram', 'Trans VRAM (GB)'),
        ('trans_words', 'Trans Words'), ('trans_lex_div', 'Trans Lex Div'),
        ('trans_peak_rate', 'Trans Peak Rep'), ('trans_peak_n', 'Trans Peak N'),
        ('trans_max_char_repeat', 'Trans Max Char Rep'), ('trans_max_word_repeat', 'Trans Max Word Rep'),
        ('trans_verdict', 'Trans Verdict'),
        ('sum_time', 'Sum Time (s)'), ('sum_vram', 'Sum VRAM (GB)'),
        ('sum_bullets', 'Sum Bullets'), ('sum_words', 'Sum Words'), ('sum_lex_div', 'Sum Lex Div'),
        ('sum_peak_rate', 'Sum Peak Rep'), ('sum_peak_n', 'Sum Peak N'),
        ('sum_max_char_repeat', 'Sum Max Char Rep'), ('sum_max_word_repeat', 'Sum Max Word Rep'),
        ('sum_verdict', 'Sum Verdict'),
        ('pod_time', 'Pod Time (s)'), ('pod_vram', 'Pod VRAM (GB)'),
        ('pod_lines', 'Pod Lines'), ('pod_words', 'Pod Words'), ('pod_lex_div', 'Pod Lex Div'),
        ('pod_peak_rate', 'Pod Peak Rep'), ('pod_peak_n', 'Pod Peak N'),
        ('pod_max_char_repeat', 'Pod Max Char Rep'), ('pod_max_word_repeat', 'Pod Max Word Rep'),
        ('pod_verdict', 'Pod Verdict'),
        ('total_time', 'Total Time (s)'), ('vram_peak_gb', 'Peak VRAM (GB)'), ('total_words', 'Total Words'),
        ('verdict', 'Overall Verdict'),
    ]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([col[1] for col in columns])

    for d in quant_data:
        row = []
        for key, _ in columns:
            value = d.get(key, '')
            if key == 'model':
                value = MODEL_SHORT.get(value, value)
            elif key in ['trans_time', 'sum_time', 'pod_time', 'total_time', 'trans_vram', 'sum_vram', 'pod_vram', 'vram_peak_gb']:
                value = f"{value:.2f}" if value else ''
            elif key in ['trans_peak_rate', 'sum_peak_rate', 'pod_peak_rate', 'trans_lex_div', 'sum_lex_div', 'pod_lex_div']:
                value = f"{value:.1%}" if value else ''
            elif key == 'quantization':
                value = 'None' if value == 'none' else '4-bit'
            row.append(value)
        writer.writerow(row)

    csv_content = output.getvalue()
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        f.write(csv_content)
    return csv_content


def export_quant_to_json(quant_data: list, output_path: Path) -> dict:
    """
    Export quantization comparison data to JSON with full metrics structure.

    Returns a structured dict with:
    - metadata: export timestamp, thresholds, etc.
    - models: list of model results organized for thesis tables
    """
    # Sort by model order and quantization
    quant_data = sorted(quant_data, key=lambda x: (
        MODEL_ORDER.index(x['model']) if x['model'] in MODEL_ORDER else 99,
        x['quantization'] != 'none'
    ))

    # Build structured data for each model
    models_data = []
    for d in quant_data:
        model_entry = {
            'exp_id': d.get('exp_id', ''),
            'model': d.get('model', ''),
            'model_short': MODEL_SHORT.get(d.get('model', ''), d.get('model', '')),
            'quantization': 'None' if d.get('quantization') == 'none' else '4-bit',

            # Overall verdict
            'verdict': d.get('verdict', 'Unknown'),
            'fail_reason': d.get('overall_fail_reason', ''),

            # Totals
            'totals': {
                'time_seconds': round(d.get('total_time', 0), 2),
                'vram_peak_gb': round(d.get('vram_peak_gb', 0), 2),
                'total_words': d.get('total_words', 0),
            },

            # Per-stage data
            'translation': {
                'time_seconds': round(d.get('trans_time', 0), 2),
                'vram_gb': round(d.get('trans_vram', 0), 2),
                'words': d.get('trans_words', 0),
                'lexical_diversity': round(d.get('trans_lex_div', 0), 4),
                'verdict': d.get('trans_verdict', 'Unknown'),
                'fail_reason': d.get('trans_fail_reason', ''),
                'degeneration': {
                    'char_repeat': {'count': d.get('trans_char_count', 0), 'pattern': d.get('trans_char', '')},
                    'word_repeat': {'count': d.get('trans_word_count', 0), 'pattern': d.get('trans_word_pattern', '')},
                    '2gram_repeat': {'count': d.get('trans_2gram_count', 0), 'pattern': d.get('trans_2gram_pattern', '')},
                    '3gram_repeat': {'count': d.get('trans_3gram_count', 0), 'pattern': d.get('trans_3gram_pattern', '')},
                    '4gram_repeat': {'count': d.get('trans_4gram_count', 0), 'pattern': d.get('trans_4gram_pattern', '')},
                    'ngram_global': d.get('trans_top_ngrams', []),
                },
            },
            'summary': {
                'time_seconds': round(d.get('sum_time', 0), 2),
                'vram_gb': round(d.get('sum_vram', 0), 2),
                'bullets': d.get('sum_bullets', 0),
                'words': d.get('sum_words', 0),
                'lexical_diversity': round(d.get('sum_lex_div', 0), 4),
                'verdict': d.get('sum_verdict', 'Unknown'),
                'fail_reason': d.get('sum_fail_reason', ''),
                'degeneration': {
                    'char_repeat': {'count': d.get('sum_char_count', 0), 'pattern': d.get('sum_char', '')},
                    'word_repeat': {'count': d.get('sum_word_count', 0), 'pattern': d.get('sum_word_pattern', '')},
                    '2gram_repeat': {'count': d.get('sum_2gram_count', 0), 'pattern': d.get('sum_2gram_pattern', '')},
                    '3gram_repeat': {'count': d.get('sum_3gram_count', 0), 'pattern': d.get('sum_3gram_pattern', '')},
                    '4gram_repeat': {'count': d.get('sum_4gram_count', 0), 'pattern': d.get('sum_4gram_pattern', '')},
                    'ngram_global': d.get('sum_top_ngrams', []),
                },
            },
            'podcast': {
                'time_seconds': round(d.get('pod_time', 0), 2),
                'vram_gb': round(d.get('pod_vram', 0), 2),
                'lines': d.get('pod_lines', 0),
                'words': d.get('pod_words', 0),
                'lexical_diversity': round(d.get('pod_lex_div', 0), 4),
                'verdict': d.get('pod_verdict', 'Unknown'),
                'fail_reason': d.get('pod_fail_reason', ''),
                'degeneration': {
                    'char_repeat': {'count': d.get('pod_char_count', 0), 'pattern': d.get('pod_char', '')},
                    'word_repeat': {'count': d.get('pod_word_count', 0), 'pattern': d.get('pod_word_pattern', '')},
                    '2gram_repeat': {'count': d.get('pod_2gram_count', 0), 'pattern': d.get('pod_2gram_pattern', '')},
                    '3gram_repeat': {'count': d.get('pod_3gram_count', 0), 'pattern': d.get('pod_3gram_pattern', '')},
                    '4gram_repeat': {'count': d.get('pod_4gram_count', 0), 'pattern': d.get('pod_4gram_pattern', '')},
                    'ngram_global': d.get('pod_top_ngrams', []),
                },
            },
        }
        models_data.append(model_entry)

    # Build final JSON structure
    result = {
        'metadata': {
            'export_type': 'quantization_comparison',
            'model_count': len(models_data),
            'thresholds': {
                'ngram_count_warn': 3,
                'ngram_count_fail': 6,
                'char_repeat_warn': 5,
                'char_repeat_fail': 10,
                'consecutive_repeat_warn': 3,
                'consecutive_repeat_fail': 5,
            },
        },
        'models': models_data,
    }

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def generate_quant_latex_table(json_data: dict, table_type: str = 'vram') -> str:
    """
    Generate LaTeX table code from quantization JSON data.

    Args:
        json_data: The JSON data from export_quant_to_json
        table_type: Type of table to generate:
            - 'vram': VRAM and time comparison table
            - 'quality': Quality metrics table per stage

    Returns:
        LaTeX code string for the table
    """
    models = json_data.get('models', [])

    if table_type == 'vram':
        # Group by model
        by_model = {}
        for m in models:
            name = m['model_short']
            if name not in by_model:
                by_model[name] = {}
            by_model[name][m['quantization']] = m

        lines = []
        lines.append(r"\begin{tabular}{l|c|r|r|r}")
        lines.append(r"    \hline")
        lines.append(r"    \textbf{Model} & \textbf{Quant} & \textbf{VRAM (GB)} & \textbf{Time (s)} & \textbf{VRAM Savings} \\")
        lines.append(r"    \hline")

        for model_name in MODEL_SHORT.values():
            if model_name not in by_model:
                continue
            data = by_model[model_name]

            none_data = data.get('None', {})
            quant_data = data.get('4-bit', {})

            none_vram = none_data.get('totals', {}).get('vram_peak_gb', 0)
            none_time = none_data.get('totals', {}).get('time_seconds', 0)
            quant_vram = quant_data.get('totals', {}).get('vram_peak_gb', 0)
            quant_time = quant_data.get('totals', {}).get('time_seconds', 0)

            # Calculate savings
            if none_vram > 0 and quant_vram > 0:
                savings = (1 - quant_vram / none_vram) * 100
                savings_str = f"{savings:.0f}\\%"
            else:
                savings_str = "---"

            lines.append(f"    {model_name} & None & {none_vram:.2f} & {none_time:.1f} & --- \\\\")
            lines.append(f"    {model_name} & 4-bit & {quant_vram:.2f} & {quant_time:.1f} & {savings_str} \\\\")
            lines.append(r"    \hline")

        lines.append(r"\end{tabular}")
        return "\n".join(lines)

    elif table_type == 'quality':
        # Quality comparison table showing verdicts
        lines = []
        lines.append(r"\begin{tabular}{l|c|c|c|c|c}")
        lines.append(r"    \hline")
        lines.append(r"    \textbf{Model} & \textbf{Quant} & \textbf{Trans} & \textbf{Sum} & \textbf{Pod} & \textbf{Overall} \\")
        lines.append(r"    \hline")

        for m in models:
            # Color-code verdicts
            def verdict_cell(v):
                if v == 'OK':
                    return r"\cellcolor{bestcell}OK"
                elif v == 'WARNING':
                    return r"\cellcolor{warncell}WARN"
                elif v == 'FAIL':
                    return r"\cellcolor{failcell}FAIL"
                return v

            trans_v = verdict_cell(m['translation']['verdict'])
            sum_v = verdict_cell(m['summary']['verdict'])
            pod_v = verdict_cell(m['podcast']['verdict'])
            overall_v = verdict_cell(m['verdict'])

            lines.append(f"    {m['model_short']} & {m['quantization']} & {trans_v} & {sum_v} & {pod_v} & {overall_v} \\\\")

        lines.append(r"    \hline")
        lines.append(r"\end{tabular}")
        return "\n".join(lines)

    return ""


def load_output_text(exp_id: str, outputs_dir: Path, stage: str) -> str:
    """Load output text from the outputs directory."""
    filenames = {
        'translation': 'translation_output.txt',
        'summary': 'summary_output.txt',
        'podcast': 'podcast_script.txt',
    }
    output_file = outputs_dir / exp_id / filenames.get(stage, '')
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


# =============================================================================
# EXPERIMENT DATA COLLECTION FUNCTIONS
# =============================================================================

def collect_llm_experiment_data(
    results: Dict[str, Any],
    outputs_dir: Path,
    experiment_type: str,
    assess_quality_fn
) -> list:
    """
    Collect and process experiment data for LLM-based experiments.

    Handles translation, summary, and podcast stages with quality assessment.
    Used by quantization, repetition_penalty, summary_mode, chunk_size, and temperature analyses.

    Args:
        results: Evaluation results dictionary
        outputs_dir: Path to outputs directory
        experiment_type: Type of experiment to filter (e.g., 'quantization')
        assess_quality_fn: Function to assess text quality

    Returns:
        List of processed experiment data dictionaries
    """
    experiments = []

    for exp_id, exp in results.get('experiments', {}).items():
        if exp.get('experiment_type') != experiment_type or 'error' in exp:
            continue

        config = exp.get('config', {})
        stages_data = exp.get('stages_data', {})

        # Get metrics from each stage
        trans_m = stages_data.get('translation', {}).get('metrics', {})
        trans_o = stages_data.get('translation', {}).get('output', {})
        sum_m = stages_data.get('summary', {}).get('metrics', {})
        sum_o = stages_data.get('summary', {}).get('output', {})
        pod_m = stages_data.get('podcast', {}).get('metrics', {})
        pod_o = stages_data.get('podcast', {}).get('output', {})

        # Load text outputs from files (with fallback to stages_data)
        trans_text = load_output_text(exp_id, outputs_dir, 'translation') or trans_o.get('translated_text', '')
        sum_text = load_output_text(exp_id, outputs_dir, 'summary') or sum_o.get('summary', '')
        pod_text = load_output_text(exp_id, outputs_dir, 'podcast') or pod_o.get('script', '')

        # Assess quality for each stage
        trans_quality = assess_quality_fn(trans_text)
        sum_quality = assess_quality_fn(sum_text)
        pod_quality = assess_quality_fn(pod_text)

        # Calculate totals
        total_time = trans_m.get('time_seconds', 0) + sum_m.get('time_seconds', 0) + pod_m.get('time_seconds', 0)
        max_vram = max(trans_m.get('vram_peak_gb', 0), sum_m.get('vram_peak_gb', 0), pod_m.get('vram_peak_gb', 0))
        total_words = trans_quality['word_count'] + sum_quality['word_count'] + pod_quality['word_count']

        # Determine overall verdict
        all_verdicts = [trans_quality['verdict'], sum_quality['verdict'], pod_quality['verdict']]
        if 'FAIL' in all_verdicts:
            overall_verdict = 'FAIL'
            fail_stages = [s for s, v in [('Trans', trans_quality['verdict']), ('Sum', sum_quality['verdict']), ('Pod', pod_quality['verdict'])] if v == 'FAIL']
            overall_fail_reason = f"{', '.join(fail_stages)} failed"
        elif 'WARNING' in all_verdicts:
            overall_verdict = 'WARNING'
            warn_stages = [s for s, v in [('Trans', trans_quality['verdict']), ('Sum', sum_quality['verdict']), ('Pod', pod_quality['verdict'])] if v == 'WARNING']
            overall_fail_reason = f"{', '.join(warn_stages)} warned"
        elif 'NO DATA' in all_verdicts:
            overall_verdict = 'NO DATA'
            overall_fail_reason = 'Missing data'
        else:
            overall_verdict = 'OK'
            overall_fail_reason = ''

        exp_data = {
            'exp_id': exp_id,
            'config': config,
            'model': config.get('llm_model_id', '').split('/')[-1],
            'quantization': config.get('quantization') or 'none',
            'repetition_penalty': config.get('repetition_penalty'),
            'summary_mode': config.get('summary_mode', 'greedy'),
            'chunk_size': config.get('chunk_size_chars', 2000),
            'temperature': config.get('podcast_creativity_temp', 0.3),
            # Per-stage timing
            'trans_time': trans_m.get('time_seconds', 0),
            'sum_time': sum_m.get('time_seconds', 0),
            'pod_time': pod_m.get('time_seconds', 0),
            # Per-stage VRAM
            'trans_vram': trans_m.get('vram_peak_gb', 0),
            'sum_vram': sum_m.get('vram_peak_gb', 0),
            'pod_vram': pod_m.get('vram_peak_gb', 0),
            # Text outputs
            'trans_text': trans_text,
            'sum_text': sum_text,
            'pod_text': pod_text,
            # Translation quality
            'trans_words': trans_quality['word_count'],
            'trans_lex_div': trans_quality['lexical_diversity'],
            'trans_verdict': trans_quality['verdict'],
            'trans_fail_reason': trans_quality['fail_reason'],
            'trans_top_ngrams': trans_quality.get('top_ngrams', []),
            'trans_char_count': trans_quality['max_char_repeat'],
            'trans_char': trans_quality.get('max_char', ''),
            'trans_word_count': trans_quality.get('consec_word_count', 0),
            'trans_word_pattern': trans_quality.get('consec_word_pattern', ''),
            'trans_2gram_count': trans_quality.get('consec_2gram_count', 0),
            'trans_2gram_pattern': trans_quality.get('consec_2gram_pattern', ''),
            'trans_3gram_count': trans_quality.get('consec_3gram_count', 0),
            'trans_3gram_pattern': trans_quality.get('consec_3gram_pattern', ''),
            'trans_4gram_count': trans_quality.get('consec_4gram_count', 0),
            'trans_4gram_pattern': trans_quality.get('consec_4gram_pattern', ''),
            # Summary quality
            'sum_bullets': sum_o.get('bullet_count', 0),
            'sum_words': sum_quality['word_count'],
            'sum_lex_div': sum_quality['lexical_diversity'],
            'sum_verdict': sum_quality['verdict'],
            'sum_fail_reason': sum_quality['fail_reason'],
            'sum_top_ngrams': sum_quality.get('top_ngrams', []),
            'sum_char_count': sum_quality['max_char_repeat'],
            'sum_char': sum_quality.get('max_char', ''),
            'sum_word_count': sum_quality.get('consec_word_count', 0),
            'sum_word_pattern': sum_quality.get('consec_word_pattern', ''),
            'sum_2gram_count': sum_quality.get('consec_2gram_count', 0),
            'sum_2gram_pattern': sum_quality.get('consec_2gram_pattern', ''),
            'sum_3gram_count': sum_quality.get('consec_3gram_count', 0),
            'sum_3gram_pattern': sum_quality.get('consec_3gram_pattern', ''),
            'sum_4gram_count': sum_quality.get('consec_4gram_count', 0),
            'sum_4gram_pattern': sum_quality.get('consec_4gram_pattern', ''),
            # Podcast quality
            'pod_lines': pod_o.get('total_lines', 0),
            'pod_words': pod_quality['word_count'],
            'pod_lex_div': pod_quality['lexical_diversity'],
            'pod_verdict': pod_quality['verdict'],
            'pod_fail_reason': pod_quality['fail_reason'],
            'pod_top_ngrams': pod_quality.get('top_ngrams', []),
            'pod_char_count': pod_quality['max_char_repeat'],
            'pod_char': pod_quality.get('max_char', ''),
            'pod_word_count': pod_quality.get('consec_word_count', 0),
            'pod_word_pattern': pod_quality.get('consec_word_pattern', ''),
            'pod_2gram_count': pod_quality.get('consec_2gram_count', 0),
            'pod_2gram_pattern': pod_quality.get('consec_2gram_pattern', ''),
            'pod_3gram_count': pod_quality.get('consec_3gram_count', 0),
            'pod_3gram_pattern': pod_quality.get('consec_3gram_pattern', ''),
            'pod_4gram_count': pod_quality.get('consec_4gram_count', 0),
            'pod_4gram_pattern': pod_quality.get('consec_4gram_pattern', ''),
            # Totals
            'total_time': total_time,
            'vram_peak_gb': max_vram,
            'total_words': total_words,
            'verdict': overall_verdict,
            'overall_fail_reason': overall_fail_reason,
        }
        experiments.append(exp_data)

    return experiments


# =============================================================================
# ANALYSIS DISPLAY FUNCTIONS
# =============================================================================

def display_quantization_analysis(evaluation) -> None:
    """
    Display quantization impact analysis.

    Compares 4-bit quantization vs full precision across all LLMs.
    Shows text outputs, metrics tables, and comparison summary.

    Args:
        evaluation: EvaluationNotebook instance
    """
    from IPython.display import display, HTML
    from .evaluation_analysis import assess_quality

    results = evaluation.load_results()
    outputs_dir = evaluation.outputs_dir

    if not results.get('experiments'):
        print("No evaluation results found. Run the experiments first.")
        return

    # Collect quantization experiments
    quant_experiments = collect_llm_experiment_data(
        results, outputs_dir, 'quantization', assess_quality
    )

    if not quant_experiments:
        print("No quantization comparison experiments found.")
        return

    print(f"Found {len(quant_experiments)} quantization experiments")
    for exp in sorted(quant_experiments, key=lambda x: (x['model'], x['quantization'])):
        quant_label = 'none' if exp['quantization'] == 'none' else '4-bit'
        print(f"  - {exp['model']} ({quant_label}): {exp['verdict']}")
    print()

    # Organize data by model and quantization
    data_by_model = {}
    for exp in quant_experiments:
        model = exp['model']
        quant = exp['quantization']
        if model not in data_by_model:
            data_by_model[model] = {}
        data_by_model[model][quant] = exp

    # Export to JSON
    json_path = evaluation.drive_base / 'quantization_comparison.json'
    json_data = export_quant_to_json(quant_experiments, json_path)
    print(f"JSON saved: {json_path}")

    # Create download button for JSON
    json_content = json.dumps(json_data, indent=2, ensure_ascii=False)
    json_b64 = base64.b64encode(json_content.encode('utf-8')).decode('utf-8')
    download_html = f'''
    <div style="margin: 15px 0;">
        <a href="data:application/json;base64,{json_b64}" download="quantization_comparison.json"
           class="download-btn" style="
                  border-radius: 20px; cursor: pointer; font-size: 14px; font-weight: 500;
                  text-decoration: none; display: inline-block;
                  box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
            📥 Download JSON
        </a>
        <span style="margin-left: 10px; font-size: 12px; opacity: 0.7;">quantization_comparison.json</span>
    </div>
    '''
    inject_evaluation_css()
    display(HTML(download_html))

    # Section 1: No Quantization (Full Precision)
    display(HTML(load_template('section_header.html', title="No Quantization (Full Precision)", color_class="primary")))
    display(HTML(create_text_boxes_row("Translation Output", data_by_model, 'none', 'trans')))
    display(HTML(create_text_boxes_row("Summary Output", data_by_model, 'none', 'sum')))
    display(HTML(create_text_boxes_row("Podcast Script Output", data_by_model, 'none', 'pod')))
    display(HTML(create_stage_metrics_table(data_by_model, 'none', 'Translation', 'trans')))
    display(HTML(create_stage_metrics_table(data_by_model, 'none', 'Summary', 'sum',
                 extra_metrics=[('sum_bullets', 'Bullet Points', lambda v: f"{v}", False)])))
    display(HTML(create_stage_metrics_table(data_by_model, 'none', 'Podcast', 'pod',
                 extra_metrics=[('pod_lines', 'Script Lines', lambda v: f"{v:,}", False)])))
    display(HTML(create_totals_table(data_by_model, 'none')))

    # Section 2: 4-bit Quantization
    display(HTML(load_template('section_header.html', title="4-bit Quantization (NF4)", color_class="success")))
    display(HTML(create_text_boxes_row("Translation Output", data_by_model, '4-bit', 'trans')))
    display(HTML(create_text_boxes_row("Summary Output", data_by_model, '4-bit', 'sum')))
    display(HTML(create_text_boxes_row("Podcast Script Output", data_by_model, '4-bit', 'pod')))
    display(HTML(create_stage_metrics_table(data_by_model, '4-bit', 'Translation', 'trans')))
    display(HTML(create_stage_metrics_table(data_by_model, '4-bit', 'Summary', 'sum',
                 extra_metrics=[('sum_bullets', 'Bullet Points', lambda v: f"{v}", False)])))
    display(HTML(create_stage_metrics_table(data_by_model, '4-bit', 'Podcast', 'pod',
                 extra_metrics=[('pod_lines', 'Script Lines', lambda v: f"{v:,}", False)])))
    display(HTML(create_totals_table(data_by_model, '4-bit')))

    # Section 3: Comparison Summary
    display(HTML(create_comparison_summary_table(data_by_model)))


def _create_generic_text_boxes_row(
    title: str,
    data: dict,
    columns: list,
    stage_key: str
) -> str:
    """
    Create a row of text boxes for any comparison analysis.

    Args:
        title: Section title
        data: Dict mapping column keys to experiment data
        columns: List of (key, label) tuples defining column order
        stage_key: Stage prefix ('trans', 'sum', 'pod')

    Returns:
        HTML string
    """
    html = f'<h4 style="margin-top: 20px; margin-bottom: 10px;">{title}</h4>'
    html += f'<div style="display: grid; grid-template-columns: repeat({len(columns)}, 1fr); gap: 15px;">'

    text_key = f'{stage_key}_text'
    verdict_key = f'{stage_key}_verdict'
    fail_reason_key = f'{stage_key}_fail_reason'

    for key, label in columns:
        if key in data:
            exp = data[key]
            text = exp.get(text_key, '')
            verdict = exp.get(verdict_key, 'Unknown')
            fail_reason = exp.get(fail_reason_key, '')
            word_count = len(text.split()) if text else 0

            box_html = create_scrollable_box(
                f"{label} ({word_count:,} words)",
                text if text else "(No output)",
                "300px",
                verdict,
                fail_reason
            )
            html += f'<div>{box_html}</div>'
        else:
            html += f'<div style="padding: 10px; opacity: 0.5;">No data for {label}</div>'

    html += '</div>'
    return html


def _create_generic_metrics_table(
    data: dict,
    columns: list,
    stage_name: str,
    stage_prefix: str,
    extra_metrics: list = None
) -> str:
    """
    Create a metrics table for any comparison analysis.

    Args:
        data: Dict mapping column keys to experiment data
        columns: List of (key, label) tuples defining column order
        stage_name: Display name for the stage
        stage_prefix: Stage prefix ('trans', 'sum', 'pod')
        extra_metrics: Optional list of (key, label, format_fn, lower_is_better) tuples

    Returns:
        HTML string
    """
    html = f'<h4 style="margin-top: 25px; margin-bottom: 10px;">{stage_name} Stage Metrics</h4>'

    html += '''<table style="border-collapse: collapse; font-size: 13px; margin-bottom: 15px;">
    <thead>
        <tr style="background: rgba(128,128,128,0.15);">
            <th style="padding: 8px; text-align: left; border-bottom: 2px solid rgba(128,128,128,0.3);"></th>'''

    for key, label in columns:
        html += f'<th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(128,128,128,0.3);">{label}</th>'

    html += '</tr></thead><tbody>'

    def get_val(key, field):
        return data.get(key, {}).get(field)

    # Performance metrics
    html += f'<tr style="background: var(--color-bg-secondary);"><td colspan="{len(columns)+1}" style="padding: 8px 6px; font-weight: bold; font-style: italic; opacity: 0.7;">Performance</td></tr>'

    # Time
    html += f'<tr><td style="padding: 6px; border-bottom: 1px solid var(--color-border-light);">Time (s)</td>'
    for key, _ in columns:
        val = get_val(key, f'{stage_prefix}_time')
        html += f'<td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">{f"{val:.1f}" if val else "—"}</td>'
    html += '</tr>'

    # VRAM
    html += f'<tr><td style="padding: 6px; border-bottom: 1px solid var(--color-border-light);">VRAM (GB)</td>'
    for key, _ in columns:
        val = get_val(key, f'{stage_prefix}_vram')
        html += f'<td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">{f"{val:.2f}" if val else "—"}</td>'
    html += '</tr>'

    # Output metrics
    html += f'<tr style="background: var(--color-bg-secondary);"><td colspan="{len(columns)+1}" style="padding: 8px 6px; font-weight: bold; font-style: italic; opacity: 0.7;">Output Quality</td></tr>'

    # Words
    html += f'<tr><td style="padding: 6px; border-bottom: 1px solid var(--color-border-light);">Words</td>'
    for key, _ in columns:
        val = get_val(key, f'{stage_prefix}_words')
        html += f'<td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">{f"{val:,}" if val else "—"}</td>'
    html += '</tr>'

    # Lexical diversity
    html += f'<tr><td style="padding: 6px; border-bottom: 1px solid var(--color-border-light);">Lexical Div.</td>'
    for key, _ in columns:
        val = get_val(key, f'{stage_prefix}_lex_div')
        html += f'<td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">{f"{val:.1%}" if val else "—"}</td>'
    html += '</tr>'

    # Extra metrics
    if extra_metrics:
        for metric_key, metric_label, fmt_fn, _ in extra_metrics:
            html += f'<tr><td style="padding: 6px; border-bottom: 1px solid var(--color-border-light);">{metric_label}</td>'
            for key, _ in columns:
                val = get_val(key, metric_key)
                html += f'<td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">{fmt_fn(val) if val is not None else "—"}</td>'
            html += '</tr>'

    # Degeneration metrics
    html += f'<tr style="background: var(--color-bg-secondary);"><td colspan="{len(columns)+1}" style="padding: 8px 6px; font-weight: bold; font-style: italic; opacity: 0.7;">Degeneration Detection</td></tr>'

    # 5-gram repeat
    html += f'<tr><td style="padding: 6px; border-bottom: 1px solid var(--color-border-light);">5-gram Repeat</td>'
    for key, _ in columns:
        top_ngrams = get_val(key, f'{stage_prefix}_top_ngrams') or []
        if top_ngrams:
            count = top_ngrams[0].get('repeat_count', 0)
            color = 'var(--color-status-fail)' if count > 6 else ('var(--color-status-warn)' if count > 3 else 'inherit')
            html += f'<td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light); color: {color};">{count}×</td>'
        else:
            html += '<td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">—</td>'
    html += '</tr>'

    # Verdict
    html += f'<tr style="background: rgba(128,128,128,0.05);"><td style="padding: 8px; font-weight: bold;">Verdict</td>'
    for key, _ in columns:
        verdict = get_val(key, f'{stage_prefix}_verdict')
        verdict_class = get_verdict_class(verdict)
        html += f'<td class="{verdict_class}" style="padding: 8px; text-align: center; font-weight: bold;">{verdict or "—"}</td>'
    html += '</tr>'

    html += '</tbody></table>'
    return html


def _create_generic_totals_table(data: dict, columns: list) -> str:
    """Create a totals table for any comparison analysis."""
    html = '<h4 style="margin-top: 25px; margin-bottom: 10px;">Overall Results</h4>'

    html += '''<table style="border-collapse: collapse; font-size: 13px; margin-bottom: 15px;">
    <thead>
        <tr style="background: rgba(128,128,128,0.15);">
            <th style="padding: 8px; text-align: left; border-bottom: 2px solid rgba(128,128,128,0.3);"></th>'''

    for key, label in columns:
        html += f'<th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(128,128,128,0.3);">{label}</th>'

    html += '</tr></thead><tbody>'

    def get_val(key, field):
        return data.get(key, {}).get(field)

    # Total time
    html += f'<tr><td style="padding: 6px; border-bottom: 1px solid var(--color-border-light);">Total Time (s)</td>'
    for key, _ in columns:
        val = get_val(key, 'total_time')
        html += f'<td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">{f"{val:.1f}" if val else "—"}</td>'
    html += '</tr>'

    # Peak VRAM
    html += f'<tr><td style="padding: 6px; border-bottom: 1px solid var(--color-border-light);">Peak VRAM (GB)</td>'
    for key, _ in columns:
        val = get_val(key, 'vram_peak_gb')
        html += f'<td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">{f"{val:.2f}" if val else "—"}</td>'
    html += '</tr>'

    # Total words
    html += f'<tr><td style="padding: 6px; border-bottom: 1px solid var(--color-border-light);">Total Words</td>'
    for key, _ in columns:
        val = get_val(key, 'total_words')
        html += f'<td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">{f"{val:,}" if val else "—"}</td>'
    html += '</tr>'

    # Overall verdict
    html += f'<tr style="background: rgba(128,128,128,0.05);"><td style="padding: 8px; font-weight: bold;">Overall Verdict</td>'
    for key, _ in columns:
        verdict = get_val(key, 'verdict')
        verdict_class = get_verdict_class(verdict)
        html += f'<td class="{verdict_class}" style="padding: 8px; text-align: center; font-weight: bold;">{verdict or "—"}</td>'
    html += '</tr>'

    html += '</tbody></table>'
    return html


def display_repetition_penalty_analysis(evaluation) -> None:
    """
    Display repetition penalty impact analysis.

    Compares repetition penalty (None vs 1.2) on Qwen2-0.5B and Mistral-7B.
    Shows text outputs, metrics tables, and overall results.

    Args:
        evaluation: EvaluationNotebook instance
    """
    from IPython.display import display, HTML
    from .evaluation_analysis import assess_quality

    results = evaluation.load_results()
    outputs_dir = evaluation.outputs_dir

    if not results.get('experiments'):
        print("No evaluation results found. Run the experiments first.")
        return

    # Collect repetition penalty experiments
    penalty_experiments = collect_llm_experiment_data(
        results, outputs_dir, 'repetition_penalty', assess_quality
    )

    if not penalty_experiments:
        print("No repetition penalty experiments found.")
        return

    print(f"Found {len(penalty_experiments)} repetition penalty experiments")
    for exp in sorted(penalty_experiments, key=lambda x: (x['model'], str(x['repetition_penalty']))):
        penalty_label = 'none' if exp['repetition_penalty'] is None else str(exp['repetition_penalty'])
        print(f"  - {exp['model']} (penalty={penalty_label}): {exp['verdict']}")
    print()

    # Organize by (model, penalty) key
    data_by_key = {}
    for exp in penalty_experiments:
        penalty_str = 'none' if exp['repetition_penalty'] is None else str(exp['repetition_penalty'])
        key = (exp['model'], penalty_str)
        data_by_key[key] = exp

    # Define columns for this analysis
    columns = [
        (('Qwen2-0.5B-Instruct', 'none'), 'Qwen2-0.5B (None)'),
        (('Qwen2-0.5B-Instruct', '1.2'), 'Qwen2-0.5B (1.2)'),
        (('Mistral-7B-Instruct-v0.3', 'none'), 'Mistral-7B (None)'),
        (('Mistral-7B-Instruct-v0.3', '1.2'), 'Mistral-7B (1.2)'),
    ]

    # Export to JSON
    json_path = evaluation.drive_base / 'repetition_penalty_comparison.json'
    json_data = _export_penalty_to_json(penalty_experiments, json_path)
    print(f"JSON saved: {json_path}")

    # Create download button for JSON
    json_content = json.dumps(json_data, indent=2, ensure_ascii=False)
    json_b64 = base64.b64encode(json_content.encode('utf-8')).decode('utf-8')
    download_html = f'''
    <div style="margin: 15px 0;">
        <a href="data:application/json;base64,{json_b64}" download="repetition_penalty_comparison.json"
           class="download-btn" style="background: var(--color-accent-purple);
                  border-radius: 20px; cursor: pointer; font-size: 14px; font-weight: 500;
                  text-decoration: none; display: inline-block;
                  box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
            📥 Download JSON
        </a>
        <span style="margin-left: 10px; font-size: 12px; opacity: 0.7;">repetition_penalty_comparison.json</span>
    </div>
    '''
    inject_evaluation_css()
    display(HTML(download_html))

    # Display text boxes
    display(HTML(load_template('section_header.html', title="Repetition Penalty Comparison", color_class="purple")))
    display(HTML(_create_generic_text_boxes_row("Translation Output", data_by_key, columns, 'trans')))
    display(HTML(_create_generic_text_boxes_row("Summary Output", data_by_key, columns, 'sum')))
    display(HTML(_create_generic_text_boxes_row("Podcast Script Output", data_by_key, columns, 'pod')))

    # Display metrics tables (detailed version with n-gram analysis)
    display(HTML(_create_detailed_stage_metrics_table(data_by_key, columns, 'Translation', 'trans',
                 header_color='var(--color-accent-purple)')))
    display(HTML(_create_detailed_stage_metrics_table(data_by_key, columns, 'Summary', 'sum',
                 header_color='var(--color-accent-purple)',
                 extra_metrics=[('sum_bullets', 'Bullet Points', lambda v: f"{v}", False)])))
    display(HTML(_create_detailed_stage_metrics_table(data_by_key, columns, 'Podcast', 'pod',
                 header_color='var(--color-accent-purple)',
                 extra_metrics=[('pod_lines', 'Script Lines', lambda v: f"{v:,}", False)])))

    # Display totals
    display(HTML(_create_generic_totals_table(data_by_key, columns)))


def _export_penalty_to_json(penalty_data: list, output_path: Path) -> dict:
    """Export repetition penalty comparison data to JSON."""
    result = {
        'metadata': {
            'export_type': 'repetition_penalty_comparison',
            'experiment_count': len(penalty_data),
        },
        'experiments': [{
            'exp_id': d.get('exp_id', ''),
            'model': d.get('model', ''),
            'penalty': 'None' if d.get('repetition_penalty') is None else str(d.get('repetition_penalty')),
            'verdict': d.get('verdict', 'Unknown'),
            'totals': {
                'time_seconds': round(d.get('total_time', 0), 2),
                'vram_peak_gb': round(d.get('vram_peak_gb', 0), 2),
            },
            'stages': {
                'translation': _extract_stage_metrics(d, 'trans'),
                'summary': _extract_stage_metrics(d, 'sum'),
                'podcast': _extract_stage_metrics(d, 'pod'),
            },
        } for d in penalty_data],
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def _export_summary_mode_to_json(mode_data: list, output_path: Path) -> dict:
    """Export summary mode comparison data to JSON."""
    result = {
        'metadata': {
            'export_type': 'summary_mode_comparison',
            'experiment_count': len(mode_data),
        },
        'experiments': [{
            'exp_id': d.get('exp_id', ''),
            'model': d.get('model', ''),
            'summary_mode': d.get('summary_mode', ''),
            'verdict': d.get('verdict', 'Unknown'),
            'totals': {
                'time_seconds': round(d.get('total_time', 0), 2),
                'vram_peak_gb': round(d.get('vram_peak_gb', 0), 2),
            },
            'stages': {
                'summary': _extract_stage_metrics(d, 'sum'),
            },
        } for d in mode_data],
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def _export_asr_to_json(asr_data: list, output_path: Path) -> dict:
    """Export ASR comparison data to JSON."""
    result = {
        'metadata': {
            'export_type': 'asr_comparison',
            'experiment_count': len(asr_data),
        },
        'experiments': [{
            'exp_id': d.get('exp_id', ''),
            'model': d.get('model', ''),
            'verdict': d.get('verdict', 'Unknown'),
            'performance': {
                'time_seconds': round(d.get('time_seconds', 0), 2),
                'realtime_factor': round(d.get('rtf', 0), 3),
                'vram_peak_gb': round(d.get('vram_peak_gb', 0), 2),
            },
            'output': {
                'word_count': d.get('word_count', 0),
                'char_count': d.get('char_count', 0),
                'sentence_count': d.get('sentence_count', 0),
                'capitalized_words': d.get('capitalized_words', 0),
                'missing_caps': d.get('missing_caps', 0),
                'acronyms': d.get('acronyms', 0),
            },
            'quality': {
                'lexical_diversity': round(d.get('lexical_diversity', 0), 4),
                'ngram_repeat_count': d.get('ngram_repeat_count', 0),
                'longest_repeat_count': d.get('longest_repeat_count', 0),
                'longest_repeat_pattern': d.get('longest_repeat_pattern', ''),
                'max_char_repeat': d.get('max_char_repeat', 0),
            },
        } for d in asr_data],
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def _export_chunk_size_to_json(chunk_data: list, output_path: Path) -> dict:
    """Export chunk size comparison data to JSON."""
    result = {
        'metadata': {
            'export_type': 'chunk_size_comparison',
            'experiment_count': len(chunk_data),
        },
        'experiments': [{
            'exp_id': d.get('exp_id', ''),
            'model': d.get('model', ''),
            'chunk_size': d.get('chunk_size', 0),
            'verdict': d.get('verdict', 'Unknown'),
            'totals': {
                'time_seconds': round(d.get('total_time', 0), 2),
                'vram_peak_gb': round(d.get('vram_peak_gb', 0), 2),
            },
            'stages': {
                'translation': _extract_stage_metrics(d, 'trans'),
                'summary': _extract_stage_metrics(d, 'sum'),
                'podcast': _extract_stage_metrics(d, 'pod'),
            },
        } for d in chunk_data],
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def _extract_stage_metrics(data: dict, stage_prefix: str) -> dict:
    """Extract metrics for a stage from experiment data."""
    top_ngrams = data.get(f'{stage_prefix}_top_ngrams') or []
    peak_ngram = top_ngrams[0] if top_ngrams else None

    return {
        'time_seconds': round(data.get(f'{stage_prefix}_time', 0), 2),
        'vram_peak_gb': round(data.get(f'{stage_prefix}_vram', 0), 2),
        'word_count': data.get(f'{stage_prefix}_words', 0),
        'lexical_diversity': round(data.get(f'{stage_prefix}_lex_div', 0), 4),
        'verdict': data.get(f'{stage_prefix}_verdict', 'Unknown'),
        'peak_ngram': {
            'n': peak_ngram.get('n', 0) if peak_ngram else 0,
            'repeat_count': peak_ngram.get('repeat_count', 0) if peak_ngram else 0,
            'rate': round(peak_ngram.get('rate', 0), 4) if peak_ngram else 0,
            'pattern': peak_ngram.get('most_repeated', '') if peak_ngram else '',
        },
        'max_char_repeat': data.get(f'{stage_prefix}_char_count', 0),
        'max_word_repeat': data.get(f'{stage_prefix}_word_count', 0),
        'max_2gram_repeat': data.get(f'{stage_prefix}_2gram_count', 0),
        'max_3gram_repeat': data.get(f'{stage_prefix}_3gram_count', 0),
        'max_4gram_repeat': data.get(f'{stage_prefix}_4gram_count', 0),
    }


def display_summary_mode_analysis(evaluation) -> None:
    """
    Display summary mode impact analysis.

    Compares greedy and sampled summary modes on Phi-3-mini (no quantization).

    Args:
        evaluation: EvaluationNotebook instance
    """
    from IPython.display import display, HTML
    from .evaluation_analysis import assess_quality

    results = evaluation.load_results()
    outputs_dir = evaluation.outputs_dir

    if not results.get('experiments'):
        print("No evaluation results found. Run the experiments first.")
        return

    # Collect summary mode experiments
    mode_experiments = collect_llm_experiment_data(
        results, outputs_dir, 'summary_mode', assess_quality
    )

    if not mode_experiments:
        print("No summary mode experiments found.")
        return

    print(f"Found {len(mode_experiments)} summary mode experiments")
    for exp in sorted(mode_experiments, key=lambda x: x['summary_mode']):
        # Show summary stage verdict (not overall) since that's what varies
        print(f"  - {exp['summary_mode']}: {exp['sum_verdict']}")
    print()

    # Organize by mode
    data_by_mode = {exp['summary_mode']: exp for exp in mode_experiments}

    # Define columns
    columns = [
        ('greedy', 'Greedy'),
        ('sampled', 'Sampled'),
    ]

    # Export to JSON
    json_path = evaluation.drive_base / 'summary_mode_comparison.json'
    json_data = _export_summary_mode_to_json(mode_experiments, json_path)
    print(f"JSON saved: {json_path}")

    # Create download button for JSON
    json_content = json.dumps(json_data, indent=2, ensure_ascii=False)
    json_b64 = base64.b64encode(json_content.encode('utf-8')).decode('utf-8')
    download_html = f'''
    <div style="margin: 15px 0;">
        <a href="data:application/json;base64,{json_b64}" download="summary_mode_comparison.json"
           class="download-btn" style="background: var(--color-accent-teal);
                  border-radius: 20px; cursor: pointer; font-size: 14px; font-weight: 500;
                  text-decoration: none; display: inline-block;
                  box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
            📥 Download JSON
        </a>
        <span style="margin-left: 10px; font-size: 12px; opacity: 0.7;">summary_mode_comparison.json</span>
    </div>
    '''
    inject_evaluation_css()
    display(HTML(download_html))

    # Display (summary mode only affects summary stage)
    display(HTML(load_template('section_header.html', title="Summary Mode Comparison (Phi-3-mini, no quant)", color_class="teal")))
    display(HTML(_create_generic_text_boxes_row("Summary Output", data_by_mode, columns, 'sum')))
    display(HTML(_create_detailed_stage_metrics_table(data_by_mode, columns, 'Summary', 'sum',
                 header_color='var(--color-accent-teal)',
                 extra_metrics=[('sum_bullets', 'Bullet Points', lambda v: f"{v}", False)])))


def display_chunk_size_analysis(evaluation) -> None:
    """
    Display chunk size impact analysis.

    Compares 2000 vs 4000 character chunks on Phi-3-mini (no quantization).

    Args:
        evaluation: EvaluationNotebook instance
    """
    from IPython.display import display, HTML
    from .evaluation_analysis import assess_quality

    results = evaluation.load_results()
    outputs_dir = evaluation.outputs_dir

    if not results.get('experiments'):
        print("No evaluation results found. Run the experiments first.")
        return

    # Collect chunk size experiments
    chunk_experiments = collect_llm_experiment_data(
        results, outputs_dir, 'chunk_size', assess_quality
    )

    if not chunk_experiments:
        print("No chunk size experiments found.")
        return

    print(f"Found {len(chunk_experiments)} chunk size experiments")
    for exp in sorted(chunk_experiments, key=lambda x: x['chunk_size']):
        print(f"  - {exp['chunk_size']} chars: {exp['verdict']}")
    print()

    # Organize by chunk size
    data_by_chunk = {exp['chunk_size']: exp for exp in chunk_experiments}

    # Define columns
    columns = [
        (2000, '2000 chars'),
        (4000, '4000 chars'),
    ]

    # Export to JSON
    json_path = evaluation.drive_base / 'chunk_size_comparison.json'
    json_data = _export_chunk_size_to_json(chunk_experiments, json_path)
    print(f"JSON saved: {json_path}")

    # Create download button for JSON
    json_content = json.dumps(json_data, indent=2, ensure_ascii=False)
    json_b64 = base64.b64encode(json_content.encode('utf-8')).decode('utf-8')
    download_html = f'''
    <div style="margin: 15px 0;">
        <a href="data:application/json;base64,{json_b64}" download="chunk_size_comparison.json"
           class="download-btn" style="background: var(--color-warning-dark);
                  border-radius: 20px; cursor: pointer; font-size: 14px; font-weight: 500;
                  text-decoration: none; display: inline-block;
                  box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
            📥 Download JSON
        </a>
        <span style="margin-left: 10px; font-size: 12px; opacity: 0.7;">chunk_size_comparison.json</span>
    </div>
    '''
    inject_evaluation_css()
    display(HTML(download_html))

    # Display
    display(HTML(load_template('section_header.html', title="Chunk Size Comparison (Phi-3-mini, no quant)", color_class="warning")))
    display(HTML(_create_generic_text_boxes_row("Translation Output", data_by_chunk, columns, 'trans')))
    display(HTML(_create_generic_text_boxes_row("Summary Output", data_by_chunk, columns, 'sum')))
    display(HTML(_create_generic_text_boxes_row("Podcast Script Output", data_by_chunk, columns, 'pod')))
    display(HTML(_create_detailed_stage_metrics_table(data_by_chunk, columns, 'Translation', 'trans',
                 header_color='var(--color-warning-dark)')))
    display(HTML(_create_detailed_stage_metrics_table(data_by_chunk, columns, 'Summary', 'sum',
                 header_color='var(--color-warning-dark)',
                 extra_metrics=[('sum_bullets', 'Bullet Points', lambda v: f"{v}", False)])))
    display(HTML(_create_detailed_stage_metrics_table(data_by_chunk, columns, 'Podcast', 'pod',
                 header_color='var(--color-warning-dark)',
                 extra_metrics=[('pod_lines', 'Script Lines', lambda v: f"{v:,}", False)])))
    display(HTML(_create_generic_totals_table(data_by_chunk, columns)))


def _export_temperature_to_json(temp_data: list, output_path: Path) -> dict:
    """Export temperature comparison data to JSON."""
    result = {
        'metadata': {
            'export_type': 'temperature_comparison',
            'experiment_count': len(temp_data),
        },
        'experiments': [],
    }

    for d in temp_data:
        exp_entry = {
            'exp_id': d.get('exp_id', ''),
            'temperature': d.get('temperature'),
            'model': d.get('model', ''),
            'verdict': d.get('verdict', 'Unknown'),
            'pod_verdict': d.get('pod_verdict', 'Unknown'),
            'totals': {
                'time_seconds': round(d.get('total_time', 0), 2),
                'vram_peak_gb': round(d.get('vram_peak_gb', 0), 2),
            },
            'podcast': {
                'time_seconds': round(d.get('pod_time', 0), 2),
                'vram_gb': round(d.get('pod_vram', 0), 2),
                'words': d.get('pod_words', 0),
                'lines': d.get('pod_lines', 0),
                'lexical_diversity': round(d.get('pod_lex_div', 0), 4),
                'verdict': d.get('pod_verdict', 'Unknown'),
                'degeneration': {
                    'top_ngrams': d.get('pod_top_ngrams', []),
                    'char_repeat': {
                        'count': d.get('pod_char_count', 0),
                        'char': d.get('pod_char', ''),
                    },
                    'word_repeat': {
                        'count': d.get('pod_word_count', 0),
                        'pattern': d.get('pod_word_pattern', ''),
                    },
                    '2gram_repeat': {
                        'count': d.get('pod_2gram_count', 0),
                        'pattern': d.get('pod_2gram_pattern', ''),
                    },
                    '3gram_repeat': {
                        'count': d.get('pod_3gram_count', 0),
                        'pattern': d.get('pod_3gram_pattern', ''),
                    },
                    '4gram_repeat': {
                        'count': d.get('pod_4gram_count', 0),
                        'pattern': d.get('pod_4gram_pattern', ''),
                    },
                },
            },
        }
        result['experiments'].append(exp_entry)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def _create_detailed_stage_metrics_table(
    data: dict,
    columns: list,
    stage_name: str,
    stage_prefix: str,
    header_color: str = 'var(--color-accent-indigo)',
    extra_metrics: list = None
) -> str:
    """
    Create a detailed metrics table with n-gram and repetition info.

    Similar to create_stage_metrics_table but column-keyed instead of model-keyed.
    """
    # Common cell border style
    cell_border = "border-left: 1px solid var(--color-border-light);"
    header_cell_border = f"border-left: 1px solid color-mix(in srgb, {header_color} 25%, transparent);"

    def get_val(key, field):
        return data.get(key, {}).get(field)

    html = f'''
    <div style="margin: 20px 0; padding: 15px; border: 1px solid var(--color-border-light); border-radius: 8px;">
    <h4 style="margin: 0 0 15px 0; color: {header_color};">{stage_name} Stage</h4>
    <table style="border-collapse: collapse; font-size: 12px;">
    <thead><tr style="background: color-mix(in srgb, {header_color} 10%, transparent);">
        <th style="padding: 8px; text-align: left; border-bottom: 2px solid {header_color};">Metric</th>'''

    for key, label in columns:
        html += f'<th style="padding: 8px; text-align: center; border-bottom: 2px solid {header_color}; {header_cell_border}">{label}</th>'
    html += '</tr></thead><tbody>'

    # Core metrics
    metrics = [
        (f'{stage_prefix}_time', 'Processing Time', lambda v: f"{v:.1f}s", True),
        (f'{stage_prefix}_vram', 'Peak VRAM', lambda v: f"{v:.2f} GB", True),
        (f'{stage_prefix}_words', 'Word Count', lambda v: f"{v:,}", False),
        (f'{stage_prefix}_lex_div', 'Lexical Diversity', lambda v: f"{v:.1%}", False),
    ]

    if extra_metrics:
        metrics.extend(extra_metrics)

    for key, label, fmt, lower_better in metrics:
        html += f'<tr><td style="padding: 6px; border-bottom: 1px solid var(--color-border-light);">{label}</td>'
        for col_key, _ in columns:
            val = get_val(col_key, key)
            formatted = fmt(val) if val else "—"
            style = f"padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light); {cell_border}"
            if formatted == "—":
                style += " color: var(--color-muted); font-style: italic;"
            html += f'<td style="{style}">{formatted}</td>'
        html += '</tr>'

    # Peak N-gram section
    html += f'<tr style="background: var(--color-bg-secondary);"><td colspan="{len(columns)+1}" style="padding: 8px 6px; font-weight: bold; font-style: italic; opacity: 0.7;">Peak N-gram Repetition</td></tr>'

    html += f'<tr><td style="padding: 6px; border-bottom: 1px solid var(--color-border-light);">Peak N-gram</td>'
    for col_key, _ in columns:
        top_ngrams = get_val(col_key, f'{stage_prefix}_top_ngrams') or []
        if top_ngrams:
            ng = top_ngrams[0]
            rate = ng.get('rate', 0)
            n = ng.get('n', 0)
            count = ng.get('repeat_count', 0)
            pattern = ng.get('most_repeated', '')
            pattern = pattern.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            color = 'var(--color-status-fail)' if count > 6 else ('var(--color-status-warn)' if count > 3 else 'inherit')
            html += f'<td style="padding: 6px; border-bottom: 1px solid var(--color-border-light); font-size: 11px; text-align: left; color: {color}; {cell_border}">'
            html += f'<b>{count}x</b> ({rate:.1%}) n={n}<br><code style="font-size: 10px; background: var(--color-bg-secondary); padding: 1px 4px; border-radius: 3px; word-break: break-all;">{pattern}</code></td>'
        else:
            html += f'<td style="padding: 6px; border-bottom: 1px solid var(--color-border-light); text-align: center; color: var(--color-muted); font-size: 11px; {cell_border}">—</td>'
    html += '</tr>'

    # Consecutive repetition section
    html += f'<tr style="background: var(--color-bg-secondary);"><td colspan="{len(columns)+1}" style="padding: 8px 6px; font-weight: bold; font-style: italic; opacity: 0.7;">Consecutive Repetitions</td></tr>'

    def format_count_pattern(count, pattern):
        if not count or count < 2:
            return "—"
        escaped = pattern.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;') if pattern else ''
        return f'{count}x<br><code style="font-size: 10px; background: var(--color-bg-secondary); padding: 1px 4px; border-radius: 3px;">{escaped}</code>'

    # Char repetition
    html += f'<tr><td style="padding: 6px; border-bottom: 1px solid var(--color-border-light);">Max Char Repeat</td>'
    for col_key, _ in columns:
        count = get_val(col_key, f'{stage_prefix}_char_count') or 0
        char = get_val(col_key, f'{stage_prefix}_char') or ''
        formatted = format_count_pattern(count, char)
        style = f"padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light); {cell_border}"
        if formatted == "—":
            style += " color: var(--color-muted); font-style: italic;"
        elif count > 10:
            style += " color: var(--color-status-fail);"
        elif count > 5:
            style += " color: var(--color-status-warn);"
        html += f'<td style="{style}">{formatted}</td>'
    html += '</tr>'

    # Word repetition
    html += f'<tr><td style="padding: 6px; border-bottom: 1px solid var(--color-border-light);">Max Word Repeat</td>'
    for col_key, _ in columns:
        count = get_val(col_key, f'{stage_prefix}_word_count') or 0
        pattern = get_val(col_key, f'{stage_prefix}_word_pattern') or ''
        formatted = format_count_pattern(count, pattern)
        style = f"padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light); {cell_border}"
        if formatted == "—":
            style += " color: var(--color-muted); font-style: italic;"
        elif count > 5:
            style += " color: var(--color-status-fail);"
        elif count > 3:
            style += " color: var(--color-status-warn);"
        html += f'<td style="{style}">{formatted}</td>'
    html += '</tr>'

    # 2-gram repetition
    html += f'<tr><td style="padding: 6px; border-bottom: 1px solid var(--color-border-light);">Max 2-gram Repeat</td>'
    for col_key, _ in columns:
        count = get_val(col_key, f'{stage_prefix}_2gram_count') or 0
        pattern = get_val(col_key, f'{stage_prefix}_2gram_pattern') or ''
        formatted = format_count_pattern(count, pattern)
        style = f"padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light); {cell_border}"
        if formatted == "—":
            style += " color: var(--color-muted); font-style: italic;"
        elif count > 5:
            style += " color: var(--color-status-fail);"
        elif count > 3:
            style += " color: var(--color-status-warn);"
        html += f'<td style="{style}">{formatted}</td>'
    html += '</tr>'

    # 3-gram repetition
    html += f'<tr><td style="padding: 6px; border-bottom: 1px solid var(--color-border-light);">Max 3-gram Repeat</td>'
    for col_key, _ in columns:
        count = get_val(col_key, f'{stage_prefix}_3gram_count') or 0
        pattern = get_val(col_key, f'{stage_prefix}_3gram_pattern') or ''
        formatted = format_count_pattern(count, pattern)
        style = f"padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light); {cell_border}"
        if formatted == "—":
            style += " color: var(--color-muted); font-style: italic;"
        elif count > 5:
            style += " color: var(--color-status-fail);"
        elif count > 3:
            style += " color: var(--color-status-warn);"
        html += f'<td style="{style}">{formatted}</td>'
    html += '</tr>'

    # 4-gram repetition
    html += f'<tr><td style="padding: 6px; border-bottom: 1px solid var(--color-border-light);">Max 4-gram Repeat</td>'
    for col_key, _ in columns:
        count = get_val(col_key, f'{stage_prefix}_4gram_count') or 0
        pattern = get_val(col_key, f'{stage_prefix}_4gram_pattern') or ''
        formatted = format_count_pattern(count, pattern)
        style = f"padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light); {cell_border}"
        if formatted == "—":
            style += " color: var(--color-muted); font-style: italic;"
        elif count > 5:
            style += " color: var(--color-status-fail);"
        elif count > 3:
            style += " color: var(--color-status-warn);"
        html += f'<td style="{style}">{formatted}</td>'
    html += '</tr>'

    # Verdict row
    html += f'<tr style="background: rgba(128,128,128,0.05);"><td style="padding: 8px; font-weight: bold;">Verdict</td>'
    for col_key, _ in columns:
        verdict = get_val(col_key, f'{stage_prefix}_verdict')
        verdict_class = get_verdict_class(verdict)
        html += f'<td class="{verdict_class}" style="padding: 8px; text-align: center; font-weight: bold;">{verdict or "—"}</td>'
    html += '</tr>'

    html += '</tbody></table></div>'
    return html


def display_temperature_analysis(evaluation) -> None:
    """
    Display temperature impact analysis.

    Compares temperature 0.2/0.5 on Mistral-7B podcast generation.
    Temperature only affects the podcast stage.

    Args:
        evaluation: EvaluationNotebook instance
    """
    from IPython.display import display, HTML
    from .evaluation_analysis import assess_quality

    results = evaluation.load_results()
    outputs_dir = evaluation.outputs_dir

    if not results.get('experiments'):
        print("No evaluation results found. Run the experiments first.")
        return

    # Collect temperature experiments
    temp_experiments = collect_llm_experiment_data(
        results, outputs_dir, 'temperature', assess_quality
    )

    if not temp_experiments:
        print("No temperature experiments found.")
        return

    print(f"Found {len(temp_experiments)} temperature experiments")
    for exp in sorted(temp_experiments, key=lambda x: x['temperature']):
        # Show podcast stage verdict (not overall) since that's what temperature affects
        print(f"  - temp={exp['temperature']}: {exp['pod_verdict']}")
    print()

    # Organize by temperature
    data_by_temp = {exp['temperature']: exp for exp in temp_experiments}

    # Define columns
    columns = [
        (0.2, 'Temp 0.2'),
        (0.5, 'Temp 0.5'),
    ]

    # Export to JSON
    json_path = evaluation.drive_base / 'temperature_comparison.json'
    json_data = _export_temperature_to_json(temp_experiments, json_path)
    print(f"JSON saved: {json_path}")

    # Create download button for JSON
    json_content = json.dumps(json_data, indent=2, ensure_ascii=False)
    json_b64 = base64.b64encode(json_content.encode('utf-8')).decode('utf-8')
    download_html = f'''
    <div style="margin: 15px 0;">
        <a href="data:application/json;base64,{json_b64}" download="temperature_comparison.json"
           class="download-btn" style="background: var(--color-accent-indigo);
                  border-radius: 20px; cursor: pointer; font-size: 14px; font-weight: 500;
                  text-decoration: none; display: inline-block;
                  box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
            📥 Download JSON
        </a>
        <span style="margin-left: 10px; font-size: 12px; opacity: 0.7;">temperature_comparison.json</span>
    </div>
    '''
    inject_evaluation_css()
    display(HTML(download_html))

    # Display (temperature only affects podcast stage)
    display(HTML(load_template('section_header.html', title="Temperature Comparison (Mistral-7B, Podcast Only)", color_class="indigo")))
    display(HTML(_create_generic_text_boxes_row("Podcast Script Output", data_by_temp, columns, 'pod')))
    display(HTML(_create_detailed_stage_metrics_table(data_by_temp, columns, 'Podcast', 'pod',
                 header_color='var(--color-accent-indigo)',
                 extra_metrics=[('pod_lines', 'Script Lines', lambda v: f"{v:,}", False)])))


def display_asr_analysis(evaluation) -> None:
    """
    Display ASR model comparison analysis.

    Compares Whisper tiny/small/large-v3 transcription results.
    Includes hallucination detection based on repetition patterns.

    Args:
        evaluation: EvaluationNotebook instance
    """
    import re
    from IPython.display import display, HTML
    from .evaluation_analysis import (
        compute_repetition_rate,
        compute_lexical_diversity,
        find_longest_word_repetition,
        find_longest_char_repetition,
        compute_ngram_analysis,
        count_capitalized_words,
        count_missing_capitalizations,
        count_acronyms,
    )

    results = evaluation.load_results()
    outputs_dir = evaluation.outputs_dir

    if not results.get('experiments'):
        print("No evaluation results found. Run the experiments first.")
        return

    # Helper functions for ASR metrics
    def load_transcript(exp_id):
        path = outputs_dir / exp_id / "asr_transcript.txt"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""

    def compute_asr_metrics(text):
        if not text or not text.strip():
            return {
                'word_count': 0, 'char_count': 0, 'sentence_count': 0,
                'lexical_diversity': 0, 'repetition_rate': 0,
                'longest_repeat_count': 0, 'longest_repeat_pattern': '',
                'capitalized_words': 0, 'missing_caps': 0, 'acronyms': 0,
                'ngram_repeat_count': 0, 'max_char_repeat': 0,
                'verdict': 'NO DATA'
            }

        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        lex_div = compute_lexical_diversity(text)
        rep_rate = compute_repetition_rate(text, n=3)
        longest_pattern, longest_count = find_longest_word_repetition(text)

        # New metrics
        char_pattern, char_count = find_longest_char_repetition(text)
        ngram_analysis = compute_ngram_analysis(text, n=5)
        ngram_repeat = ngram_analysis['top_ngrams'][0]['repeat_count'] if ngram_analysis['top_ngrams'] else 0
        cap_words = count_capitalized_words(text, exclude_sentence_start=True)
        missing_caps = count_missing_capitalizations(text)
        acronyms = count_acronyms(text)

        # Determine verdict
        if longest_count >= 10 or rep_rate > 0.20:
            verdict = 'FAIL'
        elif longest_count >= 5 or rep_rate > 0.10:
            verdict = 'WARNING'
        else:
            verdict = 'OK'

        return {
            'word_count': len(words),
            'char_count': len(text),
            'sentence_count': len(sentences) if sentences else 1,
            'lexical_diversity': lex_div,
            'repetition_rate': rep_rate,
            'longest_repeat_count': longest_count,
            'longest_repeat_pattern': longest_pattern,
            'capitalized_words': cap_words,
            'missing_caps': missing_caps,
            'acronyms': acronyms,
            'ngram_repeat_count': ngram_repeat,
            'max_char_repeat': char_count,
            'verdict': verdict,
        }

    # Collect ASR experiments
    asr_experiments = []
    for exp_id, exp in results.get('experiments', {}).items():
        if exp.get('experiment_type') != 'asr_model' or 'error' in exp:
            continue

        config = exp.get('config', {})
        asr_data = exp.get('stages_data', {}).get('asr', {})
        metrics = asr_data.get('metrics', {})
        output = asr_data.get('output', {})

        transcript = load_transcript(exp_id) or output.get('transcript', '')
        text_metrics = compute_asr_metrics(transcript)

        asr_experiments.append({
            'exp_id': exp_id,
            'model': config.get('asr_model_id', '').split('/')[-1],
            'time_seconds': metrics.get('time_seconds', 0),
            'rtf': metrics.get('realtime_factor', 0),
            'vram_peak_gb': metrics.get('vram_peak_gb', 0),
            'transcript': transcript,
            **text_metrics,
        })

    if not asr_experiments:
        print("No ASR experiments found.")
        return

    # Sort by model name
    model_order = ['whisper-tiny', 'whisper-small', 'whisper-large-v3']
    asr_experiments = sorted(asr_experiments, key=lambda x: (
        model_order.index(x['model']) if x['model'] in model_order else 99
    ))

    print(f"Found {len(asr_experiments)} ASR experiments")
    for exp in asr_experiments:
        print(f"  - {exp['model']}: {exp['word_count']} words, {exp['verdict']}")
    print()

    # Export to JSON
    json_path = evaluation.drive_base / 'asr_comparison.json'
    json_data = _export_asr_to_json(asr_experiments, json_path)
    print(f"JSON saved: {json_path}")

    # Create download button for JSON
    json_content = json.dumps(json_data, indent=2, ensure_ascii=False)
    json_b64 = base64.b64encode(json_content.encode('utf-8')).decode('utf-8')
    download_html = f'''
    <div style="margin: 15px 0;">
        <a href="data:application/json;base64,{json_b64}" download="asr_comparison.json"
           class="download-btn" style="background: var(--color-primary-dark);
                  border-radius: 20px; cursor: pointer; font-size: 14px; font-weight: 500;
                  text-decoration: none; display: inline-block;
                  box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
            📥 Download JSON
        </a>
        <span style="margin-left: 10px; font-size: 12px; opacity: 0.7;">asr_comparison.json</span>
    </div>
    '''
    inject_evaluation_css()
    display(HTML(download_html))

    # Build metrics table HTML
    html = '<h2 style="border-bottom: 2px solid var(--color-primary-dark); padding-bottom: 10px; margin-top: 20px;">ASR Model Comparison</h2>'

    # Performance table
    html += '''<h4 style="margin-top: 20px;">Performance Metrics</h4>
    <table style="border-collapse: collapse; font-size: 13px;">
    <thead><tr style="background: rgba(128,128,128,0.15);">
        <th style="padding: 8px; text-align: left; border-bottom: 2px solid rgba(128,128,128,0.3);">Model</th>
        <th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(128,128,128,0.3);">Time (s)</th>
        <th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(128,128,128,0.3);">RTF</th>
        <th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(128,128,128,0.3);">VRAM (GB)</th>
        <th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(128,128,128,0.3);">Words</th>
        <th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(128,128,128,0.3);">Sentences</th>
        <th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(128,128,128,0.3);">Lex. Div.</th>
        <th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(128,128,128,0.3);">Cap. Words</th>
        <th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(128,128,128,0.3);">Missing Caps</th>
        <th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(128,128,128,0.3);">Acronyms</th>
        <th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(128,128,128,0.3);">5-gram Rep.</th>
        <th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(128,128,128,0.3);">Consec. Rep.</th>
        <th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(128,128,128,0.3);">Max Char Rep.</th>
        <th style="padding: 8px; text-align: center; border-bottom: 2px solid rgba(128,128,128,0.3);">Verdict</th>
    </tr></thead><tbody>'''

    for exp in asr_experiments:
        verdict = exp['verdict']
        verdict_class = get_verdict_class(verdict)

        html += f'''<tr>
            <td style="padding: 6px; border-bottom: 1px solid var(--color-border-light);">{exp['model']}</td>
            <td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">{exp['time_seconds']:.1f}</td>
            <td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">{exp['rtf']:.2f}</td>
            <td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">{exp['vram_peak_gb']:.2f}</td>
            <td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">{exp['word_count']:,}</td>
            <td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">{exp['sentence_count']:,}</td>
            <td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">{exp['lexical_diversity']:.1%}</td>
            <td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">{exp['capitalized_words']:,}</td>
            <td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">{exp['missing_caps']:,}</td>
            <td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">{exp['acronyms']:,}</td>
            <td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">{exp['ngram_repeat_count']}</td>
            <td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">{exp['longest_repeat_count']}</td>
            <td style="padding: 6px; text-align: center; border-bottom: 1px solid var(--color-border-light);">{exp['max_char_repeat']}</td>
            <td class="{verdict_class}" style="padding: 6px; text-align: center; font-weight: bold;">{verdict}</td>
        </tr>'''

    html += '</tbody></table>'

    # Transcript boxes
    html += '<h4 style="margin-top: 30px;">Transcript Outputs</h4>'
    html += f'<div style="display: grid; grid-template-columns: repeat({len(asr_experiments)}, 1fr); gap: 15px;">'

    for exp in asr_experiments:
        box_html = create_scrollable_box(
            f"{exp['model']} ({exp['word_count']:,} words)",
            exp['transcript'] if exp['transcript'] else "(No transcript)",
            "400px",
            exp['verdict'],
            f"Rep: {exp['longest_repeat_count']}x" if exp['longest_repeat_count'] > 3 else None
        )
        html += f'<div>{box_html}</div>'

    html += '</div>'
    display(HTML(html))
