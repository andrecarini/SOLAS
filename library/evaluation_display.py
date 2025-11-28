"""
Results display and export functions for SOLAS evaluation.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict

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
