"""
Results display and export functions for SOLAS evaluation.
"""

import csv
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict


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
