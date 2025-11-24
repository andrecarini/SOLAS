"""
Main evaluation runner for SOLAS experiments.
"""

import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Callable
from tqdm.std import tqdm  # Use std (not auto) to avoid CDN widget notice in notebooks

from .evaluation_config import sort_experiments_for_efficiency
from .evaluation_utils import (
    get_memory_snapshot, clear_gpu_memory,
    config_hash, unload_asr, unload_all_models,
    load_cached_stage, save_stage_cache
)


def run_evaluation(
    experiments: List[Dict],
    hardware_info: Dict[str, Any],
    audio_path: str,
    results_file: Path,
    transcript_cache_file: Path,
    outputs_dir: Path,
    cache_base: Path,
    load_results_fn: Callable,
    save_results_fn: Callable,
    save_experiment_output_fn: Callable,
    is_experiment_complete_fn: Callable,
    load_cached_transcript_fn: Callable,
    save_cached_transcript_fn: Callable,
    run_asr_stage_fn: Callable,
    run_translation_stage_fn: Callable,
    run_summary_stage_fn: Callable,
    run_podcast_stage_fn: Callable,
    load_asr_fn: Callable,
    load_llm_fn: Callable,
    StageMetricsCollector: type,
    log_fn: Callable,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Run all remaining experiments with full tracking.

    Args:
        experiments: List of experiment configurations
        hardware_info: Hardware information dict
        audio_path: Path to audio file
        results_file: Path to results JSON file
        transcript_cache_file: Path to transcript cache file
        outputs_dir: Directory for experiment outputs
        cache_base: Base directory for stage caching
        load_results_fn: Function to load results
        save_results_fn: Function to save results
        save_experiment_output_fn: Function to save experiment outputs
        is_experiment_complete_fn: Function to check if experiment is complete
        load_cached_transcript_fn: Function to load cached transcript
        save_cached_transcript_fn: Function to save cached transcript
        run_asr_stage_fn: Function to run ASR stage
        run_translation_stage_fn: Function to run translation stage
        run_summary_stage_fn: Function to run summary stage
        run_podcast_stage_fn: Function to run podcast stage
        load_asr_fn: Function to load ASR model
        load_llm_fn: Function to load LLM model
        StageMetricsCollector: Metrics collector class
        log_fn: Logging function
        dry_run: If True, only print what would be run

    Returns:
        Results dictionary
    """
    results = load_results_fn()

    remaining = []
    skipped_complete = 0

    for exp in experiments:
        if is_experiment_complete_fn(results, exp['id']):
            skipped_complete += 1
            continue
        # Allow duplicate configs to run - they'll use cached stage outputs
        remaining.append(exp)

    remaining = sort_experiments_for_efficiency(remaining)

    log_fn("Evaluation Status", 'header')
    log_fn(f"Total experiments defined: {len(experiments)}", 'info')
    log_fn(f"Already completed: {skipped_complete}", 'info')
    log_fn(f"Remaining to run: {len(remaining)}", 'info')
    log_fn(f"Hardware: {hardware_info.get('gpu_type', 'Unknown')} ({hardware_info.get('gpu_name', 'N/A')})", 'info')

    if not remaining:
        log_fn("All experiments complete!", 'success')
        return results

    if dry_run:
        log_fn("Remaining experiments (dry run):", 'info')
        for exp in remaining[:15]:
            log_fn(f"  - {exp['id']}: {exp['description']}", 'detail')
        if len(remaining) > 15:
            log_fn(f"  ... and {len(remaining) - 15} more", 'detail')
        return results

    log_fn("Starting Execution", 'header')

    transcript_cache = load_cached_transcript_fn()

    # Create progress bar for experiments (text-based, no widgets)
    pbar = tqdm(
        remaining,
        desc="Experiments",
        unit="exp",
        ncols=100,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )

    for i, exp in enumerate(pbar):
        exp_id = exp['id']
        config = exp['config']
        stages = exp['stages']

        # Update progress bar description
        pbar.set_description(f"Exp {i+1}/{len(remaining)}: {exp['description'][:40]}")

        log_fn(f"[{i+1}/{len(remaining)}] {exp['description']}", 'stage')
        log_fn(f"Stages: {stages}", 'detail')

        try:
            exp_result = {
                'id': exp_id,
                'experiment_type': exp['experiment_type'],
                'description': exp['description'],
                'config': config,
                'config_hash': config_hash(config, hardware_info, include_hardware=True),
                'stages_run': stages,
                'stages_data': {},
                'hardware_snapshot': get_memory_snapshot(),
                'hardware_info': {
                    'gpu_type': hardware_info.get('gpu_type'),
                    'gpu_name': hardware_info.get('gpu_name'),
                },
                'started_at': datetime.now().isoformat(),
            }

            if stages == ['asr']:
                log_fn("Running ASR...", 'progress')
                asr_result = run_asr_stage_fn(config, load_asr_fn, StageMetricsCollector, log_fn, audio_path)
                exp_result['stages_data']['asr'] = asr_result
                save_experiment_output_fn(exp_id, 'asr', 'transcript', asr_result['output']['transcript'])
                m = asr_result['metrics']
                log_fn(f"ASR complete: {asr_result['output']['transcript_words']} words in {m['time_seconds']:.1f}s (VRAM peak: {m['vram_peak_gb']:.2f} GB)", 'success')

                if config.get('asr_model_id') == 'openai/whisper-large-v3':
                    save_cached_transcript_fn(asr_result['output']['transcript'], config['asr_model_id'], audio_path, asr_result['metrics'])
            else:
                if transcript_cache and transcript_cache.get('asr_model') == 'openai/whisper-large-v3':
                    log_fn("Using cached transcript from whisper-large-v3", 'detail')
                    transcript = transcript_cache['transcript']
                    exp_result['transcript_source'] = 'cache'
                else:
                    log_fn("Generating transcript with whisper-large-v3...", 'progress')
                    asr_config = {**config, 'asr_model_id': 'openai/whisper-large-v3'}
                    asr_result = run_asr_stage_fn(asr_config, load_asr_fn, StageMetricsCollector, log_fn, audio_path)
                    transcript = asr_result['output']['transcript']
                    exp_result['transcript_source'] = 'generated'
                    save_cached_transcript_fn(transcript, 'openai/whisper-large-v3', audio_path, asr_result['metrics'])
                    transcript_cache = load_cached_transcript_fn()
                    unload_asr()

                # --- TRANSLATION ---
                # Check if we need translation (either in stages or as input for summary/podcast)
                needs_translation = 'translation' in stages or 'summary' in stages or 'podcast' in stages
                translated = transcript  # Default to transcript if no translation needed

                if needs_translation:
                    # Check cache first
                    cached_trans = load_cached_stage('translation', config, hardware_info, cache_base)
                    if cached_trans:
                        log_fn("Using cached translation", 'detail')
                        translated = cached_trans['output']['translated_text']
                        if 'translation' in stages:
                            exp_result['stages_data']['translation'] = cached_trans
                            exp_result['stages_data']['translation']['from_cache'] = True
                            save_experiment_output_fn(exp_id, 'translation', 'output', translated)
                    else:
                        log_fn("Running Translation...", 'progress')
                        trans_result = run_translation_stage_fn(transcript, config, load_llm_fn, StageMetricsCollector, log_fn)
                        translated = trans_result['output']['translated_text']

                        # Cache the result
                        save_stage_cache('translation', config, hardware_info,
                                         trans_result['output'], trans_result['metrics'], cache_base)
                        log_fn("Translation cached", 'detail')

                        if 'translation' in stages:
                            exp_result['stages_data']['translation'] = trans_result
                            save_experiment_output_fn(exp_id, 'translation', 'output', translated)
                            m = trans_result['metrics']
                            log_fn(f"Translation complete: {trans_result['output']['translated_words']} words in {m['time_seconds']:.1f}s", 'success')

                # --- SUMMARY ---
                summary = ''
                if 'summary' in stages:
                    # Check cache first
                    cached_sum = load_cached_stage('summary', config, hardware_info, cache_base)
                    if cached_sum:
                        log_fn("Using cached summary", 'detail')
                        summary = cached_sum['output']['summary']
                        exp_result['stages_data']['summary'] = cached_sum
                        exp_result['stages_data']['summary']['from_cache'] = True
                        save_experiment_output_fn(exp_id, 'summary', 'output', summary)
                    else:
                        log_fn("Running Summary...", 'progress')
                        sum_result = run_summary_stage_fn(translated, config, load_llm_fn, StageMetricsCollector, log_fn)
                        summary = sum_result['output']['summary']

                        # Cache the result
                        save_stage_cache('summary', config, hardware_info,
                                         sum_result['output'], sum_result['metrics'], cache_base)
                        log_fn("Summary cached", 'detail')

                        exp_result['stages_data']['summary'] = sum_result
                        save_experiment_output_fn(exp_id, 'summary', 'output', summary)
                        m = sum_result['metrics']
                        log_fn(f"Summary complete: {sum_result['output']['bullet_count']} bullets in {m['time_seconds']:.1f}s", 'success')

                # --- PODCAST ---
                if 'podcast' in stages:
                    # Check cache first
                    cached_pod = load_cached_stage('podcast', config, hardware_info, cache_base)
                    if cached_pod:
                        log_fn("Using cached podcast", 'detail')
                        exp_result['stages_data']['podcast'] = cached_pod
                        exp_result['stages_data']['podcast']['from_cache'] = True
                        save_experiment_output_fn(exp_id, 'podcast', 'script', cached_pod['output']['script'])
                    else:
                        log_fn("Running Podcast Script...", 'progress')
                        pod_result = run_podcast_stage_fn(translated, summary, config, load_llm_fn, StageMetricsCollector, log_fn)

                        # Cache the result
                        save_stage_cache('podcast', config, hardware_info,
                                         pod_result['output'], pod_result['metrics'], cache_base)
                        log_fn("Podcast cached", 'detail')

                        exp_result['stages_data']['podcast'] = pod_result
                        save_experiment_output_fn(exp_id, 'podcast', 'script', pod_result['output']['script'])
                        m = pod_result['metrics']
                        log_fn(f"Podcast complete: {pod_result['output']['total_lines']} lines in {m['time_seconds']:.1f}s", 'success')

            exp_result['completed_at'] = datetime.now().isoformat()
            exp_result['hardware_snapshot_end'] = get_memory_snapshot()

            results['experiments'][exp_id] = exp_result
            save_results_fn(results)
            log_fn("Saved to Drive", 'success')

        except Exception as e:
            log_fn(f"ERROR: {e}", 'error')
            traceback.print_exc()
            results['experiments'][exp_id] = {
                'id': exp_id,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'completed_at': datetime.now().isoformat()
            }
            save_results_fn(results)

        clear_gpu_memory()

    # Close progress bar
    pbar.close()

    unload_all_models()

    log_fn("Evaluation Complete", 'header')
    return results
