"""
Notebook interface for SOLAS evaluation.
Provides a simplified API for the Jupyter notebook.
"""

from pathlib import Path
from typing import Dict, Any, Optional

from .evaluation_logging import log
from .evaluation_config import generate_experiments
from .evaluation_stages import run_asr_stage, run_translation_stage, run_summary_stage, run_podcast_stage
from .evaluation_utils import (
    get_hardware_info, StageMetricsCollector,
    load_llm, load_asr,
    load_results as util_load_results,
    save_results as util_save_results,
    save_experiment_output as util_save_experiment_output,
    is_experiment_complete as util_is_experiment_complete,
    load_cached_transcript as util_load_cached_transcript,
    save_cached_transcript as util_save_cached_transcript,
)
from .evaluation_runner import run_evaluation
from .evaluation_display import display_results_summary, export_for_analysis


class EvaluationNotebook:
    """
    Simplified interface for running SOLAS evaluations from a Jupyter notebook.
    Handles all the wiring between library components and notebook state.
    """

    def __init__(self, solas_dir: Path, drive_base: Path):
        """
        Initialize the evaluation notebook interface.

        Args:
            solas_dir: Path to SOLAS repository
            drive_base: Path to Google Drive results directory
        """
        self.solas_dir = solas_dir
        self.drive_base = drive_base
        self.outputs_dir = drive_base / 'outputs'
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # Paths
        self.audio_path = str(solas_dir / 'input_audio_samples' / 'long.ogg')
        self.results_file = drive_base / 'evaluation_results.json'
        self.transcript_cache_file = drive_base / 'cached_transcript.json'

        # Configuration
        self.hardware_info = get_hardware_info()
        self.experiments = generate_experiments()

        # Cached results (so we don't reload from disk every time)
        self._results_cache: Optional[Dict[str, Any]] = None

    def load_results(self) -> Dict[str, Any]:
        """Load results from disk and ensure hardware info is set."""
        results = util_load_results(self.results_file)
        if 'hardware' not in results.get('metadata', {}):
            results.setdefault('metadata', {})['hardware'] = self.hardware_info
        self._results_cache = results
        return results

    def save_results(self, results: Dict[str, Any]):
        """Save results to disk and update cache."""
        util_save_results(results, self.results_file)
        self._results_cache = results

    def run_evaluation(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Run all remaining experiments.

        Args:
            dry_run: If True, only show what would be run

        Returns:
            Results dictionary
        """
        # Wrapper functions for model loading with logging
        def load_llm_with_log(model_id: str, quantization):
            return load_llm(model_id, quantization, log_fn=log)

        def load_asr_with_log(model_id: str):
            return load_asr(model_id, log_fn=log)

        # Wrapper functions for results management
        def save_experiment_output_fn(exp_id: str, stage: str, output_type: str, content: str):
            util_save_experiment_output(exp_id, stage, output_type, content, self.outputs_dir)

        def load_cached_transcript_fn():
            return util_load_cached_transcript(self.transcript_cache_file)

        def save_cached_transcript_fn(transcript: str, asr_model: str, audio_path: str, metrics: Dict):
            util_save_cached_transcript(transcript, asr_model, audio_path, metrics, self.transcript_cache_file)

        # Run evaluation
        results = run_evaluation(
            experiments=self.experiments,
            hardware_info=self.hardware_info,
            audio_path=self.audio_path,
            results_file=self.results_file,
            transcript_cache_file=self.transcript_cache_file,
            outputs_dir=self.outputs_dir,
            load_results_fn=self.load_results,
            save_results_fn=self.save_results,
            save_experiment_output_fn=save_experiment_output_fn,
            is_experiment_complete_fn=util_is_experiment_complete,
            load_cached_transcript_fn=load_cached_transcript_fn,
            save_cached_transcript_fn=save_cached_transcript_fn,
            run_asr_stage_fn=run_asr_stage,
            run_translation_stage_fn=run_translation_stage,
            run_summary_stage_fn=run_summary_stage,
            run_podcast_stage_fn=run_podcast_stage,
            load_asr_fn=load_asr_with_log,
            load_llm_fn=load_llm_with_log,
            StageMetricsCollector=StageMetricsCollector,
            log_fn=log,
            dry_run=dry_run
        )

        self._results_cache = results
        return results

    def display_results(self):
        """Display evaluation results summary."""
        if self._results_cache is None:
            self._results_cache = self.load_results()
        display_results_summary(self._results_cache, self.results_file, self.outputs_dir, log)

    def export_results(self):
        """Export results as CSV files."""
        if self._results_cache is None:
            self._results_cache = self.load_results()
        export_for_analysis(self._results_cache, self.drive_base, log)

    def print_setup_info(self):
        """Print setup information."""
        log("Setup Complete", 'header')
        log(f"SOLAS directory: {self.solas_dir}", 'info')
        log(f"Results file: {self.results_file}", 'info')
        log(f"Outputs directory: {self.outputs_dir}", 'info')

        log(f"\nTotal experiments defined: {len(self.experiments)}", 'info')

        exp_counts = {}
        for exp in self.experiments:
            exp_type = exp['experiment_type']
            exp_counts[exp_type] = exp_counts.get(exp_type, 0) + 1

        log("Breakdown:", 'info')
        for exp_type, count in exp_counts.items():
            log(f"  - {exp_type}: {count} tests", 'detail')

        log(f"\nTo preview: evaluation.run_evaluation(dry_run=True)", 'info')
        log(f"To run all: evaluation.run_evaluation()", 'info')
