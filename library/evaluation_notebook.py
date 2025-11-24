"""
Notebook interface for SOLAS evaluation.
Provides a simplified API for the Jupyter notebook.

Note: Heavy imports (evaluation_stages, evaluation_runner) are done lazily
to allow the Analysis notebook to load without requiring ML dependencies
(torch, transformers, etc.). The Analysis notebook only displays results
and doesn't need to run the pipeline.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

from .evaluation_logging import log
from .evaluation_config import generate_experiments
from .evaluation_utils import (
    get_hardware_info,
    load_results as util_load_results,
    save_results as util_save_results,
    save_experiment_output as util_save_experiment_output,
    is_experiment_complete as util_is_experiment_complete,
    load_cached_transcript as util_load_cached_transcript,
    save_cached_transcript as util_save_cached_transcript,
    setup_gdrive_mount,
)
from .evaluation_display import (
    display_results_summary,
    export_for_analysis,
    run_thesis_analysis,
    setup_notebooklm_comparison,
    display_asr_analysis,
    display_quantization_analysis,
    display_repetition_penalty_analysis,
    display_summary_mode_analysis,
    display_chunk_size_analysis,
    display_temperature_analysis,
)


class EvaluationNotebook:
    """
    Simplified interface for running SOLAS evaluations from a Jupyter notebook.
    Handles all the wiring between library components and notebook state.
    """

    def __init__(
        self,
        solas_dir: Optional[Path] = None,
        use_gdrive: Optional[bool] = None,
        gdrive_mount_point: str = '/gdrive',
        gdrive_folder: str = 'SOLAS',
        gdrive_symlink: str = '/content/gdrive',
        local_dir: str = './evaluation_results'
    ):
        """
        Initialize the evaluation notebook interface.

        Args:
            solas_dir: Path to SOLAS repository (default: ./SOLAS or /content/SOLAS)
            use_gdrive: Use Google Drive for storage (default: auto-detect Colab)
            gdrive_mount_point: Where to mount Google Drive (default: /gdrive)
            gdrive_folder: Folder name in Google Drive MyDrive (default: SOLAS)
            gdrive_symlink: Symlink path for easy access (default: /content/gdrive)
            local_dir: Local directory for results when not using Google Drive
        """
        # Auto-detect environment
        in_colab = 'google.colab' in sys.modules
        if use_gdrive is None:
            use_gdrive = in_colab

        # Determine SOLAS directory
        if solas_dir is None:
            solas_dir = Path('/content/SOLAS' if in_colab else './SOLAS')
        self.solas_dir = Path(solas_dir)

        # Setup storage based on environment
        if use_gdrive:
            # Mount Google Drive and get symlink path
            symlink_path = setup_gdrive_mount(
                mount_point=gdrive_mount_point,
                folder_name=gdrive_folder,
                symlink_path=gdrive_symlink
            )
            if symlink_path:
                self.drive_base = symlink_path / 'evaluation_results'
            else:
                # Fallback if mounting failed
                log("Google Drive mounting failed, using local storage", 'warning')
                self.drive_base = Path(local_dir)
        else:
            # Use local storage
            self.drive_base = Path(local_dir)

        # Create output directories
        self.outputs_dir = self.drive_base / 'outputs'
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # Paths
        self.audio_path = str(self.solas_dir / 'input_audio_samples' / 'long.ogg')
        self.results_file = self.drive_base / 'evaluation_results.json'
        self.transcript_cache_file = self.drive_base / 'cached_transcript.json'

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

        Note: Heavy imports are done lazily here to allow Analysis notebook
        to load without ML dependencies (only needed for running experiments,
        not viewing results).
        """
        # Lazy imports - only needed when actually running experiments
        from .evaluation_stages import (
            run_asr_stage, run_translation_stage,
            run_summary_stage, run_podcast_stage
        )
        from .evaluation_runner import run_evaluation as _run_evaluation
        from .evaluation_utils import load_llm, load_asr, StageMetricsCollector

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
        results = _run_evaluation(
            experiments=self.experiments,
            hardware_info=self.hardware_info,
            audio_path=self.audio_path,
            results_file=self.results_file,
            transcript_cache_file=self.transcript_cache_file,
            outputs_dir=self.outputs_dir,
            cache_base=self.drive_base,
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

    def run_thesis_analysis(
        self,
        solas_podcast_path: Optional[Path] = None,
        notebooklm_podcast_path: Optional[Path] = None,
    ):
        """
        Run comprehensive analysis for thesis inclusion.
        Generates plots, LaTeX tables, and optional NotebookLM comparison.

        Args:
            solas_podcast_path: Optional path to SOLAS podcast script for comparison
            notebooklm_podcast_path: Optional path to NotebookLM transcript for comparison
        """
        if self._results_cache is None:
            self._results_cache = self.load_results()

        return run_thesis_analysis(
            results=self._results_cache,
            drive_base=self.drive_base,
            log_fn=log,
            solas_podcast_path=solas_podcast_path,
            notebooklm_podcast_path=notebooklm_podcast_path,
        )

    def setup_notebooklm_comparison(self):
        """
        Set up directories and instructions for NotebookLM comparison.
        Creates folder structure and copies SOLAS outputs.
        """
        if self._results_cache is None:
            self._results_cache = self.load_results()

        return setup_notebooklm_comparison(
            results=self._results_cache,
            drive_base=self.drive_base,
            log_fn=log,
        )

    # =========================================================================
    # Analysis display methods
    # =========================================================================

    def asr_analysis(self):
        """Display ASR model comparison analysis."""
        display_asr_analysis(self)

    def quantization_analysis(self):
        """Display quantization impact analysis."""
        display_quantization_analysis(self)

    def repetition_penalty_analysis(self):
        """Display repetition penalty impact analysis."""
        display_repetition_penalty_analysis(self)

    def summary_mode_analysis(self):
        """Display summary mode impact analysis."""
        display_summary_mode_analysis(self)

    def chunk_size_analysis(self):
        """Display chunk size impact analysis."""
        display_chunk_size_analysis(self)

    def temperature_analysis(self):
        """Display temperature impact analysis."""
        display_temperature_analysis(self)
