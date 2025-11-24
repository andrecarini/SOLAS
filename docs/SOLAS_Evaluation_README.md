# SOLAS Evaluation Module

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrecarini/SOLAS/blob/main/SOLAS_Evaluation.ipynb)

The Evaluation module runs controlled experiments to benchmark pipeline performance across different configurations, producing data for systematic analysis.

## Overview

The Evaluation module is designed for systematic experimentation:

- **Controlled experiments**: Each experiment varies one parameter while holding others fixed at baseline values.
- **Config-based caching**: Configurations are hashed (including hardware type) to detect duplicates and skip redundant runs.
- **Resumable execution**: Automatically resumes from the last completed experiment after runtime restarts.
- **Persistent storage**: Google Colab sessions are ephemeral; files are lost when the runtime disconnects. The notebook uses Google Drive integration to persist all results immediately after each experiment. Local execution stores files in the working directory.
- **Detailed metrics**: Time, RAM, VRAM, hardware info, and full outputs are recorded for every stage.
- **Separate analysis**: Results visualization is available in the [SOLAS_Analysis.ipynb](../SOLAS_Analysis.ipynb) notebook, which does not require a GPU.

## Requirements

### Hardware

The evaluation requires a GPU with sufficient VRAM:
- **Google Colab T4** (16GB VRAM): Sufficient for all experiments with 4-bit quantization. Full-precision 7B models may cause out-of-memory errors.
- **Google Colab A100** (40GB VRAM): Recommended for full-precision experiments on larger models.
- **Local GPU**: Any NVIDIA GPU with 16GB+ VRAM and CUDA support.

### Software

#### Google Colab

No manual installation required. The notebook automatically clones the repository, installs dependencies, mounts Google Drive, and initializes the evaluation framework.

#### Local Installation

```bash
# Clone the repository
git clone https://github.com/andrecarini/SOLAS.git
cd SOLAS

# Install system dependencies (Ubuntu/Debian)
sudo apt install ffmpeg espeak-ng libsndfile1

# Install Python dependencies
pip install -r library/requirements.txt

# Run Jupyter
jupyter notebook SOLAS_Evaluation.ipynb
```

### Storage

#### Google Drive (Colab)

A Google Drive account is required for persistent storage on Colab. The notebook creates the following structure:

```
MyDrive/
  SOLAS/
    evaluation_results/
      evaluation_results.json    # All experiment metrics
      cached_transcript.json     # Cached ASR transcript
      outputs/                   # Text outputs per experiment
```

#### Local Storage

When running locally, results are stored in the `evaluation_results` subdirectory of the working directory:

```
SOLAS/
  evaluation_results/
    evaluation_results.json
    cached_transcript.json
    outputs/
```

## Experiment Categories

The evaluation runs 21 experiments across 6 categories:

| Category | Purpose | Tests |
|----------|---------|-------|
| ASR Models | Compare Whisper tiny/small/large-v3 | 3 |
| Quantization | None vs 4-bit on all LLMs | 8 |
| Repetition Penalty | None vs 1.2 on Qwen2-0.5B and Mistral-7B | 4 |
| Summary Mode | Greedy/sampled (Phi-3, no quant) | 2 |
| Chunk Size | 2000 vs 4000 chars (Phi-3, no quant) | 2 |
| Temperature | 0.2/0.5 for podcast (Mistral-7B) | 2 |
| **Total** | | **21** |

## Step-by-Step Execution

### Cell 1: Setup & Engine

Prepares the environment:
- Clones or updates the SOLAS repository
- Installs dependencies (may require runtime restart)
- Mounts Google Drive for persistent storage
- Initializes the evaluation framework and configures paths

**Important**: If a restart warning appears after installing dependencies, restart the runtime (**Runtime → Restart session**) and re-run this cell.

### Cell 2: Run All Experiments

Executes all remaining experiments automatically:
- Skips already-completed experiments
- Displays progress bar with experiment count
- Saves results to Drive after each experiment
- Handles runtime disconnects gracefully (resume on re-run)

### Cell 3: View Results

Displays a summary table of all completed experiments with experiment ID, description, time and VRAM usage per stage, and quality verdicts (OK/WARNING/FAIL).

## Results Analysis

After running experiments, use the **[SOLAS_Analysis.ipynb](../SOLAS_Analysis.ipynb)** notebook to visualize and analyze results. The Analysis notebook does not require a GPU and provides:

- ASR Model Comparison
- Quantization Impact Analysis
- Repetition Penalty Impact Analysis
- Summary Mode Impact Analysis
- Chunk Size Impact Analysis
- Temperature Impact Analysis

See [SOLAS_Analysis_README.md](SOLAS_Analysis_README.md) for details.

## Quality Metrics

Each stage output is evaluated using automated metrics:

- **Lexical diversity**: Ratio of unique words to total words. Lower values indicate repetition.
- **Character repetition**: Count of repeated character sequences (e.g., "aaaaaa").
- **N-gram repetition rate**: Percentage of 5-grams that appear multiple times.

Verdicts are assigned based on thresholds:
- **OK**: No significant quality issues detected.
- **WARNING**: Minor repetition or low diversity (may still be usable).
- **FAIL**: Severe degeneration (repetition loops, incoherent output).

## Output Files

Results are stored in Google Drive (or locally if not using Colab):

- `evaluation_results.json`: Complete metrics for all experiments, including configuration, timing, memory usage, and quality assessments.
- `outputs/<exp_id>/`: Text outputs for each experiment (transcript, translation, summary, podcast script).
- `stage_cache/`: Cached stage outputs keyed by configuration hash, enabling reuse across experiments with shared configurations.

## Tips

- **Runtime disconnects**: The notebook automatically resumes from the last saved experiment. Simply re-run cells 1-2.
- **Partial runs**: To run specific experiment categories, modify the experiment list in `library/evaluation_config.py`.
- **Custom audio**: Replace the test audio file and clear the transcript cache to evaluate with different input.
- **Dry run**: Set `dry_run=True` in the "Run All" cell to preview which experiments would run without executing them.
