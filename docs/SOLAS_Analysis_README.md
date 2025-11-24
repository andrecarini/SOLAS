# SOLAS Analysis Module

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrecarini/SOLAS/blob/main/SOLAS_Analysis.ipynb)

The Analysis module visualizes and analyzes evaluation results from completed SOLAS experiments. **Does not require a GPU.**

## Overview

The Analysis module is designed for results visualization:

- **No GPU required**: Analysis runs entirely on CPU, making it suitable for any Colab runtime or local machine.
- **Interactive displays**: Visual tables, scrollable text boxes, and side-by-side comparisons.
- **Quality metrics**: Automated degeneration detection using n-gram repetition, lexical diversity, and character repetition.
- **JSON export**: Download structured data for further analysis or LaTeX table generation.

## Prerequisites

Before using the Analysis notebook, you must:

1. Run experiments using the **[SOLAS_Evaluation.ipynb](../SOLAS_Evaluation.ipynb)** notebook
2. Have results saved to Google Drive (Colab) or local storage

## Requirements

### Hardware

No GPU required. The Analysis notebook runs entirely on CPU.

### Software

#### Google Colab

No manual installation required. The notebook automatically clones the repository and loads saved results.

#### Local Installation

```bash
# Clone the repository
git clone https://github.com/andrecarini/SOLAS.git
cd SOLAS

# Run Jupyter
jupyter notebook SOLAS_Analysis.ipynb
```

### Storage

The Analysis notebook reads results from:

- **Google Colab**: `MyDrive/SOLAS/evaluation_results/`
- **Local**: `./evaluation_results/`

## Available Analyses

### ASR Model Comparison

Visual comparison of Whisper tiny, small, and large-v3 transcription results:
- Side-by-side transcript outputs
- Quality metrics (word count, lexical diversity)
- Degeneration detection verdicts

### Quantization Impact

Compares 4-bit quantization vs full precision across all LLMs:
- Text outputs for translation, summary, and podcast stages
- Detailed metrics tables with n-gram repetition analysis
- VRAM savings comparison

### Repetition Penalty Impact

Compares repetition penalty (None vs 1.2) on Qwen2-0.5B and Mistral-7B:
- Effect on output quality and degeneration patterns
- Stage-by-stage metrics comparison
- Download button for JSON export

### Summary Mode Impact

Compares greedy and sampled summary modes:
- Summary outputs with bullet point counts
- Quality metrics for each mode
- Download button for JSON export

### Chunk Size Impact

Compares 2000 vs 4000 character chunks:
- Effect on translation, summary, and podcast quality
- Processing time comparison
- Download button for JSON export

### Temperature Impact

Compares temperature settings (0.2/0.5) for podcast generation:
- Podcast script outputs at different creativity levels
- Quality metrics comparison
- Download button for JSON export

## Quality Metrics

Each stage output is evaluated using automated metrics:

- **Lexical diversity**: Ratio of unique words to total words. Lower values indicate repetition.
- **Peak N-gram**: The most repeated 5-gram pattern with repeat count and percentage.
- **Max character repeat**: Longest sequence of repeated characters.
- **Max word repeat**: Longest sequence of repeated words.
- **Max 2-4 gram repeat**: Longest sequences of repeated n-grams.

Verdicts are assigned based on thresholds:
- **OK**: No significant quality issues detected.
- **WARNING**: Minor repetition or low diversity (may still be usable).
- **FAIL**: Severe degeneration (repetition loops, incoherent output).

## Workflow

1. **Run Setup cell**: Clones/updates SOLAS, loads evaluation results
2. **Run analysis cells**: Each cell visualizes one experiment category
3. **Download JSON**: Use download buttons to export data for further analysis

## Tips

- **Missing results**: If "No evaluation results found" appears, run experiments first using the Evaluation notebook.
- **Refresh results**: Re-run the Setup cell to reload results after running new experiments.
- **Local analysis**: Results can be copied from Google Drive to analyze locally without Colab.
