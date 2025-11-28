"""
Comprehensive analysis and visualization for SOLAS evaluation results.
Generates plots, tables, and comparison metrics for thesis inclusion.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

# Optional imports - gracefully handle missing dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# =============================================================================
# QUALITY METRICS COMPUTATION
# =============================================================================

def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate between reference and hypothesis transcripts.
    WER = (S + D + I) / N where S=substitutions, D=deletions, I=insertions, N=reference words

    Args:
        reference: Reference transcript (e.g., from NotebookLM)
        hypothesis: Hypothesis transcript (e.g., from Whisper)

    Returns:
        WER as a float (0.0 = perfect, 1.0 = 100% error)
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Use dynamic programming for edit distance
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[n][m] / n if n > 0 else 0.0


def compute_rouge_l(reference: str, candidate: str) -> float:
    """
    Compute ROUGE-L score (longest common subsequence based).

    Args:
        reference: Reference text
        candidate: Candidate text to evaluate

    Returns:
        ROUGE-L F1 score (0.0 to 1.0)
    """
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()

    if not ref_words or not cand_words:
        return 0.0

    # LCS using dynamic programming
    n, m = len(ref_words), len(cand_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == cand_words[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_length = dp[n][m]

    # Compute precision, recall, F1
    precision = lcs_length / m if m > 0 else 0
    recall = lcs_length / n if n > 0 else 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return f1


def compute_rouge_1(reference: str, candidate: str) -> float:
    """
    Compute ROUGE-1 score (unigram overlap).

    Args:
        reference: Reference text
        candidate: Candidate text to evaluate

    Returns:
        ROUGE-1 F1 score (0.0 to 1.0)
    """
    ref_words = set(reference.lower().split())
    cand_words = set(candidate.lower().split())

    if not ref_words or not cand_words:
        return 0.0

    overlap = len(ref_words & cand_words)
    precision = overlap / len(cand_words)
    recall = overlap / len(ref_words)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return f1


def compute_repetition_rate(text: str, n: int = 3) -> float:
    """
    Compute repetition rate as percentage of repeated n-grams.

    Args:
        text: Text to analyze
        n: N-gram size (default 3 for trigrams)

    Returns:
        Repetition rate (0.0 to 1.0, higher = more repetition)
    """
    words = text.lower().split()
    if len(words) < n:
        return 0.0

    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))

    # Repetition rate = 1 - (unique / total)
    return 1 - (unique / total) if total > 0 else 0.0


def compute_lexical_diversity(text: str) -> float:
    """
    Compute lexical diversity (Type-Token Ratio).

    Args:
        text: Text to analyze

    Returns:
        Lexical diversity (0.0 to 1.0, higher = more diverse)
    """
    words = text.lower().split()
    if not words:
        return 0.0

    return len(set(words)) / len(words)


def compute_unique_ngram_ratio(text: str, n: int = 3) -> float:
    """
    Compute ratio of unique n-grams to total n-grams.

    Args:
        text: Text to analyze
        n: N-gram size

    Returns:
        Unique n-gram ratio (0.0 to 1.0)
    """
    words = text.lower().split()
    if len(words) < n:
        return 1.0

    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 1.0


def is_garbage_output(text: str, rep_threshold: float = 0.30) -> bool:
    """
    Detect if output is degenerate/garbage (excessive repetition).

    Args:
        text: Text to analyze
        rep_threshold: Repetition rate threshold (default 30%)

    Returns:
        True if output appears to be garbage
    """
    rep_rate = compute_repetition_rate(text, n=3)
    return rep_rate > rep_threshold


def compute_coherence_score(chunks: List[str]) -> float:
    """
    Compute coherence score between adjacent chunks using word overlap.
    Higher score indicates better flow between chunks.

    Args:
        chunks: List of text chunks

    Returns:
        Average coherence score (0.0 to 1.0)
    """
    if len(chunks) < 2:
        return 1.0

    coherence_scores = []
    for i in range(len(chunks) - 1):
        words1 = set(chunks[i].lower().split())
        words2 = set(chunks[i+1].lower().split())

        if words1 and words2:
            # Jaccard similarity
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            coherence_scores.append(intersection / union if union > 0 else 0)

    return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 1.0


def compute_redundancy_rate(chunks: List[str], n: int = 3) -> float:
    """
    Compute redundancy rate across chunks (repeated n-grams across chunks).

    Args:
        chunks: List of text chunks
        n: N-gram size

    Returns:
        Redundancy rate (0.0 to 1.0, higher = more redundancy)
    """
    if len(chunks) < 2:
        return 0.0

    all_ngrams = []
    for chunk in chunks:
        words = chunk.lower().split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    # Count how many n-grams appear in multiple chunks
    from collections import Counter
    ngram_counts = Counter(all_ngrams)
    repeated = sum(1 for count in ngram_counts.values() if count > 1)

    return repeated / len(ngram_counts) if ngram_counts else 0.0


# =============================================================================
# DATA LOADING AND TRANSFORMATION
# =============================================================================

def load_results_as_dataframe(results: Dict[str, Any]) -> 'pd.DataFrame':
    """Convert evaluation results dict to a pandas DataFrame."""
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for analysis. Install with: pip install pandas")

    rows = []
    for exp_id, exp in results.get('experiments', {}).items():
        if 'error' in exp:
            continue

        cfg = exp.get('config', {})
        sd = exp.get('stages_data', {})
        hw = exp.get('hardware_info', {})

        # Extract metrics from each stage
        asr_m = sd.get('asr', {}).get('metrics', {})
        asr_o = sd.get('asr', {}).get('output', {})
        trans_m = sd.get('translation', {}).get('metrics', {})
        trans_o = sd.get('translation', {}).get('output', {})
        sum_m = sd.get('summary', {}).get('metrics', {})
        sum_o = sd.get('summary', {}).get('output', {})
        pod_m = sd.get('podcast', {}).get('metrics', {})
        pod_o = sd.get('podcast', {}).get('output', {})

        # Calculate derived metrics
        total_llm_time = (
            trans_m.get('time_seconds', 0) +
            sum_m.get('time_seconds', 0) +
            pod_m.get('time_seconds', 0)
        )
        max_vram = max(
            asr_m.get('vram_peak_gb', 0),
            trans_m.get('vram_peak_gb', 0),
            sum_m.get('vram_peak_gb', 0),
            pod_m.get('vram_peak_gb', 0)
        )

        # Host balance ratio (1.0 = perfect balance)
        host_a = pod_o.get('host_a_lines', 0)
        host_b = pod_o.get('host_b_lines', 0)
        if host_a + host_b > 0:
            host_balance = min(host_a, host_b) / max(host_a, host_b) if max(host_a, host_b) > 0 else 0
        else:
            host_balance = 0

        rows.append({
            # Identifiers
            'exp_id': exp_id,
            'exp_type': exp.get('experiment_type', ''),
            'description': exp.get('description', ''),

            # Configuration
            'asr_model': cfg.get('asr_model_id', '').split('/')[-1],
            'llm_model': cfg.get('llm_model_id', '').split('/')[-1],
            'quantization': cfg.get('quantization') or 'none',
            'chunk_size': cfg.get('chunk_size_chars', 2000),
            'rep_penalty': cfg.get('repetition_penalty') if cfg.get('repetition_penalty') is not None else 'none',
            'summary_mode': cfg.get('summary_mode', 'greedy'),
            'temperature': cfg.get('podcast_creativity_temp', 0.3),

            # Hardware
            'gpu_type': hw.get('gpu_type', 'Unknown'),

            # ASR metrics
            'asr_time': asr_m.get('time_seconds', 0),
            'asr_vram': asr_m.get('vram_peak_gb', 0),
            'asr_rtf': asr_m.get('realtime_factor', 0),
            'transcript_words': asr_o.get('transcript_words', 0),

            # Translation metrics
            'trans_time': trans_m.get('time_seconds', 0),
            'trans_vram': trans_m.get('vram_peak_gb', 0),
            'trans_words': trans_o.get('translated_words', 0),

            # Summary metrics
            'sum_time': sum_m.get('time_seconds', 0),
            'sum_vram': sum_m.get('vram_peak_gb', 0),
            'sum_bullets': sum_o.get('bullet_count', 0),

            # Podcast metrics
            'pod_time': pod_m.get('time_seconds', 0),
            'pod_vram': pod_m.get('vram_peak_gb', 0),
            'pod_lines': pod_o.get('total_lines', 0),
            'pod_host_a': host_a,
            'pod_host_b': host_b,
            'pod_host_balance': host_balance,

            # Derived metrics
            'total_llm_time': total_llm_time,
            'max_vram': max_vram,
        })

    return pd.DataFrame(rows)


def filter_by_experiment_type(df: 'pd.DataFrame', exp_type: str) -> 'pd.DataFrame':
    """Filter DataFrame to specific experiment type."""
    return df[df['exp_type'] == exp_type].copy()


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def setup_plot_style():
    """Configure matplotlib for thesis-quality plots."""
    if not MATPLOTLIB_AVAILABLE:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (8, 5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_asr_comparison(df: 'pd.DataFrame', output_dir: Path) -> Path:
    """
    Create bar chart comparing ASR models.
    Returns path to saved figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    setup_plot_style()
    asr_df = filter_by_experiment_type(df, 'asr_model')

    if asr_df.empty:
        return None

    # Sort by model size (tiny < small < large)
    size_order = {'whisper-tiny': 0, 'whisper-small': 1, 'whisper-large-v3': 2}
    asr_df = asr_df.copy()
    asr_df['sort_key'] = asr_df['asr_model'].map(lambda x: size_order.get(x, 99))
    asr_df = asr_df.sort_values('sort_key')

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    models = asr_df['asr_model'].tolist()
    x = range(len(models))

    # Plot 1: Time
    axes[0].bar(x, asr_df['asr_time'], color='steelblue', edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace('whisper-', '') for m in models], rotation=15)
    axes[0].set_ylabel('Time (seconds)')
    axes[0].set_title('Transcription Time')

    # Plot 2: Real-time factor
    axes[1].bar(x, asr_df['asr_rtf'], color='coral', edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.replace('whisper-', '') for m in models], rotation=15)
    axes[1].set_ylabel('Real-time Factor (x)')
    axes[1].set_title('Processing Speed')
    axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Real-time')
    axes[1].legend()

    # Plot 3: VRAM
    axes[2].bar(x, asr_df['asr_vram'], color='mediumseagreen', edgecolor='black')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([m.replace('whisper-', '') for m in models], rotation=15)
    axes[2].set_ylabel('VRAM Peak (GB)')
    axes[2].set_title('Memory Usage')

    plt.tight_layout()

    output_path = output_dir / 'asr_comparison.pdf'
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_quantization_impact(df: 'pd.DataFrame', output_dir: Path) -> Path:
    """
    Create grouped bar chart showing quantization impact across models.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    setup_plot_style()
    quant_df = filter_by_experiment_type(df, 'quantization')

    if quant_df.empty:
        return None

    # Pivot for grouped bars
    models = quant_df['llm_model'].unique()
    model_order = ['Qwen2-0.5B-Instruct', 'Qwen2-1.5B-Instruct',
                   'phi-3-mini-4k-instruct', 'Mistral-7B-Instruct-v0.3']
    models = [m for m in model_order if m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = range(len(models))
    width = 0.35

    # Get data for each quantization type
    none_data = quant_df[quant_df['quantization'] == 'none']
    quant_data = quant_df[quant_df['quantization'] == '4-bit']

    # Plot 1: VRAM comparison
    none_vram = [none_data[none_data['llm_model'] == m]['max_vram'].values[0]
                 if len(none_data[none_data['llm_model'] == m]) > 0 else 0 for m in models]
    quant_vram = [quant_data[quant_data['llm_model'] == m]['max_vram'].values[0]
                  if len(quant_data[quant_data['llm_model'] == m]) > 0 else 0 for m in models]

    axes[0].bar([i - width/2 for i in x], none_vram, width, label='No Quantization', color='coral', edgecolor='black')
    axes[0].bar([i + width/2 for i in x], quant_vram, width, label='4-bit NF4', color='steelblue', edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.split('-')[0] for m in models], rotation=15)
    axes[0].set_ylabel('Peak VRAM (GB)')
    axes[0].set_title('Memory Usage by Quantization')
    axes[0].legend()
    axes[0].axhline(y=16, color='red', linestyle='--', alpha=0.5, label='T4 VRAM Limit')

    # Plot 2: Time comparison
    none_time = [none_data[none_data['llm_model'] == m]['total_llm_time'].values[0]
                 if len(none_data[none_data['llm_model'] == m]) > 0 else 0 for m in models]
    quant_time = [quant_data[quant_data['llm_model'] == m]['total_llm_time'].values[0]
                  if len(quant_data[quant_data['llm_model'] == m]) > 0 else 0 for m in models]

    axes[1].bar([i - width/2 for i in x], none_time, width, label='No Quantization', color='coral', edgecolor='black')
    axes[1].bar([i + width/2 for i in x], quant_time, width, label='4-bit NF4', color='steelblue', edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.split('-')[0] for m in models], rotation=15)
    axes[1].set_ylabel('Total LLM Time (seconds)')
    axes[1].set_title('Processing Time by Quantization')
    axes[1].legend()

    plt.tight_layout()

    output_path = output_dir / 'quantization_impact.pdf'
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_best_params_comparison(df: 'pd.DataFrame', output_dir: Path) -> Path:
    """
    Create stacked bar chart comparing best params across models.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    setup_plot_style()
    best_df = filter_by_experiment_type(df, 'best_params')

    if best_df.empty:
        return None

    # Sort by model size
    model_order = ['Qwen2-0.5B-Instruct', 'Qwen2-1.5B-Instruct',
                   'phi-3-mini-4k-instruct', 'Mistral-7B-Instruct-v0.3']
    best_df = best_df.copy()
    best_df['sort_key'] = best_df['llm_model'].map(lambda x: model_order.index(x) if x in model_order else 99)
    best_df = best_df.sort_values('sort_key')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = best_df['llm_model'].tolist()
    short_names = [m.split('-')[0] for m in models]
    x = range(len(models))

    # Plot 1: Stacked time breakdown
    trans_times = best_df['trans_time'].tolist()
    sum_times = best_df['sum_time'].tolist()
    pod_times = best_df['pod_time'].tolist()

    axes[0].bar(x, trans_times, label='Translation', color='steelblue', edgecolor='black')
    axes[0].bar(x, sum_times, bottom=trans_times, label='Summary', color='coral', edgecolor='black')
    axes[0].bar(x, pod_times, bottom=[t+s for t,s in zip(trans_times, sum_times)],
                label='Podcast', color='mediumseagreen', edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, rotation=15)
    axes[0].set_ylabel('Time (seconds)')
    axes[0].set_title('Pipeline Time by Stage')
    axes[0].legend()

    # Plot 2: VRAM and quality proxy
    vram = best_df['max_vram'].tolist()
    bullets = best_df['sum_bullets'].tolist()

    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    bars = ax2.bar(x, vram, color='steelblue', edgecolor='black', alpha=0.7, label='VRAM')
    line = ax2_twin.plot(x, bullets, 'o-', color='coral', linewidth=2, markersize=8, label='Bullets')

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=15)
    ax2.set_ylabel('Peak VRAM (GB)', color='steelblue')
    ax2_twin.set_ylabel('Summary Bullets', color='coral')
    ax2.set_title('VRAM vs Output Quality Proxy')

    # Combined legend
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    plt.tight_layout()

    output_path = output_dir / 'best_params_comparison.pdf'
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_temperature_impact(df: 'pd.DataFrame', output_dir: Path) -> Path:
    """
    Create line plot showing temperature impact on podcast generation.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    setup_plot_style()
    temp_df = filter_by_experiment_type(df, 'temperature')

    if temp_df.empty:
        return None

    temp_df = temp_df.sort_values('temperature')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    temps = temp_df['temperature'].tolist()

    # Plot 1: Line counts
    axes[0].plot(temps, temp_df['pod_lines'], 'o-', color='steelblue', linewidth=2, markersize=8)
    axes[0].set_xlabel('Temperature')
    axes[0].set_ylabel('Total Lines')
    axes[0].set_title('Podcast Length vs Temperature')
    axes[0].set_xticks(temps)

    # Plot 2: Host balance
    axes[1].plot(temps, temp_df['pod_host_balance'], 's-', color='coral', linewidth=2, markersize=8)
    axes[1].set_xlabel('Temperature')
    axes[1].set_ylabel('Host Balance Ratio')
    axes[1].set_title('Dialogue Balance vs Temperature')
    axes[1].set_xticks(temps)
    axes[1].set_ylim(0, 1.1)
    axes[1].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect balance')
    axes[1].legend()

    plt.tight_layout()

    output_path = output_dir / 'temperature_impact.pdf'
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_all_experiments(df: 'pd.DataFrame', output_dir: Path) -> Dict[str, Path]:
    """Generate all plots and return paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plots = {}

    # Generate each plot type
    try:
        path = plot_asr_comparison(df, output_dir)
        if path:
            plots['asr_comparison'] = path
    except Exception as e:
        print(f"Warning: Could not generate ASR plot: {e}")

    try:
        path = plot_quantization_impact(df, output_dir)
        if path:
            plots['quantization_impact'] = path
    except Exception as e:
        print(f"Warning: Could not generate quantization plot: {e}")

    try:
        path = plot_best_params_comparison(df, output_dir)
        if path:
            plots['best_params'] = path
    except Exception as e:
        print(f"Warning: Could not generate best params plot: {e}")

    try:
        path = plot_temperature_impact(df, output_dir)
        if path:
            plots['temperature'] = path
    except Exception as e:
        print(f"Warning: Could not generate temperature plot: {e}")

    return plots


# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================

def generate_latex_asr_table(df: 'pd.DataFrame') -> str:
    """Generate LaTeX table for ASR comparison."""
    asr_df = filter_by_experiment_type(df, 'asr_model')

    if asr_df.empty:
        return "% No ASR data available"

    # Sort by model size
    size_order = {'whisper-tiny': 0, 'whisper-small': 1, 'whisper-large-v3': 2}
    asr_df = asr_df.copy()
    asr_df['sort_key'] = asr_df['asr_model'].map(lambda x: size_order.get(x, 99))
    asr_df = asr_df.sort_values('sort_key')

    lines = [
        r"\begin{table}[h]",
        r"    \caption{ASR model comparison results}",
        r"    \centering",
        r"    \begin{tabular}{l|r|r|r|r}",
        r"        \hline",
        r"        \textbf{Model} & \textbf{Time (s)} & \textbf{RTF} & \textbf{Words} & \textbf{VRAM (GB)} \\",
        r"        \hline",
        r"        \hline",
    ]

    for _, row in asr_df.iterrows():
        model_name = row['asr_model'].replace('whisper-', 'Whisper-').replace('-v3', '-V3')
        lines.append(
            f"        {model_name} & {row['asr_time']:.1f} & {row['asr_rtf']:.2f}x & "
            f"{row['transcript_words']:,} & {row['asr_vram']:.2f} \\\\"
        )

    lines.extend([
        r"        \hline",
        r"    \end{tabular}",
        r"    \label{tbl:asr-results}",
        r"    \legend{Source: The Author}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_latex_quantization_table(df: 'pd.DataFrame') -> str:
    """Generate LaTeX table for quantization comparison."""
    quant_df = filter_by_experiment_type(df, 'quantization')

    if quant_df.empty:
        return "% No quantization data available"

    lines = [
        r"\begin{table}[h]",
        r"    \caption{Quantization impact on VRAM and processing time}",
        r"    \centering",
        r"    \begin{tabular}{l|c|r|r|r}",
        r"        \hline",
        r"        \textbf{Model} & \textbf{Quant} & \textbf{VRAM (GB)} & \textbf{Time (s)} & \textbf{Savings} \\",
        r"        \hline",
        r"        \hline",
    ]

    model_order = ['Qwen2-0.5B-Instruct', 'Qwen2-1.5B-Instruct',
                   'phi-3-mini-4k-instruct', 'Mistral-7B-Instruct-v0.3']

    for model in model_order:
        model_data = quant_df[quant_df['llm_model'] == model]
        if model_data.empty:
            continue

        none_row = model_data[model_data['quantization'] == 'none']
        quant_row = model_data[model_data['quantization'] == '4-bit']

        short_name = model.split('-')[0]

        if not none_row.empty:
            lines.append(
                f"        {short_name} & None & {none_row['max_vram'].values[0]:.2f} & "
                f"{none_row['total_llm_time'].values[0]:.1f} & --- \\\\"
            )

        if not quant_row.empty:
            savings = ""
            if not none_row.empty:
                vram_save = (1 - quant_row['max_vram'].values[0] / none_row['max_vram'].values[0]) * 100
                savings = f"{vram_save:.0f}\\%"
            lines.append(
                f"        {short_name} & 4-bit & {quant_row['max_vram'].values[0]:.2f} & "
                f"{quant_row['total_llm_time'].values[0]:.1f} & {savings} \\\\"
            )

        lines.append(r"        \hline")

    lines.extend([
        r"    \end{tabular}",
        r"    \label{tbl:quantization-results}",
        r"    \legend{Source: The Author}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_latex_best_params_table(df: 'pd.DataFrame') -> str:
    """Generate LaTeX table for best params comparison."""
    best_df = filter_by_experiment_type(df, 'best_params')

    if best_df.empty:
        return "% No best params data available"

    model_order = ['Qwen2-0.5B-Instruct', 'Qwen2-1.5B-Instruct',
                   'phi-3-mini-4k-instruct', 'Mistral-7B-Instruct-v0.3']
    best_df = best_df.copy()
    best_df['sort_key'] = best_df['llm_model'].map(lambda x: model_order.index(x) if x in model_order else 99)
    best_df = best_df.sort_values('sort_key')

    lines = [
        r"\begin{table}[h]",
        r"    \caption{Best parameters comparison across LLMs}",
        r"    \centering",
        r"    \begin{tabular}{l|c|r|r|r|r|r}",
        r"        \hline",
        r"        \textbf{Model} & \textbf{Quant} & \textbf{Trans} & \textbf{Sum} & \textbf{Pod} & \textbf{Total} & \textbf{VRAM} \\",
        r"        \hline",
        r"        \hline",
    ]

    for _, row in best_df.iterrows():
        short_name = row['llm_model'].split('-')[0]
        quant = row['quantization'] if row['quantization'] != 'none' else '---'
        total = row['trans_time'] + row['sum_time'] + row['pod_time']
        lines.append(
            f"        {short_name} & {quant} & {row['trans_time']:.0f}s & {row['sum_time']:.0f}s & "
            f"{row['pod_time']:.0f}s & {total:.0f}s & {row['max_vram']:.1f} \\\\"
        )

    lines.extend([
        r"        \hline",
        r"    \end{tabular}",
        r"    \label{tbl:best-params-results}",
        r"    \legend{Source: The Author}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_all_latex_tables(df: 'pd.DataFrame') -> Dict[str, str]:
    """Generate all LaTeX tables."""
    return {
        'asr': generate_latex_asr_table(df),
        'quantization': generate_latex_quantization_table(df),
        'best_params': generate_latex_best_params_table(df),
    }


# =============================================================================
# NOTEBOOKLM COMPARISON FRAMEWORK
# =============================================================================

@dataclass
class ComparisonMetrics:
    """Metrics for comparing SOLAS vs NotebookLM outputs."""
    source: str
    word_count: int
    unique_words: int
    lexical_diversity: float
    avg_sentence_length: float
    bullet_count: int  # For summaries
    host_a_lines: int  # For podcasts
    host_b_lines: int
    host_balance: float


def analyze_text(text: str, source: str) -> ComparisonMetrics:
    """Analyze a text and compute comparison metrics."""
    words = text.split()
    word_count = len(words)
    unique_words = len(set(w.lower() for w in words))
    lexical_diversity = unique_words / word_count if word_count > 0 else 0

    # Sentence analysis (simple split on period)
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

    # Bullet count (for summaries)
    bullet_count = len(re.findall(r'^[-*]\s', text, re.MULTILINE))

    # Host line analysis (for podcasts)
    host_a_lines = len(re.findall(r'^Host A:', text, re.MULTILINE))
    host_b_lines = len(re.findall(r'^Host B:', text, re.MULTILINE))
    total_host_lines = host_a_lines + host_b_lines
    host_balance = min(host_a_lines, host_b_lines) / max(host_a_lines, host_b_lines) if total_host_lines > 0 and max(host_a_lines, host_b_lines) > 0 else 0

    return ComparisonMetrics(
        source=source,
        word_count=word_count,
        unique_words=unique_words,
        lexical_diversity=lexical_diversity,
        avg_sentence_length=avg_sentence_length,
        bullet_count=bullet_count,
        host_a_lines=host_a_lines,
        host_b_lines=host_b_lines,
        host_balance=host_balance,
    )


def compare_outputs(solas_text: str, notebooklm_text: str, output_type: str = 'podcast') -> Dict[str, Any]:
    """
    Compare SOLAS output with NotebookLM output.

    Args:
        solas_text: Text generated by SOLAS
        notebooklm_text: Text from NotebookLM (transcript of audio or briefing doc)
        output_type: 'podcast' or 'summary'

    Returns:
        Dictionary with comparison metrics and analysis
    """
    solas_metrics = analyze_text(solas_text, 'SOLAS')
    nlm_metrics = analyze_text(notebooklm_text, 'NotebookLM')

    # Compute relative metrics
    length_ratio = solas_metrics.word_count / nlm_metrics.word_count if nlm_metrics.word_count > 0 else 0
    diversity_diff = solas_metrics.lexical_diversity - nlm_metrics.lexical_diversity

    return {
        'solas': solas_metrics.__dict__,
        'notebooklm': nlm_metrics.__dict__,
        'comparison': {
            'length_ratio': length_ratio,
            'diversity_difference': diversity_diff,
            'solas_more_verbose': length_ratio > 1,
            'solas_more_diverse': diversity_diff > 0,
        }
    }


def generate_comparison_latex_table(comparison: Dict[str, Any]) -> str:
    """Generate LaTeX table comparing SOLAS and NotebookLM."""
    solas = comparison['solas']
    nlm = comparison['notebooklm']

    lines = [
        r"\begin{table}[h]",
        r"    \caption{SOLAS vs NotebookLM output comparison}",
        r"    \centering",
        r"    \begin{tabular}{l|r|r}",
        r"        \hline",
        r"        \textbf{Metric} & \textbf{SOLAS} & \textbf{NotebookLM} \\",
        r"        \hline",
        r"        \hline",
        f"        Word Count & {solas['word_count']:,} & {nlm['word_count']:,} \\\\",
        f"        Unique Words & {solas['unique_words']:,} & {nlm['unique_words']:,} \\\\",
        f"        Lexical Diversity & {solas['lexical_diversity']:.3f} & {nlm['lexical_diversity']:.3f} \\\\",
        f"        Avg Sentence Length & {solas['avg_sentence_length']:.1f} & {nlm['avg_sentence_length']:.1f} \\\\",
    ]

    if solas['host_a_lines'] > 0 or nlm['host_a_lines'] > 0:
        lines.extend([
            f"        Host A Lines & {solas['host_a_lines']} & {nlm['host_a_lines']} \\\\",
            f"        Host B Lines & {solas['host_b_lines']} & {nlm['host_b_lines']} \\\\",
            f"        Host Balance & {solas['host_balance']:.2f} & {nlm['host_balance']:.2f} \\\\",
        ])

    if solas['bullet_count'] > 0 or nlm['bullet_count'] > 0:
        lines.append(f"        Bullet Points & {solas['bullet_count']} & {nlm['bullet_count']} \\\\")

    lines.extend([
        r"        \hline",
        r"    \end{tabular}",
        r"    \label{tbl:solas-vs-notebooklm}",
        r"    \legend{Source: The Author}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def create_blind_evaluation_template(solas_text: str, notebooklm_text: str, output_dir: Path) -> Path:
    """
    Create a blind evaluation template with randomized order.
    Returns path to the template file.
    """
    import random

    texts = [('A', solas_text), ('B', notebooklm_text)]
    random.shuffle(texts)

    # Store mapping for later reveal
    mapping = {texts[0][0]: 'SOLAS' if texts[0][1] == solas_text else 'NotebookLM',
               texts[1][0]: 'SOLAS' if texts[1][1] == solas_text else 'NotebookLM'}

    template = f"""# Blind Evaluation Template

## Instructions
Rate each output on a scale of 1-5 for each criterion.
Do NOT look at the answer key until you have completed your ratings.

## Evaluation Criteria

| Criterion | Description |
|-----------|-------------|
| Accuracy | Factual correctness of the content |
| Completeness | Coverage of key points from source material |
| Fluency | Natural language flow and readability |
| Structure | Logical organization and progression |
| Engagement | How interesting/engaging the content is |

---

## Output A

```
{texts[0][1][:2000]}...
```

### Ratings for Output A

| Criterion | Rating (1-5) | Notes |
|-----------|--------------|-------|
| Accuracy | | |
| Completeness | | |
| Fluency | | |
| Structure | | |
| Engagement | | |

**Overall Score A**: ___ / 25

---

## Output B

```
{texts[1][1][:2000]}...
```

### Ratings for Output B

| Criterion | Rating (1-5) | Notes |
|-----------|--------------|-------|
| Accuracy | | |
| Completeness | | |
| Fluency | | |
| Structure | | |
| Engagement | | |

**Overall Score B**: ___ / 25

---

## Answer Key (DO NOT READ UNTIL EVALUATION IS COMPLETE)

<details>
<summary>Click to reveal which system produced each output</summary>

- Output A: {mapping['A']}
- Output B: {mapping['B']}

</details>
"""

    output_path = output_dir / 'blind_evaluation_template.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)

    return output_path


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_full_analysis(
    results: Dict[str, Any],
    output_dir: Path,
    solas_podcast: Optional[str] = None,
    notebooklm_podcast: Optional[str] = None,
    log_fn=None
) -> Dict[str, Any]:
    """
    Run complete analysis pipeline.

    Args:
        results: Evaluation results dictionary
        output_dir: Directory for output files
        solas_podcast: Optional SOLAS podcast script for comparison
        notebooklm_podcast: Optional NotebookLM transcript for comparison
        log_fn: Logging function

    Returns:
        Dictionary with paths to all generated files
    """
    if log_fn is None:
        log_fn = lambda msg, level='info': print(f"[{level.upper()}] {msg}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = {
        'plots': {},
        'tables': {},
        'comparison': {},
    }

    # Load data as DataFrame
    log_fn("Loading results as DataFrame...", 'info')
    try:
        df = load_results_as_dataframe(results)
        log_fn(f"Loaded {len(df)} experiments", 'success')
    except Exception as e:
        log_fn(f"Failed to load DataFrame: {e}", 'error')
        return generated_files

    # Generate plots
    if MATPLOTLIB_AVAILABLE:
        log_fn("Generating plots...", 'info')
        plots_dir = output_dir / 'plots'
        plots = plot_all_experiments(df, plots_dir)
        generated_files['plots'] = plots
        log_fn(f"Generated {len(plots)} plots", 'success')
    else:
        log_fn("matplotlib not available, skipping plots", 'warning')

    # Generate LaTeX tables
    log_fn("Generating LaTeX tables...", 'info')
    tables = generate_all_latex_tables(df)
    tables_dir = output_dir / 'latex_tables'
    tables_dir.mkdir(exist_ok=True)

    for name, latex in tables.items():
        table_path = tables_dir / f'{name}_table.tex'
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write(latex)
        generated_files['tables'][name] = table_path

    log_fn(f"Generated {len(tables)} LaTeX tables", 'success')

    # NotebookLM comparison (if data provided)
    if solas_podcast and notebooklm_podcast:
        log_fn("Running NotebookLM comparison...", 'info')
        comparison_dir = output_dir / 'comparison'
        comparison_dir.mkdir(exist_ok=True)

        comparison = compare_outputs(solas_podcast, notebooklm_podcast, 'podcast')

        # Save comparison metrics
        metrics_path = comparison_dir / 'comparison_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2)
        generated_files['comparison']['metrics'] = metrics_path

        # Generate comparison table
        comparison_latex = generate_comparison_latex_table(comparison)
        latex_path = comparison_dir / 'comparison_table.tex'
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(comparison_latex)
        generated_files['comparison']['latex'] = latex_path

        # Create blind evaluation template
        template_path = create_blind_evaluation_template(solas_podcast, notebooklm_podcast, comparison_dir)
        generated_files['comparison']['blind_template'] = template_path

        log_fn("NotebookLM comparison complete", 'success')

    # Save summary report
    log_fn("Saving analysis summary...", 'info')
    summary = {
        'total_experiments': len(df),
        'experiment_types': df['exp_type'].value_counts().to_dict() if PANDAS_AVAILABLE else {},
        'generated_files': {k: {kk: str(vv) for kk, vv in v.items()} if isinstance(v, dict) else str(v)
                          for k, v in generated_files.items()},
    }

    summary_path = output_dir / 'analysis_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    log_fn(f"Analysis complete. Results saved to: {output_dir}", 'success')

    return generated_files
