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


def count_capitalized_words(text: str, exclude_sentence_start: bool = True) -> int:
    """
    Count capitalized words in text.

    Args:
        text: Text to analyze
        exclude_sentence_start: If True, exclude words that appear after sentence-ending
                                punctuation (. ! ?) as these are expected to be capitalized.

    Returns:
        Count of capitalized words (excluding sentence starters if specified)
    """
    import re

    if not text:
        return 0

    words = text.split()
    if not words:
        return 0

    count = 0
    after_sentence_end = True  # First word is always a sentence start

    for word in words:
        # Check if word starts with uppercase letter
        if word and word[0].isupper():
            # Only count if not after sentence-ending punctuation (or if we're not excluding)
            if not exclude_sentence_start or not after_sentence_end:
                count += 1

        # Check if this word ends a sentence (contains . ! ?)
        after_sentence_end = bool(re.search(r'[.!?]$', word))

    return count


def count_missing_capitalizations(text: str) -> int:
    """
    Count words that should be capitalized but aren't (sentence/paragraph starters).

    Args:
        text: Text to analyze

    Returns:
        Count of missing capitalizations at sentence or paragraph starts
    """
    import re

    if not text:
        return 0

    words = text.split()
    if not words:
        return 0

    count = 0
    after_sentence_end = True  # First word should be capitalized

    for word in words:
        # Check if this word should be capitalized (after sentence end) but isn't
        if after_sentence_end and word and word[0].islower():
            count += 1

        # Check if this word ends a sentence (contains . ! ?)
        after_sentence_end = bool(re.search(r'[.!?]$', word))

    return count


def count_acronyms(text: str) -> int:
    """
    Count acronyms (words with multiple consecutive uppercase letters).

    Args:
        text: Text to analyze

    Returns:
        Count of acronyms (e.g., API, HTML, USB)
    """
    import re

    if not text:
        return 0

    # Match words with 2+ consecutive uppercase letters
    acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
    return len(acronyms)


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


def compute_ngram_analysis(text: str, n: int = 5, **kwargs) -> dict:
    """
    Compute n-gram repetition rate for a fixed n value.

    We use n=5 by default because smaller n-grams (words, bigrams, trigrams,
    4-grams) have naturally high repetition rates in normal text due to common
    phrases. For n<5, consecutive repetition detection (find_consecutive_ngram_repetition)
    is used instead, which detects patterns like "the the the" or "I am here I am here".

    Note: Larger n values will always have lower or equal repetition rates compared
    to smaller n values (a repeated 6-gram necessarily contains a repeated 5-gram),
    so only n=5 is checked.

    Args:
        text: Text to analyze
        n: N-gram size (default 5)
        **kwargs: Ignored (for backward compatibility with old min_n/max_n/top_k params)

    Returns:
        dict with:
        - ngram_rates: dict mapping n -> repetition rate (only contains n=5)
        - peak_n: the n value used
        - peak_rate: the repetition rate found
        - significant_peaks: list of (n, rate) tuples where rate > 0.15
        - top_ngrams: list with single dict {n, rate, most_repeated, repeat_count}
    """
    from collections import Counter

    words = text.lower().split()
    word_count = len(words)

    if word_count < n:
        return {
            'ngram_rates': {},
            'peak_n': 0,
            'peak_rate': 0.0,
            'significant_peaks': [],
            'top_ngrams': []
        }

    # Compute n-grams for just n=5
    ngrams = [tuple(words[i:i+n]) for i in range(word_count - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    rate = 1 - (unique / total) if total > 0 else 0.0
    rate = round(rate, 4)

    # Find the most repeated n-gram
    ngram_counts = Counter(ngrams)
    most_common_ngram, most_common_count = ngram_counts.most_common(1)[0] if ngram_counts else ((), 0)

    ngram_rates = {n: rate}
    significant_peaks = [(n, rate)] if rate > 0.15 else []

    top_ngrams = [{
        'n': n,
        'rate': rate,
        'most_repeated': ' '.join(most_common_ngram) if most_common_ngram else '',
        'repeat_count': most_common_count
    }]

    return {
        'ngram_rates': ngram_rates,
        'peak_n': n,
        'peak_rate': rate,
        'significant_peaks': significant_peaks,
        'top_ngrams': top_ngrams
    }


def find_longest_char_repetition(text: str) -> Tuple[str, int]:
    """
    Find the longest consecutive repetition of the same character.

    Args:
        text: Text to analyze

    Returns:
        Tuple of (character, count) for the longest repetition
    """
    if not text:
        return ('', 0)

    max_char = ''
    max_count = 1
    current_char = text[0]
    current_count = 1

    for i in range(1, len(text)):
        if text[i] == current_char:
            current_count += 1
        else:
            if current_count > max_count:
                max_count = current_count
                max_char = current_char
            current_char = text[i]
            current_count = 1

    # Check last run
    if current_count > max_count:
        max_count = current_count
        max_char = current_char

    return (max_char, max_count)


def find_longest_word_repetition(text: str, max_pattern_len: int = 5) -> Tuple[str, int]:
    """
    Find the longest consecutive repetition of word patterns (1 to max_pattern_len words).

    This detects patterns like:
    - Single word: "the the the the"
    - 2-gram: "hello world hello world hello world"
    - Longer: "I am here I am here I am here"

    Args:
        text: Text to analyze
        max_pattern_len: Maximum pattern length in words (default 5)

    Returns:
        Tuple of (pattern_string, repetition_count) for the longest repetition
    """
    words = text.split()
    if len(words) < 2:
        return ("", 0)

    max_pattern, max_count = "", 0

    for pattern_len in range(1, min(max_pattern_len + 1, len(words) // 2)):
        i = 0
        while i < len(words) - pattern_len:
            pattern = tuple(words[i:i + pattern_len])
            count = 1
            j = i + pattern_len
            while j + pattern_len <= len(words):
                if tuple(words[j:j + pattern_len]) == pattern:
                    count += 1
                    j += pattern_len
                else:
                    break
            if count > max_count and count >= 3:
                max_count = count
                max_pattern = " ".join(pattern)
            i += 1

    return (max_pattern, max_count)


def find_consecutive_ngram_repetition(text: str, n: int) -> Tuple[str, int]:
    """
    Find the longest consecutive repetition of exactly n-word patterns.

    Args:
        text: Text to analyze
        n: Exact n-gram size (1=word, 2=bigram, 4=4-gram)

    Returns:
        Tuple of (pattern_string, repetition_count) for the longest repetition
    """
    words = text.split()
    if len(words) < n * 2:
        return ("", 0)

    max_pattern, max_count = "", 0
    i = 0

    while i <= len(words) - n:
        pattern = tuple(words[i:i + n])
        count = 1
        j = i + n
        while j + n <= len(words):
            if tuple(words[j:j + n]) == pattern:
                count += 1
                j += n
            else:
                break
        if count > max_count and count >= 2:
            max_count = count
            max_pattern = " ".join(pattern)
        i += 1

    return (max_pattern, max_count)


def assess_quality(
    text: str,
    ngram_count_fail: int = 6,
    ngram_count_warn: int = 3,
    char_repeat_fail: int = 10,
    char_repeat_warn: int = 5,
    word_repeat_fail: int = 5,
    word_repeat_warn: int = 3
) -> dict:
    """
    Assess text quality with multiple degeneration detection methods:

    1. N-gram repetition analysis (n=5) to detect repeated phrases/paragraphs
    2. Consecutive character repetition (e.g., "aaaaaaa")
    3. Consecutive word/pattern repetition (1-5 words, e.g., "the the the", "I am here I am here")

    For n<5, we use consecutive pattern detection instead of global n-gram analysis
    because smaller n-grams naturally repeat often in normal text.

    Args:
        text: Text to analyze
        ngram_count_fail: Peak n-gram repeat count for FAIL (default 6, so >6 = FAIL)
        ngram_count_warn: Peak n-gram repeat count for WARNING (default 3, so >3 = WARN)
        char_repeat_fail: Character repetition count for FAIL (default 10)
        char_repeat_warn: Character repetition count for WARNING (default 5)
        word_repeat_fail: Word pattern repetition count for FAIL (default 5)
        word_repeat_warn: Word pattern repetition count for WARNING (default 3)

    Returns:
        dict with quality metrics and verdict (OK/WARNING/FAIL)
    """
    if not text or not text.strip():
        return {
            'word_count': 0,
            'lexical_diversity': 0,
            'repetition_rate': 0,
            'peak_ngram_n': 0,
            'peak_ngram_rate': 0,
            'ngram_rates': {},
            'top_ngrams': [],
            'max_char_repeat': 0,
            'max_char': '',
            'max_word_repeat': 0,
            'max_word_pattern': '',
            'is_degenerate': True,
            'verdict': 'NO DATA',
            'fail_reason': 'No text'
        }

    words = text.split()
    word_count = len(words)
    lex_div = compute_lexical_diversity(text)

    # N-gram analysis starting at n=5 (smaller n handled by consecutive detection)
    ngram_analysis = compute_ngram_analysis(text, min_n=5, max_n=40, top_k=5)
    peak_n = ngram_analysis['peak_n']
    peak_rate = ngram_analysis['peak_rate']
    ngram_rates = ngram_analysis['ngram_rates']
    top_ngrams = ngram_analysis['top_ngrams']

    # Standard 5-gram rate (minimum n for global analysis)
    rep_rate_5gram = ngram_rates.get(5, 0.0)

    # Check for consecutive character repetition
    max_char, max_char_count = find_longest_char_repetition(text)

    # Check for consecutive word repetition (any pattern length 1-5)
    max_word_pattern, max_word_count = find_longest_word_repetition(text)

    # Check for consecutive n-gram repetitions at specific sizes
    word1_pattern, word1_count = find_consecutive_ngram_repetition(text, 1)
    bigram_pattern, bigram_count = find_consecutive_ngram_repetition(text, 2)
    gram3_pattern, gram3_count = find_consecutive_ngram_repetition(text, 3)
    gram4_pattern, gram4_count = find_consecutive_ngram_repetition(text, 4)

    # Determine verdict based on all checks
    fail_reasons = []
    warn_reasons = []

    # Get peak n-gram count
    peak_ngram_count = top_ngrams[0]['repeat_count'] if top_ngrams else 0

    # Check peak n-gram repetition count (the most sensitive detector)
    if peak_ngram_count > ngram_count_fail:
        fail_reasons.append(f"5-gram x{peak_ngram_count}")
    elif peak_ngram_count > ngram_count_warn:
        warn_reasons.append(f"5-gram x{peak_ngram_count}")

    # Check character repetition (ignore spaces and newlines)
    if max_char not in [' ', '\n', '\t'] and max_char_count >= char_repeat_fail:
        fail_reasons.append(f"Char '{max_char}' x{max_char_count}")
    elif max_char not in [' ', '\n', '\t'] and max_char_count >= char_repeat_warn:
        warn_reasons.append(f"Char '{max_char}' x{max_char_count}")

    # Check word repetition
    if max_word_count >= word_repeat_fail:
        fail_reasons.append(f"Pattern x{max_word_count}")
    elif max_word_count >= word_repeat_warn:
        warn_reasons.append(f"Pattern x{max_word_count}")

    if fail_reasons:
        verdict = 'FAIL'
        fail_reason = '; '.join(fail_reasons)
        is_degenerate = True
    elif warn_reasons:
        verdict = 'WARNING'
        fail_reason = '; '.join(warn_reasons)
        is_degenerate = False
    else:
        verdict = 'OK'
        fail_reason = ''
        is_degenerate = False

    # Get peak n-gram info (just the top one)
    peak_ngram = top_ngrams[0] if top_ngrams else {'n': 0, 'rate': 0, 'most_repeated': '', 'repeat_count': 0}

    return {
        'word_count': word_count,
        'lexical_diversity': round(lex_div, 4),
        'repetition_rate': rep_rate_5gram,  # Standard n=5 rate
        'peak_ngram_n': peak_n,
        'peak_ngram_rate': peak_rate,
        'peak_ngram_pattern': peak_ngram.get('most_repeated', ''),
        'peak_ngram_count': peak_ngram.get('repeat_count', 0),
        'ngram_rates': ngram_rates,
        'top_ngrams': top_ngrams,
        # Consecutive character repetition
        'max_char_repeat': max_char_count,
        'max_char': max_char if max_char not in [' ', '\n', '\t'] else '',
        # Consecutive word (1-gram) repetition
        'consec_word_count': word1_count,
        'consec_word_pattern': word1_pattern,
        # Consecutive 2-gram repetition
        'consec_2gram_count': bigram_count,
        'consec_2gram_pattern': bigram_pattern,
        # Consecutive 3-gram repetition
        'consec_3gram_count': gram3_count,
        'consec_3gram_pattern': gram3_pattern,
        # Consecutive 4-gram repetition
        'consec_4gram_count': gram4_count,
        'consec_4gram_pattern': gram4_pattern,
        # Legacy fields (kept for compatibility)
        'max_word_repeat': max_word_count,
        'max_word_pattern': max_word_pattern,
        'is_degenerate': is_degenerate,
        'verdict': verdict,
        'fail_reason': fail_reason
    }


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
