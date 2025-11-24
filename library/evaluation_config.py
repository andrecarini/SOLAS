"""
Experiment configuration and generation for SOLAS evaluation.
"""

from typing import Dict, Any, List


# Model options
ASR_MODELS = [
    'openai/whisper-tiny',
    'openai/whisper-small',
    'openai/whisper-large-v3',
]

LLM_MODELS = [
    'Qwen/Qwen2-0.5B-Instruct',
    'Qwen/Qwen2-1.5B-Instruct',
    'microsoft/phi-3-mini-4k-instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
]

# Summary modes
SUMMARY_MODES = ['greedy', 'sampled']

# Baseline configuration
BASELINE = {
    'asr_model_id': 'openai/whisper-large-v3',
    'llm_model_id': 'microsoft/phi-3-mini-4k-instruct',
    'quantization': '4-bit',
    'chunk_size_chars': 2000,
    'repetition_penalty': None,
    'source_language': 'Portuguese',
    'target_language': 'English',
    'summary_mode': 'greedy',
    'translation_max_new_tokens': 1024,
    'summary_max_new_tokens': 512,
    'podcast_max_new_tokens': 1024,
    'podcast_creativity_temp': 0.3,
}

def generate_experiments() -> List[Dict[str, Any]]:
    """Generate all experiment configurations."""
    experiments = []

    # EXPERIMENT 1: ASR Model Comparison
    for asr_model in ASR_MODELS:
        experiments.append({
            'id': f'asr_{asr_model.split("/")[-1]}',
            'experiment_type': 'asr_model',
            'description': f'ASR: {asr_model.split("/")[-1]}',
            'stages': ['asr'],
            'config': {**BASELINE, 'asr_model_id': asr_model},
        })

    # EXPERIMENT 2: Quantization Impact
    for llm_model in LLM_MODELS:
        for quant in [None, '4-bit']:
            quant_label = 'none' if quant is None else '4bit'
            experiments.append({
                'id': f'quant_{llm_model.split("/")[-1]}_{quant_label}',
                'experiment_type': 'quantization',
                'description': f'Quant: {llm_model.split("/")[-1]} / {quant_label}',
                'stages': ['translation', 'summary', 'podcast'],
                'config': {**BASELINE, 'llm_model_id': llm_model, 'quantization': quant},
            })

    # EXPERIMENT 3: Repetition Penalty Impact
    # Only test smallest (Qwen2-0.5B) and largest (Mistral-7B) models
    penalty_test_models = [
        'Qwen/Qwen2-0.5B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3',
    ]
    for llm_model in penalty_test_models:
        for penalty in [None, 1.2]:
            penalty_label = 'none' if penalty is None else str(penalty)
            experiments.append({
                'id': f'penalty_{llm_model.split("/")[-1]}_{penalty_label}',
                'experiment_type': 'repetition_penalty',
                'description': f'Penalty: {llm_model.split("/")[-1]} / {penalty_label}',
                'stages': ['translation', 'summary', 'podcast'],
                'config': {**BASELINE, 'llm_model_id': llm_model, 'repetition_penalty': penalty},
            })

    # EXPERIMENT 4: Summary Mode Impact (only summary stage - uses cached translation)
    # Uses Phi-3-mini without quantization (quantization degrades Phi-3 output)
    for summary_mode in SUMMARY_MODES:
        experiments.append({
            'id': f'summary_mode_{summary_mode}',
            'experiment_type': 'summary_mode',
            'description': f'Summary mode: {summary_mode} (phi-3-mini, no quant)',
            'stages': ['summary'],  # Only summary - translation from cache
            'config': {**BASELINE, 'summary_mode': summary_mode, 'quantization': None},
        })

    # EXPERIMENT 5: Chunk Size Impact
    # Uses Phi-3-mini without quantization (quantization degrades Phi-3 output)
    for chunk_size in [2000, 4000]:
        experiments.append({
            'id': f'chunk_{chunk_size}',
            'experiment_type': 'chunk_size',
            'description': f'Chunk size: {chunk_size} chars (phi-3-mini, no quant)',
            'stages': ['translation', 'summary', 'podcast'],
            'config': {**BASELINE, 'chunk_size_chars': chunk_size, 'quantization': None},
        })

    # EXPERIMENT 6: Temperature Impact
    for temp in [0.2, 0.5]:
        experiments.append({
            'id': f'temp_{temp}',
            'experiment_type': 'temperature',
            'description': f'Temperature: {temp} (Mistral-7B)',
            'stages': ['translation', 'summary', 'podcast'],
            'config': {
                **BASELINE,
                'llm_model_id': 'mistralai/Mistral-7B-Instruct-v0.3',
                'podcast_creativity_temp': temp,
            },
        })

    return experiments


def sort_experiments_for_efficiency(experiments: List[Dict]) -> List[Dict]:
    """
    Sort experiments to minimize model loading overhead.
    Groups by: ASR experiments first, then LLM experiments grouped by model+quantization.
    """
    def sort_key(exp):
        config = exp['config']
        stages = exp['stages']

        # ASR-only experiments first (group 0)
        if stages == ['asr']:
            return (0, config.get('asr_model_id', ''), '', '')

        # LLM experiments grouped by model+quantization (group 1)
        return (
            1,
            config.get('llm_model_id', ''),
            str(config.get('quantization', '')),
            exp['id'],
        )

    return sorted(experiments, key=sort_key)
