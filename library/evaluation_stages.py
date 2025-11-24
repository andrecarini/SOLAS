"""
Stage execution functions for SOLAS evaluation.
These implement each pipeline stage with explicit model management for accurate metrics.
"""

import time
import torch
import numpy as np
from typing import Dict, Any, Tuple
from tqdm.std import tqdm  # Use std (not auto) to avoid CDN widget notice in notebooks
from transformers import GenerationConfig

from .pipeline_stages import load_and_preprocess_audio
from .pipeline_utils import chunk_text


def run_asr_stage(
    config: Dict[str, Any],
    load_asr_fn,
    metrics_collector_cls,
    log_fn,
    audio_path: str
) -> Dict[str, Any]:
    """
    Run ASR transcription with full metrics collection.

    Args:
        config: Experiment configuration
        load_asr_fn: Function to load ASR model
        metrics_collector_cls: StageMetricsCollector class
        log_fn: Logging function
        audio_path: Path to audio file

    Returns:
        Dictionary with stage results, metrics, and outputs
    """
    asr_model_id = config['asr_model_id']
    source_language = config.get('source_language', 'Portuguese')

    lang_code_map = {
        'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
        'Italian': 'it', 'Portuguese': 'pt', 'Russian': 'ru', 'Chinese': 'zh',
        'Japanese': 'ja', 'Korean': 'ko',
    }
    lang_code = lang_code_map.get(source_language, source_language)

    result = {'stage': 'asr', 'config_used': config.copy()}

    # Load audio
    log_fn("Loading and preprocessing audio...", 'detail')
    audio_tensor, sample_rate = load_and_preprocess_audio(audio_path)
    audio_duration = len(audio_tensor.squeeze()) / sample_rate
    result['input'] = {
        'audio_path': audio_path,
        'audio_duration_seconds': audio_duration,
        'sample_rate': sample_rate,
    }

    # Load ASR model (with timing)
    pipe, model_load_time = load_asr_fn(asr_model_id)
    result['model_load_time_seconds'] = model_load_time

    # Run ASR with metrics
    audio_np = audio_tensor.squeeze(0).cpu().numpy().astype(np.float32)

    generate_kwargs = {'task': 'transcribe'}
    if lang_code != 'auto':
        generate_kwargs['language'] = lang_code

    log_fn("Running ASR inference...", 'detail')
    with metrics_collector_cls('asr_inference') as collector:
        asr_result = pipe(audio_np, generate_kwargs=generate_kwargs, return_timestamps=True)

    transcript = asr_result['text'].strip()

    result['metrics'] = collector.metrics
    result['metrics']['realtime_factor'] = collector.metrics['time_seconds'] / audio_duration if audio_duration > 0 else 0
    result['output'] = {
        'transcript': transcript,
        'transcript_chars': len(transcript),
        'transcript_words': len(transcript.split()),
    }

    return result


def run_translation_stage(
    transcript: str,
    config: Dict[str, Any],
    load_llm_fn,
    metrics_collector_cls,
    log_fn
) -> Dict[str, Any]:
    """
    Run translation with full metrics collection.

    Args:
        transcript: Input transcript to translate
        config: Experiment configuration
        load_llm_fn: Function to load LLM
        metrics_collector_cls: StageMetricsCollector class
        log_fn: Logging function

    Returns:
        Dictionary with stage results, metrics, and outputs
    """
    llm_model_id = config['llm_model_id']
    quantization = config.get('quantization')
    chunk_size_chars = config.get('chunk_size_chars', 2000)
    # FIX: Proper handling to allow explicit None
    repetition_penalty = config["repetition_penalty"] if "repetition_penalty" in config else 1.2
    target_language = config.get('target_language', 'English')
    max_new_tokens = config.get('translation_max_new_tokens', 1024)

    result = {'stage': 'translation', 'config_used': config.copy()}
    result['input'] = {
        'transcript_chars': len(transcript),
        'transcript_words': len(transcript.split()),
    }

    # Load LLM
    tokenizer, model, model_load_time = load_llm_fn(llm_model_id, quantization)
    result['model_load_time_seconds'] = model_load_time

    # Chunk text
    chunks = chunk_text(transcript, chunk_size_chars)
    result['input']['num_chunks'] = len(chunks)

    translated_parts = []

    with metrics_collector_cls('translation') as collector:
        for chunk in tqdm(chunks, desc="Translating", unit="chunk", leave=False):
            # Calculate dynamic max_new_tokens based on input length
            # Translation can be up to 1.5x longer than original
            chunk_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
            dynamic_max_tokens = max(max_new_tokens, int(chunk_tokens * 1.5))

            system_prompt = (
                f"You are a professional technical translator. Translate the provided transcript into {target_language}. "
                "Be faithful, precise, and complete. Preserve technical terms whenever possible. "
                "Do not summarize, omit, or add content. Do not include explanations. Output only the translation."
            )
            user_prompt = (
                f"Translate the following transcript segment into {target_language}. "
                "Only output the translation, nothing else.\n\n"
                f"{chunk}"
            )

            messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]

            try:
                input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
            except:
                fallback_prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
                input_ids = tokenizer(fallback_prompt, return_tensors='pt').input_ids

            model_device = next(model.parameters()).device
            input_ids = input_ids.to(model_device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model_device)

            # Build GenerationConfig to ensure params are applied correctly
            trans_gen_config_dict = {
                'max_new_tokens': dynamic_max_tokens,
                'do_sample': False,
                'pad_token_id': tokenizer.eos_token_id,
                'eos_token_id': tokenizer.eos_token_id,
            }
            if repetition_penalty is not None:
                trans_gen_config_dict['repetition_penalty'] = repetition_penalty

            trans_gen_config = GenerationConfig(**trans_gen_config_dict)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=trans_gen_config
                )

            gen_only = output_ids[0, input_ids.shape[1]:]
            translated = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
            translated_parts.append(translated)

    translated_text = '\n'.join(translated_parts).strip()

    result['metrics'] = collector.metrics
    result['output'] = {
        'translated_text': translated_text,
        'translated_chars': len(translated_text),
        'translated_words': len(translated_text.split())
    }

    return result


def run_summary_stage(
    translated_text: str,
    config: Dict[str, Any],
    load_llm_fn,
    metrics_collector_cls,
    log_fn
) -> Dict[str, Any]:
    """
    Run summarization with full metrics collection.

    Process:
    1. Generate partial summaries for each chunk
    2. Normalize bullets
    3. Aggregate partial summaries into final summary
    4. Normalize final bullets

    Args:
        translated_text: Input text to summarize
        config: Experiment configuration
        load_llm_fn: Function to load LLM
        metrics_collector_cls: StageMetricsCollector class
        log_fn: Logging function

    Returns:
        Dictionary with stage results, metrics, and outputs
    """
    llm_model_id = config['llm_model_id']
    quantization = config.get('quantization')
    chunk_size_chars = config.get('chunk_size_chars', 2000)
    # FIX: Proper handling to allow explicit None
    repetition_penalty = config["repetition_penalty"] if "repetition_penalty" in config else 1.2
    target_language = config.get('target_language', 'English')
    max_new_tokens = config.get('summary_max_new_tokens', 512)
    summary_mode = config.get('summary_mode', 'greedy')

    result = {'stage': 'summary', 'config_used': config.copy()}
    result['input'] = {
        'text_chars': len(translated_text),
        'text_words': len(translated_text.split())
    }

    tokenizer, model, model_load_time = load_llm_fn(llm_model_id, quantization)
    result['model_load_time_seconds'] = model_load_time

    chunks = chunk_text(translated_text, chunk_size_chars)
    result['input']['num_chunks'] = len(chunks)

    partial_bullets = []

    # Configure chunk generation based on summary_mode
    if summary_mode == 'sampled' or summary_mode == 'hybrid':
        # Sampled and hybrid both use sampling for initial chunked summarization
        chunk_do_sample = True
        chunk_temperature = 0.2
        chunk_top_p = 0.9
    else:  # greedy
        chunk_do_sample = False
        chunk_temperature = None
        chunk_top_p = None

    # Configure aggregation generation based on summary_mode
    if summary_mode == 'hybrid':
        # Hybrid: sampled for chunks, greedy for aggregation
        agg_do_sample = False
        agg_temperature = None
        agg_top_p = None
    else:
        # Sampled and greedy use the same config for both passes
        agg_do_sample = chunk_do_sample
        agg_temperature = chunk_temperature
        agg_top_p = chunk_top_p

    chunk_prompt_header = (
        f"Extract ONLY the most important key points from the transcript segment below in {target_language}. "
        "Focus on core concepts, definitions, equations, conclusions, and actionable takeaways. "
        f"Write a concise markdown bulleted list (use '-' bullets) in {target_language}. Do not include any preamble or epilogue."
    )

    with metrics_collector_cls('summary') as collector:
        # PHASE 1: Generate partial summaries for each chunk
        for chunk in tqdm(chunks, desc="Summarizing", unit="chunk", leave=False):
            messages = [
                {'role': 'system', 'content': f"You are an expert technical summarizer. Provide summaries in {target_language}."},
                {'role': 'user', 'content': f"{chunk_prompt_header}\n\nTranscript segment:\n\n{chunk}"},
            ]

            try:
                input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
            except:
                fallback_prompt = f"System: You are an expert technical summarizer.\nUser: {chunk_prompt_header}\n\nTranscript segment:\n\n{chunk}\nAssistant:"
                input_ids = tokenizer(fallback_prompt, return_tensors='pt').input_ids

            model_device = next(model.parameters()).device
            input_ids = input_ids.to(model_device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model_device)

            # Build GenerationConfig to ensure params are applied correctly
            # (direct kwargs can be ignored by model's default config)
            # Dynamic token limit based on input chunk size
            chunk_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
            dynamic_max_tokens = max(max_new_tokens, int(chunk_tokens * 0.5))  # Summary is shorter than input
            chunk_gen_config_dict = {
                'max_new_tokens': dynamic_max_tokens,
                'do_sample': chunk_do_sample,
                'pad_token_id': tokenizer.eos_token_id,
                'eos_token_id': tokenizer.eos_token_id,
            }
            if chunk_temperature is not None:
                chunk_gen_config_dict['temperature'] = chunk_temperature
            if chunk_top_p is not None:
                chunk_gen_config_dict['top_p'] = chunk_top_p
            if repetition_penalty is not None:
                chunk_gen_config_dict['repetition_penalty'] = repetition_penalty

            chunk_gen_config = GenerationConfig(**chunk_gen_config_dict)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=chunk_gen_config
                )

            gen_only = output_ids[0, input_ids.shape[1]:]
            partial = tokenizer.decode(gen_only, skip_special_tokens=True).strip()

            # Normalize bullets
            norm_lines = []
            for line in partial.splitlines():
                s = line.strip()
                if not s:
                    continue
                if s.startswith("- "):
                    s = s[2:]
                s = s.replace("**Key Point**:", "").replace("**Key Point**", "").lstrip("- ").strip()
                norm_lines.append(f"- {s}")

            partial_bullets.append("\n".join(norm_lines))

        # Phase 2: Aggregate partial summaries
        aggregation_prompt = (
            f"You will be given multiple markdown bullet lists produced from segments of the same lecture transcript in {target_language}. "
            "Merge and deduplicate them into a single, concise, well-structured set of bullets. "
            f"Prioritize clarity and completeness, keep only the most salient points, and maintain markdown '-' bullets in {target_language}."
        )

        merged_input = "\n\n".join(partial_bullets)
        messages = [
            {'role': 'system', 'content': f"You are an expert editor and summarizer. Work in {target_language}."},
            {'role': 'user', 'content': f"{aggregation_prompt}\n\nHere are the partial bullet lists:\n\n{merged_input}"},
        ]

        try:
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
        except:
            fallback_prompt = f"System: You are an expert editor and summarizer.\nUser: {aggregation_prompt}\n\nHere are the partial bullet lists:\n\n{merged_input}\nAssistant:"
            input_ids = tokenizer(fallback_prompt, return_tensors='pt').input_ids

        model_device = next(model.parameters()).device
        input_ids = input_ids.to(model_device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model_device)

        # Build GenerationConfig for aggregation
        # Dynamic token limit - needs enough space to merge all bullets
        merged_tokens = len(tokenizer.encode(merged_input, add_special_tokens=False))
        agg_max_tokens = max(max_new_tokens, int(merged_tokens * 0.75))  # Merged summary can be up to 75% of input
        agg_gen_config_dict = {
            'max_new_tokens': agg_max_tokens,
            'do_sample': agg_do_sample,
            'pad_token_id': tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        }
        if agg_temperature is not None:
            agg_gen_config_dict['temperature'] = agg_temperature
        if agg_top_p is not None:
            agg_gen_config_dict['top_p'] = agg_top_p
        if repetition_penalty is not None:
            agg_gen_config_dict['repetition_penalty'] = repetition_penalty

        agg_gen_config = GenerationConfig(**agg_gen_config_dict)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=agg_gen_config
            )

        gen_only = output_ids[0, input_ids.shape[1]:]
        summary = tokenizer.decode(gen_only, skip_special_tokens=True).strip()

        # Normalize final bullets
        final_lines = []
        for line in summary.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("- "):
                s = s[2:]
            s = s.replace("**Key Point**:", "").replace("**Key Point**", "").lstrip("- ").strip()
            final_lines.append(f"- {s}")

        summary = "\n".join(final_lines)

    bullet_count = len(final_lines)

    result['metrics'] = collector.metrics
    result['output'] = {
        'summary': summary,
        'summary_chars': len(summary),
        'bullet_count': bullet_count
    }

    return result


def run_podcast_stage(
    translated_text: str,
    summary: str,
    config: Dict[str, Any],
    load_llm_fn,
    metrics_collector_cls,
    log_fn
) -> Dict[str, Any]:
    """
    Run podcast script generation with full metrics collection.

    Args:
        translated_text: Translated transcript
        summary: Key points summary
        config: Experiment configuration
        load_llm_fn: Function to load LLM
        metrics_collector_cls: StageMetricsCollector class
        log_fn: Logging function

    Returns:
        Dictionary with stage results, metrics, and outputs
    """
    llm_model_id = config['llm_model_id']
    quantization = config.get('quantization')
    chunk_size_chars = config.get('chunk_size_chars', 2000)
    # FIX: Proper handling to allow explicit None
    repetition_penalty = config["repetition_penalty"] if "repetition_penalty" in config else 1.2
    target_language = config.get('target_language', 'English')
    max_new_tokens = config.get('podcast_max_new_tokens', 1024)
    podcast_creativity_temp = config.get('podcast_creativity_temp', 0.3)

    result = {'stage': 'podcast', 'config_used': config.copy()}
    result['input'] = {
        'translated_chars': len(translated_text),
        'summary_chars': len(summary)
    }

    tokenizer, model, model_load_time = load_llm_fn(llm_model_id, quantization)
    result['model_load_time_seconds'] = model_load_time

    chunks = chunk_text(translated_text.strip(), chunk_size_chars)
    total_chunks = len(chunks)
    result['input']['num_chunks'] = total_chunks

    system_prompt = (
        f"You are a scriptwriter for an educational podcast. Create a two-host conversation in {target_language} that is engaging, "
        "technically accurate, and pedagogically effective. Use plain language and helpful analogies. "
        "The dialogue must include clear speaker labels at the start of each line: 'Host A:' and 'Host B:'. "
        "Avoid meta commentary, stage directions, or non-dialogue text. "
        f"All dialogue content must be written in {target_language}."
    )

    # Build GenerationConfig for podcast - use sampling with temperature for creativity
    podcast_gen_config_dict = {
        'max_new_tokens': max_new_tokens,
        'do_sample': True,
        'temperature': podcast_creativity_temp,
        'top_p': 0.9,
        'pad_token_id': tokenizer.eos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }
    if repetition_penalty is not None:
        podcast_gen_config_dict['repetition_penalty'] = repetition_penalty

    podcast_gen_config = GenerationConfig(**podcast_gen_config_dict)

    model_device = next(model.parameters()).device
    script_segments = []
    previous_context = ''

    with metrics_collector_cls('podcast') as collector:
        pbar = tqdm(enumerate(chunks), total=total_chunks, desc="Podcast script", unit="chunk", leave=False)
        for i, chunk in pbar:
            # Build prompt based on position
            if i == 0:
                # First chunk: introduce the podcast
                user_prompt = (
                    f"Write the BEGINNING of a podcast script in {target_language}.\n"
                    "Start with a brief introduction where Host A welcomes listeners and introduces the topic.\n"
                    f"Use the transcript segment and summary below to create the opening discussion.\n\n"
                    f"Summary (for structure):\n{summary}\n\n"
                    f"Transcript segment:\n{chunk}\n\n"
                    "Requirements:\n"
                    "- Host A explains; Host B asks concise, clarifying questions.\n"
                    f"- All dialogue must be in {target_language}.\n"
                    "- Strictly output alternating lines beginning with 'Host A:' or 'Host B:' and nothing else."
                )
            elif i == total_chunks - 1:
                # Last chunk: conclude the podcast
                user_prompt = (
                    f"Continue and CONCLUDE the podcast script in {target_language}.\n"
                    f"Previous context:\n{previous_context}\n\n"
                    f"Transcript segment:\n{chunk}\n\n"
                    "Requirements:\n"
                    "- Continue naturally from the previous dialogue.\n"
                    "- Cover the remaining content and wrap up with a brief conclusion.\n"
                    "- Host A thanks listeners; Host B adds a final thought.\n"
                    f"- All dialogue must be in {target_language}.\n"
                    "- Strictly output alternating lines beginning with 'Host A:' or 'Host B:' and nothing else."
                )
            else:
                # Middle chunks: continue the conversation
                user_prompt = (
                    f"Continue the podcast script in {target_language}.\n"
                    f"Previous context:\n{previous_context}\n\n"
                    f"Transcript segment:\n{chunk}\n\n"
                    "Requirements:\n"
                    "- Continue naturally from the previous dialogue.\n"
                    "- Cover the new content in this segment.\n"
                    f"- All dialogue must be in {target_language}.\n"
                    "- Strictly output alternating lines beginning with 'Host A:' or 'Host B:' and nothing else."
                )

            messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]

            try:
                input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
            except:
                fallback_prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
                input_ids = tokenizer(fallback_prompt, return_tensors='pt').input_ids

            input_ids = input_ids.to(model_device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model_device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=podcast_gen_config
                )

            gen_only = output_ids[0, input_ids.shape[1]:]
            segment_script = tokenizer.decode(gen_only, skip_special_tokens=True).strip()

            # Enforce label format for this segment
            lines = [line.strip() for line in segment_script.splitlines() if line.strip()]
            fixed_lines = []
            expected = "Host A:"
            for line in lines:
                if line.startswith("Host A:") or line.startswith("Host B:"):
                    fixed_lines.append(line)
                    expected = "Host B:" if line.startswith("Host A:") else "Host A:"
                else:
                    fixed_lines.append(f"{expected} {line}")
                    expected = "Host B:" if expected == "Host A:" else "Host A:"

            segment_text = "\n".join(fixed_lines)
            script_segments.append(segment_text)

            # Keep last 2-3 lines as context for next chunk
            context_lines = fixed_lines[-3:] if len(fixed_lines) >= 3 else fixed_lines
            previous_context = "\n".join(context_lines)

    script = '\n'.join(script_segments)
    all_lines = [l for l in script.split('\n') if l.strip()]
    host_a_lines = sum(1 for l in all_lines if l.strip().startswith('Host A:'))
    host_b_lines = sum(1 for l in all_lines if l.strip().startswith('Host B:'))

    result['metrics'] = collector.metrics
    result['output'] = {
        'script': script,
        'script_chars': len(script),
        'total_lines': len(all_lines),
        'host_a_lines': host_a_lines,
        'host_b_lines': host_b_lines
    }

    return result
