"""Frame-level RVQ code generation and language-model feedback.

One Music 3 frame is a semantic code plus seven residual codes. The depth decoder
produces the residual codes and the hidden states the acoustic stage needs, and the
language model consumes the finished frame as feedback for the next step. A
reference prefix reuses the same feedback path with codes it did not sample.
"""

from __future__ import annotations

import mlx.core as mx

from .models.cache import KVCache
from .models.qwen3 import Qwen3ForCausalLM
from .models.rvq_depth import RVQDepthDecoder
from .prompting import AUDIO_CODE_OFFSET
from .sampling import Sampler, SeedSchedule, classifier_free_guidance


def generate_depth_codes(
    language_model: Qwen3ForCausalLM,
    decoder: RVQDepthDecoder,
    last_hidden: mx.array,
    semantic_token: mx.array,
    *,
    frame_index: int,
    cfg_scale: float,
    top_k: int,
    seeds: SeedSchedule,
    sampler: Sampler,
) -> tuple[mx.array, mx.array]:
    semantic_code = semantic_token - AUDIO_CODE_OFFSET
    paired_semantic = mx.repeat(semantic_code, 2, axis=0)
    semantic_embedding = language_model.model.embed_tokens(
        paired_semantic + AUDIO_CODE_OFFSET
    )
    first_inputs = mx.stack(
        (
            decoder.projection(last_hidden),
            decoder.projection(semantic_embedding),
        ),
        axis=1,
    )
    cache = decoder.make_cache()
    hidden = decoder(first_inputs, cache=cache)[:, -1]

    sampled_codes = [semantic_code]
    hidden_parts = []
    for codebook_index in range(1, decoder.config.num_codebooks):
        hidden_parts.append(hidden[:1])
        logits = decoder.logits(hidden, codebook_index=codebook_index)
        guided = classifier_free_guidance(logits, scale=cfg_scale)
        sampled = sampler(
            guided,
            top_k=min(top_k, decoder.config.audio_vocab_size),
            seed=seeds.sampling_seed,
            position=seeds.position(
                frame_index=frame_index,
                codebook_index=codebook_index,
            ),
        )
        sampled_codes.append(sampled)
        if codebook_index < decoder.config.num_codebooks - 1:
            paired = mx.repeat(sampled, 2, axis=0)
            embedding = decoder.embed_residual_code(
                paired, codebook_index=codebook_index
            )
            projected = decoder.projection(embedding)[:, None, :]
            hidden = decoder(projected, cache=cache)[:, -1]

    codes = mx.stack(sampled_codes, axis=-1)
    depth_hiddens = mx.concatenate(hidden_parts, axis=-1)
    return codes, depth_hiddens


def embed_audio_frame(
    language_model: Qwen3ForCausalLM,
    decoder: RVQDepthDecoder,
    codes: mx.array,
) -> mx.array:
    paired_codes = mx.repeat(codes, 2, axis=0)
    semantic = language_model.model.embed_tokens(
        paired_codes[:, :1] + AUDIO_CODE_OFFSET
    )
    offsets = (
        mx.arange(decoder.config.num_residual_codebooks, dtype=mx.int32)
        * decoder.config.audio_vocab_size
    )[None, :]
    residual = decoder.audio_embeddings(paired_codes[:, 1:] + offsets)
    residual = residual.sum(axis=1, keepdims=True)
    return (semantic + residual.astype(semantic.dtype)) * (
        decoder.config.num_codebooks**-0.5
    )


def advance_frame(
    language_model: Qwen3ForCausalLM,
    decoder: RVQDepthDecoder,
    codes: mx.array,
    cache: list[KVCache],
) -> mx.array:
    """Feed one complete frame back into the language model and return its state."""

    feedback = embed_audio_frame(language_model, decoder, codes)
    return language_model.model(inputs_embeds=feedback, cache=cache)[:, -1]


def prefill_frame_codes(
    language_model: Qwen3ForCausalLM,
    decoder: RVQDepthDecoder,
    last_hidden: mx.array,
    code_frame: tuple[int, ...],
    *,
    frame_index: int,
    cfg_scale: float,
    top_k: int,
    seeds: SeedSchedule,
    sampler: Sampler,
) -> mx.array:
    """Return one prefix frame's codes, resynthesizing residuals only if absent."""

    if len(code_frame) == decoder.config.num_codebooks:
        return mx.array([code_frame], dtype=mx.int32)

    # A semantic-only reference carries no residual codes, so the depth decoder
    # produces them from the injected semantic code. The frame is context, not
    # output, so its depth hidden states are discarded.
    codes, _ = generate_depth_codes(
        language_model,
        decoder,
        last_hidden,
        mx.array([code_frame[0] + AUDIO_CODE_OFFSET], dtype=mx.int32),
        frame_index=frame_index,
        cfg_scale=cfg_scale,
        top_k=top_k,
        seeds=seeds,
        sampler=sampler,
    )
    return codes


__all__ = [
    "advance_frame",
    "embed_audio_frame",
    "generate_depth_codes",
    "prefill_frame_codes",
]
