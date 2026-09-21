"""Miniature autoregressive models for weightless generation tests."""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mlx_minimax_music3.autoregressive import AutoregressiveConfig
from mlx_minimax_music3.config import Qwen3Config, RVQDepthDecoderConfig
from mlx_minimax_music3.models.qwen3 import Qwen3ForCausalLM
from mlx_minimax_music3.models.rvq_depth import RVQDepthDecoder
from mlx_minimax_music3.tokenizer import TokenizedPrompt

HIDDEN_SIZE = 16
NUM_CODEBOOKS = 4
# Scale the sampling heads so logits dominate the Gumbel column noise. Parameters
# drawn at unit scale leave every draw a coin flip, which hides real steering.
_HEAD_SCALE = 2.0
_BODY_SCALE = 0.5


def build_tiny_models(
    *,
    max_position_embeddings: int = 16,
    head_seed: int | None = None,
) -> tuple[Qwen3ForCausalLM, RVQDepthDecoder]:
    """Build a tiny model pair that spans the real semantic code vocabulary.

    Without `head_seed` every sampling head is zeroed, so the loop's structure can
    be asserted without a distribution. With one, every parameter is drawn from
    that seed's key, so the sampled codes differ between frames and repeat exactly
    on the next run.
    """

    language_model = Qwen3ForCausalLM(
        Qwen3Config(
            hidden_size=HIDDEN_SIZE,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=170_000,
            max_position_embeddings=max_position_embeddings,
            published_dtype="float32",
        )
    )
    decoder = RVQDepthDecoder(
        RVQDepthDecoderConfig(
            hidden_size=HIDDEN_SIZE,
            intermediate_size=32,
            num_layers=1,
            num_attention_heads=4,
            audio_vocab_size=32,
            num_codebooks=NUM_CODEBOOKS,
            max_position_embeddings=8,
        )
    )
    if head_seed is None:
        language_model.lm_head.weight = mx.zeros_like(language_model.lm_head.weight)
        for head in decoder.audio_heads:
            head.weight = mx.zeros_like(head.weight)
        return language_model, decoder

    _draw_parameters(language_model, seed=head_seed)
    _draw_parameters(decoder, seed=head_seed + 1)
    mx.eval(language_model.parameters(), decoder.parameters())
    return language_model, decoder


def tiny_prompt() -> TokenizedPrompt:
    """Return the shared conditional and unconditional token rows."""

    return TokenizedPrompt(conditional=(1, 2, 3), unconditional=(1, 4, 3))


def fixed_length_config(frames: int, **overrides) -> AutoregressiveConfig:
    """Build a config that generates exactly `frames` frames, stop masked out."""

    duration = frames / 25
    return AutoregressiveConfig(
        audio_duration=duration,
        min_audio_duration=duration,
        buffer_flush_interval=1,
        **overrides,
    )


def _draw_parameters(model, *, seed: int) -> None:
    """Replace every parameter with values drawn from one reproducible key."""

    flat = tree_flatten(model.parameters())
    keys = mx.random.split(mx.random.key(seed), num=len(flat))
    model.update(
        tree_unflatten(
            [
                (name, _parameter_value(name, value, key))
                for (name, value), key in zip(flat, keys, strict=True)
            ]
        )
    )


def _parameter_value(name: str, value: mx.array, key: mx.array) -> mx.array:
    if name.endswith("norm.weight"):
        # Normalization gains stay at one so hidden states keep a usable scale.
        return mx.ones_like(value)
    scale = _HEAD_SCALE if "head" in name else _BODY_SCALE
    return mx.random.normal(value.shape, key=key) * scale


__all__ = [
    "HIDDEN_SIZE",
    "NUM_CODEBOOKS",
    "build_tiny_models",
    "fixed_length_config",
    "tiny_prompt",
]
