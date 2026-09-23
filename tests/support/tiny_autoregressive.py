"""Miniature autoregressive models for weightless generation tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from mlx_minimax_music3 import autoregressive
from mlx_minimax_music3.autoregressive import AutoregressiveConfig
from mlx_minimax_music3.config import Qwen3Config, RVQDepthDecoderConfig
from mlx_minimax_music3.models.qwen3 import Qwen3ForCausalLM
from mlx_minimax_music3.models.rvq_depth import RVQDepthDecoder
from mlx_minimax_music3.prompting import SEMANTIC_VOCAB_SIZE
from mlx_minimax_music3.sampling import sample_top_k
from mlx_minimax_music3.tokenizer import TokenizedPrompt

HIDDEN_SIZE = 16
NUM_CODEBOOKS = 4
# The stop token takes column zero of the narrowed c0 row, so column k + 1 carries
# semantic code k.
C0_COLUMNS = 1 + SEMANTIC_VOCAB_SIZE
# Bound at import so a recorder built while another one's spy is installed still
# wraps the loop's own projection.
_PROJECT = autoregressive._restricted_semantic_logits

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


@dataclass(frozen=True, slots=True)
class SemanticDraw:
    """One recorded c0 draw: its sampling width, reachable columns, and result."""

    top_k: int
    columns: tuple[int, ...]
    drawn: int


class DrawRecorder:
    """Record every draw the loop makes through `sample_top_k`, keyed by loop frame.

    With `monkeypatch`, the recorder also spies on the c0 projection. The loop
    projects the c0 logits once per semantic draw and samples right after, so the
    projection spied on always belongs to the next c0 draw. Its row zero is the
    conditional row that the top-k window ranks.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch | None = None) -> None:
        self.semantic: dict[int, SemanticDraw] = {}
        self.conditional: dict[int, list[float]] = {}
        self.residual_frames: set[int] = set()
        self._pending: mx.array | None = None
        if monkeypatch is None:
            return

        def spy(*args, **kwargs) -> mx.array:
            logits = _PROJECT(*args, **kwargs)
            self._pending = logits[0]
            return logits

        monkeypatch.setattr(autoregressive, "_restricted_semantic_logits", spy)

    def __call__(
        self, logits: mx.array, *, top_k: int, seed: int, position: int
    ) -> mx.array:
        sampled = sample_top_k(logits, top_k=top_k, seed=seed, position=position)
        frame = position // NUM_CODEBOOKS
        if logits.shape[-1] != C0_COLUMNS:
            self.residual_frames.add(frame)
            return sampled
        mx.eval(logits, sampled)
        self.semantic[frame] = SemanticDraw(
            top_k,
            tuple(
                column
                for column, value in enumerate(logits[0].tolist())
                if math.isfinite(value)
            ),
            int(sampled.item()),
        )
        if self._pending is not None:
            mx.eval(self._pending)
            self.conditional[frame] = self._pending.tolist()
            self._pending = None
        return sampled

    def window(self, frame: int, top_k: int) -> set[int]:
        """Return the columns at or above the frame's kth largest conditional logit."""

        values = self.conditional[frame]
        threshold = sorted(values, reverse=True)[top_k - 1]
        return {column for column, value in enumerate(values) if value >= threshold}


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
    "C0_COLUMNS",
    "HIDDEN_SIZE",
    "NUM_CODEBOOKS",
    "DrawRecorder",
    "SemanticDraw",
    "build_tiny_models",
    "fixed_length_config",
    "tiny_prompt",
]
