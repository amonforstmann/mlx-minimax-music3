"""Global and depth autoregressive generation for MiniMax Music 3."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx

from .frames import advance_frame, generate_depth_codes, prefill_frame_codes
from .models.qwen3 import Qwen3ForCausalLM
from .models.rvq_depth import RVQDepthDecoder
from .prompting import (
    AUDIO_CODE_OFFSET,
    AUDIO_END_TOKEN_ID,
    MAX_AUDIO_FRAMES,
    SEMANTIC_VOCAB_SIZE,
)
from .reference import (
    MIN_REFERENCE_INTERVAL,
    ReferenceCodes,
    ReferenceConstraint,
    ReferenceMode,
    ReferencePlan,
    cover_logits,
    guidance_logits,
    stop_allowed,
    validate_reference_controls,
    validate_reference_stream,
)
from .sampling import (
    Sampler,
    SeedSchedule,
    classifier_free_guidance,
    sample_top_k,
)
from .tokenizer import TokenizedPrompt

FRAME_RATE = 25
AR_CFG_SCALE = 1.5
AR_TOP_K = 50


@dataclass(frozen=True, slots=True)
class AutoregressiveConfig:
    """Request-local controls for the fixed Music 3 generation recipe."""

    audio_duration: float = 60.0
    seed: int = 0
    cfg_scale: float = AR_CFG_SCALE
    top_k: int = AR_TOP_K
    frame_rate: int = FRAME_RATE
    buffer_flush_interval: int = 32
    min_audio_duration: float = 0.0
    reference_codes: ReferenceCodes | None = None
    reference_mode: ReferenceMode = ReferenceMode.GUIDANCE
    reference_interval: int = MIN_REFERENCE_INTERVAL

    def __post_init__(self) -> None:
        if not math.isfinite(self.audio_duration) or self.audio_duration <= 0:
            raise ValueError("audio_duration must be finite and positive")
        if not math.isfinite(self.min_audio_duration) or self.min_audio_duration < 0:
            raise ValueError("min_audio_duration must be finite and non-negative")
        if self.min_audio_duration > self.audio_duration:
            raise ValueError("min_audio_duration cannot exceed audio_duration")
        if not math.isfinite(self.cfg_scale) or self.cfg_scale < 0:
            raise ValueError("cfg_scale must be finite and non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= self.seed < 2**64:
            raise ValueError("seed must be a non-negative 64-bit integer")
        for name in ("top_k", "frame_rate", "buffer_flush_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        validate_reference_controls(self.reference_mode, self.reference_interval)
        if self.reference_codes is not None and not isinstance(
            self.reference_codes, ReferenceCodes
        ):
            raise TypeError("reference_codes must be a ReferenceCodes instance")

    @property
    def reference_plan(self) -> ReferencePlan | None:
        """Return the resolved reference plan, or `None` without a reference."""

        if self.reference_codes is None:
            return None
        return ReferencePlan(
            codes=self.reference_codes,
            mode=self.reference_mode,
            interval=self.reference_interval,
        )

    @property
    def max_frames(self) -> int:
        frames = min(int(self.audio_duration * self.frame_rate), MAX_AUDIO_FRAMES)
        if frames == 0:
            raise ValueError(
                "audio_duration is shorter than one autoregressive frame"
            )
        return frames

    @property
    def min_frames(self) -> int:
        """Return the number of frames protected from early stopping."""

        return min(
            math.ceil(self.min_audio_duration * self.frame_rate),
            self.max_frames,
        )


@dataclass(frozen=True, slots=True)
class GenerationProgress:
    """Progress emitted only at safe completed-frame boundaries."""

    completed_frames: int
    maximum_frames: int


@dataclass(frozen=True, slots=True)
class AutoregressiveResult:
    """Evaluated codes and acoustic-conditioning hidden states."""

    codes: mx.array
    frame_hiddens: mx.array
    stopped_on_audio_end: bool

    @property
    def num_frames(self) -> int:
        return self.frame_hiddens.shape[1]


class _FrameBuffer:
    """Preallocated, periodically evaluated durable stage output."""

    def __init__(
        self,
        *,
        max_frames: int,
        num_codebooks: int,
        hidden_size: int,
        dtype: mx.Dtype,
        flush_interval: int,
    ) -> None:
        self._hiddens = mx.zeros(
            (1, max_frames, num_codebooks * hidden_size), dtype=dtype
        )
        self._codes = mx.zeros((1, max_frames, num_codebooks), dtype=mx.int32)
        self._max_frames = max_frames
        self._flush_interval = flush_interval
        self.count = 0

    def append(self, codes: mx.array, hidden_states: mx.array) -> None:
        if self.count >= self._max_frames:
            raise RuntimeError("Frame buffer is full")
        self._codes[:, self.count, :] = codes
        self._hiddens[:, self.count, :] = hidden_states
        self.count += 1
        if self.count % self._flush_interval == 0:
            mx.eval(self._codes, self._hiddens)

    def finish(self) -> tuple[mx.array, mx.array]:
        if self.count == 0:
            raise ValueError(
                "MiniMax Music 3 generated zero frames; the prompt ended immediately"
            )
        mx.eval(self._codes, self._hiddens)
        if self.count == self._max_frames:
            return self._codes, self._hiddens
        codes = mx.contiguous(self._codes[:, : self.count])
        hiddens = mx.contiguous(self._hiddens[:, : self.count])
        mx.eval(codes, hiddens)
        return codes, hiddens


def _restricted_semantic_logits(
    language_model: Qwen3ForCausalLM,
    last_hidden: mx.array,
    *,
    allow_stop: bool,
) -> mx.array:
    """Project hidden states onto the only vocabulary rows Music 3 can sample."""

    weight = language_model.lm_head.weight
    stop_logits = last_hidden @ weight[AUDIO_END_TOKEN_ID : AUDIO_END_TOKEN_ID + 1].T
    semantic_logits = (
        last_hidden
        @ weight[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE].T
    )
    if not allow_stop:
        stop_logits = mx.full(stop_logits.shape, -mx.inf, dtype=stop_logits.dtype)
    return mx.concatenate((stop_logits, semantic_logits), axis=-1).astype(mx.float32)


def _sample_semantic_code(
    language_model: Qwen3ForCausalLM,
    last_hidden: mx.array,
    *,
    allow_stop: bool,
    cfg_scale: float,
    top_k: int,
    seed: int,
    position: int,
    sampler: Sampler,
    reference: ReferenceConstraint | None = None,
) -> mx.array:
    # Project only the rows that c0 sampling can return. The dense checkpoint has
    # a 200,000-token language head, while Music 3 uses 16,384 semantic codes and
    # one stop token. Computing the other logits adds work without changing the
    # distribution.
    # Preserve the reference's narrowed-column order for deterministic sampling:
    # stop first, followed by c0 codes 0 through 16,383.
    allowed_logits = _restricted_semantic_logits(
        language_model,
        last_hidden,
        allow_stop=allow_stop,
    )
    guided = classifier_free_guidance(allowed_logits, scale=cfg_scale)
    columns = guided.shape[-1]
    if reference is not None and reference.mode is ReferenceMode.COVER:
        # A covered frame has to emit its reference code, so only that column stays
        # reachable and the draw ranges over the reference set alone.
        guided = cover_logits(guided, reference.candidates)
        draw_top_k = len(reference.candidates)
    else:
        conditional = allowed_logits[:1]
        window = mx.where(
            conditional
            < mx.topk(
                conditional,
                k=min(top_k, conditional.shape[-1]),
                axis=-1,
            )[..., -1:],
            -mx.inf,
            guided,
        )
        if reference is None:
            guided = window
            draw_top_k = min(top_k, columns)
        else:
            # The draw spans the model's window plus every candidate, so no
            # reference code is dropped before the sampler sees it.
            guided = guidance_logits(guided, window, reference.candidates)
            draw_top_k = min(columns, top_k + len(reference.candidates))
    local_index = sampler(
        guided,
        top_k=draw_top_k,
        seed=seed,
        position=position,
    )
    return mx.where(
        local_index == 0,
        mx.array(AUDIO_END_TOKEN_ID, dtype=mx.int32),
        local_index + AUDIO_CODE_OFFSET - 1,
    ).astype(mx.int32)


def generate_autoregressive(
    language_model: Qwen3ForCausalLM,
    decoder: RVQDepthDecoder,
    prompt: TokenizedPrompt,
    config: AutoregressiveConfig,
    *,
    sampler: Sampler = sample_top_k,
    progress: Callable[[GenerationProgress], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> AutoregressiveResult:
    """Generate Music 3 frame codes and hidden-state conditioning.

    A `reference_codes` stream on the config steers the semantic codebook as
    described in `reference.py`. Without one, the loop is free-running and its
    output depends only on the prompt, the seed, and the sampling controls.

    A `CONTINUE` prefix is context: it runs before the first emitted frame, extends
    the key-value cache beyond the requested duration, and is absent from the
    result. `GUIDANCE` and `COVER` emit the frames they steer, and a live reference
    window masks the stop token so the reference cannot be cut short.
    """

    if language_model.config.hidden_size != decoder.config.hidden_size:
        raise ValueError("Language model and RVQ decoder hidden sizes must match")
    if language_model.config.vocab_size < AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE:
        raise ValueError("Language-model vocabulary cannot represent Music 3 codes")

    max_frames = config.max_frames
    plan = config.reference_plan
    prefix_frames = 0
    if plan is not None:
        validate_reference_stream(
            plan,
            num_codebooks=decoder.config.num_codebooks,
            audio_vocab_size=decoder.config.audio_vocab_size,
            max_frames=max_frames,
            top_k=config.top_k,
        )
        prefix_frames = plan.prefix_frames

    text_ids = mx.array(prompt.rows(), dtype=mx.int32)
    cache = language_model.make_cache(
        capacity=prompt.length + max_frames + prefix_frames
    )
    last_hidden = language_model.model(text_ids, cache=cache)[:, -1]
    mx.eval(last_hidden)

    seeds = SeedSchedule(config.seed, decoder.config.num_codebooks)
    frames = _FrameBuffer(
        max_frames=max_frames,
        num_codebooks=decoder.config.num_codebooks,
        hidden_size=decoder.config.hidden_size,
        dtype=last_hidden.dtype,
        flush_interval=config.buffer_flush_interval,
    )
    stopped_on_audio_end = False

    # Frame zero advances past <|audio_start|>; it is feedback, not output.
    for frame_index in range(max_frames + prefix_frames + 1):
        if cancelled is not None and cancelled():
            raise InterruptedError("Music 3 autoregressive generation was cancelled")
        # Reference frame zero belongs to the frame after the feedback frame.
        reference_index = frame_index - 1
        if plan is not None and plan.prefills(reference_index):
            codes = prefill_frame_codes(
                language_model,
                decoder,
                last_hidden,
                plan.codes.code_frame(reference_index),
                frame_index=frame_index,
                cfg_scale=config.cfg_scale,
                top_k=config.top_k,
                seeds=seeds,
                sampler=sampler,
            )
            last_hidden = advance_frame(language_model, decoder, codes, cache)
            continue

        semantic_token = _sample_semantic_code(
            language_model,
            last_hidden,
            allow_stop=stop_allowed(
                completed_frames=frames.count,
                frame_index=frame_index,
                min_frames=config.min_frames,
                plan=plan,
            ),
            cfg_scale=config.cfg_scale,
            top_k=config.top_k,
            seed=seeds.sampling_seed,
            position=seeds.position(frame_index=frame_index, codebook_index=0),
            sampler=sampler,
            reference=None if plan is None else plan.constraint(reference_index),
        )
        if int(semantic_token.item()) == AUDIO_END_TOKEN_ID:
            stopped_on_audio_end = True
            break

        codes, depth_hiddens = generate_depth_codes(
            language_model,
            decoder,
            last_hidden,
            semantic_token,
            frame_index=frame_index,
            cfg_scale=config.cfg_scale,
            top_k=config.top_k,
            seeds=seeds,
            sampler=sampler,
        )
        if frame_index > 0:
            frame_hidden = mx.concatenate((last_hidden[:1], depth_hiddens), axis=-1)
            frames.append(codes, frame_hidden)
            if progress is not None:
                progress(GenerationProgress(frames.count, max_frames))
            if frames.count >= max_frames:
                break

        last_hidden = advance_frame(language_model, decoder, codes, cache)

    codes, frame_hiddens = frames.finish()
    return AutoregressiveResult(
        codes=codes,
        frame_hiddens=frame_hiddens,
        stopped_on_audio_end=stopped_on_audio_end,
    )
