"""Reference-code conditioning for the autoregressive semantic head.

A reference is one code stream per generated frame, taken from an encoded track or
from an earlier generation. A complete stream carries every codebook of each frame.
A semantic-only stream carries the semantic code (c0) alone. Optional per-frame
candidates carry the semantic alternates that a guidance pass biases towards.

Only the semantic codebook is steered. Residual codebooks c1 through c7 and every
hidden state stay model-generated under `GUIDANCE` and `COVER`, so the acoustic
stage receives the conditioning it receives for a text-only request. `CONTINUE` is
the exception: it injects whole reference frames as context, and those frames are
never part of the result.

Reference frame zero aligns with the first frame after the `<|audio_start|>`
feedback frame, which is never steered. `GUIDANCE` and `COVER` emit the frames
they steer, so the requested duration covers the reference window and the free
frames together. `CONTINUE` prefills its prefix outside the requested duration, so
that duration always counts new frames. Frames past the end of a stream free-run,
and a stream longer than the generated length has its tail unused.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import mlx.core as mx

from .prompting import MAX_AUDIO_FRAMES, SEMANTIC_VOCAB_SIZE

MIN_REFERENCE_INTERVAL = 1
MAX_REFERENCE_INTERVAL = 10
# Logit units subtracted from every non-candidate column on a guided frame, which
# leaves a non-candidate code reachable and keeps GUIDANCE distinct from COVER.
# Calibrated on the selective-q8 checkpoint over 100 frames and 25 guided frames,
# against a reference captured from a different prompt (plausible to the model) and
# a reference shifted by 4,096 codes (implausible). Guided frames that followed the
# reference: 8 units gave 8 of 25 plausible and 0 of 25 implausible, 16 gave 22 and
# 1, 24 gave 24 and 13, 32 gave 25 and 17. The value below follows a plausible
# reference while still refusing an implausible one. Listening validation has not
# been done.
GUIDANCE_LOGIT_PENALTY = 16.0


class ReferenceQualityWarning(UserWarning):
    """Warn when a valid reference cannot be applied as the caller intended."""


class ReferenceMode(StrEnum):
    """How a reference code stream steers autoregressive generation.

    `GUIDANCE` biases the semantic draw towards the reference candidates on every
    `reference_interval`-th covered frame and leaves the frames between them free.
    That bias is finite, so the model can still leave the reference. `COVER`
    restricts the semantic code to the reference code on every covered frame.
    `CONTINUE` prefills the stream as context and then free-runs without steering.
    """

    GUIDANCE = "guidance"
    COVER = "cover"
    CONTINUE = "continue"


@dataclass(frozen=True, slots=True)
class ReferenceCodes:
    """Per-frame reference codes and their optional semantic alternates.

    `code_frames[frame]` is one frame of RVQ codes, semantic code first. A stream
    produced by an encoder carries every codebook. A stream captured from an earlier
    generation's semantic head carries one code per frame and cannot prefill
    residual codebooks. `semantic_candidates`, when present, lists the
    semantic alternates of each frame ordered best first, with the frame's own
    semantic code at position zero. Candidate order is the caller's ranking and is
    only checked for that first entry and for duplicates.
    """

    code_frames: tuple[tuple[int, ...], ...]
    semantic_candidates: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code_frames, tuple) or not self.code_frames:
            raise ValueError("reference codes must hold at least one frame")
        _validate_rows(self.code_frames, label="reference frame")
        width = len(self.code_frames[0])
        for frame in self.code_frames:
            if len(frame) != width:
                raise ValueError("every reference frame needs the same codebook count")
            _validate_semantic_code(frame[0])
            for code in frame[1:]:
                if isinstance(code, bool) or not isinstance(code, int) or code < 0:
                    raise ValueError(
                        "residual reference codes must be non-negative integers"
                    )
        if self.semantic_candidates is None:
            return
        if len(self.semantic_candidates) != len(self.code_frames):
            raise ValueError("semantic candidates must cover every reference frame")
        _validate_rows(self.semantic_candidates, label="reference candidate frame")
        candidate_width = len(self.semantic_candidates[0])
        for frame, candidates in zip(
            self.code_frames, self.semantic_candidates, strict=True
        ):
            if len(candidates) != candidate_width:
                raise ValueError("every reference frame needs the same candidate count")
            if len(set(candidates)) != len(candidates):
                raise ValueError("reference candidates must be distinct")
            if candidates[0] != frame[0]:
                raise ValueError(
                    "the first candidate must be the frame's own semantic code"
                )
            for code in candidates:
                _validate_semantic_code(code)

    @property
    def num_frames(self) -> int:
        return len(self.code_frames)

    @property
    def num_codebooks(self) -> int:
        return len(self.code_frames[0])

    @property
    def has_residual_codes(self) -> bool:
        """Whether the stream can prefill residual codebooks."""

        return self.num_codebooks > 1

    @property
    def candidates_per_frame(self) -> int:
        if self.semantic_candidates is None:
            return 1
        return len(self.semantic_candidates[0])

    def code_frame(self, frame_index: int) -> tuple[int, ...]:
        return self.code_frames[frame_index]

    def semantic_code(self, frame_index: int) -> int:
        return self.code_frames[frame_index][0]

    def candidates_at(self, frame_index: int) -> tuple[int, ...]:
        """Return the semantic codes a guided frame may be biased towards."""

        if self.semantic_candidates is None:
            return self.code_frames[frame_index][:1]
        return self.semantic_candidates[frame_index]

    @classmethod
    def from_code_frames(
        cls,
        codes: mx.array | Sequence[Sequence[int]],
        *,
        semantic_candidates: mx.array | Sequence[Sequence[int]] | None = None,
    ) -> ReferenceCodes:
        """Build a stream from complete `[frames, codebooks]` reference codes."""

        return cls(
            _code_rows(codes, label="reference code frames"),
            None
            if semantic_candidates is None
            else _code_rows(semantic_candidates, label="reference candidates"),
        )

    @classmethod
    def from_semantic_codes(
        cls,
        codes: mx.array | Sequence[int],
        *,
        semantic_candidates: mx.array | Sequence[Sequence[int]] | None = None,
    ) -> ReferenceCodes:
        """Build a semantic-only stream from one semantic code per frame."""

        return cls(
            tuple(
                (int(code),)
                for code in _rows(codes, rank=1, label="reference semantic codes")
            ),
            None
            if semantic_candidates is None
            else _code_rows(semantic_candidates, label="reference candidates"),
        )

    @classmethod
    def from_semantic_candidates(
        cls,
        candidates: mx.array | Sequence[Sequence[int]],
        *,
        candidates_per_frame: int,
    ) -> ReferenceCodes:
        """Build a semantic-only stream from ranked candidates, best first.

        `candidates_per_frame` is required because a `[frames, codebooks]` code
        grid and a `[frames, k]` candidate grid have the same shape. Reading
        residual codes as semantic alternates would corrupt guidance silently, so
        the caller states the candidate count and every frame is checked against
        it.
        """

        if (
            isinstance(candidates_per_frame, bool)
            or not isinstance(candidates_per_frame, int)
            or candidates_per_frame < 1
        ):
            raise ValueError("candidates_per_frame must be a positive integer")
        rows = _code_rows(candidates, label="reference candidates")
        for row in rows:
            if len(row) != candidates_per_frame:
                raise ValueError(
                    f"every reference frame needs {candidates_per_frame} candidates"
                )
        return cls(tuple(row[:1] for row in rows), rows)


@dataclass(frozen=True, slots=True)
class ReferenceConstraint:
    """The semantic codes that steer one frame, and the mode that applies them."""

    mode: ReferenceMode
    candidates: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ReferencePlan:
    """Resolve which frames a reference stream steers and how."""

    codes: ReferenceCodes
    mode: ReferenceMode = ReferenceMode.GUIDANCE
    interval: int = MIN_REFERENCE_INTERVAL

    def __post_init__(self) -> None:
        if not isinstance(self.codes, ReferenceCodes):
            raise TypeError("reference_codes must be a ReferenceCodes instance")
        validate_reference_controls(self.mode, self.interval)

    @property
    def prefix_frames(self) -> int:
        """Return the frames prefilled as context before any frame is emitted."""

        return self.codes.num_frames if self.mode is ReferenceMode.CONTINUE else 0

    def covers(self, frame_index: int) -> bool:
        return 0 <= frame_index < self.codes.num_frames

    def prefills(self, frame_index: int) -> bool:
        """Whether this frame is injected from the reference instead of sampled."""

        return self.mode is ReferenceMode.CONTINUE and self.covers(frame_index)

    def constraint(self, frame_index: int) -> ReferenceConstraint | None:
        """Return the semantic-head constraint for one frame, or `None` to free-run.

        `CONTINUE` never returns a constraint: its prefix frames bypass the
        semantic head, and the frames after the prefix are free.
        """

        if not self.covers(frame_index) or self.mode is ReferenceMode.CONTINUE:
            return None
        if self.mode is ReferenceMode.COVER:
            return ReferenceConstraint(
                self.mode, (self.codes.semantic_code(frame_index),)
            )
        if frame_index % self.interval:
            return None
        return ReferenceConstraint(self.mode, self.codes.candidates_at(frame_index))


def validate_reference_controls(mode: object, interval: object) -> None:
    """Validate the reference mode and interval without a code stream."""

    if not isinstance(mode, ReferenceMode):
        raise TypeError("reference_mode must be a ReferenceMode")
    if isinstance(interval, bool) or not isinstance(interval, int):
        raise TypeError("reference_interval must be an integer")
    if not MIN_REFERENCE_INTERVAL <= interval <= MAX_REFERENCE_INTERVAL:
        raise ValueError(
            "reference_interval must be in "
            f"[{MIN_REFERENCE_INTERVAL}, {MAX_REFERENCE_INTERVAL}], got {interval}"
        )
    if mode is not ReferenceMode.GUIDANCE and interval != MIN_REFERENCE_INTERVAL:
        raise ValueError(
            "reference_interval applies only to ReferenceMode.GUIDANCE. "
            f"{mode} steers every covered frame"
        )


def _reference_columns(shape: tuple[int, ...], codes: tuple[int, ...]) -> mx.array:
    """Mark the narrowed-vocabulary columns of a reference's semantic codes."""

    # Column zero is the stop token, so semantic code k lives at column k + 1.
    columns = mx.array([code + 1 for code in codes], dtype=mx.int32)
    marked = mx.zeros(shape, dtype=mx.bool_)
    marked[:, columns] = True
    return marked


def cover_logits(guided: mx.array, codes: tuple[int, ...]) -> mx.array:
    """Leave only the reference's semantic columns reachable."""

    return mx.where(_reference_columns(guided.shape, codes), guided, -mx.inf)


def guidance_logits(
    guided: mx.array, window: mx.array, codes: tuple[int, ...]
) -> mx.array:
    """Bias a frame towards the reference without masking the model's own window.

    Reference candidates keep their guided logit even when the model ranks them
    outside its top-k window. Every other reachable column drops by a fixed penalty,
    so the model can still leave the reference. That finite bias is what separates
    guidance from cover.
    """

    return mx.where(
        _reference_columns(guided.shape, codes),
        guided,
        window - GUIDANCE_LOGIT_PENALTY,
    )


def stop_allowed(
    *,
    completed_frames: int,
    frame_index: int,
    min_frames: int,
    plan: ReferencePlan | None,
) -> bool:
    """Whether the semantic head may emit the audio-end token on this frame.

    A live reference window floors early stopping: the reference decides its own
    frames, and the loop's first frame is feedback for `<|audio_start|>`, so a stop
    there would void the reference and report an empty generation instead.
    """

    if completed_frames < min_frames:
        return False
    return plan is None or frame_index > plan.codes.num_frames


def validate_reference_stream(
    plan: ReferencePlan,
    *,
    num_codebooks: int,
    audio_vocab_size: int,
    max_frames: int,
    top_k: int,
) -> None:
    """Reject unusable reference streams and warn about degraded conditioning."""

    codes = plan.codes
    if codes.has_residual_codes:
        if codes.num_codebooks != num_codebooks:
            raise ValueError(
                f"reference frames carry {codes.num_codebooks} codebooks but the "
                f"depth decoder expects {num_codebooks}"
            )
        if any(
            code >= audio_vocab_size
            for frame in codes.code_frames
            for code in frame[1:]
        ):
            raise ValueError(
                f"residual reference codes must be below {audio_vocab_size}"
            )
    if codes.candidates_per_frame > top_k:
        warnings.warn(
            f"the reference carries {codes.candidates_per_frame} candidates per "
            f"frame, more than top_k={top_k}. The guided draw widens to keep every "
            "candidate reachable",
            ReferenceQualityWarning,
            stacklevel=3,
        )
    if plan.mode is not ReferenceMode.CONTINUE:
        if codes.num_frames > max_frames:
            warnings.warn(
                f"the reference covers {codes.num_frames} frames but the request "
                f"generates {max_frames}. The reference tail is unused",
                ReferenceQualityWarning,
                stacklevel=3,
            )
        return
    if not codes.has_residual_codes:
        warnings.warn(
            "a semantic-only reference cannot prefill residual codebooks. The depth "
            "decoder resynthesizes them for every prefix frame",
            ReferenceQualityWarning,
            stacklevel=3,
        )
    if plan.prefix_frames + max_frames > MAX_AUDIO_FRAMES:
        warnings.warn(
            f"the prefix of {plan.prefix_frames} frames plus {max_frames} generated "
            f"frames exceeds the {MAX_AUDIO_FRAMES}-frame model maximum",
            ReferenceQualityWarning,
            stacklevel=3,
        )


def _validate_semantic_code(code: object) -> None:
    if isinstance(code, bool) or not isinstance(code, int):
        raise TypeError("reference semantic codes must be integers")
    if not 0 <= code < SEMANTIC_VOCAB_SIZE:
        raise ValueError(
            "reference semantic codes must be in "
            f"[0, {SEMANTIC_VOCAB_SIZE - 1}], got {code}"
        )


def _validate_rows(rows: tuple[object, ...], *, label: str) -> None:
    for row in rows:
        if not isinstance(row, tuple) or not row:
            raise ValueError(f"every {label} needs at least one code")


def _rows(values: mx.array | Sequence[object], *, rank: int, label: str) -> list:
    if isinstance(values, mx.array):
        if values.ndim != rank:
            raise ValueError(f"{label} must have rank {rank}")
        mx.eval(values)
        return values.tolist()
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be an MLX array or a sequence")
    return list(values)


def _code_rows(
    values: mx.array | Sequence[Sequence[int]], *, label: str
) -> tuple[tuple[int, ...], ...]:
    rows = _rows(values, rank=2, label=label)
    for row in rows:
        if isinstance(row, str | bytes) or not isinstance(row, Sequence):
            raise TypeError(f"{label} must hold one sequence of codes per frame")
    return tuple(tuple(int(code) for code in row) for row in rows)


__all__ = [
    "GUIDANCE_LOGIT_PENALTY",
    "MAX_REFERENCE_INTERVAL",
    "MIN_REFERENCE_INTERVAL",
    "ReferenceCodes",
    "ReferenceConstraint",
    "ReferenceMode",
    "ReferencePlan",
    "ReferenceQualityWarning",
    "cover_logits",
    "guidance_logits",
    "stop_allowed",
    "validate_reference_controls",
    "validate_reference_stream",
]
