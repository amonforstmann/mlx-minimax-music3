"""Verify reference-code conditioning against a local Music 3 checkpoint.

The pytest suite stays weightless, so this script carries the checkpoint-backed
evidence for the three reference modes. It runs the autoregressive stage only and
compares emitted codes, which is where reference steering acts.

Every run pins `min_audio_duration` to the requested duration so no run can end
early and misalign the comparison. The continue run injects the baseline's own
frames, so its result must replay the baseline's tail exactly.

`--penalties` sweeps `GUIDANCE_LOGIT_PENALTY`, the value the released constant is
calibrated from. Each penalty guides the prompt towards a plausible reference (the
semantic codes of a baseline rendered from `--plausible-caption` and
`--plausible-lyrics`), an implausible one (the baseline shifted by 4,096 codes),
and, with `--reference-npz`, an encoder stream of real audio. A guided frame is on
the reference when its emitted code is one of that frame's reference candidates.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import mlx.core as mx

from mlx_minimax_music3 import reference
from mlx_minimax_music3.autoregressive import (
    AutoregressiveConfig,
    AutoregressiveResult,
    generate_autoregressive,
)
from mlx_minimax_music3.loading import load_language_model, load_rvq_depth_decoder
from mlx_minimax_music3.models.qwen3 import Qwen3ForCausalLM
from mlx_minimax_music3.models.rvq_depth import RVQDepthDecoder
from mlx_minimax_music3.reference import (
    GUIDANCE_LOGIT_PENALTY,
    ReferenceCodes,
    ReferenceMode,
)
from mlx_minimax_music3.tokenizer import Qwen2BPETokenizer, TokenizedPrompt

_SEMANTIC_VOCABULARY = 16_384
_PREFIX_FRAMES = 25
_IMPLAUSIBLE_SHIFT = 4_096

Render = Callable[..., AutoregressiveResult]


def _semantic_codes(result: AutoregressiveResult) -> tuple[int, ...]:
    mx.eval(result.codes)
    return tuple(result.codes[0, :, 0].tolist())


def _matched_frames(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(1 for one, other in zip(left, right, strict=False) if one == other)


def _shifted(codes: tuple[int, ...], offset: int) -> tuple[int, ...]:
    return tuple((code + offset) % _SEMANTIC_VOCABULARY for code in codes)


def _penalties(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"penalties must be comma-separated numbers, got {text!r}"
        ) from error
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise argparse.ArgumentTypeError("penalties must be finite and not negative")
    return values


def _render(
    language_model: Qwen3ForCausalLM,
    decoder: RVQDepthDecoder,
    prompt: TokenizedPrompt,
    *,
    duration: float,
    seed: int,
    **overrides: object,
) -> AutoregressiveResult:
    config = AutoregressiveConfig(
        audio_duration=duration,
        min_audio_duration=duration,
        seed=seed,
        **overrides,
    )
    result = generate_autoregressive(language_model, decoder, prompt, config)
    mx.eval(result.codes, result.frame_hiddens)
    return result


def _load_encoder_stream(path: Path, frames: int) -> ReferenceCodes:
    """Load the `codes` and `semantic_candidates` arrays an encoder run saved."""

    arrays = mx.load(str(path))
    # The request generates `frames` frames, so a longer stream's tail is unused.
    return ReferenceCodes.from_code_frames(
        arrays["codes"][:frames],
        semantic_candidates=arrays["semantic_candidates"][:frames],
    )


def _frames_on_reference(
    emitted: tuple[int, ...], codes: ReferenceCodes, frames: range
) -> int:
    return sum(1 for frame in frames if emitted[frame] in codes.candidates_at(frame))


def _guided_with_penalty(
    run: Render, codes: ReferenceCodes, *, interval: int, penalty: float
) -> tuple[int, ...]:
    # `guidance_logits` reads the module global on every guided frame.
    released = reference.GUIDANCE_LOGIT_PENALTY
    reference.GUIDANCE_LOGIT_PENALTY = penalty
    try:
        return _semantic_codes(
            run(
                reference_codes=codes,
                reference_mode=ReferenceMode.GUIDANCE,
                reference_interval=interval,
            )
        )
    finally:
        reference.GUIDANCE_LOGIT_PENALTY = released


def _penalty_sweep(
    run: Render,
    captured: tuple[int, ...],
    references: dict[str, ReferenceCodes],
    *,
    interval: int,
    penalties: Sequence[float],
) -> dict[str, object]:
    """Guide towards each reference at each penalty and count what followed."""

    sweep: dict[str, object] = {}
    for name, codes in references.items():
        guided_frames = range(0, min(codes.num_frames, len(captured)), interval)
        rows: dict[str, dict[str, int]] = {}
        for penalty in penalties:
            emitted = _guided_with_penalty(
                run, codes, interval=interval, penalty=penalty
            )
            rows[str(penalty)] = {
                "guided_frames": len(guided_frames),
                "frames_on_reference": _frames_on_reference(
                    emitted, codes, guided_frames
                ),
                "frames_changed": sum(
                    1
                    for one, other in zip(emitted, captured, strict=True)
                    if one != other
                ),
            }
        sweep[name] = {
            "baseline_frames_on_reference": _frames_on_reference(
                captured, codes, guided_frames
            ),
            "penalties": rows,
        }
    return sweep


def verify_reference_conditioning(
    checkpoint: Path,
    *,
    caption: str,
    lyrics: str,
    duration: float,
    seed: int,
    interval: int,
    penalties: Sequence[float] = (),
    plausible_caption: str | None = None,
    plausible_lyrics: str | None = None,
    reference_npz: Path | None = None,
) -> dict[str, object]:
    """Run one free baseline and one run per reference mode, then compare codes.

    With `penalties`, also sweep the guidance penalty against a plausible, an
    implausible, and optionally a real reference, which needs the plausible prompt.
    """

    if penalties and (plausible_caption is None or plausible_lyrics is None):
        raise ValueError("a penalty sweep needs a plausible caption and lyrics")
    tokenizer = Qwen2BPETokenizer.from_directory(checkpoint)
    prompt = tokenizer.encode_prompt(caption, lyrics)
    language_model = load_language_model(checkpoint)
    decoder = load_rvq_depth_decoder(checkpoint)
    render = partial(
        _render, language_model, decoder, duration=duration, seed=seed
    )
    run = partial(render, prompt)

    baseline = run()
    captured = _semantic_codes(baseline)

    null_reference = run(reference_mode=ReferenceMode.COVER)
    cover = run(
        reference_codes=ReferenceCodes.from_semantic_codes(captured),
        reference_mode=ReferenceMode.COVER,
    )
    foreign = _shifted(captured, _IMPLAUSIBLE_SHIFT)
    foreign_codes = ReferenceCodes.from_semantic_codes(foreign)
    foreign_cover = run(
        reference_codes=foreign_codes,
        reference_mode=ReferenceMode.COVER,
    )
    guidance = run(
        reference_codes=foreign_codes,
        reference_mode=ReferenceMode.GUIDANCE,
        reference_interval=interval,
    )
    dense_guidance = run(
        reference_codes=foreign_codes,
        reference_mode=ReferenceMode.GUIDANCE,
    )
    prefix = baseline.codes[0, :_PREFIX_FRAMES]
    continued = run(
        reference_codes=ReferenceCodes.from_code_frames(prefix),
        reference_mode=ReferenceMode.CONTINUE,
    )

    guided_frames = range(0, len(foreign), interval)
    guidance_codes = _semantic_codes(guidance)
    overlap = baseline.num_frames - _PREFIX_FRAMES
    report: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "duration": duration,
        "seed": seed,
        "interval": interval,
        "guidance_logit_penalty": GUIDANCE_LOGIT_PENALTY,
        "baseline_frames": baseline.num_frames,
        "null_reference_codes_identical": bool(
            mx.array_equal(null_reference.codes, baseline.codes).item()
        ),
        "null_reference_hiddens_identical": bool(
            mx.array_equal(null_reference.frame_hiddens, baseline.frame_hiddens).item()
        ),
        "cover_captured_frames": len(captured),
        "cover_matched_frames": _matched_frames(_semantic_codes(cover), captured),
        "cover_codes_identical": bool(
            mx.array_equal(cover.codes, baseline.codes).item()
        ),
        "cover_foreign_matched_frames": _matched_frames(
            _semantic_codes(foreign_cover), foreign
        ),
        "guidance_guided_frames": len(guided_frames),
        "guidance_frames_on_reference": sum(
            1 for frame in guided_frames if guidance_codes[frame] == foreign[frame]
        ),
        "guidance_frames_changed": sum(
            1
            for emitted, original in zip(guidance_codes, captured, strict=True)
            if emitted != original
        ),
        "guidance_free_frames_changed": sum(
            1
            for frame, (emitted, original) in enumerate(
                zip(guidance_codes, captured, strict=True)
            )
            if frame % interval and emitted != original
        ),
        "dense_guidance_frames_on_reference": _matched_frames(
            _semantic_codes(dense_guidance), foreign
        ),
        "dense_guidance_matches_cover": bool(
            mx.array_equal(dense_guidance.codes, foreign_cover.codes).item()
        ),
        "continue_prefix_frames": _PREFIX_FRAMES,
        "continue_frames": continued.num_frames,
        "continue_tail_identical": bool(
            mx.array_equal(
                continued.codes[0, :overlap],
                baseline.codes[0, _PREFIX_FRAMES:],
            ).item()
        ),
    }
    if not penalties:
        return report

    plausible_prompt = tokenizer.encode_prompt(plausible_caption, plausible_lyrics)
    references = {
        "plausible": ReferenceCodes.from_semantic_codes(
            _semantic_codes(render(plausible_prompt))
        ),
        "implausible": foreign_codes,
    }
    if reference_npz is not None:
        references["real"] = _load_encoder_stream(reference_npz, baseline.num_frames)
    report["penalty_sweep"] = _penalty_sweep(
        run, captured, references, interval=interval, penalties=penalties
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--lyrics", required=True)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument(
        "--penalties",
        type=_penalties,
        default=(),
        help="comma-separated guidance penalties to sweep, for example 8,16,24",
    )
    parser.add_argument("--plausible-caption")
    parser.add_argument("--plausible-lyrics")
    parser.add_argument(
        "--reference-npz",
        type=Path,
        help="encoder stream with `codes` [frames, 8] and "
        "`semantic_candidates` [frames, k] int32 arrays",
    )
    args = parser.parse_args()
    sweep_inputs = (args.plausible_caption, args.plausible_lyrics, args.reference_npz)
    if args.penalties and (
        args.plausible_caption is None or args.plausible_lyrics is None
    ):
        parser.error("--penalties needs --plausible-caption and --plausible-lyrics")
    if not args.penalties and any(value is not None for value in sweep_inputs):
        parser.error(
            "--plausible-caption, --plausible-lyrics, and --reference-npz "
            "apply only with --penalties"
        )
    report = verify_reference_conditioning(
        args.checkpoint,
        caption=args.caption,
        lyrics=args.lyrics,
        duration=args.duration,
        seed=args.seed,
        interval=args.interval,
        penalties=args.penalties,
        plausible_caption=args.plausible_caption,
        plausible_lyrics=args.plausible_lyrics,
        reference_npz=args.reference_npz,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
