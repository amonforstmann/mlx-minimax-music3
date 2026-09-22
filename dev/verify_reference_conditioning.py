"""Verify reference-code conditioning against a local Music 3 checkpoint.

The pytest suite stays weightless, so this script carries the checkpoint-backed
evidence for the three reference modes. It runs the autoregressive stage only and
compares emitted codes, which is where reference steering acts.

Every run pins `min_audio_duration` to the requested duration so no run can end
early and misalign the comparison. The continue run injects the baseline's own
frames, so its result must replay the baseline's tail exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from mlx_minimax_music3.autoregressive import (
    AutoregressiveConfig,
    AutoregressiveResult,
    generate_autoregressive,
)
from mlx_minimax_music3.loading import load_language_model, load_rvq_depth_decoder
from mlx_minimax_music3.reference import (
    GUIDANCE_LOGIT_PENALTY,
    ReferenceCodes,
    ReferenceMode,
)
from mlx_minimax_music3.tokenizer import Qwen2BPETokenizer

_SEMANTIC_VOCABULARY = 16_384
_PREFIX_FRAMES = 25


def _semantic_codes(result: AutoregressiveResult) -> tuple[int, ...]:
    mx.eval(result.codes)
    return tuple(result.codes[0, :, 0].tolist())


def _matched_frames(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(1 for one, other in zip(left, right, strict=False) if one == other)


def _shifted(codes: tuple[int, ...], offset: int) -> tuple[int, ...]:
    return tuple((code + offset) % _SEMANTIC_VOCABULARY for code in codes)


def verify_reference_conditioning(
    checkpoint: Path,
    *,
    caption: str,
    lyrics: str,
    duration: float,
    seed: int,
    interval: int,
) -> dict[str, object]:
    """Run one free baseline and one run per reference mode, then compare codes."""

    tokenizer = Qwen2BPETokenizer.from_directory(checkpoint)
    prompt = tokenizer.encode_prompt(caption, lyrics)
    language_model = load_language_model(checkpoint)
    decoder = load_rvq_depth_decoder(checkpoint)

    def run(**overrides: object) -> AutoregressiveResult:
        config = AutoregressiveConfig(
            audio_duration=duration,
            min_audio_duration=duration,
            seed=seed,
            **overrides,
        )
        result = generate_autoregressive(language_model, decoder, prompt, config)
        mx.eval(result.codes, result.frame_hiddens)
        return result

    baseline = run()
    captured = _semantic_codes(baseline)

    null_reference = run(reference_mode=ReferenceMode.COVER)
    cover = run(
        reference_codes=ReferenceCodes.from_semantic_codes(captured),
        reference_mode=ReferenceMode.COVER,
    )
    foreign = _shifted(captured, 4_096)
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
    return {
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--lyrics", required=True)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interval", type=int, default=4)
    args = parser.parse_args()
    report = verify_reference_conditioning(
        args.checkpoint,
        caption=args.caption,
        lyrics=args.lyrics,
        duration=args.duration,
        seed=args.seed,
        interval=args.interval,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
