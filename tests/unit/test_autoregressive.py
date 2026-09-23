from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_minimax_music3.autoregressive import (
    AutoregressiveConfig,
    GenerationProgress,
    _restricted_semantic_logits,
    generate_autoregressive,
)
from mlx_minimax_music3.prompting import (
    AUDIO_CODE_OFFSET,
    AUDIO_END_TOKEN_ID,
    SEMANTIC_VOCAB_SIZE,
)
from mlx_minimax_music3.reference import ReferenceCodes, ReferenceMode
from tests.support.tiny_autoregressive import (
    HIDDEN_SIZE,
    NUM_CODEBOOKS,
    build_tiny_models,
    fixed_length_config,
    tiny_prompt,
)


def _argmax_sampler(
    logits: mx.array, *, top_k: int, seed: int, position: int
) -> mx.array:
    del top_k, seed, position
    # c0 column zero is the stop token in the narrowed reference vocabulary.
    return mx.array([1 if logits.shape[-1] == 16_385 else 0], dtype=mx.int32)


def _literal_argmax_sampler(
    logits: mx.array, *, top_k: int, seed: int, position: int
) -> mx.array:
    del top_k, seed, position
    return mx.argmax(logits, axis=-1).astype(mx.int32)


def test_tiny_autoregressive_loop_produces_aligned_frames() -> None:
    language_model, decoder = build_tiny_models()
    prompt = tiny_prompt()

    result = generate_autoregressive(
        language_model,
        decoder,
        prompt,
        AutoregressiveConfig(audio_duration=2 / 25, buffer_flush_interval=1),
        sampler=_argmax_sampler,
    )
    mx.eval(result.codes, result.frame_hiddens)

    assert result.codes.shape == (1, 2, 4)
    assert result.frame_hiddens.shape == (1, 2, NUM_CODEBOOKS * HIDDEN_SIZE)
    assert not result.stopped_on_audio_end
    assert mx.array_equal(result.codes, mx.zeros_like(result.codes)).item()


def test_minimum_duration_masks_early_stop_until_required_frames() -> None:
    language_model, decoder = build_tiny_models()
    prompt = tiny_prompt()

    result = generate_autoregressive(
        language_model,
        decoder,
        prompt,
        AutoregressiveConfig(
            audio_duration=4 / 25,
            min_audio_duration=2 / 25,
            buffer_flush_interval=1,
        ),
        sampler=_literal_argmax_sampler,
    )
    mx.eval(result.codes, result.frame_hiddens)

    assert result.codes.shape == (1, 2, 4)
    assert result.stopped_on_audio_end


def test_progress_without_a_reference_reports_an_empty_prefix() -> None:
    language_model, decoder = build_tiny_models()
    events: list[GenerationProgress] = []

    generate_autoregressive(
        language_model,
        decoder,
        tiny_prompt(),
        AutoregressiveConfig(audio_duration=3 / 25, buffer_flush_interval=1),
        sampler=_argmax_sampler,
        progress=events.append,
    )

    assert [
        (
            event.completed_frames,
            event.maximum_frames,
            event.prefilled_frames,
            event.prefix_frames,
        )
        for event in events
    ] == [(1, 3, 0, 0), (2, 3, 0, 0), (3, 3, 0, 0)]


def test_generation_progress_defaults_the_prefill_fields_to_zero() -> None:
    progress = GenerationProgress(4, 10)

    assert progress == GenerationProgress(
        completed_frames=4,
        maximum_frames=10,
        prefilled_frames=0,
        prefix_frames=0,
    )


def test_restricted_semantic_head_matches_full_projection() -> None:
    language_model, _ = build_tiny_models()
    weight = language_model.lm_head.weight
    rows = mx.arange(weight.shape[0], dtype=mx.float32)[:, None]
    columns = mx.arange(weight.shape[1], dtype=mx.float32)[None, :]
    language_model.lm_head.weight = mx.sin(rows * 0.001 + columns * 0.01).astype(
        mx.bfloat16
    )
    hidden = (mx.arange(32, dtype=mx.float32).reshape(2, 16) / 32).astype(mx.bfloat16)
    full = language_model.lm_head(hidden).astype(mx.float32)
    ids = mx.concatenate(
        (
            mx.array([AUDIO_END_TOKEN_ID], dtype=mx.int32),
            mx.arange(
                AUDIO_CODE_OFFSET,
                AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE,
                dtype=mx.int32,
            ),
        )
    )
    expected = full[:, ids]
    actual = _restricted_semantic_logits(
        language_model,
        hidden,
        allow_stop=True,
    )
    mx.eval(expected, actual)

    assert mx.allclose(actual, expected, rtol=0, atol=0).item()


def test_reference_controls_without_codes_match_the_free_running_baseline() -> None:
    language_model, decoder = build_tiny_models(
        head_seed=17,
        max_position_embeddings=64,
    )
    prompt = tiny_prompt()

    baseline = generate_autoregressive(
        language_model,
        decoder,
        prompt,
        fixed_length_config(6),
    )
    steerable = generate_autoregressive(
        language_model,
        decoder,
        prompt,
        fixed_length_config(6, reference_mode=ReferenceMode.COVER),
    )
    mx.eval(
        baseline.codes,
        baseline.frame_hiddens,
        steerable.codes,
        steerable.frame_hiddens,
    )

    assert mx.array_equal(steerable.codes, baseline.codes).item()
    assert mx.array_equal(steerable.frame_hiddens, baseline.frame_hiddens).item()
    assert steerable.stopped_on_audio_end == baseline.stopped_on_audio_end


@pytest.mark.parametrize("interval", [0, 11])
def test_autoregressive_config_rejects_out_of_range_reference_interval(
    interval: int,
) -> None:
    with pytest.raises(ValueError, match="reference_interval must be in"):
        AutoregressiveConfig(reference_interval=interval)


def test_autoregressive_config_rejects_a_reference_mode_string() -> None:
    with pytest.raises(TypeError, match="reference_mode must be a ReferenceMode"):
        AutoregressiveConfig(reference_mode="cover")


def test_autoregressive_config_rejects_unwrapped_reference_codes() -> None:
    with pytest.raises(TypeError, match="reference_codes must be a ReferenceCodes"):
        AutoregressiveConfig(reference_codes=((1,),))


def test_reference_plan_is_absent_without_a_code_stream() -> None:
    config = AutoregressiveConfig(reference_mode=ReferenceMode.COVER)

    assert config.reference_plan is None


def test_reference_plan_carries_the_configured_controls() -> None:
    codes = ReferenceCodes.from_semantic_codes((3, 4))

    plan = AutoregressiveConfig(
        reference_codes=codes,
        reference_mode=ReferenceMode.GUIDANCE,
        reference_interval=2,
    ).reference_plan

    assert plan is not None
    assert plan.codes == codes
    assert plan.mode is ReferenceMode.GUIDANCE
    assert plan.interval == 2


def test_prefix_frames_is_zero_without_a_reference() -> None:
    config = AutoregressiveConfig(reference_mode=ReferenceMode.CONTINUE)

    assert config.prefix_frames == 0


@pytest.mark.parametrize(
    ("mode", "prefix_frames"),
    [
        (ReferenceMode.GUIDANCE, 0),
        (ReferenceMode.COVER, 0),
        (ReferenceMode.CONTINUE, 2),
    ],
)
def test_prefix_frames_counts_only_a_continue_reference(
    mode: ReferenceMode,
    prefix_frames: int,
) -> None:
    config = AutoregressiveConfig(
        reference_codes=ReferenceCodes.from_semantic_codes((3, 4)),
        reference_mode=mode,
    )

    assert config.prefix_frames == prefix_frames


def test_autoregressive_config_rejects_an_interval_outside_guidance() -> None:
    with pytest.raises(ValueError, match="applies only to ReferenceMode.GUIDANCE"):
        AutoregressiveConfig(reference_mode=ReferenceMode.COVER, reference_interval=3)


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_autoregressive_config_rejects_seed_outside_uint64(seed: int) -> None:
    with pytest.raises(ValueError, match="64-bit"):
        AutoregressiveConfig(seed=seed)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_autoregressive_config_rejects_non_finite_controls(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        AutoregressiveConfig(audio_duration=value)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_autoregressive_config_rejects_invalid_minimum_duration(value: float) -> None:
    with pytest.raises(ValueError, match="min_audio_duration"):
        AutoregressiveConfig(min_audio_duration=value)


def test_autoregressive_config_rejects_minimum_above_ceiling() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        AutoregressiveConfig(audio_duration=1.0, min_audio_duration=1.1)
