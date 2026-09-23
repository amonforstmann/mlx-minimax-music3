"""Reference conditioning over the real loop, sampler, and request objects."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_minimax_music3 import reference
from mlx_minimax_music3.autoregressive import (
    AutoregressiveConfig,
    AutoregressiveResult,
    generate_autoregressive,
)
from mlx_minimax_music3.models.qwen3 import Qwen3ForCausalLM
from mlx_minimax_music3.models.rvq_depth import RVQDepthDecoder
from mlx_minimax_music3.pipeline import GenerationRequest
from mlx_minimax_music3.reference import (
    ReferenceCodes,
    ReferenceMode,
    ReferenceQualityWarning,
)
from mlx_minimax_music3.tokenizer import TokenizedPrompt
from tests.support.tiny_autoregressive import build_tiny_models

pytestmark = pytest.mark.integration

_FRAMES = 8
_SEED = 4
_PREFIX = 3
_SEMANTIC_VOCABULARY = 16_384
_PROMPT = TokenizedPrompt(conditional=(1, 2, 3), unconditional=(1, 4, 3))


@pytest.fixture(scope="module")
def models() -> tuple[Qwen3ForCausalLM, RVQDepthDecoder]:
    return build_tiny_models(head_seed=23, max_position_embeddings=64)


def _config(frames: int, **overrides) -> AutoregressiveConfig:
    duration = frames / 25
    return AutoregressiveConfig(
        audio_duration=duration,
        min_audio_duration=duration,
        seed=_SEED,
        buffer_flush_interval=1,
        **overrides,
    )


def _generate(
    models: tuple[Qwen3ForCausalLM, RVQDepthDecoder],
    config: AutoregressiveConfig,
) -> AutoregressiveResult:
    language_model, decoder = models
    result = generate_autoregressive(language_model, decoder, _PROMPT, config)
    mx.eval(result.codes, result.frame_hiddens)
    return result


def _semantic_codes(result: AutoregressiveResult) -> tuple[int, ...]:
    return tuple(result.codes[0, :, 0].tolist())


def _shifted(codes: tuple[int, ...], offset: int) -> tuple[int, ...]:
    return tuple((code + offset) % _SEMANTIC_VOCABULARY for code in codes)


def test_null_reference_reproduces_the_free_running_run(models) -> None:
    baseline = _generate(models, _config(_FRAMES))
    repeated = _generate(models, _config(_FRAMES, reference_mode=ReferenceMode.COVER))

    assert mx.array_equal(repeated.codes, baseline.codes).item()
    assert mx.array_equal(repeated.frame_hiddens, baseline.frame_hiddens).item()


def test_cover_replays_the_captured_semantic_codes(models) -> None:
    baseline = _generate(models, _config(_FRAMES))
    captured = _semantic_codes(baseline)

    replayed = _generate(
        models,
        _config(
            _FRAMES,
            reference_codes=ReferenceCodes.from_semantic_codes(captured),
            reference_mode=ReferenceMode.COVER,
        ),
    )

    assert _semantic_codes(replayed) == captured
    # Forcing the semantic code back onto its own history leaves the residual
    # codebooks and the hidden states unchanged, so the whole frame replays.
    assert mx.array_equal(replayed.codes, baseline.codes).item()
    assert mx.array_equal(replayed.frame_hiddens, baseline.frame_hiddens).item()


def test_cover_replays_a_foreign_reference_stream(models) -> None:
    baseline = _generate(models, _config(_FRAMES))
    foreign = _shifted(_semantic_codes(baseline), 1_000)

    covered = _generate(
        models,
        _config(
            _FRAMES,
            reference_codes=ReferenceCodes.from_semantic_codes(foreign),
            reference_mode=ReferenceMode.COVER,
        ),
    )

    assert _semantic_codes(covered) == foreign


def test_guidance_displaces_guided_and_free_frames_when_the_bias_wins(
    models,
    monkeypatch,
) -> None:
    interval = 2
    baseline = _generate(models, _config(_FRAMES))
    captured = _semantic_codes(baseline)
    foreign = _shifted(captured, 4_096)
    # `GUIDANCE_LOGIT_PENALTY` is calibrated for the released checkpoint's logit
    # scale; this miniature model's gaps sit outside that band, so each guidance
    # test pins one end of it. Here the bias wins.
    monkeypatch.setattr(reference, "GUIDANCE_LOGIT_PENALTY", 64.0)

    guided = _generate(
        models,
        _config(
            _FRAMES,
            reference_codes=ReferenceCodes.from_semantic_codes(foreign),
            reference_mode=ReferenceMode.GUIDANCE,
            reference_interval=interval,
        ),
    )
    emitted = _semantic_codes(guided)
    guided_frames = range(0, _FRAMES, interval)
    free_frames = [frame for frame in range(_FRAMES) if frame % interval]

    assert emitted != captured
    # Steering one frame changes the history every later frame reads, so the free
    # frames move too. A free frame draws from its whole top-k window with the
    # baseline's seed and position, so a changed code there comes from the history.
    assert any(emitted[frame] != captured[frame] for frame in free_frames)
    assert sum(1 for frame in guided_frames if emitted[frame] == foreign[frame]) > 0


def test_guidance_bias_is_finite_and_loses_to_a_confident_model(
    models,
    monkeypatch,
) -> None:
    baseline = _generate(models, _config(_FRAMES))
    captured = _semantic_codes(baseline)
    foreign = _shifted(captured, 4_096)
    codes = ReferenceCodes.from_semantic_codes(foreign)
    # The other end of the calibrated band: a bias this small loses, which is the
    # property that separates guidance from cover.
    monkeypatch.setattr(reference, "GUIDANCE_LOGIT_PENALTY", 1.0)

    guided = _generate(
        models,
        _config(
            _FRAMES,
            reference_codes=codes,
            reference_mode=ReferenceMode.GUIDANCE,
        ),
    )
    covered = _generate(
        models,
        _config(
            _FRAMES,
            reference_codes=codes,
            reference_mode=ReferenceMode.COVER,
        ),
    )

    assert _semantic_codes(covered) == foreign
    assert _semantic_codes(guided) == captured


def test_continue_prefill_extends_the_captured_run(models) -> None:
    baseline = _generate(models, _config(_FRAMES))
    prefix = ReferenceCodes.from_code_frames(baseline.codes[0, :_PREFIX])

    continued = _generate(
        models,
        _config(
            _FRAMES,
            reference_codes=prefix,
            reference_mode=ReferenceMode.CONTINUE,
        ),
    )

    assert continued.num_frames == _FRAMES
    # The prefix is context, so the result starts where the captured run had
    # reached. Injecting a run's own frames therefore replays its tail exactly.
    overlap = _FRAMES - _PREFIX
    assert mx.array_equal(
        continued.codes[0, :overlap],
        baseline.codes[0, _PREFIX:],
    ).item()


def test_continue_from_semantic_codes_warns_about_resynthesis(models) -> None:
    baseline = _generate(models, _config(_FRAMES))
    prefix = ReferenceCodes.from_semantic_codes(
        _semantic_codes(baseline)[:_PREFIX]
    )

    with pytest.warns(ReferenceQualityWarning, match="resynthesizes"):
        continued = _generate(
            models,
            _config(
                _FRAMES,
                reference_codes=prefix,
                reference_mode=ReferenceMode.CONTINUE,
            ),
        )

    assert continued.num_frames == _FRAMES


def test_request_threads_reference_controls_into_the_stage_config() -> None:
    codes = ReferenceCodes.from_code_frames([[1, 2], [3, 4]])

    request = GenerationRequest(
        caption="caption",
        lyrics="lyrics",
        audio_duration=1.0,
        reference_codes=codes,
        reference_mode=ReferenceMode.CONTINUE,
    )
    plan = request.autoregressive_config.reference_plan

    assert plan is not None
    assert plan.codes == codes
    assert plan.mode is ReferenceMode.CONTINUE
    assert plan.prefix_frames == 2


def test_request_rejects_an_interval_outside_the_supported_range() -> None:
    with pytest.raises(ValueError, match="reference_interval must be in"):
        GenerationRequest(
            caption="caption",
            lyrics="lyrics",
            reference_interval=0,
        )


def test_request_rejects_an_interval_outside_guidance() -> None:
    with pytest.raises(ValueError, match="applies only to ReferenceMode.GUIDANCE"):
        GenerationRequest(
            caption="caption",
            lyrics="lyrics",
            reference_mode=ReferenceMode.COVER,
            reference_interval=2,
        )
