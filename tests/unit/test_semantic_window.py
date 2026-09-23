"""The semantic top-k window as the autoregressive loop hands it to the sampler."""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

from mlx_minimax_music3.autoregressive import AR_TOP_K, generate_autoregressive
from mlx_minimax_music3.models.qwen3 import Qwen3ForCausalLM
from mlx_minimax_music3.models.rvq_depth import RVQDepthDecoder
from mlx_minimax_music3.prompting import AUDIO_CODE_OFFSET, SEMANTIC_VOCAB_SIZE
from mlx_minimax_music3.reference import ReferenceCodes, ReferenceMode
from tests.support.tiny_autoregressive import (
    C0_COLUMNS,
    DrawRecorder,
    build_tiny_models,
    fixed_length_config,
    tiny_prompt,
)

_FRAMES = 4
_SEEDS = range(8)
_CANDIDATES = [(code, code + 1, code + 2) for code in (100, 200, 300, 400)]
_TIE_GROUP = 4

_Models = tuple[Qwen3ForCausalLM, RVQDepthDecoder]


@pytest.fixture(scope="module")
def models() -> _Models:
    return build_tiny_models(head_seed=23, max_position_embeddings=64)


@pytest.fixture(scope="module")
def tied_models() -> _Models:
    """Return the tiny pair with each run of four semantic codes sharing a head row.

    Identical rows give identical logits for every hidden state, so the conditional
    and guided rows of every frame tie in groups of four. A `top_k` that is not a
    multiple of four then ends the window inside a tied group on every frame.
    """

    language_model, decoder = build_tiny_models(
        head_seed=23, max_position_embeddings=64
    )
    weight = language_model.lm_head.weight
    start, stop = AUDIO_CODE_OFFSET, AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE
    shared = mx.repeat(weight[start:stop:_TIE_GROUP], _TIE_GROUP, axis=0)
    language_model.lm_head.weight = mx.concatenate(
        (weight[:start], shared, weight[stop:])
    )
    mx.eval(language_model.lm_head.weight)
    return language_model, decoder


def _record(
    models: _Models,
    monkeypatch: pytest.MonkeyPatch,
    frames: int,
    **overrides,
) -> DrawRecorder:
    language_model, decoder = models
    recorder = DrawRecorder(monkeypatch)
    generate_autoregressive(
        language_model,
        decoder,
        tiny_prompt(),
        fixed_length_config(frames, **overrides),
        sampler=recorder,
    )
    return recorder


def _record_guided(
    models: _Models, monkeypatch: pytest.MonkeyPatch, top_k: int
) -> DrawRecorder:
    return _record(
        models,
        monkeypatch,
        len(_CANDIDATES),
        top_k=top_k,
        reference_codes=ReferenceCodes.from_semantic_candidates(
            _CANDIDATES, candidates_per_frame=3
        ),
        reference_mode=ReferenceMode.GUIDANCE,
    )


@pytest.mark.parametrize("top_k", [1, 5, AR_TOP_K])
def test_free_semantic_draw_reaches_the_whole_top_k_window(
    models, monkeypatch, top_k: int
) -> None:
    recorder = _record(models, monkeypatch, _FRAMES, top_k=top_k)

    assert sorted(recorder.semantic) == list(range(_FRAMES + 1))
    for frame, draw in recorder.semantic.items():
        assert len(draw.columns) == min(top_k, C0_COLUMNS)
        assert set(draw.columns) == recorder.window(frame, top_k)


@pytest.mark.filterwarnings("ignore::mlx_minimax_music3.reference.ReferenceQualityWarning")
@pytest.mark.parametrize("top_k", [1, 5, AR_TOP_K])
def test_guided_semantic_draw_reaches_the_window_and_every_candidate(
    models, monkeypatch, top_k: int
) -> None:
    recorder = _record_guided(models, monkeypatch, top_k)

    # Loop frame zero is feedback for <|audio_start|>, so reference frame zero is
    # guided at loop frame one. Column k + 1 carries semantic code k.
    for reference_frame, candidates in enumerate(_CANDIDATES):
        frame = reference_frame + 1
        expected = recorder.window(frame, top_k) | {code + 1 for code in candidates}
        assert set(recorder.semantic[frame].columns) == expected
        assert recorder.semantic[frame].top_k == len(expected)
    assert set(recorder.semantic[0].columns) == recorder.window(0, top_k)


@pytest.mark.parametrize("top_k", [1, 5, AR_TOP_K])
def test_semantic_window_keeps_every_column_tied_at_its_boundary(
    tied_models, monkeypatch, top_k: int
) -> None:
    # Regression for amonforstmann/sonido-studio#24: SGLang masks only columns
    # below the kth largest conditional logit, so a tie there widens the window.
    recorder = _record(tied_models, monkeypatch, _FRAMES, top_k=top_k)

    for frame, draw in recorder.semantic.items():
        window = recorder.window(frame, top_k)
        assert len(window) == math.ceil(top_k / _TIE_GROUP) * _TIE_GROUP
        assert set(draw.columns) == window


def test_semantic_draw_reaches_every_column_tied_at_the_boundary(
    tied_models, monkeypatch
) -> None:
    # Regression for amonforstmann/sonido-studio#24: the draw kept exactly top_k
    # columns, so at top_k 1 it drew one of four tied codes on every seed.
    draws = set()
    for seed in range(32):
        recorder = _record(tied_models, monkeypatch, 1, seed=seed, top_k=1)
        draws.add(recorder.semantic[0].drawn)

    assert draws == recorder.window(0, 1)
    assert len(draws) == _TIE_GROUP


@pytest.mark.filterwarnings("ignore::mlx_minimax_music3.reference.ReferenceQualityWarning")
@pytest.mark.parametrize("top_k", [1, 5, AR_TOP_K])
def test_guided_draw_widened_by_boundary_ties_reaches_every_candidate(
    tied_models, monkeypatch, top_k: int
) -> None:
    # Regression for amonforstmann/sonido-studio#24: a guided draw sized as top_k
    # plus the candidate count cut columns once a boundary tie widened the window.
    recorder = _record_guided(tied_models, monkeypatch, top_k)

    for reference_frame, candidates in enumerate(_CANDIDATES):
        frame = reference_frame + 1
        window = recorder.window(frame, top_k)
        draw = recorder.semantic[frame]
        assert len(window) > top_k
        assert set(draw.columns) == window | {code + 1 for code in candidates}
        assert draw.top_k == len(draw.columns)


def _first_semantic_draw(
    models: _Models,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seed: int,
    top_k: int,
) -> tuple[int, DrawRecorder]:
    # The first draw reads the prompt alone, so only the seed and the window can
    # change it between runs.
    recorder = _record(models, monkeypatch, 1, seed=seed, top_k=top_k)
    return recorder.semantic[0].drawn, recorder


def test_first_semantic_draw_varies_with_the_seed_above_top_k_one(
    models, monkeypatch
) -> None:
    draws = {
        _first_semantic_draw(models, monkeypatch, seed=seed, top_k=AR_TOP_K)[0]
        for seed in _SEEDS
    }

    assert len(draws) > 1


def test_first_semantic_draw_at_top_k_one_is_the_conditional_argmax(
    models, monkeypatch
) -> None:
    draws = set()
    for seed in _SEEDS:
        drawn, recorder = _first_semantic_draw(models, monkeypatch, seed=seed, top_k=1)
        assert {drawn} == recorder.window(0, 1)
        draws.add(drawn)

    assert len(draws) == 1
