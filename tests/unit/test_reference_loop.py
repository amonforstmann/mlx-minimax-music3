"""Reference conditioning as the autoregressive loop applies it."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import pytest

from mlx_minimax_music3 import autoregressive
from mlx_minimax_music3.autoregressive import AR_TOP_K, generate_autoregressive
from mlx_minimax_music3.prompting import SEMANTIC_VOCAB_SIZE
from mlx_minimax_music3.reference import (
    ReferenceCodes,
    ReferenceMode,
    ReferenceQualityWarning,
)
from mlx_minimax_music3.sampling import sample_top_k
from tests.support.tiny_autoregressive import (
    NUM_CODEBOOKS,
    build_tiny_models,
    fixed_length_config,
    tiny_prompt,
)

_C0_COLUMNS = 1 + SEMANTIC_VOCAB_SIZE


@dataclass(frozen=True, slots=True)
class _SemanticDraw:
    """One recorded c0 draw: its sampling width and its reachable columns."""

    top_k: int
    columns: tuple[int, ...]


class _DrawRecorder:
    """Record every sampler draw the loop makes, keyed by loop frame."""

    def __init__(self) -> None:
        self.semantic: dict[int, _SemanticDraw] = {}
        self.residual_frames: set[int] = set()

    def __call__(
        self, logits: mx.array, *, top_k: int, seed: int, position: int
    ) -> mx.array:
        frame = position // NUM_CODEBOOKS
        if logits.shape[-1] == _C0_COLUMNS:
            mx.eval(logits)
            self.semantic[frame] = _SemanticDraw(
                top_k,
                tuple(
                    column
                    for column, value in enumerate(logits[0].tolist())
                    if value != float("-inf")
                ),
            )
        else:
            self.residual_frames.add(frame)
        return sample_top_k(logits, top_k=top_k, seed=seed, position=position)


def test_cover_reference_forces_the_semantic_code_on_every_frame() -> None:
    language_model, decoder = build_tiny_models(
        head_seed=17,
        max_position_embeddings=64,
    )
    codes = (11, 4_242, 777, 16_383, 0, 8_192)

    result = generate_autoregressive(
        language_model,
        decoder,
        tiny_prompt(),
        fixed_length_config(
            len(codes),
            reference_codes=ReferenceCodes.from_semantic_codes(codes),
            reference_mode=ReferenceMode.COVER,
        ),
    )
    mx.eval(result.codes)

    assert tuple(result.codes[0, :, 0].tolist()) == codes


def test_reference_shorter_than_the_request_frees_the_remaining_frames() -> None:
    language_model, decoder = build_tiny_models(
        head_seed=17,
        max_position_embeddings=64,
    )
    codes = (5_000, 5_001)
    recorder = _DrawRecorder()

    result = generate_autoregressive(
        language_model,
        decoder,
        tiny_prompt(),
        fixed_length_config(
            5,
            reference_codes=ReferenceCodes.from_semantic_codes(codes),
            reference_mode=ReferenceMode.COVER,
        ),
        sampler=recorder,
    )
    mx.eval(result.codes)
    emitted = result.codes[0, :, 0].tolist()

    assert len(emitted) == 5
    assert tuple(emitted[: len(codes)]) == codes
    # Loop frame zero is feedback for <|audio_start|>, so reference frame zero is
    # steered at loop frame one. Column k + 1 carries semantic code k.
    reference_columns = {(codes[0] + 1,), (codes[1] + 1,)}
    assert sorted(recorder.semantic) == [0, 1, 2, 3, 4, 5]
    assert recorder.semantic[1].columns == (codes[0] + 1,)
    assert recorder.semantic[2].columns == (codes[1] + 1,)
    assert recorder.semantic[1].top_k == 1
    assert all(
        recorder.semantic[frame].columns not in reference_columns
        and recorder.semantic[frame].top_k == AR_TOP_K
        for frame in (0, 3, 4, 5)
    )


def test_reference_longer_than_the_request_warns_and_keeps_its_tail_unused() -> None:
    language_model, decoder = build_tiny_models(
        head_seed=17,
        max_position_embeddings=64,
    )
    codes = (6_000, 6_001, 6_002, 6_003, 6_004)

    with pytest.warns(ReferenceQualityWarning, match="tail is unused"):
        result = generate_autoregressive(
            language_model,
            decoder,
            tiny_prompt(),
            fixed_length_config(
                3,
                reference_codes=ReferenceCodes.from_semantic_codes(codes),
                reference_mode=ReferenceMode.COVER,
            ),
        )
    mx.eval(result.codes)

    assert tuple(result.codes[0, :, 0].tolist()) == codes[:3]


def test_guidance_biases_the_candidates_only_on_interval_frames() -> None:
    language_model, decoder = build_tiny_models(
        head_seed=17,
        max_position_embeddings=64,
    )
    candidates = [
        (code, code + 1, code + 2) for code in (100, 200, 300, 400, 500, 600)
    ]
    recorder = _DrawRecorder()

    generate_autoregressive(
        language_model,
        decoder,
        tiny_prompt(),
        fixed_length_config(
            len(candidates),
            reference_codes=ReferenceCodes.from_semantic_candidates(
                candidates,
                candidates_per_frame=3,
            ),
            reference_mode=ReferenceMode.GUIDANCE,
            reference_interval=3,
        ),
        sampler=recorder,
    )

    assert sorted(recorder.semantic) == [0, 1, 2, 3, 4, 5, 6]
    for frame, frame_candidates in ((1, candidates[0]), (4, candidates[3])):
        draw = recorder.semantic[frame]
        columns = set(draw.columns)
        # Every candidate is reachable and the model's own window survives, so
        # guidance biases the draw instead of replacing it.
        assert {code + 1 for code in frame_candidates} <= columns
        assert len(columns) > len(frame_candidates)
        assert draw.top_k == AR_TOP_K + len(frame_candidates)
    assert all(
        recorder.semantic[frame].top_k == AR_TOP_K for frame in (0, 2, 3, 5, 6)
    )


def test_continue_prefix_is_context_and_stays_out_of_the_result() -> None:
    language_model, decoder = build_tiny_models(
        head_seed=17,
        max_position_embeddings=64,
    )
    prefix = ReferenceCodes.from_code_frames(
        [[1_234, 1, 2, 3], [2_345, 4, 5, 6], [3_456, 7, 8, 9]]
    )
    recorder = _DrawRecorder()

    result = generate_autoregressive(
        language_model,
        decoder,
        tiny_prompt(),
        fixed_length_config(
            4,
            reference_codes=prefix,
            reference_mode=ReferenceMode.CONTINUE,
        ),
        sampler=recorder,
    )
    mx.eval(result.codes)

    assert result.num_frames == 4
    # The prefix occupies loop frames one through three and draws nothing: no
    # semantic code and no residual codes.
    assert sorted(recorder.semantic) == [0, 4, 5, 6, 7]
    assert recorder.residual_frames == {0, 4, 5, 6, 7}


def test_continue_prefill_evaluates_every_prefix_frame(monkeypatch) -> None:
    # Regression for amonforstmann/sonido-studio#17: an unevaluated prefill
    # chained a 1,500-frame prefix and its key-value writes into one lazy graph
    # that exhausted unified memory before the first emitted frame.
    language_model, decoder = build_tiny_models(
        head_seed=17,
        max_position_embeddings=64,
    )
    prefix_frames = 8
    prefix = ReferenceCodes.from_code_frames(
        [[1_000 + frame, 1, 2, 3] for frame in range(prefix_frames)]
    )
    evaluate = mx.eval
    feed = autoregressive.advance_frame
    fed_hiddens: list[mx.array] = []
    evaluations_after_feed: list[int] = []

    def recording_eval(*arrays):
        # Count only evaluations of the hidden state the last feed returned,
        # because every key-value write is an input of that array.
        if fed_hiddens and any(array is fed_hiddens[-1] for array in arrays):
            evaluations_after_feed[-1] += 1
        return evaluate(*arrays)

    def recording_feed(*args, **kwargs):
        hidden = feed(*args, **kwargs)
        fed_hiddens.append(hidden)
        evaluations_after_feed.append(0)
        return hidden

    monkeypatch.setattr(mx, "eval", recording_eval)
    monkeypatch.setattr(autoregressive, "advance_frame", recording_feed)

    generate_autoregressive(
        language_model,
        decoder,
        tiny_prompt(),
        fixed_length_config(
            2,
            reference_codes=prefix,
            reference_mode=ReferenceMode.CONTINUE,
        ),
    )

    # Feed zero is the <|audio_start|> feedback frame. The prefix frames draw no
    # code, so an evaluation after each of their feeds is what bounds the graph.
    assert len(evaluations_after_feed) >= 1 + prefix_frames
    assert 0 not in evaluations_after_feed[1 : prefix_frames + 1]


def test_continue_from_semantic_codes_warns_and_resynthesizes_residuals() -> None:
    language_model, decoder = build_tiny_models(
        head_seed=17,
        max_position_embeddings=64,
    )
    recorder = _DrawRecorder()

    with pytest.warns(ReferenceQualityWarning, match="resynthesizes"):
        result = generate_autoregressive(
            language_model,
            decoder,
            tiny_prompt(),
            fixed_length_config(
                3,
                reference_codes=ReferenceCodes.from_semantic_codes((1_234, 2_345)),
                reference_mode=ReferenceMode.CONTINUE,
            ),
            sampler=recorder,
        )

    assert result.num_frames == 3
    # The prefix frames skip the semantic head but still need residual codes.
    assert sorted(recorder.semantic) == [0, 3, 4, 5]
    assert recorder.residual_frames == {0, 1, 2, 3, 4, 5}


def test_reference_with_a_foreign_codebook_count_is_rejected() -> None:
    language_model, decoder = build_tiny_models(
        head_seed=17,
        max_position_embeddings=64,
    )

    with pytest.raises(ValueError, match="codebooks"):
        generate_autoregressive(
            language_model,
            decoder,
            tiny_prompt(),
            fixed_length_config(
                2,
                reference_codes=ReferenceCodes.from_code_frames([[1, 2], [3, 4]]),
                reference_mode=ReferenceMode.COVER,
            ),
        )


def test_reference_with_out_of_range_residual_codes_is_rejected() -> None:
    language_model, decoder = build_tiny_models(
        head_seed=17,
        max_position_embeddings=64,
    )

    with pytest.raises(ValueError, match="below 32"):
        generate_autoregressive(
            language_model,
            decoder,
            tiny_prompt(),
            fixed_length_config(
                2,
                reference_codes=ReferenceCodes.from_code_frames([[1, 2, 3, 99]]),
                reference_mode=ReferenceMode.CONTINUE,
            ),
        )


def test_more_candidates_than_top_k_warns_about_the_widened_draw() -> None:
    language_model, decoder = build_tiny_models(
        head_seed=17,
        max_position_embeddings=64,
    )
    candidates = [tuple(range(1_000, 1_004))]

    with pytest.warns(ReferenceQualityWarning, match="more than top_k=3"):
        generate_autoregressive(
            language_model,
            decoder,
            tiny_prompt(),
            fixed_length_config(
                2,
                top_k=3,
                reference_codes=ReferenceCodes.from_semantic_candidates(
                    candidates,
                    candidates_per_frame=4,
                ),
                reference_mode=ReferenceMode.GUIDANCE,
            ),
        )
