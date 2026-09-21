from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_minimax_music3.prompting import SEMANTIC_VOCAB_SIZE
from mlx_minimax_music3.reference import (
    GUIDANCE_LOGIT_PENALTY,
    ReferenceCodes,
    ReferenceMode,
    ReferencePlan,
    cover_logits,
    guidance_logits,
    stop_allowed,
    validate_reference_controls,
)


def test_semantic_stream_holds_one_code_per_frame() -> None:
    codes = ReferenceCodes.from_semantic_codes(mx.array([7, 9, 11], dtype=mx.int32))

    assert codes.code_frames == ((7,), (9,), (11,))
    assert codes.num_frames == 3
    assert codes.num_codebooks == 1
    assert not codes.has_residual_codes
    assert codes.candidates_per_frame == 1
    assert codes.candidates_at(1) == (9,)


def test_code_frame_stream_holds_every_codebook() -> None:
    grid = mx.array([[5, 1, 2], [6, 3, 4]], dtype=mx.int32)

    codes = ReferenceCodes.from_code_frames(grid)

    assert codes.code_frames == ((5, 1, 2), (6, 3, 4))
    assert codes.num_codebooks == 3
    assert codes.has_residual_codes
    assert codes.code_frame(1) == (6, 3, 4)
    assert codes.semantic_code(1) == 6
    # Without candidates a guided frame falls back to the frame's own code.
    assert codes.candidates_at(1) == (6,)


def test_code_frames_accept_ranked_semantic_candidates() -> None:
    codes = ReferenceCodes.from_code_frames(
        [[5, 1], [6, 2]],
        semantic_candidates=[[5, 40], [6, 41]],
    )

    assert codes.candidates_per_frame == 2
    assert codes.candidates_at(0) == (5, 40)
    assert codes.semantic_code(0) == 5


def test_candidate_stream_requires_the_declared_candidate_count() -> None:
    # A [frames, codebooks] code grid has the same shape as a [frames, k] candidate
    # grid, so the count is declared and checked instead of inferred.
    with pytest.raises(ValueError, match="needs 2 candidates"):
        ReferenceCodes.from_semantic_candidates(
            [[11, 12, 13], [14, 15, 16]],
            candidates_per_frame=2,
        )


def test_candidate_stream_derives_its_codes_from_the_best_candidate() -> None:
    codes = ReferenceCodes.from_semantic_candidates(
        [[11, 12], [14, 15]],
        candidates_per_frame=2,
    )

    assert codes.code_frames == ((11,), (14,))
    assert codes.candidates_at(1) == (14, 15)


def test_candidate_stream_rejects_a_non_positive_candidate_count() -> None:
    with pytest.raises(ValueError, match="candidates_per_frame"):
        ReferenceCodes.from_semantic_candidates([[1]], candidates_per_frame=0)


def test_stream_rejects_codes_outside_the_semantic_vocabulary() -> None:
    with pytest.raises(ValueError, match="reference semantic codes must be in"):
        ReferenceCodes.from_semantic_codes([SEMANTIC_VOCAB_SIZE])


def test_stream_rejects_negative_residual_codes() -> None:
    with pytest.raises(ValueError, match="residual reference codes"):
        ReferenceCodes.from_code_frames([[1, -2]])


def test_stream_rejects_frames_with_different_codebook_counts() -> None:
    with pytest.raises(ValueError, match="same codebook count"):
        ReferenceCodes(((1, 2), (3,)))


def test_stream_rejects_an_empty_frame_list() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        ReferenceCodes(())


def test_stream_rejects_a_frame_without_codes() -> None:
    with pytest.raises(ValueError, match="at least one code"):
        ReferenceCodes(((5,), ()))


def test_stream_rejects_boolean_codes() -> None:
    with pytest.raises(TypeError, match="must be integers"):
        ReferenceCodes(((True,),))


def test_stream_rejects_duplicate_candidates() -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        ReferenceCodes(((5,),), ((5, 5),))


def test_stream_rejects_candidates_that_do_not_lead_with_the_frame_code() -> None:
    with pytest.raises(ValueError, match="first candidate"):
        ReferenceCodes(((5,),), ((6, 5),))


def test_stream_rejects_candidates_that_miss_a_frame() -> None:
    with pytest.raises(ValueError, match="cover every reference frame"):
        ReferenceCodes(((5,), (6,)), ((5, 7),))


def test_semantic_stream_rejects_a_two_dimensional_array() -> None:
    with pytest.raises(ValueError, match="rank 1"):
        ReferenceCodes.from_semantic_codes(mx.zeros((2, 2), dtype=mx.int32))


def test_code_frame_stream_rejects_a_one_dimensional_array() -> None:
    with pytest.raises(ValueError, match="rank 2"):
        ReferenceCodes.from_code_frames(mx.zeros((2,), dtype=mx.int32))


def test_cover_plan_restricts_every_covered_frame_to_its_own_code() -> None:
    plan = ReferencePlan(
        codes=ReferenceCodes.from_code_frames([[4, 1], [6, 2]]),
        mode=ReferenceMode.COVER,
    )

    for frame_index, expected in enumerate(((4,), (6,))):
        constraint = plan.constraint(frame_index)
        assert constraint is not None
        assert constraint.mode is ReferenceMode.COVER
        assert constraint.candidates == expected
    assert plan.prefix_frames == 0
    assert not plan.prefills(0)


def test_guidance_plan_steers_only_every_interval_th_frame() -> None:
    plan = ReferencePlan(
        codes=ReferenceCodes.from_semantic_candidates(
            [(index, index + 100) for index in range(7)],
            candidates_per_frame=2,
        ),
        mode=ReferenceMode.GUIDANCE,
        interval=3,
    )

    steered = [
        frame_index
        for frame_index in range(7)
        if plan.constraint(frame_index) is not None
    ]

    assert steered == [0, 3, 6]
    first = plan.constraint(0)
    assert first is not None
    assert first.mode is ReferenceMode.GUIDANCE
    assert first.candidates == (0, 100)


def test_continue_plan_prefills_the_prefix_and_never_constrains() -> None:
    plan = ReferencePlan(
        codes=ReferenceCodes.from_code_frames([[12, 1], [13, 2]]),
        mode=ReferenceMode.CONTINUE,
    )

    assert plan.prefix_frames == 2
    assert plan.prefills(0)
    assert plan.prefills(1)
    assert not plan.prefills(2)
    assert plan.constraint(0) is None
    assert plan.constraint(1) is None


def test_plan_leaves_frames_outside_the_stream_free() -> None:
    plan = ReferencePlan(
        codes=ReferenceCodes.from_semantic_codes((1,)),
        mode=ReferenceMode.COVER,
    )

    assert plan.constraint(-1) is None
    assert plan.constraint(1) is None
    assert not plan.covers(1)


@pytest.mark.parametrize("interval", [0, -1, 11])
def test_controls_reject_intervals_outside_one_to_ten(interval: int) -> None:
    with pytest.raises(ValueError, match="reference_interval must be in"):
        validate_reference_controls(ReferenceMode.GUIDANCE, interval)


@pytest.mark.parametrize("interval", [True, 2.0, "2"])
def test_controls_reject_non_integer_intervals(interval: object) -> None:
    with pytest.raises(TypeError, match="reference_interval must be an integer"):
        validate_reference_controls(ReferenceMode.GUIDANCE, interval)


@pytest.mark.parametrize("mode", [ReferenceMode.COVER, ReferenceMode.CONTINUE])
def test_controls_reject_an_interval_outside_guidance(mode: ReferenceMode) -> None:
    with pytest.raises(ValueError, match="applies only to ReferenceMode.GUIDANCE"):
        validate_reference_controls(mode, 2)


def test_controls_reject_a_mode_that_is_not_the_enum() -> None:
    with pytest.raises(TypeError, match="reference_mode must be a ReferenceMode"):
        validate_reference_controls("cover", 1)


def test_plan_rejects_a_stream_that_is_not_reference_codes() -> None:
    with pytest.raises(TypeError, match="ReferenceCodes"):
        ReferencePlan(codes=((1,),))


def test_cover_masks_every_column_outside_the_reference() -> None:
    guided = mx.array([[1.0, 2.0, 3.0, 4.0]])

    masked = cover_logits(guided, (1,))
    mx.eval(masked)

    assert masked.tolist() == [[-float("inf"), -float("inf"), 3.0, -float("inf")]]


def test_guidance_penalizes_only_the_columns_outside_the_reference() -> None:
    guided = mx.array([[1.0, 2.0, 3.0, 4.0]])
    window = mx.array([[-float("inf"), 2.0, 3.0, 4.0]])

    biased = guidance_logits(guided, window, (0,))
    mx.eval(biased)

    assert biased.tolist() == [
        [
            -float("inf"),
            2.0,
            3.0 - GUIDANCE_LOGIT_PENALTY,
            4.0 - GUIDANCE_LOGIT_PENALTY,
        ]
    ]


def test_stop_stays_masked_until_the_reference_window_ends() -> None:
    plan = ReferencePlan(
        codes=ReferenceCodes.from_semantic_codes((5_000, 5_001)),
        mode=ReferenceMode.COVER,
    )

    # Loop frames zero to two are the feedback frame and the two covered frames.
    assert not any(
        stop_allowed(
            completed_frames=2,
            frame_index=frame_index,
            min_frames=0,
            plan=plan,
        )
        for frame_index in (0, 1, 2)
    )
    assert stop_allowed(
        completed_frames=2,
        frame_index=3,
        min_frames=0,
        plan=plan,
    )


def test_stop_floor_still_honours_the_minimum_duration() -> None:
    plan = ReferencePlan(
        codes=ReferenceCodes.from_semantic_codes((1,)),
        mode=ReferenceMode.COVER,
    )

    assert not stop_allowed(
        completed_frames=1,
        frame_index=9,
        min_frames=4,
        plan=plan,
    )


def test_stop_is_unchanged_without_a_reference() -> None:
    assert stop_allowed(
        completed_frames=0,
        frame_index=0,
        min_frames=0,
        plan=None,
    )
    assert not stop_allowed(
        completed_frames=1,
        frame_index=5,
        min_frames=2,
        plan=None,
    )
