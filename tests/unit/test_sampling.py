from __future__ import annotations

import math

import mlx.core as mx
import pytest

from mlx_minimax_music3.sampling import (
    SeedSchedule,
    classifier_free_guidance,
    derive_acoustic_seed,
    derive_sampling_seed,
    kth_largest,
    murmur_hash32,
    restrict_top_k,
    sample_top_k,
)

# `mx.topk(_UNSORTED, k=3)` returns [[7, 8, 9]] on MLX 0.32. The last element is
# the maximum, not the kth largest, and MLX does not document any order, so no
# fixed index is a kth-largest selector (amonforstmann/sonido-studio#24).
_UNSORTED = mx.array([[5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0]])
_DESCENDING = [9.0, 8.0, 7.0, 5.0, 3.0, 2.0, 1.0]
# Columns one through four tie at the largest value, so top_k 1 ends inside a tie.
_TIED_AT_TOP = mx.array([[1.0, 4.0, 4.0, 4.0, 4.0, 0.0]])
# The second largest value is 3.0 and three columns hold it, so top_k 2 keeps four.
_TIED_AT_SECOND = mx.array([[5.0, 3.0, 0.0, 3.0, 3.0, 1.0]])


def test_seed_schedule_is_position_stable() -> None:
    schedule = SeedSchedule(seed=42, num_codebooks=8)

    assert schedule.sampling_seed == 1_153_769_322
    assert schedule.position(frame_index=3, codebook_index=4) == 28
    assert schedule.position(frame_index=3, codebook_index=5) == 29


def test_seed_derivation_matches_sglang_reference_vectors() -> None:
    assert derive_sampling_seed("minimax-ttm-ar", 0) == 411_363_039
    assert derive_sampling_seed("minimax-ttm-ar", 3) == 122_074_423
    assert derive_acoustic_seed(3, "dit", 0) == 1_648_571_301_556_962_109
    assert derive_acoustic_seed(3, "dit", 1) == 5_080_984_550_091_710_395


def test_murmur_hash_matches_reference_vectors() -> None:
    seed = 1_153_769_322

    assert murmur_hash32(seed, 0, 0) == 1_795_620_021
    assert murmur_hash32(seed, 0, 1) == 1_827_116_861
    assert murmur_hash32(seed, 7, 1_023) == 390_523_399
    assert murmur_hash32(seed, 24, 16_384) == 3_048_668_828


def test_top_k_sampling_never_draws_a_masked_token() -> None:
    logits = mx.array([[10.0, 9.0, 8.0, 7.0]])
    samples = [
        int(
            sample_top_k(
                logits,
                top_k=2,
                seed=derive_sampling_seed("minimax-ttm-ar", seed),
                position=0,
            ).item()
        )
        for seed in range(32)
    ]

    assert set(samples) <= {0, 1}


def test_top_k_sampling_matches_reference_gumbel_vector() -> None:
    sampled = sample_top_k(
        mx.array([[1.0, 2.0, 3.0, 4.0]]),
        top_k=4,
        seed=1_809_552_049,
        position=0,
    )

    assert sampled.item() == 3


def _draw(logits: mx.array, *, top_k: int, public_seed: int, position: int) -> int:
    return int(
        sample_top_k(
            logits,
            top_k=top_k,
            seed=derive_sampling_seed("minimax-ttm-ar", public_seed),
            position=position,
        ).item()
    )


def test_top_k_sampling_draws_every_column_tied_with_the_kth_largest() -> None:
    # Regression for amonforstmann/sonido-studio#24: SGLang masks only values below
    # the kth largest, so every column tied with it stays a candidate.
    draws = {
        _draw(_TIED_AT_TOP, top_k=1, public_seed=seed, position=0)
        for seed in range(32)
    }

    assert draws == {1, 2, 3, 4}


def test_top_k_sampling_with_a_boundary_tie_matches_the_widened_draw() -> None:
    # Regression for amonforstmann/sonido-studio#24: a boundary tie widens the
    # candidate set, and each candidate keeps its own column noise.
    for seed in range(16):
        for position in range(4):
            assert _draw(
                _TIED_AT_SECOND, top_k=2, public_seed=seed, position=position
            ) == _draw(_TIED_AT_SECOND, top_k=4, public_seed=seed, position=position)


def test_classifier_free_guidance_uses_fp32() -> None:
    logits = mx.array([[3.0, 5.0], [1.0, 2.0]], dtype=mx.bfloat16)

    guided = classifier_free_guidance(logits, scale=1.5)
    mx.eval(guided)

    assert guided.dtype == mx.float32
    assert mx.allclose(guided, mx.array([[4.0, 6.5]])).item()


def test_seed_schedule_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="64-bit"):
        SeedSchedule(seed=-1, num_codebooks=8)


@pytest.mark.parametrize("top_k", [1, 3, 6, 7])
def test_restrict_top_k_keeps_the_k_largest_of_unsorted_logits(top_k: int) -> None:
    restricted = restrict_top_k(_UNSORTED, top_k)
    mx.eval(restricted)

    kept = [value for value in restricted[0].tolist() if math.isfinite(value)]

    assert sorted(kept, reverse=True) == _DESCENDING[:top_k]


@pytest.mark.parametrize("k", [1, 2, 3, 6, 7])
def test_kth_largest_ignores_the_order_topk_returns(k: int) -> None:
    threshold = kth_largest(_UNSORTED, k)

    assert threshold.tolist() == [[_DESCENDING[k - 1]]]


def test_kth_largest_selects_within_each_row_of_a_batch() -> None:
    values = mx.array(
        [
            [5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0],
            [0.5, 4.0, -2.0, 6.0, 1.5, 3.0, 2.5],
        ]
    )

    assert kth_largest(values, 2).tolist() == [[8.0], [4.0]]
    assert kth_largest(values, 4).tolist() == [[5.0], [2.5]]


@pytest.mark.parametrize("k", [0, 8])
def test_kth_largest_rejects_k_outside_the_row(k: int) -> None:
    with pytest.raises(ValueError, match=r"k must be in \[1, 7\]"):
        kth_largest(_UNSORTED, k)
