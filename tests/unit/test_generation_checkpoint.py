from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx

from mlx_minimax_music3.acoustic import LatentChunk
from mlx_minimax_music3.autoregressive import AutoregressiveResult
from mlx_minimax_music3.chunking import ChunkWindow
from mlx_minimax_music3.generation_checkpoint import (
    GenerationCheckpointStore,
    generation_fingerprint,
)
from mlx_minimax_music3.manifest import CheckpointManifest, ComponentManifest
from mlx_minimax_music3.pipeline import GenerationRequest
from mlx_minimax_music3.reference import ReferenceCodes, ReferenceMode


def _model_manifest() -> CheckpointManifest:
    return CheckpointManifest(
        profile="dense",
        source_repository="MiniMaxAI/MiniMax-Music3",
        source_revision="f" * 40,
        components=(ComponentManifest(name="transformer", files=()),),
    )


def _request() -> GenerationRequest:
    return GenerationRequest(
        caption="caption", lyrics="lyrics", audio_duration=10.0, seed=7
    )


def _store(tmp_path: Path) -> GenerationCheckpointStore:
    return GenerationCheckpointStore(
        tmp_path,
        request=_request(),
        flow_compute_dtype="float32",
        model_manifest=_model_manifest(),
    )


def test_checkpoint_round_trip_skips_completed_autoregressive_stage(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    expected = AutoregressiveResult(
        codes=mx.arange(16, dtype=mx.int32).reshape(1, 2, 8),
        frame_hiddens=mx.ones((1, 2, 32), dtype=mx.bfloat16),
        stopped_on_audio_end=True,
    )

    store.save_autoregressive(expected)
    restored = _store(tmp_path).restore(
        autoregressive_config=_request().autoregressive_config
    )

    assert restored.autoregressive is not None
    assert mx.array_equal(restored.autoregressive.codes, expected.codes).item()
    assert mx.array_equal(
        restored.autoregressive.frame_hiddens, expected.frame_hiddens
    ).item()
    assert restored.autoregressive.stopped_on_audio_end is True


def test_checkpoint_resumes_from_first_missing_acoustic_window(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    autoregressive = AutoregressiveResult(
        codes=mx.zeros((1, 250, 8), dtype=mx.int32),
        frame_hiddens=mx.zeros((1, 250, 32), dtype=mx.float32),
        stopped_on_audio_end=False,
    )
    store.save_autoregressive(autoregressive)
    first = LatentChunk(
        ChunkWindow(0, 0, 200, True, False),
        mx.ones((1, 688, 128), dtype=mx.float32),
    )
    store.save_acoustic_window(
        first,
        next_latent=mx.ones((1, 172, 128), dtype=mx.float32),
        next_condition=mx.ones((1, 172, 1024), dtype=mx.float32),
    )

    restored = _store(tmp_path).restore(
        autoregressive_config=_request().autoregressive_config
    )

    assert restored.acoustic is not None
    assert restored.acoustic.chunks[0].window == first.window
    assert mx.array_equal(
        restored.acoustic.chunks[0].latents, first.latents
    ).item()
    assert restored.acoustic.previous_latent.shape == (1, 172, 128)
    assert restored.acoustic.previous_condition.shape == (1, 172, 1024)


def test_incompatible_or_partial_checkpoint_recomputes_last_valid_boundary(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    autoregressive = AutoregressiveResult(
        codes=mx.zeros((1, 250, 8), dtype=mx.int32),
        frame_hiddens=mx.zeros((1, 250, 32), dtype=mx.float32),
        stopped_on_audio_end=False,
    )
    store.save_autoregressive(autoregressive)
    first = LatentChunk(
        ChunkWindow(0, 0, 200, True, False),
        mx.ones((1, 688, 128), dtype=mx.float32),
    )
    store.save_acoustic_window(
        first,
        next_latent=mx.ones((1, 172, 128), dtype=mx.float32),
        next_condition=mx.ones((1, 172, 1024), dtype=mx.float32),
    )
    artifact = store.directory / "acoustic-0000.safetensors"
    artifact.write_bytes(b"corrupt")

    restored = _store(tmp_path).restore(
        autoregressive_config=_request().autoregressive_config
    )

    assert restored.autoregressive is not None
    assert restored.acoustic is None

    (store.directory / "autoregressive.safetensors").write_bytes(b"corrupt")
    restored = _store(tmp_path).restore(
        autoregressive_config=_request().autoregressive_config
    )
    assert restored.autoregressive is None
    assert restored.acoustic is None


def test_reference_conditioning_changes_the_generation_fingerprint() -> None:
    text_only = _request()
    referenced = replace(
        text_only,
        reference_codes=ReferenceCodes.from_semantic_codes((5, 6)),
        reference_mode=ReferenceMode.COVER,
    )
    identity = {
        "flow_compute_dtype": "float32",
        "model_manifest": _model_manifest(),
    }

    assert generation_fingerprint(text_only, **identity) != generation_fingerprint(
        referenced, **identity
    )


def test_inconsistent_autoregressive_stop_state_invalidates_cache(
    tmp_path: Path,
) -> None:
    for stopped in (False, True):
        root = tmp_path / str(stopped)
        request = GenerationRequest(
            caption="caption",
            lyrics="lyrics",
            audio_duration=2.0,
            min_audio_duration=1.0,
        )
        store = GenerationCheckpointStore(
            root,
            request=request,
            flow_compute_dtype="float32",
            model_manifest=_model_manifest(),
        )
        store.save_autoregressive(
            AutoregressiveResult(
                codes=mx.zeros((1, 2, 8), dtype=mx.int32),
                frame_hiddens=mx.zeros((1, 2, 32), dtype=mx.float32),
                stopped_on_audio_end=stopped,
            )
        )

        restored = GenerationCheckpointStore(
            root,
            request=request,
            flow_compute_dtype="float32",
            model_manifest=_model_manifest(),
        ).restore(autoregressive_config=request.autoregressive_config)

        assert restored.autoregressive is None


def test_stopped_autoregressive_result_at_maximum_frames_is_invalid(
    tmp_path: Path,
) -> None:
    request = GenerationRequest(
        caption="caption",
        lyrics="lyrics",
        audio_duration=0.08,
    )
    store = GenerationCheckpointStore(
        tmp_path,
        request=request,
        flow_compute_dtype="float32",
        model_manifest=_model_manifest(),
    )
    store.save_autoregressive(
        AutoregressiveResult(
            codes=mx.zeros((1, 2, 8), dtype=mx.int32),
            frame_hiddens=mx.zeros((1, 2, 32), dtype=mx.float32),
            stopped_on_audio_end=True,
        )
    )

    restored = GenerationCheckpointStore(
        tmp_path,
        request=request,
        flow_compute_dtype="float32",
        model_manifest=_model_manifest(),
    ).restore(autoregressive_config=request.autoregressive_config)

    assert restored.autoregressive is None


def test_empty_final_acoustic_carry_round_trips_shape_and_dtype(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.save_autoregressive(
        AutoregressiveResult(
            codes=mx.zeros((1, 250, 8), dtype=mx.int32),
            frame_hiddens=mx.zeros((1, 250, 32), dtype=mx.float32),
            stopped_on_audio_end=False,
        )
    )
    store.save_acoustic_window(
        LatentChunk(
            ChunkWindow(0, 0, 200, True, False),
            mx.ones((1, 688, 128), dtype=mx.float32),
        ),
        next_latent=mx.ones((1, 172, 128), dtype=mx.float32),
        next_condition=mx.ones((1, 172, 1024), dtype=mx.float32),
    )
    store.save_acoustic_window(
        LatentChunk(
            ChunkWindow(1, 100, 250, False, True),
            mx.ones((1, 516, 128), dtype=mx.float32),
        ),
        next_latent=mx.zeros((1, 0, 128), dtype=mx.float16),
        next_condition=mx.zeros((1, 0, 1024), dtype=mx.float32),
    )

    restored = _store(tmp_path).restore(
        autoregressive_config=_request().autoregressive_config
    )

    assert restored.acoustic is not None
    assert len(restored.acoustic.chunks) == 2
    assert restored.acoustic.previous_latent.shape == (1, 0, 128)
    assert restored.acoustic.previous_latent.dtype == mx.float16
    assert restored.acoustic.previous_condition.shape == (1, 0, 1024)
    assert restored.acoustic.previous_condition.dtype == mx.float32


def test_invalid_utf8_manifest_is_treated_as_disposable_cache_corruption(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.directory.mkdir(parents=True)
    (store.directory / "manifest.json").write_bytes(b"\xff")

    restored = _store(tmp_path).restore(
        autoregressive_config=_request().autoregressive_config
    )

    assert restored.autoregressive is None
    assert restored.acoustic is None
