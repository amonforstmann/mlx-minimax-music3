"""Golden numerical regression tests over real dense and q8 loaders."""

from __future__ import annotations

import hashlib
from typing import Any, cast

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_minimax_music3.acoustic import FlowGenerationConfig, generate_acoustic_latents
from mlx_minimax_music3.autoregressive import AutoregressiveResult
from mlx_minimax_music3.decoding import decode_latent_chunks
from mlx_minimax_music3.generation_checkpoint import GenerationCheckpointStore
from mlx_minimax_music3.loading import (
    load_condition_encoder,
    load_flow_transformer,
    load_language_model,
    load_rvq_depth_decoder,
    load_vocoder,
)
from mlx_minimax_music3.manifest import CheckpointManifest
from mlx_minimax_music3.pipeline import GenerationRequest
from tests.support.golden_checkpoint import (
    GoldenCheckpoints,
    load_golden_contract,
)

pytestmark = pytest.mark.integration


def _topology_digest(*models) -> str:
    rows = []
    for model in models:
        rows.extend(
            f"{name}|{tuple(value.shape)}|{value.dtype}"
            for name, value in tree_flatten(model.parameters())
        )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _values(values: mx.array) -> tuple[float, ...]:
    return tuple(float(value) for value in values.astype(mx.float32).tolist())


@pytest.mark.parametrize("profile", ("dense", "q8"))
def test_golden_checkpoint_is_small_and_self_contained(
    golden_checkpoints: GoldenCheckpoints,
    profile: str,
) -> None:
    contract = load_golden_contract()
    source = cast(dict[str, str], contract["source"])
    checkpoint = getattr(golden_checkpoints, profile)
    manifest = CheckpointManifest.read(checkpoint / "manifest.json")

    manifest.verify(checkpoint, digests=True)
    assert manifest.profile == profile
    assert manifest.source_repository == source["repository"]
    assert manifest.source_revision == source["revision"]
    assert sum(
        record.size
        for component in manifest.components
        for record in component.files
    ) < int(contract["maximum_checkpoint_bytes"])

    if profile == "q8":
        assert manifest.quantization_bits == 8
        assert manifest.quantization_group_size == 64
        assert manifest.quantized_modules


@pytest.mark.parametrize(
    ("expectation", "checkpoint_profile", "flow_compute_dtype"),
    (
        ("dense", "dense", mx.float32),
        ("q8", "q8", mx.float32),
        ("runtime-f16-flow", "dense", mx.float16),
    ),
)
def test_golden_checkpoint_detects_inference_regressions(
    golden_checkpoints: GoldenCheckpoints,
    expectation: str,
    checkpoint_profile: str,
    flow_compute_dtype: mx.Dtype,
) -> None:
    contract = load_golden_contract()
    generation = cast(dict[str, Any], contract["generation"])
    expected_profiles = cast(dict[str, dict[str, Any]], contract["expected"])
    tolerances = cast(dict[str, float], contract["absolute_tolerances"])
    checkpoint = getattr(golden_checkpoints, checkpoint_profile)
    language_model = load_language_model(checkpoint)
    depth_decoder = load_rvq_depth_decoder(checkpoint)
    condition_encoder = load_condition_encoder(checkpoint)
    transformer = load_flow_transformer(
        checkpoint,
        compute_dtype=flow_compute_dtype,
    )
    vocoder = load_vocoder(checkpoint)

    language_output = language_model(
        mx.array([generation["token_ids"]], dtype=mx.int32)
    )
    language_last = language_output.last_hidden_state[:, -1]
    depth_inputs = mx.stack(
        tuple(
            language_last * float(scale)
            for scale in generation["depth_input_scales"]
        ),
        axis=1,
    )
    depth_hidden = depth_decoder(depth_inputs)[:, -1]
    frame_hiddens = mx.concatenate(
        (language_last[:, None, :], depth_hidden[:, None, :]), axis=-1
    )
    acoustic = generate_acoustic_latents(
        transformer,
        condition_encoder,
        frame_hiddens,
        seed=int(generation["flow_seed"]),
        config=FlowGenerationConfig(
            num_steps=int(generation["flow_steps"]),
            cfg_scale=float(generation["cfg_scale"]),
        ),
    )
    waveform = decode_latent_chunks(vocoder, acoustic)
    latents = acoustic.chunks[0].latents
    mx.eval(language_output.logits, depth_hidden, latents, waveform.samples)

    expected = expected_profiles[expectation]
    assert _topology_digest(
        language_model,
        depth_decoder,
        condition_encoder,
        transformer,
        vocoder,
    ) == expected["topology"]

    snapshots = {
        "language": _values(
            language_output.logits[
                0, -1, : int(generation["language_slice_size"])
            ]
        ),
        "depth": _values(
            depth_hidden[0, : int(generation["depth_slice_size"])]
        ),
        "latents": _values(latents.reshape(-1)),
        "waveform": _values(waveform.samples.reshape(-1)),
    }
    for name, actual in snapshots.items():
        assert actual == pytest.approx(
            expected[name],
            abs=tolerances[name],
            rel=0.0,
        )


def test_golden_checkpoint_resume_recomputes_only_the_missing_acoustic_suffix(
    golden_checkpoints: GoldenCheckpoints,
    tmp_path,
) -> None:
    contract = load_golden_contract()
    generation = cast(dict[str, Any], contract["generation"])
    checkpoint = golden_checkpoints.dense
    manifest = CheckpointManifest.read(checkpoint / "manifest.json")
    language_model = load_language_model(checkpoint)
    depth_decoder = load_rvq_depth_decoder(checkpoint)
    language_output = language_model(
        mx.array([generation["token_ids"]], dtype=mx.int32)
    )
    language_last = language_output.last_hidden_state[:, -1]
    depth_inputs = mx.stack(
        tuple(
            language_last * float(scale)
            for scale in generation["depth_input_scales"]
        ),
        axis=1,
    )
    depth_hidden = depth_decoder(depth_inputs)[:, -1]
    one_frame = mx.concatenate(
        (language_last[:, None, :], depth_hidden[:, None, :]), axis=-1
    )
    frame_hiddens = mx.repeat(one_frame, 250, axis=1)
    request = GenerationRequest(
        caption="golden",
        lyrics="golden",
        audio_duration=10.0,
        seed=int(generation["flow_seed"]),
        flow_steps=int(generation["flow_steps"]),
        flow_cfg_scale=float(generation["cfg_scale"]),
    )
    store = GenerationCheckpointStore(
        tmp_path / "checkpoints",
        request=request,
        flow_compute_dtype="float32",
        model_manifest=manifest,
    )
    store.save_autoregressive(
        AutoregressiveResult(
            codes=mx.zeros((1, 250, 2), dtype=mx.int32),
            frame_hiddens=frame_hiddens,
            stopped_on_audio_end=False,
        )
    )
    flow_config = request.flow_config

    def publish(completed) -> None:
        store.save_acoustic_window(
            completed.chunk,
            next_latent=completed.next_latent,
            next_condition=completed.next_condition,
        )

    clean = generate_acoustic_latents(
        load_flow_transformer(checkpoint),
        load_condition_encoder(checkpoint),
        frame_hiddens,
        seed=request.seed,
        config=flow_config,
        window_completed=publish,
    )
    (store.directory / "acoustic-0001.safetensors").unlink()
    restored = GenerationCheckpointStore(
        tmp_path / "checkpoints",
        request=request,
        flow_compute_dtype="float32",
        model_manifest=manifest,
    ).restore(autoregressive_config=request.autoregressive_config)
    assert restored.acoustic is not None
    assert len(restored.acoustic.chunks) == 1

    completed_windows = []

    def record_completed(completed) -> None:
        completed_windows.append(completed.chunk.window.index)

    assert restored.autoregressive is not None
    resumed = generate_acoustic_latents(
        load_flow_transformer(checkpoint),
        load_condition_encoder(checkpoint),
        restored.autoregressive.frame_hiddens,
        seed=request.seed,
        config=flow_config,
        resume=restored.acoustic,
        window_completed=record_completed,
    )

    assert completed_windows == [1]
    assert len(resumed.chunks) == len(clean.chunks) == 2
    vocoder = load_vocoder(checkpoint)
    for expected, actual in zip(clean.chunks, resumed.chunks, strict=True):
        expected_waveform = vocoder(expected.latents)
        actual_waveform = vocoder(actual.latents)
        mx.eval(expected.latents, actual.latents, expected_waveform, actual_waveform)
        assert mx.array_equal(actual.latents, expected.latents).item()
        assert mx.array_equal(actual_waveform, expected_waveform).item()
