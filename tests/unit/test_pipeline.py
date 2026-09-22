from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_minimax_music3 import pipeline, stage_runners
from mlx_minimax_music3.acoustic import (
    AcousticLatents,
    AcousticResumeState,
    FlowGenerationConfig,
    LatentChunk,
)
from mlx_minimax_music3.autoregressive import AutoregressiveResult
from mlx_minimax_music3.chunking import ChunkWindow
from mlx_minimax_music3.decoding import Waveform
from mlx_minimax_music3.manifest import CheckpointManifest, ComponentManifest
from mlx_minimax_music3.stages import StageMemoryPolicy
from mlx_minimax_music3.tokenizer import TokenizedPrompt


def _write_manifest(root: Path, *, profile: str = "dense") -> None:
    components = tuple(
        ComponentManifest(name=name, files=())
        for name in sorted(pipeline._REQUIRED_COMPONENTS)
    )
    CheckpointManifest(
        profile=profile,
        source_repository="MiniMaxAI/MiniMax-Music3",
        source_revision="f" * 40,
        components=components,
        quantized_modules=(
            ("language_model.model.layers.0.self_attn.q_proj",)
            if profile == "q8"
            else ()
        ),
        quantization_mode="affine" if profile == "q8" else None,
        quantization_bits=8 if profile == "q8" else None,
        quantization_group_size=64 if profile == "q8" else None,
    ).write(root / "manifest.json")


@pytest.mark.usefixtures("isolated_stage_memory")
def test_private_pipeline_orders_residency_and_writes_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_manifest(tmp_path)
    load_order = []

    class FakeTokenizer:
        def encode_prompt(self, caption: str, lyrics: str) -> TokenizedPrompt:
            assert caption == "clear caption"
            assert lyrics == "[verse]\nclear lyrics"
            return TokenizedPrompt((1, 2, 3), (1, 2, 3))

    monkeypatch.setattr(
        pipeline.Qwen2BPETokenizer,
        "from_directory",
        classmethod(lambda cls, checkpoint: FakeTokenizer()),
    )

    def load_autoregressive(checkpoint: Path):
        load_order.append("autoregressive")
        return stage_runners._AutoregressiveModels(None, None)

    def load_acoustic(checkpoint: Path, flow_compute_dtype: str):
        assert flow_compute_dtype == "float32"
        load_order.append("acoustic")
        return stage_runners._AcousticModels(None, None)

    def load_vocoder(checkpoint: Path):
        load_order.append("decode")
        return object()

    monkeypatch.setattr(
        stage_runners, "_load_autoregressive_models", load_autoregressive
    )
    monkeypatch.setattr(stage_runners, "_load_acoustic_models", load_acoustic)
    monkeypatch.setattr(stage_runners, "load_vocoder", load_vocoder)
    monkeypatch.setattr(
        stage_runners,
        "generate_autoregressive",
        lambda *args, **kwargs: AutoregressiveResult(
            codes=mx.zeros((1, 1, 8), dtype=mx.int32),
            frame_hiddens=mx.zeros((1, 1, 32), dtype=mx.bfloat16),
            stopped_on_audio_end=False,
        ),
    )
    window = ChunkWindow(0, 0, 1, True, True)
    monkeypatch.setattr(
        stage_runners,
        "generate_acoustic_latents",
        lambda *args, **kwargs: AcousticLatents(
            chunks=(LatentChunk(window, mx.zeros((1, 3, 128))),)
        ),
    )
    monkeypatch.setattr(
        stage_runners,
        "decode_latent_chunks",
        lambda *args, **kwargs: Waveform(mx.zeros((2, 8)), sample_rate=44_100),
    )
    output = tmp_path / "outputs/smoke.wav"

    result = pipeline._run_pipeline(
        tmp_path,
        pipeline.GenerationRequest(
            caption="clear caption",
            lyrics="[verse]\nclear lyrics",
            audio_duration=0.04,
            seed=7,
        ),
        output=output,
        include_footprint=False,
    )

    assert load_order == ["autoregressive", "acoustic", "decode"]
    assert result.metadata.frame_count == 1
    assert result.metadata.chunk_count == 1
    assert result.metadata.seed == 7
    assert result.metadata.flow_compute_dtype == "float32"
    assert [report.label for report in result.metadata.memory_reports] == [
        "autoregressive",
        "acoustic",
        "decode",
    ]
    assert result.audio_file is not None
    with wave.open(str(output), "rb") as audio:
        assert audio.getnchannels() == 2
        assert audio.getframerate() == 44_100
        assert audio.getnframes() == 8


def test_checkpoint_validation_accepts_manifest_declared_q8(tmp_path: Path) -> None:
    _write_manifest(tmp_path, profile="q8")

    root, manifest = pipeline._validate_checkpoint(
        tmp_path,
        verify_digests=False,
    )

    assert root == tmp_path.resolve()
    assert manifest.profile == "q8"


def test_q8_pipeline_warns_that_quality_is_experimental(
    tmp_path: Path, monkeypatch
) -> None:
    _write_manifest(tmp_path, profile="q8")
    monkeypatch.setattr(
        pipeline.Qwen2BPETokenizer,
        "from_directory",
        classmethod(lambda cls, checkpoint: object()),
    )

    with pytest.warns(
        pipeline.ExperimentalQuantizationWarning,
        match="correctness baseline",
    ):
        instance = pipeline.Music3Pipeline(tmp_path)

    assert instance.checkpoint_profile == "q8"


def test_runtime_f16_flow_warns_that_quality_is_experimental(
    tmp_path: Path, monkeypatch
) -> None:
    _write_manifest(tmp_path)
    monkeypatch.setattr(
        pipeline.Qwen2BPETokenizer,
        "from_directory",
        classmethod(lambda cls, checkpoint: object()),
    )

    with pytest.warns(
        pipeline.ExperimentalPrecisionWarning,
        match="float32",
    ):
        instance = pipeline.Music3Pipeline(
            tmp_path,
            flow_compute_dtype="float16",
        )

    assert instance.checkpoint_profile == "dense"
    assert instance.flow_compute_dtype == "float16"


def test_pipeline_rejects_unknown_flow_compute_dtype(tmp_path: Path) -> None:
    _write_manifest(tmp_path)

    with pytest.raises(ValueError, match="flow_compute_dtype"):
        pipeline.Music3Pipeline(tmp_path, flow_compute_dtype="float8")


def test_generation_request_builds_validated_stage_configs() -> None:
    request = pipeline.GenerationRequest(
        caption="caption",
        lyrics="lyrics",
        audio_duration=4.0,
        seed=11,
        autoregressive_top_k=32,
        flow_steps=12,
    )

    assert request.autoregressive_config.max_frames == 100
    assert request.autoregressive_config.seed == 11
    assert request.autoregressive_config.top_k == 32
    assert request.flow_config.num_steps == 12


@pytest.mark.usefixtures("isolated_stage_memory")
def test_restored_stages_report_defined_metadata_and_progress(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    _write_manifest(checkpoint)

    class FakeTokenizer:
        def encode_prompt(self, caption: str, lyrics: str) -> TokenizedPrompt:
            return TokenizedPrompt((1, 2), (1, 2))

    monkeypatch.setattr(
        pipeline.Qwen2BPETokenizer,
        "from_directory",
        classmethod(lambda cls, root: FakeTokenizer()),
    )
    autoregressive = AutoregressiveResult(
        codes=mx.zeros((1, 250, 8), dtype=mx.int32),
        frame_hiddens=mx.zeros((1, 250, 32), dtype=mx.float32),
        stopped_on_audio_end=True,
    )
    windows = (
        ChunkWindow(0, 0, 200, True, False),
        ChunkWindow(1, 100, 250, False, True),
    )
    acoustic = AcousticLatents(
        chunks=tuple(
            LatentChunk(window, mx.full((1, 8, 4), window.index + 1.0))
            for window in windows
        )
    )

    monkeypatch.setattr(
        pipeline,
        "run_autoregressive_stage",
        lambda *args, **kwargs: (
            autoregressive,
            SimpleNamespace(label="autoregressive"),
        ),
    )

    def run_acoustic(*args, **kwargs):
        publish = kwargs["window_completed"]
        for chunk in acoustic.chunks:
            publish(
                SimpleNamespace(
                    chunk=chunk,
                    next_latent=mx.ones((1, 2, 4)),
                    next_condition=mx.ones((1, 2, 8)),
                )
            )
        return acoustic, SimpleNamespace(label="acoustic")

    monkeypatch.setattr(pipeline, "run_acoustic_stage", run_acoustic)
    monkeypatch.setattr(
        pipeline,
        "run_decode_stage",
        lambda *args, **kwargs: (
            Waveform(mx.zeros((2, 8)), sample_rate=44_100),
            SimpleNamespace(label="decode"),
        ),
    )
    request = pipeline.GenerationRequest(
        caption="caption",
        lyrics="lyrics",
        audio_duration=11.0,
        seed=7,
    )
    cache = tmp_path / "checkpoints"
    pipeline._run_pipeline(
        checkpoint,
        request,
        generation_checkpoint_dir=cache,
        include_footprint=False,
    )

    monkeypatch.setattr(
        pipeline,
        "run_autoregressive_stage",
        lambda *args, **kwargs: pytest.fail("restored AR stage was loaded"),
    )
    monkeypatch.setattr(
        pipeline,
        "run_acoustic_stage",
        lambda *args, **kwargs: pytest.fail("restored acoustic stage was loaded"),
    )
    ar_progress = []
    flow_progress = []
    result = pipeline._run_pipeline(
        checkpoint,
        request,
        generation_checkpoint_dir=cache,
        include_footprint=False,
        autoregressive_progress=ar_progress.append,
        flow_progress=flow_progress.append,
    )

    assert result.metadata.frame_count == 250
    assert result.metadata.chunk_count == 2
    assert result.metadata.stopped_on_audio_end is True
    assert [(timing.label, timing.seconds) for timing in result.metadata.stage_timings[:2]] == [
        ("autoregressive", 0.0),
        ("acoustic", 0.0),
    ]
    assert [report.label for report in result.metadata.memory_reports] == ["decode"]
    assert ar_progress[-1].completed_frames == 250
    assert [sample.chunk_index for sample in flow_progress] == [0, 1]
    assert all(sample.step == request.flow_steps for sample in flow_progress)


def test_acoustic_runner_forwards_resume_and_window_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}
    window = ChunkWindow(0, 0, 1, True, True)
    latents = AcousticLatents(chunks=(LatentChunk(window, mx.zeros((1, 3, 128))),))

    def generate(*args, **kwargs):
        received.update(kwargs)
        return latents

    monkeypatch.setattr(
        stage_runners,
        "_load_acoustic_models",
        lambda checkpoint, flow_compute_dtype: stage_runners._AcousticModels(
            None, None
        ),
    )
    monkeypatch.setattr(stage_runners, "generate_acoustic_latents", generate)
    resume = AcousticResumeState(
        chunks=(), previous_latent=None, previous_condition=None
    )

    def on_window(completed: object) -> None:
        return None

    stage_runners.run_acoustic_stage(
        Path("unused"),
        mx.zeros((1, 1, 32)),
        seed=0,
        config=FlowGenerationConfig(num_steps=1),
        flow_compute_dtype="float32",
        policy=StageMemoryPolicy(),
        include_footprint=False,
        progress=None,
        cancelled=None,
        resume=resume,
        window_completed=on_window,
    )

    assert received["resume"] is resume
    assert received["window_completed"] is on_window

