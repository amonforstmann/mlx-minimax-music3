"""Runners for the autoregressive, acoustic, and decode stages.

Each runner loads one stage's weights inside its own `StageSession`, runs the
stage, evaluates the outputs the next stage needs, and releases the weights. It
returns those outputs with the stage's memory report. The pipeline calls the
runners in order. Model-specific loading lives here so that `stages.py` stays
model-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from .acoustic import (
    AcousticLatents,
    AcousticResumeState,
    CompletedAcousticWindow,
    FlowGenerationConfig,
    FlowProgress,
    generate_acoustic_latents,
)
from .autoregressive import (
    AutoregressiveConfig,
    AutoregressiveResult,
    GenerationProgress,
    generate_autoregressive,
)
from .decoding import Waveform, decode_latent_chunks
from .loading import (
    load_condition_encoder,
    load_flow_transformer,
    load_language_model,
    load_rvq_depth_decoder,
    load_vocoder,
)
from .models.condition_encoder import ConditionEncoder
from .models.flow_transformer import FlowTransformer
from .models.qwen3 import Qwen3ForCausalLM
from .models.rvq_depth import RVQDepthDecoder
from .stages import StageMemoryPolicy, StageMemoryReport, StageSession
from .tokenizer import TokenizedPrompt

FLOW_COMPUTE_DTYPES = {
    "float16": mx.float16,
    "float32": mx.float32,
}


@dataclass(frozen=True, slots=True)
class _AutoregressiveModels:
    language_model: Qwen3ForCausalLM
    depth_decoder: RVQDepthDecoder


@dataclass(frozen=True, slots=True)
class _AcousticModels:
    condition_encoder: ConditionEncoder
    transformer: FlowTransformer


def _load_autoregressive_models(checkpoint: Path) -> _AutoregressiveModels:
    return _AutoregressiveModels(
        language_model=load_language_model(checkpoint),
        depth_decoder=load_rvq_depth_decoder(checkpoint),
    )


def _load_acoustic_models(
    checkpoint: Path,
    flow_compute_dtype: str,
) -> _AcousticModels:
    return _AcousticModels(
        condition_encoder=load_condition_encoder(checkpoint),
        transformer=load_flow_transformer(
            checkpoint,
            compute_dtype=FLOW_COMPUTE_DTYPES[flow_compute_dtype],
        ),
    )


def run_autoregressive_stage(
    checkpoint: Path,
    prompt: TokenizedPrompt,
    config: AutoregressiveConfig,
    *,
    policy: StageMemoryPolicy,
    include_footprint: bool,
    progress: Callable[[GenerationProgress], None] | None,
    cancelled: Callable[[], bool] | None,
) -> tuple[AutoregressiveResult, StageMemoryReport]:
    if progress is not None and config.prefix_frames:
        # Loading the language model and evaluating the prompt take seconds
        # before the first prefix frame reports. This report marks the prefill
        # as it starts. A restored generation skips this stage and prefills
        # nothing, so it never sends one.
        progress(
            GenerationProgress(
                completed_frames=0,
                maximum_frames=config.max_frames,
                prefilled_frames=0,
                prefix_frames=config.prefix_frames,
            )
        )
    session = StageSession(
        "autoregressive",
        lambda: _load_autoregressive_models(checkpoint),
        policy=policy,
        include_footprint=include_footprint,
    )
    with session:
        result = generate_autoregressive(
            session.require_model().language_model,
            session.require_model().depth_decoder,
            prompt,
            config,
            progress=progress,
            cancelled=cancelled,
        )
        session.handoff(result.codes, result.frame_hiddens)
    if session.report is None:
        raise RuntimeError("Autoregressive stage did not produce a memory report")
    return result, session.report


def run_acoustic_stage(
    checkpoint: Path,
    frame_hiddens: mx.array,
    *,
    seed: int,
    config: FlowGenerationConfig,
    flow_compute_dtype: str,
    policy: StageMemoryPolicy,
    include_footprint: bool,
    progress: Callable[[FlowProgress], None] | None,
    cancelled: Callable[[], bool] | None,
    resume: AcousticResumeState | None = None,
    window_completed: Callable[[CompletedAcousticWindow], None] | None = None,
) -> tuple[AcousticLatents, StageMemoryReport]:
    session = StageSession(
        "acoustic",
        lambda: _load_acoustic_models(checkpoint, flow_compute_dtype),
        policy=policy,
        include_footprint=include_footprint,
    )
    with session:
        result = generate_acoustic_latents(
            session.require_model().transformer,
            session.require_model().condition_encoder,
            frame_hiddens,
            seed=seed,
            config=config,
            progress=progress,
            cancelled=cancelled,
            resume=resume,
            window_completed=window_completed,
        )
        session.handoff(*(chunk.latents for chunk in result.chunks))
    if session.report is None:
        raise RuntimeError("Acoustic stage did not produce a memory report")
    return result, session.report


def run_decode_stage(
    checkpoint: Path,
    acoustic: AcousticLatents,
    *,
    policy: StageMemoryPolicy,
    include_footprint: bool,
    progress: Callable[[int, int], None] | None,
    cancelled: Callable[[], bool] | None,
) -> tuple[Waveform, StageMemoryReport]:
    session = StageSession(
        "decode",
        lambda: load_vocoder(checkpoint),
        policy=policy,
        include_footprint=include_footprint,
    )
    with session:
        waveform = decode_latent_chunks(
            session.require_model(),
            acoustic,
            progress=progress,
            cancelled=cancelled,
        )
        session.handoff(waveform.samples)
    if session.report is None:
        raise RuntimeError("Decode stage did not produce a memory report")
    return waveform, session.report


__all__ = [
    "FLOW_COMPUTE_DTYPES",
    "run_acoustic_stage",
    "run_autoregressive_stage",
    "run_decode_stage",
]
