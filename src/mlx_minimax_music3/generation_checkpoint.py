"""Disposable, validated checkpoints for resumable song generation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from . import reference
from .acoustic import AcousticResumeState, LatentChunk
from .autoregressive import AutoregressiveConfig, AutoregressiveResult
from .chunking import ChunkWindow, chunk_windows
from .manifest import CheckpointManifest, sha256_file

SCHEMA_VERSION = 1
BEHAVIOR_VERSION = "music3-resume-v2"
_CACHE_ERRORS = (OSError, RuntimeError, ValueError, TypeError, KeyError)


@dataclass(frozen=True, slots=True)
class RestoredGeneration:
    autoregressive: AutoregressiveResult | None
    acoustic: AcousticResumeState | None


def _model_identity(manifest: CheckpointManifest) -> dict[str, Any]:
    identity = manifest.to_dict()
    identity["components"] = {
        component.name: {
            "files": [
                asdict(record)
                for record in sorted(component.files, key=lambda item: item.path)
            ]
        }
        for component in sorted(manifest.components, key=lambda item: item.name)
    }
    return identity


def generation_fingerprint(
    request: object,
    *,
    flow_compute_dtype: str,
    model_manifest: CheckpointManifest,
) -> str:
    """Hash every input that can change generated stage output."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "behavior_version": BEHAVIOR_VERSION,
        # Read at call time so a recalibrated penalty invalidates guided
        # checkpoints without a behavior-version bump.
        "guidance_logit_penalty": reference.GUIDANCE_LOGIT_PENALTY,
        "request": asdict(request),
        "flow_compute_dtype": flow_compute_dtype,
        "model": _model_identity(model_manifest),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _tensor_record(value: mx.array) -> dict[str, Any]:
    return {"shape": list(value.shape), "dtype": str(value.dtype)}


def _serializable_tensors(
    tensors: dict[str, mx.array],
) -> tuple[dict[str, mx.array], dict[str, dict[str, Any]]]:
    stored: dict[str, mx.array] = {}
    records: dict[str, dict[str, Any]] = {}
    for name, value in tensors.items():
        record = _tensor_record(value)
        if value.size == 0:
            # MLX safetensors rejects empty arrays. The logical shape and dtype
            # are sufficient to reconstruct an empty carry without data loss.
            stored[name] = mx.zeros((1,), dtype=value.dtype)
            record["storage"] = "empty-sentinel"
        else:
            stored[name] = value
        records[name] = record
    return stored, records


def _artifact_record(
    path: Path,
    tensor_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "filename": path.name,
        "sha256": sha256_file(path),
        "tensors": dict(sorted(tensor_records.items())),
    }


def _atomic_tensors(path: Path, tensors: dict[str, mx.array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    try:
        mx.save_safetensors(str(temporary), tensors)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class GenerationCheckpointStore:
    """Own one fingerprint directory of disposable generation artifacts."""

    def __init__(
        self,
        root: str | Path,
        *,
        request: object,
        flow_compute_dtype: str,
        model_manifest: CheckpointManifest,
    ) -> None:
        self.fingerprint = generation_fingerprint(
            request,
            flow_compute_dtype=flow_compute_dtype,
            model_manifest=model_manifest,
        )
        self.directory = Path(root) / self.fingerprint
        self._path = self.directory / "manifest.json"
        self._manifest = self._read_manifest()

    def _empty_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "behavior_version": BEHAVIOR_VERSION,
            "fingerprint": self.fingerprint,
            "autoregressive": None,
            "acoustic": [],
        }

    def _read_manifest(self) -> dict[str, Any]:
        try:
            value = json.loads(self._path.read_text(encoding="utf-8"))
        except (
            FileNotFoundError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
            return self._empty_manifest()
        if not isinstance(value, dict):
            return self._empty_manifest()
        if (
            value.get("schema_version") != SCHEMA_VERSION
            or value.get("behavior_version") != BEHAVIOR_VERSION
            or value.get("fingerprint") != self.fingerprint
            or not isinstance(value.get("acoustic"), list)
        ):
            return self._empty_manifest()
        return value

    def _write_manifest(self) -> None:
        _atomic_json(self._path, self._manifest)

    def save_autoregressive(self, result: AutoregressiveResult) -> None:
        tensors = {
            "codes": result.codes,
            "frame_hiddens": result.frame_hiddens,
        }
        mx.eval(tuple(tensors.values()))
        stored, tensor_records = _serializable_tensors(tensors)
        path = self.directory / "autoregressive.safetensors"
        _atomic_tensors(path, stored)
        record = _artifact_record(path, tensor_records)
        record.update(
            {
                "frame_count": result.num_frames,
                "stopped_on_audio_end": result.stopped_on_audio_end,
            }
        )
        self._manifest["autoregressive"] = record
        self._manifest["acoustic"] = []
        self._write_manifest()

    def save_acoustic_window(
        self,
        chunk: LatentChunk,
        *,
        next_latent: mx.array,
        next_condition: mx.array,
    ) -> None:
        records = self._manifest.get("acoustic")
        if not isinstance(records, list):
            records = []
        if chunk.window.index > len(records):
            raise ValueError("Acoustic windows must be checkpointed in prefix order")
        records = records[: chunk.window.index]
        tensors = {
            "latents": chunk.latents,
            "next_latent": next_latent,
            "next_condition": next_condition,
        }
        mx.eval(tuple(tensors.values()))
        stored, tensor_records = _serializable_tensors(tensors)
        path = self.directory / f"acoustic-{chunk.window.index:04d}.safetensors"
        _atomic_tensors(path, stored)
        record = _artifact_record(path, tensor_records)
        record["window"] = asdict(chunk.window)
        records.append(record)
        self._manifest["acoustic"] = records
        self._write_manifest()

    def _load_artifact(
        self,
        record: object,
        required: set[str],
    ) -> dict[str, mx.array]:
        if not isinstance(record, dict):
            raise TypeError("Artifact record must be an object")
        filename = record.get("filename")
        digest = record.get("sha256")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("Artifact filename must be a local basename")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Artifact digest is invalid")
        path = self.directory / filename
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError("Artifact digest mismatch")
        values = mx.load(str(path))
        if not isinstance(values, dict) or set(values) != required:
            raise ValueError("Artifact tensors do not match the manifest")
        tensor_records = record.get("tensors")
        if not isinstance(tensor_records, dict):
            raise TypeError("Artifact tensor metadata is missing")
        restored: dict[str, mx.array] = {}
        for name, value in values.items():
            expected = tensor_records.get(name)
            if not isinstance(expected, dict):
                raise TypeError("Artifact tensor metadata is incomplete")
            if expected.get("dtype") != str(value.dtype):
                raise ValueError("Artifact tensor dtype mismatch")
            shape = expected.get("shape")
            if (
                expected.get("storage") == "empty-sentinel"
                and isinstance(shape, list)
                and 0 in shape
                and value.shape == (1,)
            ):
                restored[name] = mx.zeros(tuple(shape), dtype=value.dtype)
            elif shape == list(value.shape) and "storage" not in expected:
                restored[name] = value
            else:
                raise ValueError("Artifact tensor shape mismatch")
        mx.eval(tuple(restored.values()))
        return restored

    def restore(
        self, *, autoregressive_config: AutoregressiveConfig
    ) -> RestoredGeneration:
        ar_record = self._manifest.get("autoregressive")
        try:
            ar_values = self._load_artifact(
                ar_record, {"codes", "frame_hiddens"}
            )
            if not isinstance(ar_record, dict):
                raise TypeError("Autoregressive metadata is missing")
            frame_count = ar_record.get("frame_count")
            stopped = ar_record.get("stopped_on_audio_end")
            if (
                isinstance(frame_count, bool)
                or not isinstance(frame_count, int)
                or not 0 < frame_count <= autoregressive_config.max_frames
                or not isinstance(stopped, bool)
                or ar_values["codes"].ndim != 3
                or ar_values["frame_hiddens"].ndim != 3
                or ar_values["codes"].shape[:2] != (1, frame_count)
                or ar_values["frame_hiddens"].shape[:2] != (1, frame_count)
            ):
                raise ValueError("Autoregressive checkpoint metadata is invalid")
            stopped_state_is_valid = (
                stopped
                and autoregressive_config.min_frames
                <= frame_count
                < autoregressive_config.max_frames
            ) or (
                not stopped and frame_count == autoregressive_config.max_frames
            )
            if not stopped_state_is_valid:
                raise ValueError("Autoregressive stopping state is invalid")
            autoregressive = AutoregressiveResult(
                codes=ar_values["codes"],
                frame_hiddens=ar_values["frame_hiddens"],
                stopped_on_audio_end=stopped,
            )
        except _CACHE_ERRORS:
            self._manifest = self._empty_manifest()
            if self.directory.exists():
                self._write_manifest()
            return RestoredGeneration(None, None)

        windows = chunk_windows(autoregressive.num_frames)
        chunks: list[LatentChunk] = []
        previous_latent = None
        previous_condition = None
        records = self._manifest.get("acoustic", [])
        valid_records: list[dict[str, Any]] = []
        if not isinstance(records, list):
            records = []
        for index, record in enumerate(records):
            try:
                if index >= len(windows) or not isinstance(record, dict):
                    raise ValueError("Acoustic window is outside the calculated prefix")
                raw_window = record.get("window")
                if not isinstance(raw_window, dict):
                    raise TypeError("Acoustic window metadata is missing")
                window = ChunkWindow(**raw_window)
                if window != windows[index]:
                    raise ValueError("Acoustic window metadata does not match")
                values = self._load_artifact(
                    record, {"latents", "next_latent", "next_condition"}
                )
                chunks.append(LatentChunk(window, values["latents"]))
                previous_latent = values["next_latent"]
                previous_condition = values["next_condition"]
                valid_records.append(record)
            except _CACHE_ERRORS:
                break
        if len(valid_records) != len(records):
            self._manifest["acoustic"] = valid_records
            self._write_manifest()
        acoustic = None
        if chunks:
            acoustic = AcousticResumeState(
                chunks=tuple(chunks),
                previous_latent=previous_latent,
                previous_condition=previous_condition,
            )
        return RestoredGeneration(autoregressive, acoustic)


__all__ = [
    "BEHAVIOR_VERSION",
    "SCHEMA_VERSION",
    "GenerationCheckpointStore",
    "RestoredGeneration",
    "generation_fingerprint",
]
