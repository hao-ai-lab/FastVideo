# SPDX-License-Identifier: Apache-2.0
"""Locate and validate Pika H3 Flex's released LoRA adapters.

Pika H3 Flex is MiniMax-H3 plus one of two rank-128 PEFT adapters: ``fl2va``
for text/first-frame/last-frame generation and ``ref2va`` for ordered
image/video/audio references.  The adapter is part of the model contract, not a
user-tunable accelerator, so the pipeline loads it automatically and refuses a
manifest whose rank, trunk, or sampler does not match the release.

This module stays torch-free; the manifest is a plain JSON document and the
only optional dependency is ``huggingface_hub`` for a repo-id source.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from fastvideo.configs.pipelines.pika_h3_flex import (
    PIKA_AUDIO_SCHEDULER_SHIFT,
    PIKA_DENOISING_STEPS,
    PIKA_VIDEO_SCHEDULER_SHIFT,
)

PIKA_ADAPTER_FORMAT_VERSION = 1
PIKA_ADAPTER_RANK = 128
PIKA_ADAPTER_ALPHA = 128
PIKA_BASE_REPO = "MiniMaxAI/MiniMax-H3"
PIKA_ADAPTER_FILENAME = "adapter.safetensors"
PIKA_MANIFEST_FILENAME = "manifest.json"

PIKA_SAMPLING_CONTRACT = {
    "knots": (*PIKA_DENOISING_STEPS, 0),
    "video_shift": PIKA_VIDEO_SCHEDULER_SHIFT,
    "audio_shift": PIKA_AUDIO_SCHEDULER_SHIFT,
    "method": "renoise",
}
PIKA_TARGET_MODULES = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)

TASK_TO_TRUNK = {"fl2va": "transformer", "ref2va": "transformer_ref"}


class PikaAdapterError(ValueError):
    """The Pika adapter is missing or does not describe the released model."""


@dataclass(frozen=True)
class PikaAdapter:
    """One validated released adapter."""

    task: str
    trunk: str
    path: str
    manifest_path: str
    manifest: dict[str, Any]

    @property
    def directory(self) -> str:
        return str(Path(self.path).parent)


def task_for_pipeline(ref2va: bool) -> str:
    return "ref2va" if ref2va else "fl2va"


def expected_trunk(task: str) -> str:
    try:
        return TASK_TO_TRUNK[task]
    except KeyError as error:
        raise PikaAdapterError(
            f"Unknown Pika H3 Flex task {task!r}; expected one of {sorted(TASK_TO_TRUNK)}.") from error


def _as_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PikaAdapterError(f"Pika adapter manifest field {field!r} must be an integer, got {value!r}.")
    return value


def _as_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PikaAdapterError(f"Pika adapter manifest field {field!r} must be a number, got {value!r}.")
    return float(value)


def validate_adapter_manifest(
    manifest: Mapping[str, Any],
    *,
    task: str,
    pipeline_trunk: str | None = None,
) -> None:
    """Reject a manifest that is not the released Pika adapter for ``task``."""
    trunk = expected_trunk(task)
    if pipeline_trunk is not None and pipeline_trunk != trunk:
        raise PikaAdapterError(f"Pika H3 Flex task {task!r} loads trunk {trunk!r}, but the pipeline requested "
                               f"{pipeline_trunk!r}.")

    format_version = _as_int(manifest.get("format_version"), "format_version")
    if format_version != PIKA_ADAPTER_FORMAT_VERSION:
        raise PikaAdapterError(f"Unsupported Pika adapter manifest format {format_version}; expected "
                               f"{PIKA_ADAPTER_FORMAT_VERSION}.")
    manifest_task = manifest.get("task")
    if manifest_task is not None and manifest_task != task:
        raise PikaAdapterError(f"The adapter at this path is for task {manifest_task!r}, not {task!r}.")
    rank = _as_int(manifest.get("rank"), "rank")
    alpha = _as_int(manifest.get("alpha"), "alpha")
    if rank != PIKA_ADAPTER_RANK or alpha != PIKA_ADAPTER_ALPHA:
        raise PikaAdapterError(f"Pika H3 Flex adapters are rank {PIKA_ADAPTER_RANK} with alpha "
                               f"{PIKA_ADAPTER_ALPHA}; this manifest says rank {rank}, alpha {alpha}.")
    manifest_trunk = manifest.get("trunk")
    if manifest_trunk != trunk:
        raise PikaAdapterError(f"Pika H3 Flex task {task!r} expects adapter trunk {trunk!r}, got "
                               f"{manifest_trunk!r}.")

    sampling = manifest.get("sampling")
    if not isinstance(sampling, Mapping):
        raise PikaAdapterError("Pika adapter manifest is missing its `sampling` contract.")
    knots = sampling.get("knots")
    if not isinstance(knots, list | tuple) or tuple(knots) != PIKA_SAMPLING_CONTRACT["knots"]:
        raise PikaAdapterError(f"Pika adapter sampling knots must be "
                               f"{list(PIKA_SAMPLING_CONTRACT['knots'])}, got {knots!r}.")
    for field in ("video_shift", "audio_shift"):
        value = _as_float(sampling.get(field), f"sampling.{field}")
        expected = float(PIKA_SAMPLING_CONTRACT[field])
        if abs(value - expected) > 1e-9:
            raise PikaAdapterError(f"Pika adapter sampling.{field} must be {expected:g}, got {value:g}.")
    if sampling.get("method") != PIKA_SAMPLING_CONTRACT["method"]:
        raise PikaAdapterError(f"Pika adapter sampling.method must be "
                               f"{PIKA_SAMPLING_CONTRACT['method']!r}, got {sampling.get('method')!r}.")

    target_modules = manifest.get("target_modules")
    if not isinstance(target_modules, list | tuple) or tuple(target_modules) != PIKA_TARGET_MODULES:
        raise PikaAdapterError(f"Pika adapter target_modules must be {list(PIKA_TARGET_MODULES)}, got "
                               f"{target_modules!r}.")

    base = manifest.get("base")
    if not isinstance(base, Mapping) or base.get("repo_id") != PIKA_BASE_REPO:
        raise PikaAdapterError(f"Pika adapter base must name {PIKA_BASE_REPO!r}, got {base!r}.")


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise PikaAdapterError(f"Cannot read Pika adapter manifest {str(path)!r}: {error}") from error
    except json.JSONDecodeError as error:
        raise PikaAdapterError(f"Pika adapter manifest {str(path)!r} is not valid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise PikaAdapterError(f"Pika adapter manifest {str(path)!r} must be a JSON object.")
    return manifest


def _adapter_in_directory(directory: Path, task: str) -> tuple[Path, Path] | None:
    nested = directory / task / PIKA_ADAPTER_FILENAME
    if nested.is_file():
        manifest = directory / task / PIKA_MANIFEST_FILENAME
        return nested, manifest
    flat = directory / PIKA_ADAPTER_FILENAME
    if flat.is_file():
        manifest = directory / PIKA_MANIFEST_FILENAME
        return flat, manifest
    return None


def _looks_like_repo_id(source: str) -> bool:
    parts = source.split("/")
    return len(parts) == 2 and all(parts) and "\\" not in source


def _download_adapter(source: str, task: str, revision: str | None) -> tuple[Path, Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - huggingface_hub is a runtime dependency
        raise PikaAdapterError("Downloading a Pika adapter by repo id requires huggingface_hub.") from error
    try:
        adapter = Path(hf_hub_download(
            repo_id=source,
            filename=f"{task}/{PIKA_ADAPTER_FILENAME}",
            revision=revision,
        ))
        manifest = Path(
            hf_hub_download(
                repo_id=source,
                filename=f"{task}/{PIKA_MANIFEST_FILENAME}",
                revision=revision,
            ))
    except Exception as error:
        raise PikaAdapterError(f"Could not download the Pika H3 Flex {task} adapter from {source!r}"
                               f"{f' at revision {revision!r}' if revision else ''}: {error}") from error
    return adapter, manifest


def load_adapter(source: str | os.PathLike[str], *, task: str, revision: str | None = None) -> PikaAdapter:
    """Resolve ``source`` (adapter file, release directory, task directory, or repo id)."""
    trunk = expected_trunk(task)
    raw = os.fspath(source)
    path = Path(raw).expanduser()
    if path.is_file():
        adapter_path, manifest_path = path, path.with_name(PIKA_MANIFEST_FILENAME)
    elif path.is_dir():
        found = _adapter_in_directory(path, task)
        if found is None:
            raise PikaAdapterError(f"{raw!r} has no {task}/{PIKA_ADAPTER_FILENAME} or {PIKA_ADAPTER_FILENAME}.")
        adapter_path, manifest_path = found
    elif _looks_like_repo_id(raw):
        adapter_path, manifest_path = _download_adapter(raw, task, revision)
    else:
        raise PikaAdapterError(f"Pika adapter path {raw!r} does not exist and is not a Hub repo id.")

    if not adapter_path.is_file():
        raise PikaAdapterError(f"Pika adapter weights are missing at {str(adapter_path)!r}.")
    if not manifest_path.is_file():
        raise PikaAdapterError(f"Pika adapter manifest is missing at {str(manifest_path)!r}; a bare "
                               f"{PIKA_ADAPTER_FILENAME} cannot be checked against the release contract.")
    manifest = _load_manifest(manifest_path)
    validate_adapter_manifest(manifest, task=task, pipeline_trunk=trunk)
    return PikaAdapter(
        task=task,
        trunk=trunk,
        path=str(adapter_path),
        manifest_path=str(manifest_path),
        manifest=manifest,
    )


def adapter_hf_patterns(task: str | None = None) -> list[str]:
    """The adapter files the pipeline must include in a partial Hub snapshot."""
    tasks = tuple(TASK_TO_TRUNK) if task is None else (task, )
    patterns: list[str] = []
    for item in tasks:
        expected_trunk(item)
        patterns.extend((f"{item}/{PIKA_ADAPTER_FILENAME}", f"{item}/{PIKA_MANIFEST_FILENAME}"))
    return patterns


__all__ = [
    "PIKA_ADAPTER_ALPHA",
    "PIKA_ADAPTER_FILENAME",
    "PIKA_ADAPTER_FORMAT_VERSION",
    "PIKA_ADAPTER_RANK",
    "PIKA_BASE_REPO",
    "PIKA_MANIFEST_FILENAME",
    "PIKA_SAMPLING_CONTRACT",
    "PIKA_TARGET_MODULES",
    "PikaAdapter",
    "PikaAdapterError",
    "TASK_TO_TRUNK",
    "adapter_hf_patterns",
    "expected_trunk",
    "load_adapter",
    "task_for_pipeline",
    "validate_adapter_manifest",
]
