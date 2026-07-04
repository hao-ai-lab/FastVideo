# SPDX-License-Identifier: Apache-2.0
"""Comparable identity helpers for performance benchmark records."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import importlib.metadata
import json
import os
import platform
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

RECIPE_SCHEMA_VERSION = 1
PROFILE_ID_LENGTH = 16
_PATH_EXCLUDED_GENERATION_KEYS = {"output_path", "output_video_name"}
_PROMPT_KEYS = {"prompt", "negative_prompt", "neg_prompt"}
_PROFILE_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "FASTVIDEO_ATTENTION_BACKEND",
    "IMAGE_VERSION",
    "UV_TORCH_BACKEND",
)
_PACKAGE_DISTRIBUTIONS = {
    "fastvideo_kernel": ("fastvideo-kernel", "fastvideo_kernel"),
    "flash_attn": ("flash-attn", "flash_attn"),
    "sageattention": ("sageattention",),
    "sageattn3": ("sageattn3",),
    "triton": ("triton",),
    "xformers": ("xformers",),
}


def canonical_json(payload: Any) -> str:
    """Return deterministic, compact JSON for a recipe/profile payload."""

    return json.dumps(
        _canonicalize(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def sha256_hexdigest(payload: str | bytes) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def recipe_fingerprint(recipe: Mapping[str, Any]) -> str:
    return sha256_hexdigest(canonical_json(recipe))


def profile_id(prefix: str, profile: Mapping[str, Any]) -> str:
    return f"{prefix}-{sha256_hexdigest(canonical_json(profile))[:PROFILE_ID_LENGTH]}"


def hardware_profile_id(profile: Mapping[str, Any]) -> str:
    return profile_id("hw", profile)


def software_profile_id(profile: Mapping[str, Any]) -> str:
    return profile_id("sw", profile)


def environment_fingerprint(metadata: Mapping[str, Any]) -> str:
    return profile_id("env", metadata)


def build_recipe_from_benchmark_config(
    cfg: Mapping[str, Any],
    *,
    attention_backend: str | None = None,
    resolved_attention_backend: Any | None = None,
    resolved_model_revision: Any | None = None,
    measured_prompts: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic recipe document from a benchmark config.

    The recipe captures fields that describe what benchmark workload was run. It
    intentionally excludes timings, timestamps, commit metadata, and output paths.
    """

    model = dict(cfg.get("model") or {})
    init_kwargs = dict(cfg.get("init_kwargs") or {})
    generation_kwargs = dict(cfg.get("generation_kwargs") or {})
    prompts = list(measured_prompts) if measured_prompts is not None else list(
        cfg.get("test_prompts") or ["A cinematic video."])
    if attention_backend is None:
        attention_backend = os.environ.get("FASTVIDEO_ATTENTION_BACKEND")

    negative_prompt = generation_kwargs.pop("negative_prompt", generation_kwargs.pop("neg_prompt", None))
    generation_recipe = {
        key: value
        for key, value in generation_kwargs.items()
        if key not in _PATH_EXCLUDED_GENERATION_KEYS and key not in _PROMPT_KEYS
    }

    return {
        "recipe_schema_version": RECIPE_SCHEMA_VERSION,
        "benchmark": {
            "benchmark_id": cfg.get("benchmark_id"),
            "workload_id": cfg.get("workload_id"),
            "variant_id": cfg.get("variant_id"),
            "benchmark_version": cfg.get("benchmark_version"),
        },
        "model": {
            "model_path": normalize_model_path(model.get("model_path")),
            "revision": _none_if_empty(model.get("revision") or cfg.get("revision")),
            "resolved_revision": _none_if_empty(resolved_model_revision),
        },
        "init_kwargs": init_kwargs,
        "generation_kwargs": generation_recipe,
        "inputs": {
            "prompt_count": len(prompts),
            "prompt_sha256": [_optional_digest(prompt) for prompt in prompts],
            "negative_prompt_sha256": _optional_digest(negative_prompt),
        },
        "attention": {
            "requested_backend": _none_if_empty(attention_backend) or "auto",
            "resolved_backend": _none_if_empty(resolved_attention_backend),
        },
    }


def normalize_model_path(value: Any) -> str | None:
    if value is None:
        return None
    text = os.fspath(value) if isinstance(value, os.PathLike) else str(value)
    text = text.strip()
    if not text:
        return None
    if _looks_like_local_path(text):
        return Path(text).expanduser().as_posix()
    return re.sub(r"/+", "/", text)


def resolved_revision_from_model_path(value: Any) -> str | None:
    normalized = normalize_model_path(value)
    if normalized is None:
        return None
    parts = Path(normalized).parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots":
            revision = parts[index + 1]
            if re.fullmatch(r"[0-9a-f]{40}", revision):
                return revision
    return None


def hardware_profile(
    *,
    num_gpus: int | None = None,
    gpu_devices: Sequence[Mapping[str, Any]] | None = None,
    device_type: str | None = None,
    interconnect: str | None = None,
) -> dict[str, Any]:
    """Return normalized hardware cohort metadata.

    ``gpu_devices`` is an injection point for tests and non-CUDA callers.
    """

    if gpu_devices is not None:
        devices = [_normalize_gpu_device(device) for device in gpu_devices]
        return {
            "device_type": device_type or "cuda",
            "gpu_count": num_gpus if num_gpus is not None else len(devices),
            "gpus": devices,
            "interconnect": interconnect or "unknown",
        }

    if not torch.cuda.is_available():
        return {
            "device_type": device_type or "cpu",
            "gpu_count": 0,
            "gpus": [],
            "interconnect": interconnect or "none",
        }

    gpu_count = int(num_gpus if num_gpus is not None else torch.cuda.device_count())
    devices = []
    cuda_device_count = torch.cuda.device_count()
    for device_id in range(gpu_count):
        if device_id >= cuda_device_count:
            devices.append(_normalize_gpu_device({}))
            continue
        props = torch.cuda.get_device_properties(device_id)
        capability = torch.cuda.get_device_capability(device_id)
        devices.append(
            _normalize_gpu_device({
                "name": props.name,
                "memory_bytes": props.total_memory,
                "compute_capability": f"{capability[0]}.{capability[1]}",
            }))

    return {
        "device_type": device_type or "cuda",
        "gpu_count": gpu_count,
        "gpus": devices,
        "interconnect": interconnect or _detect_cuda_interconnect(gpu_count),
    }


def software_profile(
    *,
    python_version: str | None = None,
    torch_version: str | None = None,
    cuda_version: str | None = None,
    package_versions: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Return normalized software cohort metadata.

    Python, PyTorch, and CUDA use major/minor cohorts. Attention and kernel
    packages keep exact versions because patch releases can change performance.
    """

    versions = dict(package_versions) if package_versions is not None else _installed_package_versions()
    return {
        "python": _major_minor(python_version or platform.python_version()),
        "pytorch": _major_minor(torch_version or torch.__version__),
        "cuda": _major_minor(cuda_version or torch.version.cuda),
        "packages": {
            name: str(version)
            for name, version in sorted(versions.items())
            if version
        },
    }


def environment_metadata(
    *,
    env: Mapping[str, str] | None = None,
    package_versions: Mapping[str, str | None] | None = None,
    hardware: Mapping[str, Any] | None = None,
    software: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return audit metadata kept separate from comparable identity keys."""

    source_env = env if env is not None else os.environ
    full_package_versions = dict(package_versions) if package_versions is not None else _installed_package_versions()
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "env": {
            key: source_env.get(key)
            for key in _PROFILE_ENV_VARS
            if source_env.get(key) is not None
        },
        "packages": {
            key: value
            for key, value in sorted(full_package_versions.items())
            if value
        },
        "hardware_profile": dict(hardware) if hardware is not None else hardware_profile(),
        "software_profile": dict(software) if software is not None else software_profile(
            package_versions=full_package_versions),
    }


def _canonicalize(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (set, frozenset)):
        return [_canonicalize(item) for item in sorted(value, key=lambda item: str(item))]
    if isinstance(value, tuple):
        return [_canonicalize(item) for item in value]
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, Path):
        return value.expanduser().as_posix()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    return value


def _none_if_empty(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _optional_digest(value: Any) -> str | None:
    if value is None:
        return None
    return sha256_hexdigest(str(value))


def _looks_like_local_path(value: str) -> bool:
    if value.startswith(("/", "./", "../", "~")):
        return True
    looks_like_hf_repo = re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", value)
    has_separator = "/" in value or "\\" in value or os.sep in value
    return has_separator and looks_like_hf_repo is None


def _normalize_gpu_device(device: Mapping[str, Any]) -> dict[str, Any]:
    memory_gb = device.get("memory_gb")
    if memory_gb is None and device.get("memory_bytes") is not None:
        memory_gb = round(float(device["memory_bytes"]) / (1024**3))
    return {
        "name": _none_if_empty(device.get("name")) or "unknown",
        "memory_gb": memory_gb,
        "compute_capability": _none_if_empty(device.get("compute_capability")),
    }


def _detect_cuda_interconnect(gpu_count: int) -> str:
    if gpu_count <= 1:
        return "single_gpu"
    try:
        from fastvideo.platforms import current_platform

        if current_platform.is_cuda() and current_platform.is_full_nvlink(list(range(gpu_count))):
            return "full_nvlink"
        return "none_or_partial"
    except Exception:
        return "unknown"


def _installed_package_versions() -> dict[str, str | None]:
    return {
        name: _distribution_version(*distribution_names)
        for name, distribution_names in _PACKAGE_DISTRIBUTIONS.items()
    }


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _major_minor(version: str | None) -> str | None:
    if not version:
        return None
    match = re.search(r"(\d+)\.(\d+)", version)
    if match is None:
        return version
    return f"{match.group(1)}.{match.group(2)}"


__all__ = [
    "RECIPE_SCHEMA_VERSION",
    "canonical_json",
    "environment_fingerprint",
    "environment_metadata",
    "hardware_profile",
    "hardware_profile_id",
    "normalize_model_path",
    "profile_id",
    "recipe_fingerprint",
    "resolved_revision_from_model_path",
    "build_recipe_from_benchmark_config",
    "sha256_hexdigest",
    "software_profile",
    "software_profile_id",
]
