# SPDX-License-Identifier: Apache-2.0
"""Capture runtime environment metadata for performance benchmark records.

The returned dict is embedded under the ``env`` key of each ``perf_*.json``
record. All sub-fields are best-effort: capture must NEVER raise. Missing
data is reported as ``None`` (or empty list/dict) so that older records and
records produced on partial environments still load and compare.

Two consumers care about this data:

* The Markdown summary and dashboard surface it for human inspection.
* :func:`compare_baseline._check_regressions` optionally filters the rolling
  baseline window to records whose ``env`` matches the current run, so that a
  torch/CUDA/attention-backend swap does not silently shift the median.

The set of ``env`` keys used for comparison is intentionally short
(``DEFAULT_COMPARE_KEYS``) and configurable via the ``PERF_COMPARE_KEYS``
environment variable. Everything else captured here is metadata.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from importlib import metadata
from typing import Any

import torch

# Bump when fields are added/removed/renamed in a way that downstream
# consumers must adapt to. Older records without the field are treated as
# schema_version=0.
PERF_RECORD_SCHEMA_VERSION = 1

# Packages whose installed version is most likely to influence inference
# performance numbers. Missing packages produce None instead of raising.
_KEY_PACKAGES = (
    "torch",
    "transformers",
    "diffusers",
    "huggingface_hub",
    "fastvideo",
    "fastvideo-kernel",
    "flash-attn",
    "sage-attention",
    "xformers",
    "triton",
    "vllm",
)

# Default tuple of dotted ``env``-record keys used to decide whether two
# records are "comparable" for rolling-baseline / dashboard-grouping
# purposes. The tuple is intentionally short — finer-grained fields (CUDA
# patch, torch patch, kernel build hash) live in ``env`` for human
# inspection but are not enforced. Override via ``PERF_COMPARE_KEYS``
# (comma-separated dotted paths).
DEFAULT_COMPARE_KEYS: tuple[str, ...] = (
    "gpu.name",
    "gpu.count",
    "torch.version_major_minor",
    "cuda.runtime_major",
    "attention_backend",
)


def capture_env() -> dict[str, Any]:
    """Return a metadata dict describing the runtime."""
    return {
        "schema_version": PERF_RECORD_SCHEMA_VERSION,
        "gpu": _capture_gpu(),
        "cuda": _capture_cuda(),
        "torch": _capture_torch(),
        "python": platform.python_version(),
        "os": _capture_os(),
        "attention_backend": _capture_attention_backend(),
        "key_packages": _capture_packages(),
        "container_image": _capture_container_image(),
    }


# ---------------------------------------------------------------------------
# Per-section capture helpers
# ---------------------------------------------------------------------------


def _capture_gpu() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "name": None,
            "count": 0,
            "compute_capability": None,
            "memory_total_mb": None,
            "driver_version": None,
            "uuids": [],
        }
    count = torch.cuda.device_count()
    name: str | None
    compute_capability: str | None
    memory_total_mb: float | None
    try:
        props = torch.cuda.get_device_properties(0)
        name = props.name
        compute_capability = f"{props.major}.{props.minor}"
        memory_total_mb = round(props.total_memory / (1024 * 1024), 1)
    except Exception:
        name = None
        compute_capability = None
        memory_total_mb = None

    return {
        "name": name,
        "count": count,
        "compute_capability": compute_capability,
        "memory_total_mb": memory_total_mb,
        "driver_version": _nvidia_smi_query("driver_version"),
        "uuids": _nvidia_smi_query("uuid", multi=True),
    }


def _capture_cuda() -> dict[str, Any]:
    runtime = getattr(torch.version, "cuda", None)
    runtime_major: str | None
    if isinstance(runtime, str):
        runtime_major = runtime.split(".")[0]
    else:
        runtime_major = None
    return {
        "runtime": runtime,
        "runtime_major": runtime_major,
    }


def _capture_torch() -> dict[str, Any]:
    version = getattr(torch, "__version__", None)
    major_minor: str | None = None
    if isinstance(version, str):
        match = re.match(r"(\d+\.\d+)", version)
        if match is not None:
            major_minor = match.group(1)
    return {
        "version": version,
        "version_major_minor": major_minor,
        "built_with_cuda": getattr(torch.version, "cuda", None),
    }


def _capture_os() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }


def _capture_attention_backend() -> str | None:
    # FASTVIDEO_ATTENTION_BACKEND is the user-visible knob. Empty string is
    # treated the same as unset.
    value = os.environ.get("FASTVIDEO_ATTENTION_BACKEND")
    return value if value else None


def _capture_packages() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for pkg in _KEY_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _capture_container_image() -> str | None:
    return (os.environ.get("MODAL_IMAGE_DIGEST")
            or os.environ.get("DOCKER_IMAGE_DIGEST")
            or os.environ.get("DOCKER_IMAGE")
            or None)


def _nvidia_smi_query(field: str, *, multi: bool = False) -> Any:
    """Best-effort ``nvidia-smi`` field query. Returns None / [] on failure."""
    if shutil.which("nvidia-smi") is None:
        return [] if multi else None
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return [] if multi else None
    values = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if multi:
        return values
    return values[0] if values else None


# ---------------------------------------------------------------------------
# Compare-tuple helpers (used by compare_baseline.py / dashboard.py)
# ---------------------------------------------------------------------------


def get_compare_keys() -> tuple[str, ...]:
    """Return the dotted env-keys used for comparable-record bucketing.

    Override the default by setting ``PERF_COMPARE_KEYS`` to a
    comma-separated list of dotted paths into the ``env`` block, e.g.
    ``gpu.name,torch.version_major_minor``.
    """
    raw = os.environ.get("PERF_COMPARE_KEYS")
    if not raw:
        return DEFAULT_COMPARE_KEYS
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    return parts or DEFAULT_COMPARE_KEYS


def extract_compare_tuple(
    record: dict[str, Any],
    keys: tuple[str, ...] | None = None,
) -> tuple:
    """Extract a comparable tuple from *record* using dotted *keys*.

    Records that predate ``schema_version=1`` (no ``env`` block) yield a
    tuple of ``None``. Comparator code should treat such records as
    incomparable when strict-env mode is on, and surface a clear note.
    """
    if keys is None:
        keys = get_compare_keys()
    env = record.get("env") or {}
    return tuple(_dot_get(env, key) for key in keys)


def describe_compare_tuple(
    values: tuple,
    keys: tuple[str, ...] | None = None,
) -> str:
    """Render a compare-tuple as a human-readable ``k=v, k=v`` string."""
    if keys is None:
        keys = get_compare_keys()
    return ", ".join(f"{key}={value}" for key, value in zip(keys, values, strict=False))


def _dot_get(obj: Any, dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    return cur


# ---------------------------------------------------------------------------
# Strict-env mode
# ---------------------------------------------------------------------------


def strict_env_mode() -> bool:
    """Return True if comparator should require env-tuple equality.

    Controlled by ``PERF_STRICT_ENV``. Truthy values: ``1``, ``true``,
    ``yes`` (case-insensitive). Default: off.
    """
    raw = os.environ.get("PERF_STRICT_ENV", "").strip().lower()
    return raw in ("1", "true", "yes", "on")
