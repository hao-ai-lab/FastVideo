# SPDX-License-Identifier: Apache-2.0
"""Memory-tier helpers for Apple Silicon MLX/MPS experiments.

macOS does not expose a perfect "pretend this machine only has 16 GB unified
memory" switch. MLX can cap the allocator used by the Apple-native DiT path,
and PyTorch MPS exposes process-level watermark environment variables for the
hybrid prompt/decode stages. Applying both gives benchmark and generation
entrypoints a practical, explicit way to exercise memory-tier presets.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Any

GIB = 1024**3


@dataclass(frozen=True)
class AppliedMemoryLimits:
    """Memory limits applied for one Apple Silicon benchmark/generation process."""

    mlx_memory_limit_gib: float | None = None
    mlx_cache_limit_gib: float | None = None
    mlx_disable_cache: bool = False
    mlx_wired_limit_gib: float | None = None
    torch_mps_high_watermark_ratio: float | None = None
    torch_mps_low_watermark_ratio: float | None = None
    applied_bytes: dict[str, int] = field(default_factory=dict)
    previous_bytes: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def as_metrics(self) -> dict[str, int | float | str | bool | None]:
        metrics: dict[str, int | float | str | bool | None] = {
            "mlx_memory_limit_gib": self.mlx_memory_limit_gib,
            "mlx_cache_limit_gib": self.mlx_cache_limit_gib,
            "mlx_disable_cache": self.mlx_disable_cache,
            "mlx_wired_limit_gib": self.mlx_wired_limit_gib,
            "torch_mps_high_watermark_ratio": self.torch_mps_high_watermark_ratio,
            "torch_mps_low_watermark_ratio": self.torch_mps_low_watermark_ratio,
        }
        for name, value in self.applied_bytes.items():
            metrics[f"{name}_bytes"] = value
        for name, value in self.previous_bytes.items():
            metrics[f"previous_{name}_bytes"] = value
        for name, error in self.errors.items():
            metrics[f"{name}_error"] = error
        return metrics


def gib_to_bytes(value: float | None) -> int | None:
    if value is None:
        return None
    if value <= 0:
        raise ValueError(f"Memory limit must be positive GiB, got {value}")
    return int(value * GIB)


def _set_mps_env(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    os.environ[name] = str(value)
    return value


def apply_memory_limits(
    *,
    mlx_memory_limit_gib: float | None = None,
    mlx_cache_limit_gib: float | None = None,
    mlx_disable_cache: bool = False,
    mlx_wired_limit_gib: float | None = None,
    torch_mps_high_watermark_ratio: float | None = None,
    torch_mps_low_watermark_ratio: float | None = None,
    mx_module: Any | None = None,
) -> AppliedMemoryLimits:
    """Apply optional MLX allocator limits and PyTorch MPS watermarks.

    PyTorch reads MPS watermark variables when the MPS backend initializes, so
    callers should invoke this before importing torch. If a high watermark is
    requested without a low watermark, the low watermark is set to ``0.0`` to
    avoid PyTorch's default low watermark exceeding the requested high cap.
    """
    if torch_mps_high_watermark_ratio is not None and torch_mps_low_watermark_ratio is None:
        torch_mps_low_watermark_ratio = 0.0

    high = _set_mps_env("PYTORCH_MPS_HIGH_WATERMARK_RATIO", torch_mps_high_watermark_ratio)
    low = _set_mps_env("PYTORCH_MPS_LOW_WATERMARK_RATIO", torch_mps_low_watermark_ratio)

    memory_bytes = gib_to_bytes(mlx_memory_limit_gib)
    cache_bytes = 0 if mlx_disable_cache else gib_to_bytes(mlx_cache_limit_gib)
    wired_bytes = gib_to_bytes(mlx_wired_limit_gib)

    applied: dict[str, int] = {}
    previous: dict[str, int] = {}
    errors: dict[str, str] = {}
    if memory_bytes is not None or cache_bytes is not None or wired_bytes is not None:
        if mx_module is None:
            import mlx.core as mx

            mx_module = mx

        if memory_bytes is not None:
            previous["mlx_memory_limit"] = int(mx_module.set_memory_limit(memory_bytes))
            applied["mlx_memory_limit"] = memory_bytes
        if cache_bytes is not None:
            previous["mlx_cache_limit"] = int(mx_module.set_cache_limit(cache_bytes))
            applied["mlx_cache_limit"] = cache_bytes
        if wired_bytes is not None:
            try:
                previous["mlx_wired_limit"] = int(mx_module.set_wired_limit(wired_bytes))
                applied["mlx_wired_limit"] = wired_bytes
            except Exception as exc:  # noqa: BLE001 - macOS/system-limit dependent.
                errors["mlx_wired_limit"] = f"{type(exc).__name__}: {exc}"

    return AppliedMemoryLimits(
        mlx_memory_limit_gib=mlx_memory_limit_gib,
        mlx_cache_limit_gib=mlx_cache_limit_gib,
        mlx_disable_cache=mlx_disable_cache,
        mlx_wired_limit_gib=mlx_wired_limit_gib,
        torch_mps_high_watermark_ratio=high,
        torch_mps_low_watermark_ratio=low,
        applied_bytes=applied,
        previous_bytes=previous,
        errors=errors,
    )


def add_memory_limit_args(
    parser: argparse.ArgumentParser,
    *,
    mlx_memory_limit_gib: float | None = None,
    mlx_cache_limit_gib: float | None = None,
    mlx_disable_cache: bool = False,
    mlx_wired_limit_gib: float | None = None,
    torch_mps_high_watermark_ratio: float | None = None,
    torch_mps_low_watermark_ratio: float | None = None,
) -> None:
    """Add shared Apple Silicon memory-tier flags to an argparse parser."""
    parser.add_argument("--mlx-memory-limit-gib",
                        type=float,
                        default=mlx_memory_limit_gib,
                        help="Set MLX memory limit in GiB for memory-tier testing (DiT path).")
    parser.add_argument("--mlx-cache-limit-gib",
                        type=float,
                        default=mlx_cache_limit_gib,
                        help="Set MLX cache limit in GiB. Use --mlx-disable-cache to force 0.")
    parser.add_argument("--mlx-disable-cache",
                        action="store_true",
                        default=mlx_disable_cache,
                        help="Set MLX cache limit to 0 for stricter memory-tier tests.")
    parser.add_argument("--mlx-wired-limit-gib",
                        type=float,
                        default=mlx_wired_limit_gib,
                        help="Set MLX wired-memory limit in GiB where supported by macOS/MLX.")
    parser.add_argument("--torch-mps-high-watermark-ratio",
                        type=float,
                        default=torch_mps_high_watermark_ratio,
                        help="Set PYTORCH_MPS_HIGH_WATERMARK_RATIO before importing torch.")
    parser.add_argument("--torch-mps-low-watermark-ratio",
                        type=float,
                        default=torch_mps_low_watermark_ratio,
                        help="Set PYTORCH_MPS_LOW_WATERMARK_RATIO before importing torch.")
