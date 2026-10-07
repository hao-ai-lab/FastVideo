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
import gc
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
        """Flatten the configured memory limits, applied values, previous values, and errors into a metrics dictionary.

        Returns:
            dict[str, int | float | str | bool | None]: Metrics keyed by limit names and their corresponding values.
        """
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
    """
    Convert a positive memory limit from GiB to bytes.

    Parameters:
        value (float | None): Memory limit in GiB, or `None` when unset.

    Returns:
        int | None: The memory limit in bytes, or `None` when no limit is provided.

    Raises:
        ValueError: If `value` is zero or negative.
    """
    if value is None:
        return None
    if value <= 0:
        raise ValueError(f"Memory limit must be positive GiB, got {value}")
    return int(value * GIB)


def cleanup_mlx(mx_module: Any | None = None) -> None:
    """Collect unreachable MLX objects, then release their allocator cache."""
    if mx_module is None:
        import mlx.core as mx

        mx_module = mx
    gc.collect()
    mx_module.clear_cache()


def cleanup_torch_mps(torch_module: Any | None = None) -> None:
    """Collect unreachable Torch objects, then release the MPS allocator cache."""
    if torch_module is None:
        import torch

        torch_module = torch
    gc.collect()
    if torch_module.backends.mps.is_available():
        torch_module.mps.empty_cache()


def _set_mps_env(name: str, value: float | None) -> float | None:
    """Set a PyTorch MPS watermark environment variable.

    Parameters:
        name (str): Name of the environment variable to set.
        value (float | None): Watermark ratio, or `None` to leave the variable unchanged.

    Returns:
        float | None: The configured watermark ratio, or `None` when no value is provided.

    Raises:
        ValueError: If `value` is negative.
    """
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
    call this before importing PyTorch. Specifying only a high watermark sets the
    low watermark to ``0.0``. MLX limit-setting failures are recorded in the
    result and do not prevent other limits from being applied.

    Parameters:
        mlx_memory_limit_gib (float | None): Maximum MLX memory in GiB.
        mlx_cache_limit_gib (float | None): Maximum MLX cache size in GiB.
        mlx_disable_cache (bool): Whether to disable the MLX cache.
        mlx_wired_limit_gib (float | None): Maximum MLX wired memory in GiB.
        torch_mps_high_watermark_ratio (float | None): PyTorch MPS high watermark
            ratio.
        torch_mps_low_watermark_ratio (float | None): PyTorch MPS low watermark
            ratio.

    Returns:
        AppliedMemoryLimits: Configured values, applied and previous MLX byte
            limits, MPS watermark values, and per-limit errors.
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

        # Apply each limit independently; record failures without stopping.
        limits = [
            ("mlx_memory_limit", memory_bytes, mx_module.set_memory_limit),
            ("mlx_cache_limit", cache_bytes, mx_module.set_cache_limit),
            ("mlx_wired_limit", wired_bytes, mx_module.set_wired_limit),
        ]
        for name, value, setter in limits:
            if value is not None:
                try:
                    previous[name] = int(setter(value))
                    applied[name] = value
                except Exception as exc:  # noqa: BLE001 - macOS/system-limit dependent.
                    errors[name] = f"{type(exc).__name__}: {exc}"

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
    """
    Add configurable Apple Silicon memory-limit options to an argument parser.

    Parameters:
        parser (argparse.ArgumentParser): Parser to which the options are added.
        mlx_memory_limit_gib (float | None): Default MLX memory limit in GiB.
        mlx_cache_limit_gib (float | None): Default MLX cache limit in GiB.
        mlx_disable_cache (bool): Whether the cache limit defaults to zero.
        mlx_wired_limit_gib (float | None): Default MLX wired-memory limit in GiB.
        torch_mps_high_watermark_ratio (float | None): Default PyTorch MPS high-watermark ratio.
        torch_mps_low_watermark_ratio (float | None): Default PyTorch MPS low-watermark ratio.
    """
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
