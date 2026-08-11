# SPDX-License-Identifier: Apache-2.0
"""GPU telemetry for the studio status page, via NVML (nvidia-ml-py)."""

from __future__ import annotations

import contextlib
import logging
from typing import Any

logger = logging.getLogger("fastvideo.studio.gpu")

_nvml_initialized = False


def _ensure_nvml() -> Any:
    """Import and initialize NVML once; raises on machines without it."""
    global _nvml_initialized  # noqa: PLW0603
    import pynvml
    if not _nvml_initialized:
        pynvml.nvmlInit()
        _nvml_initialized = True
    return pynvml


def _device_snapshot(pynvml: Any, index: int) -> dict[str, Any]:
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode()
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)

    # Optional sensors: not every GPU/driver exposes them.
    temperature: int | None = None
    power_watts: float | None = None
    power_limit_watts: float | None = None
    with contextlib.suppress(pynvml.NVMLError):
        temperature = int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    with contextlib.suppress(pynvml.NVMLError):
        power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        power_limit_watts = (pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0)

    return {
        "index": index,
        "name": name,
        "utilization": int(util.gpu),
        "memory_used_mib": int(mem.used / (1024 * 1024)),
        "memory_total_mib": int(mem.total / (1024 * 1024)),
        "temperature_c": temperature,
        "power_watts": power_watts,
        "power_limit_watts": power_limit_watts,
    }


def get_gpu_snapshot() -> dict[str, Any]:
    """Return {"available", "gpus", "error"} — never raises.

    ``available: False`` covers both "no NVIDIA driver/library on this host"
    and transient NVML failures; the frontend shows ``error`` as-is.
    """
    try:
        pynvml = _ensure_nvml()
        count = pynvml.nvmlDeviceGetCount()
        gpus = [_device_snapshot(pynvml, i) for i in range(count)]
        return {"available": True, "gpus": gpus, "error": None}
    except ImportError:
        return {
            "available": False,
            "gpus": [],
            "error": "nvidia-ml-py is not installed on the API server host.",
        }
    except Exception as exc:  # NVMLError, driver issues, …
        logger.warning("GPU snapshot failed: %s", exc)
        return {"available": False, "gpus": [], "error": str(exc)}
