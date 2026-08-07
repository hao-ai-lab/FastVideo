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


def _remote_gpu_probe() -> dict[str, Any]:
    """Self-contained per-node NVML probe (runs as a ray task on each node;
    no fastvideo_studio import — worker environments don't have apps/ on
    their path, so cloudpickle must carry this by value)."""
    import contextlib as _ctx
    import socket as _socket
    out: dict[str, Any] = {"hostname": _socket.gethostname(), "available": False, "gpus": [], "error": None}
    try:
        import pynvml
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            temp = power = plimit = None
            with _ctx.suppress(pynvml.NVMLError):
                temp = int(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
            with _ctx.suppress(pynvml.NVMLError):
                power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                plimit = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
            out["gpus"].append({
                "index": i,
                "name": name,
                "utilization": int(util.gpu),
                "memory_used_mib": int(mem.used / (1024 * 1024)),
                "memory_total_mib": int(mem.total / (1024 * 1024)),
                "temperature_c": temp,
                "power_watts": power,
                "power_limit_watts": plimit,
            })
        out["available"] = True
    except Exception as exc:  # noqa: BLE001 -- reported per node
        out["error"] = str(exc)
    return out


def get_cluster_snapshot() -> dict[str, Any]:
    """Cluster-wide GPU/host telemetry.

    When this process is connected to a ray cluster (a model has been
    loaded), probe every alive node via per-node ray tasks. Otherwise fall
    back to this host's NVML snapshot. Never raises.
    """
    import socket

    local = get_gpu_snapshot()
    local_node = {"hostname": socket.gethostname(), "ip": None, "is_this_host": True,
                  "cpus": None, "ray_gpus": None, **local}
    out: dict[str, Any] = {"mode": "local", "nodes": [local_node],
                           "resources": None, "error": None}
    try:
        import ray
        if not ray.is_initialized():
            out["error"] = "not connected to a ray cluster yet (load a model first); showing the API host only"
            return out
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        alive = [n for n in ray.nodes() if n.get("Alive")]
        probe = ray.remote(num_cpus=0)(_remote_gpu_probe)
        refs = [probe.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=n["NodeID"], soft=True)).remote() for n in alive]
        snaps = ray.get(refs, timeout=15)
        nodes = []
        for n, snap in zip(alive, snaps, strict=True):
            nodes.append({
                "ip": n.get("NodeManagerAddress"),
                "is_this_host": snap.get("hostname") == socket.gethostname(),
                "cpus": n.get("Resources", {}).get("CPU"),
                "ray_gpus": n.get("Resources", {}).get("GPU"),
                **snap,
            })
        out["mode"] = "ray"
        out["nodes"] = nodes
        out["resources"] = {
            "gpus_total": ray.cluster_resources().get("GPU", 0.0),
            "gpus_available": ray.available_resources().get("GPU", 0.0),
        }
    except Exception as exc:  # noqa: BLE001 -- degrade to the local view
        logger.warning("cluster snapshot failed: %s", exc)
        out["mode"] = "local"
        out["nodes"] = [local_node]
        out["error"] = f"cluster probe failed: {exc}"
    return out
