# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the GPU telemetry snapshot (NVML stubbed)."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import fastvideo_studio.gpu as gpu_mod


class _NVMLError(Exception):
    pass


def _make_fake_pynvml(count: int = 2, broken_sensors: bool = False) -> Any:
    fake = types.ModuleType("pynvml")

    class Mem:
        used = 4 * 1024 * 1024 * 1024
        total = 80 * 1024 * 1024 * 1024

    class Util:
        gpu = 37

    def fail(*_a: Any, **_k: Any) -> Any:
        raise fake.NVMLError()

    fake.NVMLError = _NVMLError
    fake.NVML_TEMPERATURE_GPU = 0
    fake.nvmlInit = lambda: None
    fake.nvmlDeviceGetCount = lambda: count
    fake.nvmlDeviceGetHandleByIndex = lambda i: i
    fake.nvmlDeviceGetName = lambda h: b"Fake GPU"
    fake.nvmlDeviceGetMemoryInfo = lambda h: Mem()
    fake.nvmlDeviceGetUtilizationRates = lambda h: Util()
    if broken_sensors:
        fake.nvmlDeviceGetTemperature = fail
        fake.nvmlDeviceGetPowerUsage = fail
        fake.nvmlDeviceGetEnforcedPowerLimit = fail
    else:
        fake.nvmlDeviceGetTemperature = lambda h, kind: 43
        fake.nvmlDeviceGetPowerUsage = lambda h: 310_000
        fake.nvmlDeviceGetEnforcedPowerLimit = lambda h: 700_000
    return fake


@pytest.fixture(autouse=True)
def _reset_nvml_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gpu_mod, "_nvml_initialized", False)
    yield
    monkeypatch.delitem(sys.modules, "pynvml", raising=False)


def test_snapshot_shapes_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", _make_fake_pynvml())
    snap = gpu_mod.get_gpu_snapshot()
    assert snap["available"] is True
    assert snap["error"] is None
    assert len(snap["gpus"]) == 2
    g = snap["gpus"][0]
    assert g == {
        "index": 0,
        "name": "Fake GPU",
        "utilization": 37,
        "memory_used_mib": 4096,
        "memory_total_mib": 81920,
        "temperature_c": 43,
        "power_watts": 310.0,
        "power_limit_watts": 700.0,
    }


def test_snapshot_tolerates_missing_sensors(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml",
                        _make_fake_pynvml(broken_sensors=True))
    snap = gpu_mod.get_gpu_snapshot()
    assert snap["available"] is True
    g = snap["gpus"][0]
    assert g["temperature_c"] is None
    assert g["power_watts"] is None
    assert g["power_limit_watts"] is None


def test_snapshot_reports_nvml_failure(
        monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_fake_pynvml()
    fake.nvmlInit = lambda: (_ for _ in ()).throw(_NVMLError("driver gone"))
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    snap = gpu_mod.get_gpu_snapshot()
    assert snap["available"] is False
    assert snap["gpus"] == []
    assert "driver gone" in snap["error"]
