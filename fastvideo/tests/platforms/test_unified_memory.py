# SPDX-License-Identifier: Apache-2.0
"""CPU tests for device-local unified-memory classification."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fastvideo.platforms.cuda import (CudaPlatformBase,
                                      _CU_DEVICE_ATTRIBUTE_INTEGRATED)
from fastvideo.platforms.interface import Platform
from fastvideo.platforms.mps import MpsPlatform


def test_base_platform_reports_separate_pools() -> None:
    # Discrete accelerators are the default, so the base must answer False and
    # leave the existing offload path alone.
    assert Platform.has_unified_memory(device_id=3) is False


def test_mps_is_unified() -> None:
    assert MpsPlatform.has_unified_memory(device_id=2) is True


def _fake_cuda_driver(integrated: bool = True):
    driver = SimpleNamespace(
        cuInit=Mock(return_value=0),
        cuDeviceGet=Mock(),
        cuDeviceGetAttribute=Mock(),
    )

    def get_device(device, device_id):
        device._obj.value = device_id + 40
        return 0

    def get_attribute(value, attribute, device):
        value._obj.value = int(integrated)
        return 0

    driver.cuDeviceGet.side_effect = get_device
    driver.cuDeviceGetAttribute.side_effect = get_attribute
    return driver


@pytest.mark.parametrize("integrated", [True, False])
def test_cuda_follows_driver_integrated_attribute(monkeypatch, integrated: bool) -> None:
    # The driver attribute is cudaDeviceProp::integrated's source of truth and
    # does not initialize a PyTorch CUDA context.
    driver = _fake_cuda_driver(integrated)
    get_device_properties = Mock(side_effect=AssertionError("torch fallback used"))
    monkeypatch.setattr("fastvideo.platforms.cuda.ctypes.CDLL", lambda library: driver)
    monkeypatch.setattr("torch.cuda.get_device_properties", get_device_properties)

    assert CudaPlatformBase.has_unified_memory(device_id=7) is integrated

    driver.cuInit.assert_called_once_with(0)
    assert driver.cuDeviceGet.call_args.args[1] == 7
    assert driver.cuDeviceGetAttribute.call_args.args[1] == _CU_DEVICE_ATTRIBUTE_INTEGRATED
    assert driver.cuDeviceGetAttribute.call_args.args[2].value == 47
    get_device_properties.assert_not_called()


@pytest.mark.parametrize("failing_call", ["cuInit", "cuDeviceGet", "cuDeviceGetAttribute"])
def test_cuda_driver_call_failure_assumes_separate_pools(monkeypatch, failing_call: str) -> None:
    driver = _fake_cuda_driver()
    getattr(driver, failing_call).side_effect = None
    getattr(driver, failing_call).return_value = 1
    monkeypatch.setattr("fastvideo.platforms.cuda.ctypes.CDLL", lambda library: driver)

    assert CudaPlatformBase.has_unified_memory(device_id=4) is False


def test_cuda_without_driver_library_assumes_separate_pools(monkeypatch) -> None:
    def unavailable(library):
        raise OSError("CUDA driver unavailable")

    monkeypatch.setattr("fastvideo.platforms.cuda.ctypes.CDLL", unavailable)

    assert CudaPlatformBase.has_unified_memory(device_id=5) is False
