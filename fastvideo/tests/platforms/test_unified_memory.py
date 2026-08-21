# SPDX-License-Identifier: Apache-2.0
"""CPU tests for ``Platform.has_unified_memory``.

The flag decides whether the text encoder loader skips FSDP CPU offload. On a
device where host and device are the same RAM, offloading holds two copies of
the model instead of moving one, so getting this wrong costs peak memory on
exactly the machines that have the least of it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from fastvideo.platforms.cuda import CudaPlatformBase
from fastvideo.platforms.interface import Platform
from fastvideo.platforms.mps import MpsPlatform


def test_base_platform_reports_separate_pools() -> None:
    # Discrete accelerators are the default, so the base must answer False and
    # leave the existing offload path alone.
    assert Platform.has_unified_memory() is False


def test_mps_is_unified() -> None:
    assert MpsPlatform.has_unified_memory() is True


@pytest.mark.parametrize("integrated, expected", [(True, True), (False, False)])
def test_cuda_follows_is_integrated(monkeypatch, integrated: bool, expected: bool) -> None:
    # cudaDeviceProp::integrated is the authority: it is set on parts whose GPU
    # reads host memory (GB10, Jetson) and clear on discrete cards.
    monkeypatch.setattr(
        "torch.cuda.get_device_properties",
        lambda device_id: SimpleNamespace(is_integrated=integrated),
    )
    assert CudaPlatformBase.has_unified_memory() is expected


def test_cuda_without_the_attribute_assumes_separate_pools(monkeypatch) -> None:
    # Older torch builds do not surface the field. Assume discrete rather than
    # silently changing offload behaviour on a machine we cannot classify.
    monkeypatch.setattr(
        "torch.cuda.get_device_properties",
        lambda device_id: SimpleNamespace(),
    )
    assert CudaPlatformBase.has_unified_memory() is False


def test_cuda_without_a_visible_device_does_not_raise(monkeypatch) -> None:
    # Called during loading, where a probe failure should not take the run down.
    def boom(device_id):
        raise RuntimeError("no CUDA device")

    monkeypatch.setattr("torch.cuda.get_device_properties", boom)
    assert CudaPlatformBase.has_unified_memory() is False
