# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

import fastvideo.platforms as platforms
from fastvideo.distributed import parallel_state


def _fake_platform(monkeypatch: pytest.MonkeyPatch, *, cuda_alike: bool = False, npu: bool = False) -> None:
    platform = SimpleNamespace(is_cuda_alike=lambda: cuda_alike, is_npu=lambda: npu, device_name="fake")
    # Set the lazy source, not the public attribute: a pinned current_platform
    # would leak into later tests that replace _current_platform.
    monkeypatch.setattr(platforms, "_current_platform", platform)
    if "current_platform" in vars(platforms):
        monkeypatch.delitem(vars(platforms), "current_platform")


@pytest.mark.parametrize("nccl, expected", [(True, "nccl"), (False, "gloo")])
def test_cuda_alike_platform_uses_nccl_only_when_torch_has_it(monkeypatch, nccl, expected):
    _fake_platform(monkeypatch, cuda_alike=True)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: nccl)
    assert parallel_state._platform_backend(world_size=1) == expected


def test_multi_gpu_without_nccl_is_refused(monkeypatch):
    _fake_platform(monkeypatch, cuda_alike=True)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: False)
    with pytest.raises(RuntimeError, match="no NCCL"):
        parallel_state._platform_backend(world_size=2)


def test_npu_platform_uses_hccl(monkeypatch):
    _fake_platform(monkeypatch, npu=True)
    assert parallel_state._platform_backend(world_size=1) == "hccl"


def test_other_platforms_use_gloo(monkeypatch):
    _fake_platform(monkeypatch)
    assert parallel_state._platform_backend(world_size=2) == "gloo"
