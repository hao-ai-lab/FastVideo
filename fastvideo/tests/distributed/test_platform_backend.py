# SPDX-License-Identifier: Apache-2.0
"""The default torch.distributed backend follows the platform and what the torch build provides."""

from types import SimpleNamespace

import pytest
import torch

import fastvideo.platforms as platforms
from fastvideo.distributed import parallel_state


def fake_platform(monkeypatch, *, cuda_alike=False, npu=False):
    platform = SimpleNamespace(is_cuda_alike=lambda: cuda_alike, is_npu=lambda: npu, device_name="fake")
    monkeypatch.setattr(platforms, "_current_platform", platform)
    return platform


@pytest.mark.parametrize("nccl, expected", [(True, "nccl"), (False, "gloo")])
def test_cuda_alike_platform_uses_nccl_only_when_torch_has_it(monkeypatch, nccl, expected):
    # ROCm torch on Windows is built without NCCL.
    fake_platform(monkeypatch, cuda_alike=True)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: nccl)
    assert parallel_state._platform_backend() == expected


def test_npu_platform_uses_hccl(monkeypatch):
    fake_platform(monkeypatch, npu=True)
    assert parallel_state._platform_backend() == "hccl"


def test_other_platforms_use_gloo(monkeypatch):
    fake_platform(monkeypatch)
    assert parallel_state._platform_backend() == "gloo"
