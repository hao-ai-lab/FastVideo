# SPDX-License-Identifier: Apache-2.0
"""CPU reference models must resolve real SDPA without a CUDA backend."""

import pytest
import torch

from fastvideo.platforms.cpu import CpuPlatform
from fastvideo.platforms.interface import AttentionBackendEnum


@pytest.mark.parametrize("requested", [None, AttentionBackendEnum.TORCH_SDPA])
def test_cpu_sdpa_resolution_through_selector(monkeypatch, requested):
    from fastvideo import platforms
    from fastvideo.attention.backends.sdpa import SDPABackend
    from fastvideo.attention.selector import _cached_get_attn_backend, get_attn_backend

    monkeypatch.setattr(platforms, "current_platform", CpuPlatform())
    _cached_get_attn_backend.cache_clear()
    try:
        backend = get_attn_backend(64, torch.float32, (AttentionBackendEnum.TORCH_SDPA,), requested=requested)
        assert backend is SDPABackend
    finally:
        _cached_get_attn_backend.cache_clear()


def test_cpu_rejects_sparse_backend_instead_of_changing_attention():
    with pytest.raises(NotImplementedError, match="not supported on CPU"):
        CpuPlatform.get_attn_backend_cls(AttentionBackendEnum.VIDEO_SPARSE_ATTN, 64, torch.float32)
