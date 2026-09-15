# SPDX-License-Identifier: Apache-2.0
"""ROCm must route the video sparse attention backends to their Triton kernels
instead of rejecting them as invalid for the platform."""

import sys
import types

import pytest
import torch

from fastvideo.platforms import AttentionBackendEnum
from fastvideo.platforms.rocm import RocmPlatform


@pytest.fixture
def kernel_stubs(monkeypatch):
    """Minimal fastvideo_kernel stand-ins, so routing is tested without a GPU build."""
    kernel = types.ModuleType("fastvideo_kernel")
    kernel.__path__ = []
    kernel.video_sparse_attn = lambda *args, **kwargs: None
    bsa_256 = types.ModuleType("fastvideo_kernel.block_sparse_attn_256")
    bsa_256.block_sparse_attn_256_bshd = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "fastvideo_kernel", kernel)
    monkeypatch.setitem(sys.modules, "fastvideo_kernel.block_sparse_attn_256", bsa_256)


@pytest.fixture
def no_kernel(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastvideo_kernel", None)
    monkeypatch.setitem(sys.modules, "fastvideo_kernel.block_sparse_attn_256", None)


def test_rocm_routes_video_sparse_attention(kernel_stubs):
    cls_str = RocmPlatform.get_attn_backend_cls(AttentionBackendEnum.VIDEO_SPARSE_ATTN, 128, torch.bfloat16)
    assert cls_str == "fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionBackend"


def test_rocm_routes_h3_video_sparse_attention(kernel_stubs):
    cls_str = RocmPlatform.get_attn_backend_cls(AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3, 128, torch.bfloat16)
    assert cls_str == "fastvideo.attention.backends.video_sparse_attn_h3.MiniMaxH3VSABackend"


@pytest.mark.parametrize("backend", [AttentionBackendEnum.VIDEO_SPARSE_ATTN, AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3])
def test_rocm_without_fastvideo_kernel_raises_actionable_import_error(no_kernel, backend):
    with pytest.raises(ImportError, match="fastvideo-kernel"):
        RocmPlatform.get_attn_backend_cls(backend, 128, torch.bfloat16)


def test_rocm_rejects_sage_attention_with_value_error():
    with pytest.raises(ValueError, match="not supported"):
        RocmPlatform.get_attn_backend_cls(AttentionBackendEnum.SAGE_ATTN, 128, torch.bfloat16)


def test_rocm_rejects_other_backends_with_value_error():
    # Used to raise TypeError from a membership test against a bare enum member.
    with pytest.raises(ValueError, match="Invalid attention backend"):
        RocmPlatform.get_attn_backend_cls(AttentionBackendEnum.BSA_ATTN, 128, torch.bfloat16)


def test_rocm_still_resolves_sdpa():
    cls_str = RocmPlatform.get_attn_backend_cls(AttentionBackendEnum.TORCH_SDPA, 128, torch.bfloat16)
    assert cls_str == "fastvideo.attention.backends.sdpa.SDPABackend"
