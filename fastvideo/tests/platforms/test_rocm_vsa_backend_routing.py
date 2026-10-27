# SPDX-License-Identifier: Apache-2.0
"""ROCm must route the video sparse attention backends to their Triton kernels
instead of rejecting them as invalid for the platform."""

import contextlib
import sys
import types

import pytest
import torch

import fastvideo.envs as envs
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.platforms import rocm
from fastvideo.platforms.rocm import RocmPlatform
from fastvideo.utils import resolve_obj_by_qualname

VSA_BACKEND = "fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionBackend"
H3_BACKEND = "fastvideo.attention.backends.video_sparse_attn_h3.MiniMaxH3VSABackend"


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


@contextlib.contextmanager
def kernel_switches(cutedsl=None, triton=None, legacy_triton=None):
    """Set fastvideo-kernel's backend switches for the block, unsetting the ones left as None."""
    with contextlib.ExitStack() as overrides:
        overrides.enter_context(envs.override_external("FASTVIDEO_VSA_CUTEDSL", cutedsl))
        overrides.enter_context(envs.override_external("FASTVIDEO_VSA_TRITON", triton))
        overrides.enter_context(envs.override_external("FASTVIDEO_KERNEL_VSA_FORCE_TRITON", legacy_triton))
        yield


def test_rocm_routes_video_sparse_attention(kernel_stubs):
    cls_str = RocmPlatform.get_attn_backend_cls(AttentionBackendEnum.VIDEO_SPARSE_ATTN, 128, torch.bfloat16)
    assert cls_str == VSA_BACKEND


def test_rocm_routes_h3_video_sparse_attention(kernel_stubs):
    # kernel_switches() unsets FASTVIDEO_VSA_CUTEDSL, so an ambient
    # FASTVIDEO_VSA_CUTEDSL=1 cannot turn this happy path into the ValueError.
    with kernel_switches():
        cls_str = RocmPlatform.get_attn_backend_cls(AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3, 128, torch.bfloat16)
    assert cls_str == H3_BACKEND


@pytest.mark.parametrize("backend, qualname", [(AttentionBackendEnum.VIDEO_SPARSE_ATTN, VSA_BACKEND),
                                               (AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3, H3_BACKEND)])
def test_routed_backend_classes_resolve(backend, qualname):
    # Attention construction imports the class behind the returned name, so a
    # wrong module path or a broken backend import has to fail here. No kernel
    # stubs: both backend modules import without a fastvideo_kernel build.
    backend_cls = resolve_obj_by_qualname(qualname)
    assert backend_cls.get_name() == backend.name


def test_rocm_rejects_the_cute_opt_in_for_h3(kernel_stubs):
    with kernel_switches(cutedsl="1"), pytest.raises(ValueError, match="FASTVIDEO_VSA_CUTEDSL"):
        RocmPlatform.get_attn_backend_cls(AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3, 128, torch.bfloat16)


@pytest.mark.parametrize("triton, legacy_triton", [("1", None), (None, "1")])
def test_rocm_keeps_h3_when_triton_is_forced_over_the_cute_opt_in(kernel_stubs, triton, legacy_triton):
    with kernel_switches(cutedsl="1", triton=triton, legacy_triton=legacy_triton):
        cls_str = RocmPlatform.get_attn_backend_cls(AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3, 128, torch.bfloat16)
    assert cls_str == H3_BACKEND


def test_rocm_cute_opt_in_leaves_the_64_token_vsa_path_alone(kernel_stubs):
    # The 64-token VSA kernels have no CuTe route, so the opt-in never reaches them.
    with kernel_switches(cutedsl="1"):
        cls_str = RocmPlatform.get_attn_backend_cls(AttentionBackendEnum.VIDEO_SPARSE_ATTN, 128, torch.bfloat16)
    assert cls_str == VSA_BACKEND


@pytest.mark.parametrize("cutedsl", [None, "1"])
@pytest.mark.parametrize("triton", [None, "1"])
@pytest.mark.parametrize("legacy_triton", [None, "1"])
def test_cute_opt_in_rule_matches_fastvideo_kernel(cutedsl, triton, legacy_triton):
    kernel_256 = pytest.importorskip("fastvideo_kernel.block_sparse_attn_256")
    with kernel_switches(cutedsl, triton, legacy_triton):
        assert rocm._vsa_cute_opt_in() == (kernel_256._resolve_backend() == "cutedsl")


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
