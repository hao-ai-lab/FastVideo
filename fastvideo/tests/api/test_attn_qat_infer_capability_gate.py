# SPDX-License-Identifier: Apache-2.0
"""ATTN_QAT_INFER selection must be gated on device capability, through
the real platform resolver.

``is_attn_qat_infer_available()`` used to test only whether the kernel
extension imports. CUDA 13 wheel builds can carry the sm_120/sm_121
extension on any host (e.g. H100 sm_90, GB200 sm_100): the import
succeeds, ``CudaPlatformBase.get_attn_backend_cls`` selects the
consumer-Blackwell backend, and the first kernel call fails with an
unsupported-capability error -- instead of the FlashAttention fallback
the QAD README documents for non-sm_120 GPUs.

These tests drive the REAL resolver (``fastvideo.platforms.cuda``) and
the REAL availability function; only the two physical facts are faked --
"does the extension import" (``_get_attn_qat_infer``) and "what GPU is
active" (``torch.cuda``). The stage-guard test
(fastvideo/tests/stages/test_kandinsky5_attention_backend_guard.py)
injects an already-resolved backend and by design cannot see this bug.

CPU-only: the ATTN_QAT_INFER branch and its fallback never require a
physical GPU to *resolve* (only to run).
"""
from __future__ import annotations

import pytest
import torch

import fastvideo.attention.backends.attn_qat_infer as attn_qat_infer_module
from fastvideo.attention.backends.attn_qat_infer import is_attn_qat_infer_available
# The concrete resolver whose device queries go straight through torch.cuda
# (the NVML variant only differs in device-info plumbing, not selection
# logic) -- the abstract CudaPlatformBase can't run the fallthrough's
# has_device_capability() check.
from fastvideo.platforms.cuda import NonNvmlCudaPlatform
from fastvideo.platforms.interface import AttentionBackendEnum

ATTN_QAT_INFER_CLS = "fastvideo.attention.backends.attn_qat_infer.AttnQatInferBackend"
# What the resolver's fallthrough legitimately returns when ATTN_QAT_INFER
# is unavailable: FlashAttention, or SDPA when flash_attn isn't installed
# in the running environment (e.g. CPU-only CI).
FALLBACK_CLASSES = {
    "fastvideo.attention.backends.flash_attn.FlashAttentionBackend",
    "fastvideo.attention.backends.sdpa.SDPABackend",
}


def _fake_gpu(monkeypatch, *, capability: tuple[int, int], extension_imports: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(
        attn_qat_infer_module,
        "_get_attn_qat_infer",
        lambda: (lambda *a, **k: None) if extension_imports else None,
    )


def _resolve() -> str:
    return NonNvmlCudaPlatform.get_attn_backend_cls(
        AttentionBackendEnum.ATTN_QAT_INFER,
        head_size=128,
        dtype=torch.bfloat16,
    )


def test_sm90_host_with_bundled_extension_falls_back(monkeypatch):
    """The reviewed failure: H100 + CUDA 13 wheel that bundles the sm_120
    extension. Import succeeds; selection must still fall back."""
    _fake_gpu(monkeypatch, capability=(9, 0), extension_imports=True)

    assert not is_attn_qat_infer_available()
    assert _resolve() in FALLBACK_CLASSES


def test_sm100_host_with_bundled_extension_falls_back(monkeypatch):
    _fake_gpu(monkeypatch, capability=(10, 0), extension_imports=True)

    assert not is_attn_qat_infer_available()
    assert _resolve() in FALLBACK_CLASSES


@pytest.mark.parametrize("capability", [(12, 0), (12, 1)])
def test_consumer_blackwell_with_extension_selects_backend(monkeypatch, capability):
    _fake_gpu(monkeypatch, capability=capability, extension_imports=True)

    assert is_attn_qat_infer_available()
    assert _resolve() == ATTN_QAT_INFER_CLS


def test_consumer_blackwell_without_extension_falls_back(monkeypatch):
    _fake_gpu(monkeypatch, capability=(12, 0), extension_imports=False)

    assert not is_attn_qat_infer_available()
    assert _resolve() in FALLBACK_CLASSES


def test_no_cuda_reports_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        attn_qat_infer_module,
        "_get_attn_qat_infer",
        lambda: (lambda *a, **k: None),
    )

    assert not is_attn_qat_infer_available()
