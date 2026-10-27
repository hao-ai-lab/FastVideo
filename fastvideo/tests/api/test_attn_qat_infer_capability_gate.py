# SPDX-License-Identifier: Apache-2.0
"""ATTN_QAT_INFER selection must be gated on device capability, through
the real platform resolver.

``is_attn_qat_infer_available()`` used to test only whether the kernel
extension imports. CUDA 13 wheel builds can carry the sm_120/sm_121
extension on any host (e.g. H100 sm_90, the Slurm runner's GB200 sm_100). Without the
capability gate, that successful import selects the consumer-Blackwell
backend and defers failure until the first unsupported kernel call.
Explicit ATTN_QAT_INFER requests must instead fail closed during
resolution.

These tests drive the REAL resolver (``fastvideo.platforms.cuda``) and
the REAL availability function; only the two physical facts are faked --
"does the extension import" (``_get_attn_qat_infer``) and "what GPU is
active" (``torch.cuda``). The stage-guard test
(fastvideo/tests/stages/test_kandinsky5_attention_backend_guard.py)
injects an already-resolved backend and by design cannot see this bug.

CPU-only: the ATTN_QAT_INFER branch and its error path never require a
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


def _fake_gpu(monkeypatch, *, capability: tuple[int, int], extension_imports: bool, fa4_imports: bool = False) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(
        attn_qat_infer_module,
        "_get_attn_qat_infer",
        lambda: (lambda *a, **k: None) if extension_imports else None,
    )
    # The FP4 FA4 probe (flash-attention-fp4, the sm_100a/sm_103a route) is
    # a physical fact of the running environment; fake it like the others so
    # these tests are deterministic on hosts with/without flash_attn.cute.
    monkeypatch.setattr(attn_qat_infer_module, "_fa4_fp4_available", lambda: fa4_imports)


def _resolve() -> str:
    return NonNvmlCudaPlatform.get_attn_backend_cls(
        AttentionBackendEnum.ATTN_QAT_INFER,
        head_size=128,
        dtype=torch.bfloat16,
    )


def test_sm90_host_with_bundled_extension_fails_closed(monkeypatch):
    """The reviewed failure: H100 + CUDA 13 wheel that bundles the sm_120
    extension. Import succeeds; explicit selection must still fail closed."""
    _fake_gpu(monkeypatch, capability=(9, 0), extension_imports=True)

    assert not is_attn_qat_infer_available()
    with pytest.raises(ImportError, match="ATTN_QAT_INFER selected but"):
        _resolve()


def test_sm100_host_with_bundled_extension_fails_closed(monkeypatch):
    """sm_100 with only the (unrunnable) bundled sm_12x extension and no
    FP4 FA4 kernel still fails closed -- the original reviewed failure class."""
    _fake_gpu(monkeypatch, capability=(10, 0), extension_imports=True, fa4_imports=False)

    assert not is_attn_qat_infer_available()
    with pytest.raises(ImportError, match="ATTN_QAT_INFER selected but"):
        _resolve()


@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
def test_datacenter_blackwell_with_fa4_selects_backend(monkeypatch, capability):
    """sm_100a/sm_103a resolve ATTN_QAT_INFER through the FP4 FA4 route
    when flash-attention-fp4 is installed (even without the sm_12x ext)."""
    _fake_gpu(monkeypatch, capability=capability, extension_imports=False, fa4_imports=True)

    assert is_attn_qat_infer_available()
    assert _resolve() == ATTN_QAT_INFER_CLS


@pytest.mark.parametrize("capability", [(12, 0), (12, 1)])
def test_consumer_blackwell_with_extension_selects_backend(monkeypatch, capability):
    _fake_gpu(monkeypatch, capability=capability, extension_imports=True)

    assert is_attn_qat_infer_available()
    assert _resolve() == ATTN_QAT_INFER_CLS


def test_consumer_blackwell_without_extension_fails_closed(monkeypatch):
    _fake_gpu(monkeypatch, capability=(12, 0), extension_imports=False)

    assert not is_attn_qat_infer_available()
    with pytest.raises(ImportError, match="ATTN_QAT_INFER selected but"):
        _resolve()


def test_no_cuda_reports_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        attn_qat_infer_module,
        "_get_attn_qat_infer",
        lambda: (lambda *a, **k: None),
    )

    assert not is_attn_qat_infer_available()
