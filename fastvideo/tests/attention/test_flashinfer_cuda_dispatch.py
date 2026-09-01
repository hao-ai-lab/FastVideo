# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for CudaPlatformBase.get_attn_backend_cls's FLASHINFER branch.

Covers the three ways FLASHINFER resolution can fail before ever touching a
GPU kernel: missing sm80 capability, a head size outside FlashInfer's safe
list, and flashinfer-python not being importable. All device/import probes
are monkeypatched, so this runs without a real GPU or the flashinfer wheel.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch

from fastvideo.platforms.cuda import CudaPlatformBase
from fastvideo.platforms.interface import AttentionBackendEnum


def _patch_capability(monkeypatch, *, supported: bool) -> None:
    monkeypatch.setattr(CudaPlatformBase, "has_device_capability",
                        classmethod(lambda cls, capability, device_id=0: supported))


def _install_fake_flashinfer(monkeypatch) -> None:
    prefill = types.ModuleType("flashinfer.prefill")
    prefill.single_prefill_with_kv_cache = lambda *a, **k: None
    flashinfer = types.ModuleType("flashinfer")
    flashinfer.prefill = prefill
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.prefill", prefill)


def _block_flashinfer_import(monkeypatch) -> None:
    # A None entry in sys.modules makes `import flashinfer.prefill` raise
    # ImportError, regardless of whether the real package is installed.
    monkeypatch.setitem(sys.modules, "flashinfer.prefill", None)
    monkeypatch.setitem(sys.modules, "flashinfer", None)


def test_flashinfer_requires_sm80(monkeypatch) -> None:
    _patch_capability(monkeypatch, supported=False)
    with pytest.raises(RuntimeError, match="sm80"):
        CudaPlatformBase.get_attn_backend_cls(AttentionBackendEnum.FLASHINFER, 128, torch.bfloat16)


def test_flashinfer_rejects_unsupported_head_size(monkeypatch) -> None:
    _patch_capability(monkeypatch, supported=True)
    _install_fake_flashinfer(monkeypatch)
    with pytest.raises(ValueError, match="does not safely support head size"):
        CudaPlatformBase.get_attn_backend_cls(AttentionBackendEnum.FLASHINFER, 96, torch.bfloat16)


def test_flashinfer_missing_install_raises_actionable_error(monkeypatch) -> None:
    _patch_capability(monkeypatch, supported=True)
    _block_flashinfer_import(monkeypatch)
    with pytest.raises(ImportError, match="flashinfer-python is not importable"):
        CudaPlatformBase.get_attn_backend_cls(AttentionBackendEnum.FLASHINFER, 128, torch.bfloat16)


@pytest.mark.parametrize("head_size", [64, 128, 256])
def test_flashinfer_resolves_for_supported_head_sizes(monkeypatch, head_size: int) -> None:
    _patch_capability(monkeypatch, supported=True)
    _install_fake_flashinfer(monkeypatch)
    backend_cls = CudaPlatformBase.get_attn_backend_cls(AttentionBackendEnum.FLASHINFER, head_size, torch.bfloat16)
    assert backend_cls == "fastvideo.attention.backends.flashinfer.FlashInferBackend"


def test_flashinfer_warns_on_dtype_cast(monkeypatch, caplog) -> None:
    _patch_capability(monkeypatch, supported=True)
    _install_fake_flashinfer(monkeypatch)
    with caplog.at_level("WARNING"):
        CudaPlatformBase.get_attn_backend_cls(AttentionBackendEnum.FLASHINFER, 128, torch.float32)
    assert any("cast" in record.message for record in caplog.records)
