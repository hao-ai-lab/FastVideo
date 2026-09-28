# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the CUDA selector's FlashAttention import policy.

Exercise the real resolver while faking device capability and backend imports;
no CUDA device or FlashAttention installation is required.
"""

import builtins
from types import SimpleNamespace

import pytest
import torch

from fastvideo.platforms import cuda
from fastvideo.platforms.cuda import NonNvmlCudaPlatform
from fastvideo.platforms.interface import AttentionBackendEnum

FLASH_ATTN_CLS = "fastvideo.attention.backends.flash_attn.FlashAttentionBackend"
SDPA_CLS = "fastvideo.attention.backends.sdpa.SDPABackend"


@pytest.fixture(autouse=True)
def _fake_device(monkeypatch):
    monkeypatch.delenv("FASTVIDEO_ATTENTION_BACKEND", raising=False)
    monkeypatch.delenv("FASTVIDEO_FA4", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 1))


def _fake_backend_import(monkeypatch, error=None):
    original_import = builtins.__import__
    imported = []
    backend = SimpleNamespace(
        FlashAttentionBackend=SimpleNamespace(get_supported_head_sizes=lambda: [64, 128]),
    )

    def import_backend(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fastvideo.attention.backends.flash_attn":
            imported.append(name)
            if error is not None:
                raise error
            return backend
        if name == "flash_attn":
            # FA3-only environments need not provide the FA2 parent package.
            raise ModuleNotFoundError("No module named 'flash_attn'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_backend)
    return imported


def _resolve(*, selected_backend=None, head_size=128, dtype=torch.bfloat16):
    return NonNvmlCudaPlatform.get_attn_backend_cls(selected_backend, head_size, dtype)


@pytest.mark.parametrize("fa4", ["0", "1"])
def test_usable_backend_does_not_require_separate_fa2_package_probe(monkeypatch, fa4):
    monkeypatch.setenv("FASTVIDEO_FA4", fa4)
    imported = _fake_backend_import(monkeypatch)

    assert _resolve() == FLASH_ATTN_CLS
    assert imported == ["fastvideo.attention.backends.flash_attn"]


@pytest.mark.parametrize("error", [
    ModuleNotFoundError("No module named 'flash_attn'"),
    ImportError("cannot import name 'flash_attn_varlen_func' from 'flash_attn.cute'"),
])
def test_automatic_import_failure_logs_actual_reason_and_falls_back(monkeypatch, error):
    messages = []
    monkeypatch.setattr(cuda.logger, "info", lambda message, *args: messages.append(message % args))
    _fake_backend_import(monkeypatch, error)

    assert _resolve() == SDPA_CLS
    assert any(f"FASTVIDEO_FA4=0): {type(error).__name__}: {error}" in message for message in messages)


def test_explicit_fa4_import_failure_does_not_silently_select_sdpa(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_FA4", "1")
    error = ImportError("cannot import name 'flash_attn_varlen_func' from 'flash_attn.cute'")
    _fake_backend_import(monkeypatch, error)

    with pytest.raises(RuntimeError, match="FASTVIDEO_FA4=1 but the FlashAttention backend failed to import") as exc_info:
        _resolve()

    assert exc_info.value.__cause__ is error
    assert str(error) in str(exc_info.value)
    assert "worker's Python environment" in str(exc_info.value)


@pytest.mark.parametrize("kwargs, capability, import_expected", [
    ({"dtype": torch.float32}, (12, 1), False),
    ({"head_size": 48}, (12, 1), True),
    ({}, (7, 5), False),
    ({"selected_backend": AttentionBackendEnum.TORCH_SDPA}, (12, 1), False),
])
def test_fa4_opt_in_preserves_layer_and_explicit_sdpa_fallbacks(monkeypatch, kwargs, capability, import_expected):
    monkeypatch.setenv("FASTVIDEO_FA4", "1")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    imported = _fake_backend_import(monkeypatch)

    assert _resolve(**kwargs) == SDPA_CLS
    assert bool(imported) is import_expected
