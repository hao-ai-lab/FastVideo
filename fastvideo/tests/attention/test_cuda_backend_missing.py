from __future__ import annotations

import builtins
import sys
from types import ModuleType

import pytest
import torch

from fastvideo.platforms.cuda import CudaPlatformBase
from fastvideo.platforms.interface import AttentionBackendEnum


class _GpuIndependentCudaPlatform(CudaPlatformBase):

    @classmethod
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        return False


@pytest.mark.parametrize(
    ("backend", "missing_module", "package_name", "install_hint"),
    [
        (
            AttentionBackendEnum.SAGE_ATTN,
            "sageattention",
            "sageattention",
            "uv pip install sageattention",
        ),
        (
            AttentionBackendEnum.SAGE_ATTN_THREE,
            "sageattn3",
            "sageattn3",
            "uv pip install sageattn3",
        ),
        (
            AttentionBackendEnum.ATTN_QAT_INFER,
            "fastvideo.attention.backends.attn_qat_infer",
            "attn_qat_infer",
            "https://hao-ai-lab.github.io/FastVideo/",
        ),
    ],
)
def test_explicit_backend_import_failure_does_not_fallback(
    monkeypatch,
    backend: AttentionBackendEnum,
    missing_module: str,
    package_name: str,
    install_hint: str,
) -> None:
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", backend.name)
    real_import = builtins.__import__

    def import_with_missing_backend(name, globals=None, locals=None, fromlist=(), level=0):
        if name == missing_module:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_with_missing_backend)

    with pytest.raises(ImportError) as excinfo:
        _GpuIndependentCudaPlatform.get_attn_backend_cls(backend, head_size=64, dtype=torch.float16)

    message = str(excinfo.value)
    assert "explicitly requested" in message
    assert backend.name in message
    assert package_name in message
    assert install_hint in message
    assert excinfo.value.__cause__ is not None


def test_explicit_attn_qat_infer_unavailable_does_not_fallback(monkeypatch) -> None:
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", AttentionBackendEnum.ATTN_QAT_INFER.name)
    module_name = "fastvideo.attention.backends.attn_qat_infer"
    unavailable_backend = ModuleType(module_name)
    unavailable_backend.AttnQatInferBackend = object
    unavailable_backend.is_attn_qat_infer_available = lambda: False
    monkeypatch.setitem(sys.modules, module_name, unavailable_backend)

    with pytest.raises(ImportError) as excinfo:
        _GpuIndependentCudaPlatform.get_attn_backend_cls(
            AttentionBackendEnum.ATTN_QAT_INFER,
            head_size=64,
            dtype=torch.float16,
        )

    message = str(excinfo.value)
    assert "explicitly requested" in message
    assert AttentionBackendEnum.ATTN_QAT_INFER.name in message
    assert "attn_qat_infer" in message
    assert "not built" in message
    assert "Install the FastVideo Attn-QAT inference kernel" in message
    assert "https://hao-ai-lab.github.io/FastVideo/" in message
