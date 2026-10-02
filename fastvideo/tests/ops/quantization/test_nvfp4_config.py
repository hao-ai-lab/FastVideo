# SPDX-License-Identifier: Apache-2.0
"""NVFP4Config import + lazy-flashinfer behavior tests.

The actual NVFP4 kernels need flashinfer + a CUDA device, so these
tests focus on the lazy-import contract: the module must load on
hosts without flashinfer, and only call sites should fail with a
clear error when flashinfer is missing.
"""
from __future__ import annotations

import importlib
import sys
import types

import pytest


def _clear_nvfp4_module(monkeypatch):
    # Load the original first, even when this test runs in isolation.
    module = importlib.import_module("fastvideo.layers.quantization.nvfp4_config")
    # A fresh import replaces both bindings; restore both at teardown.
    monkeypatch.delattr(sys.modules[module.__package__], "nvfp4_config")
    monkeypatch.delitem(sys.modules, module.__name__)


def test_nvfp4config_imports_without_flashinfer(monkeypatch):
    """Importing the module on a host without flashinfer must succeed.

    This is the contract Dreamverse depends on — the GPU worker boots
    even when flashinfer is not in the venv, because the import happens
    in `video_generation.py` before any NVFP4 kernel is invoked.
    """
    # Hide flashinfer from sys.modules and the import path.
    monkeypatch.setitem(sys.modules, "flashinfer", None)
    # Force a re-import of the target module.
    _clear_nvfp4_module(monkeypatch)
    from fastvideo.layers.quantization.nvfp4_config import NVFP4Config
    config = NVFP4Config()
    assert config.get_name() == "nvfp4"
    assert config.layer_profile == "refine"


def test_nvfp4config_layer_profile_round_trips_from_dict():
    from fastvideo.layers.quantization.nvfp4_config import NVFP4Config
    config = NVFP4Config.from_config({"layer_profile": "base"})
    assert config.layer_profile == "base"
    config = NVFP4Config.from_config({})
    assert config.layer_profile == "refine"


def test_nvfp4_kernel_call_raises_clear_error_without_flashinfer(monkeypatch):
    """A call into the NVFP4 kernels must raise an actionable
    ImportError when flashinfer is missing, not a confusing
    AttributeError or NameError."""
    # Stage a fake flashinfer that fails on import.
    monkeypatch.setitem(sys.modules, "flashinfer", _raise_module_on_import("flashinfer"))
    _clear_nvfp4_module(monkeypatch)
    from fastvideo.layers.quantization.nvfp4_config import _require_flashinfer
    with pytest.raises(ImportError, match="flashinfer"):
        _require_flashinfer()


def _raise_module_on_import(name: str) -> types.ModuleType:
    """Build a stub module that raises ImportError on any attribute
    access, so `from <name> import X` fails the way a missing package
    would."""

    class _RaisingModule(types.ModuleType):

        def __getattr__(self, item: str):
            raise ImportError(f"No module named '{name}.{item}'")

    return _RaisingModule(name)


def test_coerce_fp4_input_dtype_casts_and_rejects():
    """The FP4 input-dtype coercion (CPU-only, no flashinfer/CUDA): bf16/fp16
    pass through, other floats (the fp32 pre-attention norm in eager mode) are
    cast to bf16, and non-floating inputs are rejected fast."""
    import torch

    from fastvideo.layers.quantization.nvfp4_config import (_coerce_fp4_input_dtype)

    # bf16 / fp16 pass through untouched.
    bf16 = torch.zeros(4, 8, dtype=torch.bfloat16)
    assert _coerce_fp4_input_dtype(bf16) is bf16
    fp16 = torch.zeros(4, 8, dtype=torch.float16)
    assert _coerce_fp4_input_dtype(fp16) is fp16

    # Other floating dtypes (fp32 from an unfused norm, fp64) -> bf16.
    assert _coerce_fp4_input_dtype(torch.zeros(4, 8, dtype=torch.float32)).dtype is torch.bfloat16
    assert _coerce_fp4_input_dtype(torch.zeros(4, 8, dtype=torch.float64)).dtype is torch.bfloat16

    # Non-floating inputs are a real error, not silently cast.
    for bad in (torch.int32, torch.int64, torch.bool):
        with pytest.raises(TypeError, match="floating-point"):
            _coerce_fp4_input_dtype(torch.zeros(4, 8, dtype=bad))


@pytest.mark.parametrize("import_test", [
    test_nvfp4config_imports_without_flashinfer,
    test_nvfp4_kernel_call_raises_clear_error_without_flashinfer,
])
def test_nvfp4_import_tests_restore_module_identity(import_test):
    original = importlib.import_module("fastvideo.layers.quantization.nvfp4_config")

    with pytest.MonkeyPatch.context() as monkeypatch:
        import_test(monkeypatch)
        assert sys.modules[original.__name__] is not original

    from fastvideo.layers.quantization import nvfp4_config
    from fastvideo.layers.quantization.nvfp4_config import NVFP4Config

    assert sys.modules[original.__name__] is original
    assert nvfp4_config is original
    assert NVFP4Config is original.NVFP4Config
