# SPDX-License-Identifier: Apache-2.0
"""FP4Config import + lazy-flashinfer behavior tests.

The actual FP4 kernels need flashinfer + a CUDA device, so these tests
focus on the lazy-import contract: the module must load on hosts
without flashinfer, and only call sites should fail with a clear
error when flashinfer is missing.
"""
from __future__ import annotations

import sys
import types

import pytest


def test_fp4config_imports_without_flashinfer(monkeypatch):
    """Importing the module on a host without flashinfer must succeed.

    This is the contract Dreamverse depends on — the GPU worker boots
    even when flashinfer is not in the venv, because the import happens
    in `video_generation.py` before any FP4 kernel is invoked.
    """
    # Hide flashinfer from sys.modules and the import path.
    monkeypatch.setitem(sys.modules, "flashinfer", None)
    # Force a re-import of the target module.
    sys.modules.pop("fastvideo.layers.quantization.fp4_config", None)
    from fastvideo.layers.quantization.fp4_config import FP4Config
    config = FP4Config()
    assert config.get_name() == "fp4"
    assert config.layer_profile == "refine"


def test_fp4config_layer_profile_round_trips_from_dict():
    from fastvideo.layers.quantization.fp4_config import FP4Config
    config = FP4Config.from_config({"layer_profile": "base"})
    assert config.layer_profile == "base"
    config = FP4Config.from_config({})
    assert config.layer_profile == "refine"


def test_fp4_kernel_call_raises_clear_error_without_flashinfer(monkeypatch):
    """A call into the FP4 kernels must raise an actionable ImportError
    when flashinfer is missing, not a confusing AttributeError or
    NameError."""
    # Stage a fake flashinfer that fails on import.
    monkeypatch.setitem(sys.modules, "flashinfer",
                        _raise_module_on_import("flashinfer"))
    sys.modules.pop("fastvideo.layers.quantization.fp4_config", None)
    from fastvideo.layers.quantization.fp4_config import _require_flashinfer
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
