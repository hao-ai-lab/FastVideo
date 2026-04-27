# SPDX-License-Identifier: Apache-2.0
"""NVFP4Config import + lazy-flashinfer behavior tests.

The actual NVFP4 kernels need flashinfer + a CUDA device, so these
tests focus on the lazy-import contract: the module must load on
hosts without flashinfer, and only call sites should fail with a
clear error when flashinfer is missing.
"""
from __future__ import annotations

import sys
import types

import pytest


def test_nvfp4config_imports_without_flashinfer(monkeypatch):
    """Importing the module on a host without flashinfer must succeed.

    This is the contract Dreamverse depends on — the GPU worker boots
    even when flashinfer is not in the venv, because the import happens
    in `video_generation.py` before any NVFP4 kernel is invoked.
    """
    # Hide flashinfer from sys.modules and the import path.
    monkeypatch.setitem(sys.modules, "flashinfer", None)
    # Force a re-import of the target module.
    sys.modules.pop("fastvideo.layers.quantization.nvfp4_config", None)
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
    monkeypatch.setitem(sys.modules, "flashinfer",
                        _raise_module_on_import("flashinfer"))
    sys.modules.pop("fastvideo.layers.quantization.nvfp4_config", None)
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
