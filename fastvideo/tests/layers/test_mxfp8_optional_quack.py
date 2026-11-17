# SPDX-License-Identifier: Apache-2.0
"""The in-tree MXFP8 module must import without Quack.

``fastvideo.layers.mxfp8linear`` used to import ``to_blocked``/``to_mx`` from
``quack.mx_utils`` at module scope. Quack is optional (and newer releases moved
those helpers to ``quack.blockscaled.quantize``), so a missing or too-old Quack
made the whole module -- and everything that imports the MXFP8 layer set --
unimportable on an otherwise supported Blackwell image. Weight
prequantization uses Quack when it is installed and otherwise falls back to the
in-tree Triton quantizer (the DGX Spark image has no Quack).
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest
import torch

_MODULE_NAME = "fastvideo.layers.mxfp8linear"


def _import_blocking_quack(monkeypatch: pytest.MonkeyPatch):
    """Import the MXFP8 module while every ``quack`` import raises ImportError."""
    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "quack" or name.startswith("quack."):
            raise ImportError(f"No module named {name!r} (blocked by the test)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)
    monkeypatch.delitem(sys.modules, _MODULE_NAME, raising=False)
    try:
        return importlib.import_module(_MODULE_NAME)
    finally:
        monkeypatch.delitem(sys.modules, _MODULE_NAME, raising=False)


def test_mxfp8_module_imports_without_quack(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_blocking_quack(monkeypatch)
    assert module.to_mx is None
    assert module.to_blocked is None
    assert callable(module.quantize_mxfp8_blockwise)
    assert callable(module.swiglu_quantize_mxfp8_blockwise)
    assert callable(module.mxfp8_scaled_mm)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton MXFP8 quantizer needs a CUDA device")
def test_weight_quantization_without_quack_uses_the_triton_quantizer(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_blocking_quack(monkeypatch)
    weight = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
    values, scales = module.quantize_mxfp8_weight_blockwise(weight)
    expected_values, expected_scales = module.quantize_mxfp8_blockwise(weight)
    assert torch.equal(values.view(torch.uint8), expected_values.view(torch.uint8))
    assert torch.equal(scales.view(torch.uint8), expected_scales.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Quack's MXFP8 helpers compile for CUDA tensors")
def test_triton_weight_quantizer_matches_quack_bytes() -> None:
    pytest.importorskip("quack")
    module = importlib.import_module(_MODULE_NAME)
    if module.to_mx is None:
        pytest.skip("the installed quack has neither mx_utils nor blockscaled.quantize")
    weight = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
    quack_values, quack_scales = module.quantize_mxfp8_weight_blockwise(weight)
    triton_values, triton_scales = module.quantize_mxfp8_blockwise(weight)
    assert torch.equal(quack_values.view(torch.uint8), triton_values.view(torch.uint8))
    assert torch.equal(quack_scales.view(torch.uint8), triton_scales.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Quack's MXFP8 helpers compile for CUDA tensors")
def test_weight_quantization_with_quack_matches_blocked_layout() -> None:
    pytest.importorskip("quack")
    module = importlib.import_module(_MODULE_NAME)
    if module.to_mx is None:
        pytest.skip("the installed quack has neither mx_utils nor blockscaled.quantize")
    values, scales = module.quantize_mxfp8_weight_blockwise(torch.randn(64, 128, dtype=torch.bfloat16, device="cuda"))
    assert values.shape == (64, 128)
    assert values.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.float8_e8m0fnu
    assert scales.numel() % 512 == 0
