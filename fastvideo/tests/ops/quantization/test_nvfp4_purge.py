# SPDX-License-Identifier: Apache-2.0
"""CPU construction-level tests for the NVFP4 original-weight purge.

``convert_model_to_nvfp4`` keeps the packed FP4 buffers and, by default
(auto), purges the bf16 ``layer.weight`` of always-FP4 layers while
retaining refine-only layers (which still take the stage-profile dense
path). ``retain_original_weights`` on ``NVFP4Config`` overrides in both
directions. flashinfer and the FP4 GEMM are monkeypatched, so this runs
on any host.
"""
from __future__ import annotations

import logging
import types

import pytest
import torch
import torch.nn as nn

import fastvideo.layers.quantization.nvfp4_config as nv

_ALWAYS_FP4_PREFIX = "ltx2.blocks.0.attn1.to_q"
_REFINE_ONLY_PREFIX = "ltx2.blocks.0.audio_to_video_attn.to_q"


def _method(prefix: str, retain: bool | None) -> nv.NVFP4QuantizeMethod:
    # NVFP4QuantizeMethod.__init__ allocates x_global_sf on cuda; build the
    # object without it so the purge tests run on CPU-only hosts.
    m = object.__new__(nv.NVFP4QuantizeMethod)
    m.weight_fp4 = None
    m.weight_scale = None
    m.x_global_sf = torch.tensor(1.0, dtype=torch.float32)
    m.layer_prefix = prefix
    m._is_refine_only_layer = nv._is_ltx2_refine_only_prefix(prefix)
    m._retain_original_weights = retain
    return m


class _FakeLinear(nn.Module):

    def __init__(self, prefix: str, retain: bool | None, out_dim: int = 8, in_dim: int = 16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim, dtype=torch.bfloat16), requires_grad=False)
        self.quant_method = _method(prefix, retain)


def _fake_quantize(weight, global_sf, sfLayout=None, do_shuffle=False):
    out_dim, in_dim = weight.shape[0], weight.shape[-1]
    return (torch.zeros(out_dim, in_dim // 2, dtype=torch.int8), torch.zeros(out_dim, in_dim // 16,
                                                                             dtype=torch.uint8))


@pytest.fixture(autouse=True)
def _patch_flashinfer(monkeypatch):
    fake_sf_layout = types.SimpleNamespace(layout_128x4=None)
    monkeypatch.setattr(nv, "_require_flashinfer", lambda: (fake_sf_layout, None, None))
    monkeypatch.setattr(nv, "_nvfp4_quantize", _fake_quantize)


def _model(retain: bool | None) -> nn.Module:
    root = nn.Module()
    root.always_fp4 = _FakeLinear(_ALWAYS_FP4_PREFIX, retain)
    root.refine_only = _FakeLinear(_REFINE_ONLY_PREFIX, retain)
    return root


def test_prefix_classification_sanity() -> None:
    assert not nv._is_ltx2_refine_only_prefix(_ALWAYS_FP4_PREFIX)
    assert nv._is_ltx2_refine_only_prefix(_REFINE_ONLY_PREFIX)


def test_auto_purges_always_fp4_and_retains_refine_only(caplog) -> None:
    model = _model(retain=None)
    with caplog.at_level(logging.INFO):
        nv.convert_model_to_nvfp4(model)
    assert model.always_fp4.weight is None
    assert model.refine_only.weight is not None
    assert model.always_fp4._nvfp4_weight is not None
    receipt = [r.message for r in caplog.records if "NVFP4 weight purge receipt" in r.message]
    assert receipt and "purged 1" in receipt[0] and "retained 1" in receipt[0]


def test_retain_true_keeps_everything() -> None:
    model = _model(retain=True)
    nv.convert_model_to_nvfp4(model)
    assert model.always_fp4.weight is not None
    assert model.refine_only.weight is not None


def test_retain_false_still_retains_dense_capable_layers() -> None:
    """Refine-only layers run dense under the base stage profile by
    deployment contract (single-stage deploys included), so no flag value
    may purge them."""
    model = _model(retain=False)
    nv.convert_model_to_nvfp4(model)
    assert model.always_fp4.weight is None
    assert model.refine_only.weight is not None


def test_apply_out_dim_survives_purge(monkeypatch) -> None:
    model = _model(retain=False)
    nv.convert_model_to_nvfp4(model)
    layer = model.always_fp4
    captured = {}

    def fake_mm_fp4(x_fp4, w_t, x_scale, w_scale_t, alpha, out_dtype, out, backend):
        captured["out_dim"] = w_t.shape[-1] if w_t.dim() == 2 else None
        return torch.zeros(x_fp4.shape[0], w_t.shape[-1], dtype=torch.bfloat16)

    monkeypatch.setattr(nv, "_mm_fp4", fake_mm_fp4)
    monkeypatch.setattr(nv, "_get_ltx2_fp4_stage_profile", lambda default="refine": "refine")
    layer._weight_global_sf = torch.tensor(1.0, dtype=torch.bfloat16)
    x = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    out = layer.quant_method.apply(layer, x)
    assert out.shape == (2, 3, 8)


def test_dense_path_after_purge_raises_with_flag_named(monkeypatch) -> None:
    """Defensive guard: convert never purges dense-capable layers anymore,
    but a hand-purged module hitting the dense path must fail loudly."""
    model = _model(retain=False)
    nv.convert_model_to_nvfp4(model)
    model.refine_only.register_parameter("weight", None)  # simulate misuse
    monkeypatch.setattr(nv, "_get_ltx2_fp4_stage_profile", lambda default="refine": "base")
    x = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="retain_original_weights"):
        model.refine_only.quant_method.apply(model.refine_only, x)
