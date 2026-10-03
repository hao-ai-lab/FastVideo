# SPDX-License-Identifier: Apache-2.0
"""Exact parity of shared VAE input preparation and strided INT8 weight GEMMs."""
from unittest.mock import patch

import pytest
import torch

from fastvideo.models.vaes.minimax_h3_int8_convrot import Int8ConvRotLinear, shared_int8_projections


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for INT8 GEMM")
@pytest.mark.parametrize("rows", [3, 17, 129])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("convrot", [False, True])
def test_shared_int8_and_transpose_views_are_exact(rows, dtype, convrot, monkeypatch):
    monkeypatch.setenv("FASTVIDEO_H3_VAE_INT8_TRANSPOSE_VIEW", "0")
    torch.manual_seed(73)
    layers = tuple(Int8ConvRotLinear(256, out, bias=index != 1, convrot=convrot, group_size=256)
                   .to("cuda") for index, out in enumerate([128, 256, 64]))
    for layer in layers:
        layer.weight.random_(-127, 128)
        layer.weight_scale.uniform_(0.0001, 0.03)
        if layer.bias is not None:
            layer.bias.data.normal_()
    x = torch.randn(1, rows, 256, device="cuda", dtype=dtype)
    x[0, 0].zero_()  # clamp/padding semantics must also survive sharing
    with torch.inference_mode():
        expected = tuple(layer(x) for layer in layers)
        with patch.object(Int8ConvRotLinear, "quantize_input", autospec=True,
                          side_effect=Int8ConvRotLinear.quantize_input) as quant:
            shared = shared_int8_projections(layers, x)
        assert quant.call_count == 1
        for layer in layers:
            layer._transpose_view = True
        views = shared_int8_projections(layers, x)
    for ref, actual, view in zip(expected, shared, views, strict=True):
        assert torch.isfinite(ref).all()
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
        torch.testing.assert_close(view, ref, rtol=0, atol=0)


def test_shared_int8_keeps_cpu_fallback_exact():
    layers = tuple(Int8ConvRotLinear(16, 8, bias=False, convrot=False, group_size=16) for _ in range(3))
    for layer in layers:
        layer.weight.fill_(1)
        layer.weight_scale.fill_(0.01)
    x = torch.ones(3, 16)
    for actual, ref in zip(shared_int8_projections(layers, x), (layer(x) for layer in layers), strict=True):
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for VAE attention parity")
def test_vae_attention_shares_quantized_projections_exactly(distributed_setup, monkeypatch):
    from fastvideo.models.vaes.minimax_h3_video import MiniMaxH3VideoAttention

    monkeypatch.setenv("FASTVIDEO_H3_VAE_INT8_SHARED_QKV", "0")
    monkeypatch.setenv("FASTVIDEO_H3_VAE_INT8_TRANSPOSE_VIEW", "0")
    torch.manual_seed(49)
    attention = MiniMaxH3VideoAttention(256, 2, 128).to("cuda").eval()
    for name in ("to_q", "to_k", "to_v"):
        layer = Int8ConvRotLinear(256, 256, bias=True, convrot=True, group_size=256).to("cuda")
        layer.weight.random_(-8, 9)
        layer.weight_scale.fill_(0.01)
        layer.bias.data.normal_(std=0.1)
        setattr(attention, name, layer)
    x = torch.randn(2, 33, 256, device="cuda")
    with torch.inference_mode():
        expected = attention(x)
        attention._share_int8_qkv = True
        for layer in (attention.to_q, attention.to_k, attention.to_v):
            layer._transpose_view = True
        actual = attention(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
