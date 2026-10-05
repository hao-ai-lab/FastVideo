# SPDX-License-Identifier: Apache-2.0
"""Exact parity of shared VAE input preparation and strided INT8 weight GEMMs."""
from unittest.mock import patch

import pytest
import torch

import fastvideo.envs as envs

from fastvideo.models.vaes.minimax_h3_int8_convrot import Int8ConvRotLinear, shared_int8_projections


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for INT8 GEMM")
@pytest.mark.parametrize("rows", [3, 17, 129])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("convrot", [False, True])
def test_shared_int8_and_transpose_views_are_exact(rows, dtype, convrot, env_overrides):
    env_overrides.enter_context(envs.FASTVIDEO_H3_VAE_INT8_TRANSPOSE_VIEW.override(False))
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
            layer._fused_dequant = True
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
def test_vae_attention_shares_quantized_projections_exactly(distributed_setup, env_overrides):
    from fastvideo.models.vaes.minimax_h3_video import MiniMaxH3VideoAttention

    env_overrides.enter_context(envs.FASTVIDEO_H3_VAE_INT8_SHARED_QKV.override(False))
    env_overrides.enter_context(envs.FASTVIDEO_H3_VAE_INT8_TRANSPOSE_VIEW.override(False))
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
            layer._fused_dequant = True
        actual = attention(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused INT8 epilogue")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_bias", [False, True])
def test_fused_int8_epilogue_large_accumulators_and_small_scales(dtype, has_bias):
    from fastvideo.models.vaes.minimax_h3_int8_kernels import fused_int8_dequant_bias

    torch.manual_seed(19)
    acc = torch.randint(-400_000_000, 400_000_000, (129, 264), device="cuda", dtype=torch.int32)
    x_scale = torch.logspace(-30, -3, 129, device="cuda").view(-1, 1)
    w_scale = torch.logspace(-6, -2, 264, device="cuda").view(-1, 1)
    bias = torch.randn(264, device="cuda") if has_bias else None
    expected = acc.float() * x_scale.float() * w_scale.t().float()
    if bias is not None:
        expected = expected + bias.float()
    actual = fused_int8_dequant_bias(acc, x_scale, w_scale, bias, dtype)
    torch.testing.assert_close(actual, expected.to(dtype), rtol=0, atol=0)
