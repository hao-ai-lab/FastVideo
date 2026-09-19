# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the MiniMax-H3 Comfy int8-convrot VAE overlay."""

from __future__ import annotations

import json

import torch
import torch.nn as nn
from safetensors.torch import save_file

from fastvideo.models.vaes.minimax_h3_int8_convrot import (
    Int8ConvRotLinear,
    _int8_linear_from_tensors,
    dense_vae_safetensors,
    overlay_minimax_h3_int8_convrot_decoder,
    parse_comfy_quant_marker,
    regular_hadamard,
    rotate_activation,
)


def test_regular_hadamard_is_orthogonal() -> None:
    hadamard = regular_hadamard(16, device=torch.device("cpu"), dtype=torch.float64)
    identity = torch.eye(16, dtype=torch.float64)
    torch.testing.assert_close(hadamard @ hadamard.T, identity, atol=1e-12, rtol=0.0)


def test_convrot_rotation_is_an_involution() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 256)
    rotated = rotate_activation(x, 256)
    twice = rotate_activation(rotated, 256)
    torch.testing.assert_close(twice, x, atol=1e-5, rtol=1e-5)


def test_int8_gemm_scales_in_float32_not_fp16() -> None:
    acc = torch.tensor([[100_000]], dtype=torch.int32)
    x_scale = torch.tensor([[1.0 / 127.0]], dtype=torch.float16)
    weight_scale = torch.tensor([[1.0 / 127.0]], dtype=torch.float16)
    overflowed = acc.to(torch.float16) * x_scale * weight_scale.t()
    scaled = Int8ConvRotLinear._dequant_int8_gemm(acc, x_scale, weight_scale)
    assert torch.isinf(overflowed).all()
    assert torch.isfinite(scaled).all()
    torch.testing.assert_close(scaled, acc.float() * x_scale.float() * weight_scale.t().float())


def test_int8_linear_rejects_incompatible_convrot_group() -> None:
    marker = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    weight = torch.ones(8, 8, dtype=torch.int8)
    scale = torch.ones(8, 1)
    try:
        _int8_linear_from_tensors(weight, scale, None, marker)
    except ValueError as error:
        assert "group_size 256" in str(error)
        assert "in_features 8" in str(error)
    else:
        raise AssertionError("incompatible ConvRot overlay must be rejected")


def test_int8_linear_matches_dequantized_matmul() -> None:
    torch.manual_seed(1)
    layer = Int8ConvRotLinear(256, 32, bias=True, convrot=True, group_size=256)
    layer.weight.copy_(torch.randint(-8, 8, (32, 256), dtype=torch.int8))
    layer.weight_scale.copy_(torch.linspace(0.01, 0.02, 32).unsqueeze(1))
    layer.bias.data.copy_(torch.randn(32))
    x = torch.randn(3, 256)
    out = layer(x)
    rotated = rotate_activation(x, 256)
    expected = torch.nn.functional.linear(rotated, layer.weight.float() * layer.weight_scale, layer.bias)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_dense_vae_safetensors_drops_convrot_overlay() -> None:
    kept = dense_vae_safetensors([
        "/tmp/vae/diffusion_pytorch_model-00001-of-00003.safetensors",
        "/tmp/vae/minimax_h3_video_vae_int8_convrot.safetensors",
        "/tmp/vae/other.safetensors",
    ])
    assert kept == [
        "/tmp/vae/diffusion_pytorch_model-00001-of-00003.safetensors",
        "/tmp/vae/other.safetensors",
    ]


def test_parse_comfy_quant_marker_reads_padded_uint8() -> None:
    payload = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode()
    blob = torch.zeros(72, dtype=torch.uint8)
    blob[:len(payload)] = torch.tensor(list(payload), dtype=torch.uint8)
    marker = parse_comfy_quant_marker(blob)
    assert marker["format"] == "int8_tensorwise"
    assert marker["convrot"] is True
    assert marker["convrot_groupsize"] == 256


class _FakeH3VAE(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        decoder = nn.Module()
        decoder.proj_in = nn.Linear(24, 8, bias=True)
        decoder.register_tokens = nn.Parameter(torch.zeros(1, 4, 8), requires_grad=False)
        decoder.norm_out = nn.LayerNorm(8)
        decoder.proj_out = nn.Linear(8, 12, bias=True)
        block = nn.Module()
        block.norm1 = nn.RMSNorm(8, elementwise_affine=True)
        block.norm2 = nn.RMSNorm(8, elementwise_affine=True)
        block.scale1 = nn.Parameter(torch.zeros(8), requires_grad=False)
        block.scale2 = nn.Parameter(torch.zeros(8), requires_grad=False)
        attn = nn.Module()
        attn.heads = 2
        attn.dim_head = 4
        attn.to_q = nn.Linear(8, 8, bias=True)
        attn.to_k = nn.Linear(8, 8, bias=True)
        attn.to_v = nn.Linear(8, 8, bias=True)
        attn.to_out = nn.ModuleList([nn.Linear(8, 8, bias=True)])
        ff = nn.Module()
        swiglu = nn.Module()
        swiglu.proj = nn.Linear(8, 16, bias=True)
        ff.net = nn.ModuleList([swiglu, nn.Dropout(0.0), nn.Linear(8, 8, bias=True)])
        block.attn = attn
        block.ff = ff
        decoder.transformer_blocks = nn.ModuleList([block])
        self.decoder = decoder


def _marker_tensor() -> torch.Tensor:
    payload = json.dumps({"format": "int8_tensorwise", "convrot": False}).encode()
    blob = torch.zeros(64, dtype=torch.uint8)
    blob[:len(payload)] = torch.tensor(list(payload), dtype=torch.uint8)
    return blob


def test_overlay_splits_fused_qkv_and_ffn(tmp_path) -> None:
    tensors = {
        "decoder.x_embedder.weight": torch.ones(8, 24),
        "decoder.x_embedder.bias": torch.zeros(8),
        "decoder.register_tokens": torch.ones(1, 4, 8),
        "decoder.norm_out.weight": torch.ones(8),
        "decoder.norm_out.bias": torch.zeros(8),
        "decoder.proj_out.weight": torch.ones(12, 8),
        "decoder.proj_out.bias": torch.zeros(12),
        "decoder.transformer_blocks.0.attn.to_qkv.weight": torch.arange(24 * 8, dtype=torch.int8).reshape(24, 8),
        "decoder.transformer_blocks.0.attn.to_qkv.weight_scale": torch.ones(24, 1),
        "decoder.transformer_blocks.0.attn.to_qkv.bias": torch.zeros(24),
        "decoder.transformer_blocks.0.attn.to_qkv.comfy_quant": _marker_tensor(),
        "decoder.transformer_blocks.0.attn.to_out.weight": torch.ones(8, 8, dtype=torch.int8),
        "decoder.transformer_blocks.0.attn.to_out.weight_scale": torch.ones(8, 1),
        "decoder.transformer_blocks.0.attn.to_out.bias": torch.zeros(8),
        "decoder.transformer_blocks.0.attn.to_out.comfy_quant": _marker_tensor(),
        "decoder.transformer_blocks.0.ff.w1.weight": torch.cat(
            [torch.ones(8, 8, dtype=torch.int8), torch.full((8, 8), 2, dtype=torch.int8)], dim=0),
        "decoder.transformer_blocks.0.ff.w1.weight_scale": torch.cat(
            [torch.ones(8, 1), torch.full((8, 1), 0.5)], dim=0),
        "decoder.transformer_blocks.0.ff.w1.bias": torch.cat([torch.ones(8), torch.full((8, ), 3.0)], dim=0),
        "decoder.transformer_blocks.0.ff.w1.comfy_quant": _marker_tensor(),
        "decoder.transformer_blocks.0.ff.w2.weight": torch.ones(8, 8, dtype=torch.int8),
        "decoder.transformer_blocks.0.ff.w2.weight_scale": torch.ones(8, 1),
        "decoder.transformer_blocks.0.ff.w2.bias": torch.zeros(8),
        "decoder.transformer_blocks.0.ff.w2.comfy_quant": _marker_tensor(),
        "decoder.transformer_blocks.0.norm1.weight": torch.ones(8),
        "decoder.transformer_blocks.0.norm2.weight": torch.ones(8),
        "decoder.transformer_blocks.0.scale1": torch.ones(8),
        "decoder.transformer_blocks.0.scale2": torch.ones(8),
    }
    path = tmp_path / "minimax_h3_video_vae_int8_convrot.safetensors"
    save_file(tensors, path)
    vae = _FakeH3VAE()
    installed = overlay_minimax_h3_int8_convrot_decoder(vae, path)
    assert installed == 6
    q = vae.decoder.transformer_blocks[0].attn.to_q
    k = vae.decoder.transformer_blocks[0].attn.to_k
    v = vae.decoder.transformer_blocks[0].attn.to_v
    fused = tensors["decoder.transformer_blocks.0.attn.to_qkv.weight"].view(2, 3, 4, 8)
    assert isinstance(q, Int8ConvRotLinear)
    assert q.weight.shape == (8, 8)
    assert torch.equal(q.weight, fused[:, 0].reshape(8, 8))
    assert torch.equal(k.weight, fused[:, 1].reshape(8, 8))
    assert torch.equal(v.weight, fused[:, 2].reshape(8, 8))
    proj = vae.decoder.transformer_blocks[0].ff.net[0].proj
    assert isinstance(proj, Int8ConvRotLinear)
    assert isinstance(vae.decoder.transformer_blocks[0].ff.net[2], Int8ConvRotLinear)
    assert torch.equal(proj.weight[:8], torch.full((8, 8), 2, dtype=torch.int8))
    assert torch.equal(proj.weight[8:], torch.ones(8, 8, dtype=torch.int8))
    assert torch.equal(proj.weight_scale[:8], torch.full((8, 1), 0.5))
    assert torch.equal(proj.bias[:8], torch.full((8, ), 3.0))
    assert not hasattr(vae.decoder, "mask_token")
    assert torch.equal(vae.decoder.proj_in.weight, torch.ones(8, 24))
