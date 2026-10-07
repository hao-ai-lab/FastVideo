# SPDX-License-Identifier: Apache-2.0
"""Tile-first VSA parity on CUDA, including partial tiles and learned gates."""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import fastvideo.envs as envs

from fastvideo.layers.quantization.fp8_config import FP8Config, FP8QuantizeMethod
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.models.dits.minimax_h3_vsa_fp4 import _shared_input_projections


def _install_fp8_buffers(layer):
    weight = layer.weight.data.float()
    if layer.quant_method.granularity == "channel":
        scale = (weight.abs().amax(dim=1, keepdim=True) / 448).clamp_min(1e-6)
    else:
        scale = (weight.abs().amax().reshape(1) / 448).clamp_min(1e-6)
    layer.register_buffer("_fp8_weight", (weight / scale).to(torch.float8_e4m3fn))
    layer.register_buffer("_fp8_weight_scale", scale)
    layer.register_parameter("weight", None)


@pytest.mark.parametrize("granularity", ["tensor", "channel"])
def test_shared_fp8_projections_match_independent_quantization(granularity):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("sm89+ CUDA is required for FP8 GEMM")
    from fastvideo.layers.linear import ReplicatedLinear

    torch.manual_seed(17)
    layers = tuple(ReplicatedLinear(128, 256, bias=True, quant_config=FP8Config(granularity),
                                    prefix=f"block.attn.to_{name}") for name in ("q", "k", "v"))
    for layer in layers:
        layer.to(device="cuda", dtype=torch.bfloat16)
        layer.weight.data.normal_(std=0.1)
        layer.bias.data.normal_(std=0.1)
        _install_fp8_buffers(layer)
    x = torch.randn(1, 272, 128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        reference = [layer(x)[0] for layer in layers]
        with patch.object(FP8QuantizeMethod, "quantize_input", autospec=True,
                          side_effect=FP8QuantizeMethod.quantize_input) as quant:
            actual = _shared_input_projections(layers, x)
        assert quant.call_count == 1
        for expected, output in zip(reference, actual, strict=True):
            torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.parametrize("kernel", ["original", "bf16", "int8"])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("gate_active", [False, True])
@pytest.mark.parametrize("fused_rope", [False, True])
def test_tile_first_matches_generic_vsa_with_partial_tiles(env_overrides, distributed_setup, tmp_path,
                                                          fp8, gate_active, fused_rope, kernel):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 CUDA is required")
    if fp8 and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("sm89+ CUDA is required for FP8 GEMM")
    from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAMetadataBuilder
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.dits.minimax_h3 import MiniMaxH3Attention

    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override("VIDEO_SPARSE_ATTN_H3"))
    env_overrides.enter_context(envs.override_external("FASTVIDEO_VSA_TRITON", "1"))
    env_overrides.enter_context(envs.FASTVIDEO_VSA_SM100A.override(False))
    env_overrides.enter_context(envs.FASTVIDEO_H3_VSA_FP4.override(False))
    env_overrides.enter_context(envs.FASTVIDEO_H3_VSA_TILE_FIRST.override(False))
    env_overrides.enter_context(envs.FASTVIDEO_H3_VSA_SM89_KERNEL.override("original"))
    capture = kernel == "int8" and fp8 and gate_active and fused_rope
    if capture:
        env_overrides.enter_context(envs.FASTVIDEO_H3_CAPTURE_QKV.override(str(tmp_path)))
    torch.manual_seed(21)
    attn = MiniMaxH3Attention(256, 2, 128, 1e-5, (AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3,), FP8Config("channel") if fp8 else None,
                             "transformer_blocks.0.attn", fuse_qknorm_rope=fused_rope)
    attn.to(device="cuda", dtype=torch.bfloat16)
    for parameter in attn.parameters():
        parameter.data.normal_(std=0.1)
    if not gate_active:
        attn.to_gate_compress.weight.data.zero_()
    if fp8:
        for layer in (attn.to_q, attn.to_k, attn.to_v, attn.to_out):
            _install_fp8_buffers(layer)
    meta = MiniMaxH3VSAMetadataBuilder().build(current_timestep=999, patch_size=(1, 1, 1),
                                               VSA_sparsity=0.8, packed_segments=(65, 97, (4, 6, 10)),
                                               device=torch.device("cuda"), tile_size=64)
    length = meta.total_seq_length
    x = torch.randn(1, length, 256, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(length, 96, device="cuda")
    rope = angles.cos(), angles.sin()
    with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=meta):
        reference = attn(x, rope, length)
        attn._vsa_tile_first = True
        attn.distributed_attention.attn_impl._sm89_kernel = kernel
        actual = attn(x, rope, length)
    # Row order can choose a different GEMM reduction; neither attention nor
    # the VSA selection/padding semantics are approximated by this route.
    error = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert error < (0.02 if fp8 else 0.005), float(error)
    torch.testing.assert_close(actual, reference, rtol=0.03, atol=0.05)
    if capture:
        data = torch.load(tmp_path / "layer-0.pt", weights_only=True)
        torch.testing.assert_close(data["vbs"], meta.variable_block_sizes.cpu(), rtol=0, atol=0)
        assert data["q"].shape == (1, 2, meta.variable_block_sizes.numel() * 64, 128)
        assert data["mask"].dtype == torch.bool
