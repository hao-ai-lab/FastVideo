# SPDX-License-Identifier: Apache-2.0
"""Blackwell byte parity for H3 producer/FP4 quantization fusions."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="requires a Blackwell GPU with native FP4 conversion",
)


@pytest.mark.parametrize("batch,rows,heads,prefix", [(1, 129, 3, 0), (2, 257, 4, 64), (1, 383, 56, 192)])
def test_quantized_gate_matches_separate_gather_addcmul_and_quantization(batch, rows, heads, prefix):
    from fastvideo.layers.quantization.nvfp4_config import _nvfp4_quantize, _require_flashinfer
    from fastvideo.models.dits.minimax_h3_fusions.untile_gate import untile_gate

    torch.manual_seed(935)
    tiled = 512
    start = prefix // 128 * 128
    full = torch.randn((batch, heads, tiled, 128), device="cuda", dtype=torch.bfloat16)
    output = (full[:, :, :prefix], full[:, :, start:], start) if prefix else full
    compressed = torch.randn((batch, heads, tiled // 64, 128), device="cuda", dtype=full.dtype).permute(0, 2, 1, 3)
    gate = torch.randn((batch, rows, heads * 128), device="cuda", dtype=full.dtype)
    index = torch.randperm(tiled, device="cuda")[:rows]
    scale = torch.ones((), device="cuda")
    layout, _, _ = _require_flashinfer()
    with torch.inference_mode():
        expected = full.transpose(1, 2).index_select(1, index)
        expected.addcmul_(compressed.index_select(1, index // 64), gate.unflatten(-1, (heads, 128)))
        q, s = _nvfp4_quantize(expected.flatten(0, 1).flatten(1), scale, sfLayout=layout.layout_128x4)
        actual_q, actual_s, actual_scale = untile_gate(output, compressed, gate, index, index // 64, scale)
    assert torch.equal(q, actual_q)
    assert torch.equal(s, actual_s)
    assert actual_scale is scale


@pytest.mark.parametrize("batch,rows,width,global_scale", [(1, 129, 64, 1.0), (2, 257, 5376, 0.7079)])
def test_quantized_residual_modulation_matches_existing_fusion(batch, rows, width, global_scale):
    from fastvideo.layers.quantization.nvfp4_config import _nvfp4_quantize, _require_flashinfer
    from fastvideo.models.dits.minimax_h3_fusions.modulation import fused_residual_gate_rmsnorm_modulate
    from fastvideo.models.dits.minimax_h3_fusions.modulation_quant import residual_modulate_quantize_nvfp4

    torch.manual_seed(831)
    residual, branch = (torch.randn((batch, rows, width), device="cuda", dtype=torch.bfloat16) for _ in range(2))
    gate, scale, shift = (torch.randn((3, width), device="cuda", dtype=torch.bfloat16) for _ in range(3))
    weight = torch.randn((width,), device="cuda", dtype=torch.bfloat16)
    index = torch.randint(0, 3, (rows,), device="cuda")
    sf = torch.tensor(global_scale, device="cuda")
    args = residual, branch, gate, weight, scale, shift, index, 1e-5
    layout, _, _ = _require_flashinfer()
    with torch.inference_mode():
        expected_hidden, normalized = fused_residual_gate_rmsnorm_modulate(*args)
        q, s = _nvfp4_quantize(normalized.reshape(-1, width), sf, sfLayout=layout.layout_128x4)
        hidden, (actual_q, actual_s, actual_sf) = residual_modulate_quantize_nvfp4(*args, sf)
    assert torch.equal(hidden, expected_hidden)
    assert torch.equal(q, actual_q)
    assert torch.equal(s, actual_s)
    assert actual_sf is sf


@pytest.mark.parametrize("batch,rows,width", [(1, 129, 64), (2, 257, 5376)])
def test_quantized_modulation_matches_existing_fusion(batch, rows, width):
    from fastvideo.layers.quantization.nvfp4_config import _nvfp4_quantize, _require_flashinfer
    from fastvideo.models.dits.minimax_h3_fusions.modulation import fused_rmsnorm_modulate
    from fastvideo.models.dits.minimax_h3_fusions.modulation_quant import modulate_quantize_nvfp4

    torch.manual_seed(834)
    x = torch.randn((batch, rows, width), device="cuda", dtype=torch.bfloat16)
    scale, shift = (torch.randn((3, width), device="cuda", dtype=torch.bfloat16) for _ in range(2))
    weight = torch.randn((width,), device="cuda", dtype=torch.bfloat16)
    index = torch.randint(0, 3, (rows,), device="cuda")
    sf = torch.ones((), device="cuda")
    args = x, weight, scale, shift, index, 1e-5
    layout, _, _ = _require_flashinfer()
    with torch.inference_mode():
        normalized = fused_rmsnorm_modulate(*args)
        q, s = _nvfp4_quantize(normalized.reshape(-1, width), sf, sfLayout=layout.layout_128x4)
        actual_q, actual_s, actual_sf = modulate_quantize_nvfp4(*args, sf)
    assert torch.equal(q, actual_q)
    assert torch.equal(s, actual_s)
    assert actual_sf is sf


@pytest.mark.parametrize("batch,rows,width", [(1, 129, 64), (2, 257, 5376)])
def test_next_block_preparation_preserves_bf16_residual_rounding(batch, rows, width):
    from fastvideo.models.dits.minimax_h3 import _gated_residual
    from fastvideo.models.dits.minimax_h3_fusions.modulation_quant import (
        modulate_quantize_nvfp4, residual_modulate_quantize_nvfp4,
    )

    torch.manual_seed(9183)
    residual, branch = (torch.randn((batch, rows, width), device="cuda", dtype=torch.bfloat16) for _ in range(2))
    gate, scale, shift = (torch.randn((3, width), device="cuda", dtype=residual.dtype) for _ in range(3))
    weight = torch.randn(width, device="cuda", dtype=residual.dtype)
    index = torch.randint(0, 3, (rows,), device="cuda")
    global_scale = torch.ones((), device="cuda")
    with torch.inference_mode():
        expected_hidden = _gated_residual(residual, gate, index, branch)
        expected = modulate_quantize_nvfp4(expected_hidden, weight, scale, shift, index, 1e-5, global_scale)
        hidden, actual = residual_modulate_quantize_nvfp4(
            residual, branch, gate, weight, scale, shift, index, 1e-5, global_scale,
            round_residual_before_norm=True,
        )
    assert torch.equal(hidden, expected_hidden)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert actual[2] is global_scale
