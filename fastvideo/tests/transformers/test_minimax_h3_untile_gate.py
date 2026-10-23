# SPDX-License-Identifier: Apache-2.0
"""CUDA parity for H3's packed-order compression gate and resident AdaLN cache."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU and Triton")


@pytest.mark.parametrize("batch,rows,heads", [(1, 127, 4), (2, 257, 6), (1, 14859, 56)])
def test_untile_gate_matches_aten_addcmul(batch, rows, heads):
    from fastvideo.models.dits.minimax_h3_fusions.untile_gate import untile_gate

    torch.manual_seed(751)
    tiled_rows = ((rows + 127) // 128) * 128
    out = torch.randn((batch, heads, tiled_rows, 128), device="cuda", dtype=torch.bfloat16)
    compressed = torch.randn((batch, heads, tiled_rows // 64, 128), device="cuda", dtype=out.dtype)
    compressed = compressed.permute(0, 2, 1, 3)
    gate = torch.randn((batch, rows, heads * 128), device="cuda", dtype=out.dtype)
    untile = torch.randperm(tiled_rows, device="cuda")[:rows]
    row_tile = untile // 64
    expected = out.transpose(1, 2).index_select(1, untile)
    expected.addcmul_(compressed.index_select(1, row_tile), gate.unflatten(-1, (heads, 128)))
    actual = untile_gate(out, compressed, gate, untile, row_tile)
    assert torch.equal(actual, expected.flatten(2, 3))


def test_resident_adaln_cache_preserves_projection_and_allocation():
    from fastvideo.models.dits.minimax_h3 import MiniMaxH3AdaLayerNormModulation

    torch.manual_seed(319)
    mod = MiniMaxH3AdaLayerNormModulation(16, 12).to(device="cuda", dtype=torch.bfloat16)
    # ReplicatedLinear leaves initialization to the loader.
    with torch.no_grad():
        mod.linear.weight.normal_()
        mod.linear.bias.normal_()
    inputs = [torch.randn((3, 16), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    with torch.inference_mode():
        expected = [mod(x) for x in inputs]
        weight_ptr = mod.linear.weight.data_ptr()
        mod.enable_host_cache(keep_resident=True)
        assert mod._host_weight.data_ptr() == weight_ptr
        for _ in range(2):
            for key, x in enumerate(inputs):
                mod._cache_key = key
                assert all(torch.equal(a, b) for a, b in zip(mod(x), expected[key], strict=True))
        assert len(mod._modulation_cache) == 2


@pytest.mark.parametrize("prefix_rows", [64, 128, 192])
def test_split_untile_gate_matches_concatenation(prefix_rows):
    from fastvideo.models.dits.minimax_h3_fusions.untile_gate import untile_gate

    torch.manual_seed(457)
    batch, heads, rows, tiled_rows = 2, 3, 319, 512
    start = prefix_rows // 128 * 128
    prefix = torch.randn((batch, heads, prefix_rows, 128), device="cuda", dtype=torch.bfloat16)
    video = torch.randn((batch, heads, tiled_rows - start, 128), device="cuda", dtype=torch.bfloat16)
    compressed = torch.randn((batch, heads, tiled_rows // 64, 128), device="cuda", dtype=torch.bfloat16)
    compressed = compressed.permute(0, 2, 1, 3)
    gate = torch.randn((batch, rows, heads * 128), device="cuda", dtype=torch.bfloat16)
    index = torch.randperm(tiled_rows, device="cuda")[:rows]
    combined = torch.cat((prefix, video[:, :, prefix_rows - start:]), dim=2)
    expected = combined.transpose(1, 2).index_select(1, index)
    expected.addcmul_(compressed.index_select(1, index // 64), gate.unflatten(-1, (heads, 128)))
    actual = untile_gate((prefix, video, start), compressed, gate, index, index // 64)
    assert torch.equal(actual, expected.flatten(2, 3))


@pytest.mark.parametrize("width,scale", [(64, 1.0), (5376, 0.7079)])
def test_quantized_tile_matches_flashinfer_quantization(width, scale):
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 quantization requires Blackwell")
    from fastvideo.layers.quantization.nvfp4_config import _nvfp4_quantize, _require_flashinfer
    from fastvideo.models.dits.minimax_h3_fusions.quant_tile import tile_quantized

    torch.manual_seed(319)
    layout, _, _ = _require_flashinfer()
    x = torch.randn((129, width), device="cuda", dtype=torch.bfloat16)
    global_sf = torch.tensor(scale, device="cuda")
    index = torch.randperm(384, device="cuda")[:129]
    inverse = torch.full((384,), -1, device="cuda", dtype=torch.int64)
    inverse[index] = torch.arange(129, device="cuda")
    tiled = x.new_zeros((384, width)).index_copy_(0, index, x)
    expected = _nvfp4_quantize(tiled, global_sf, sfLayout=layout.layout_128x4)
    codes, scales = _nvfp4_quantize(x, global_sf, sfLayout=layout.layout_128x4)
    actual = tile_quantized(codes, scales, inverse)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
