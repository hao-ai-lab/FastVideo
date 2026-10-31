# SPDX-License-Identifier: Apache-2.0
"""Exact parity for FP8 Ulysses receive unpacking on a CUDA GPU with Triton."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")


@pytest.mark.parametrize("world,local_rows,heads,dim,trim", [
    (1, 17, 2, 64, 0),
    (2, 33, 3, 128, 1),
    (3, 47, 19, 128, 2),
    (2, 7430, 28, 128, 1),
])
def test_fp8_receive_tiles_match_separate_unpack_and_scatter(world, local_rows, heads, dim, trim):
    from fastvideo.models.dits.minimax_h3_fusions.sp_exchange import unpack_fp8_tiles
    from fastvideo.models.dits.minimax_h3_vsa_fp4 import _unpack_seq_fp8

    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("native FP8 conversion requires Hopper or newer")
    torch.manual_seed(178)
    sequence_length = world * local_rows - trim
    tile_rows = ((sequence_length + 127) // 128 + 1) * 128
    payload = torch.randn((world, 3, local_rows, heads, dim), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand((world, 3, local_rows, heads), device="cuda", dtype=torch.float32).clamp_min_(1e-12)
    untile = torch.randperm(tile_rows, device="cuda")[:sequence_length]
    inverse = torch.full((tile_rows,), -1, device="cuda", dtype=torch.int64)
    inverse[untile] = torch.arange(sequence_length, device="cuda")
    # Ensure special finite FP8 values and distinct source-rank boundaries are covered.
    payload[:, :, 0, 0, :4] = torch.tensor([0.0, -0.0, 448.0, -448.0], device="cuda").to(payload.dtype)
    expected = torch.zeros((3, tile_rows, heads, dim), device="cuda", dtype=torch.bfloat16)
    expected.index_copy_(1, untile, _unpack_seq_fp8(payload, scales, sequence_length))
    output = torch.full_like(expected, float("nan"))
    actual = unpack_fp8_tiles(payload, scales, inverse, output)
    assert actual.data_ptr() == output.data_ptr()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    # Reuse must overwrite old values as well as all padding rows.
    scales.mul_(0.25)
    expected.zero_().index_copy_(1, untile, _unpack_seq_fp8(payload, scales, sequence_length))
    unpack_fp8_tiles(payload, scales, inverse, output)
    assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("world,rows,heads,dim,tiles", [
    (1, 17, 2, 64, 4),
    (2, 33, 3, 128, 7),
    (3, 47, 19, 128, 11),
    (2, 7430, 28, 128, 256),
])
@pytest.mark.parametrize("peer_rank", [None, 0, 1])
def test_fp8_output_gate_matches_separate_unpack_and_compiled_gate(world, rows, heads, dim, tiles, peer_rank):
    if peer_rank is not None and world != 2:
        pytest.skip("direct-local payloads require two ranks")
    from fastvideo.models.dits.minimax_h3_fusions.sp_exchange import unpack_fp8_gate
    from fastvideo.models.dits.minimax_h3_vsa_fp4 import _apply_gate, _unpack_heads_fp8

    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("native FP8 conversion requires Hopper or newer")
    torch.manual_seed(735)
    payload = torch.randn((world, rows, heads, dim), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand((world, rows, heads), device="cuda").clamp_min_(1e-12)
    compressed = torch.randn((world, heads, tiles, dim), device="cuda", dtype=torch.bfloat16)
    gate = torch.randn((rows, world * heads, dim), device="cuda", dtype=torch.bfloat16)
    row_tile = torch.randint(tiles, (rows,), device="cuda")
    payload[:, 0, 0, :4] = torch.tensor([0.0, -0.0, 448.0, -448.0], device="cuda").to(payload.dtype)
    expected = _apply_gate(_unpack_heads_fp8(payload, scales), compressed.flatten(0, 1).transpose(0, 1),
                           row_tile, gate)
    received = payload
    if peer_rank is not None:
        from fastvideo.distributed.async_collectives import PeerSlices
        received = PeerSlices(payload[peer_rank], payload[1 - peer_rank].clone(), torch.empty_like(payload), peer_rank)
    actual = unpack_fp8_gate(received, scales, compressed, gate, row_tile)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("world,rows,heads,dim,global_scale", [
    (1, 17, 2, 64, 1.0),
    (2, 129, 3, 128, 0.7079),
    (3, 257, 19, 128, 1.0),
    (2, 7430, 28, 128, 0.7079),
])
def test_fp8_output_gate_quantization_matches_separate_fp4(world, rows, heads, dim, global_scale):
    from fastvideo.layers.quantization.nvfp4_config import _nvfp4_quantize, _require_flashinfer
    from fastvideo.models.dits.minimax_h3_fusions.sp_exchange import unpack_fp8_gate
    from fastvideo.models.dits.minimax_h3_vsa_fp4 import _apply_gate, _unpack_heads_fp8

    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("native FP4 conversion requires Blackwell")
    torch.manual_seed(156)
    tiles = 256
    payload = torch.randn((world, rows, heads, dim), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand((world, rows, heads), device="cuda").clamp_min_(1e-12)
    compressed = torch.randn((world, heads, tiles, dim), device="cuda", dtype=torch.bfloat16)
    gate = torch.randn((rows, world * heads, dim), device="cuda", dtype=torch.bfloat16)
    row_tile = torch.randint(tiles, (rows,), device="cuda")
    sf = torch.tensor(global_scale, device="cuda")
    layout, _, _ = _require_flashinfer()
    with torch.inference_mode():
        expected = _apply_gate(_unpack_heads_fp8(payload, scales), compressed.flatten(0, 1).transpose(0, 1),
                               row_tile, gate)
        q, s = _nvfp4_quantize(expected.flatten(1), sf, sfLayout=layout.layout_128x4)
        actual_q, actual_s, actual_sf = unpack_fp8_gate(payload, scales, compressed, gate, row_tile, sf)
    assert torch.equal(q, actual_q)
    assert torch.equal(s, actual_s)
    assert actual_sf is sf


@pytest.mark.parametrize("world,rows,heads,dim,chunks", [(2, 33, 3, 128, 2), (3, 17, 2, 64, 3), (2, 7430, 14, 128, 2)])
def test_chunked_output_gate_writes_full_rank_major_head_order(world, rows, heads, dim, chunks):
    from fastvideo.models.dits.minimax_h3_fusions.sp_exchange import unpack_fp8_gate
    from fastvideo.models.dits.minimax_h3_vsa_fp4 import _apply_gate, _unpack_heads_fp8

    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("native FP8 conversion requires Hopper or newer")
    torch.manual_seed(943)
    tiles = 256
    full_heads = heads * chunks
    payload = torch.randn((world, rows, full_heads, dim), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand((world, rows, full_heads), device="cuda").clamp_min_(1e-12)
    compressed = torch.randn((world, full_heads, tiles, dim), device="cuda", dtype=torch.bfloat16)
    gate = torch.randn((rows, world * full_heads, dim), device="cuda", dtype=torch.bfloat16)
    row_tile = torch.randint(tiles, (rows,), device="cuda")
    expected = _apply_gate(_unpack_heads_fp8(payload, scales), compressed.flatten(0, 1).transpose(0, 1), row_tile, gate)
    output = torch.full_like(expected, float("nan"))
    for chunk in range(chunks):
        start, stop = chunk * heads, (chunk + 1) * heads
        result = unpack_fp8_gate(payload[:, :, start:stop].contiguous(), scales[:, :, start:stop].contiguous(),
                                 compressed[:, start:stop].contiguous(), gate, row_tile, head_chunk=chunk,
                                 head_chunks=chunks, output=output)
        assert result.data_ptr() == output.data_ptr()
    assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("world,rows,heads,dim,chunks", [
    (2, 33, 3, 128, 2), (3, 17, 2, 64, 3), (2, 7430, 14, 128, 2), (2, 257, 7, 128, 4),
    (2, 7430, 4, 128, 7), (2, 7430, 2, 128, 14), (2, 7430, 1, 128, 28),
])
@pytest.mark.parametrize("peer_rank", [None, 0, 1])
def test_chunked_fp4_output_matches_full_unpack_gate_and_quantization(world, rows, heads, dim, chunks, peer_rank):
    if peer_rank is not None and world != 2:
        pytest.skip("direct-local payloads require two ranks")
    from fastvideo.layers.quantization.nvfp4_config import _nvfp4_quantize, _require_flashinfer
    from fastvideo.models.dits.minimax_h3_fusions.sp_exchange import unpack_fp8_gate
    from fastvideo.models.dits.minimax_h3_vsa_fp4 import _apply_gate, _unpack_heads_fp8

    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("native FP4 conversion requires Blackwell")
    torch.manual_seed(245)
    tiles, full_heads = 256, heads * chunks
    payload = torch.randn((world, rows, full_heads, dim), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand((world, rows, full_heads), device="cuda").clamp_min_(1e-12)
    compressed = torch.randn((world, full_heads, tiles, dim), device="cuda", dtype=torch.bfloat16)
    gate = torch.randn((rows, world * full_heads, dim), device="cuda", dtype=torch.bfloat16)
    row_tile = torch.randint(tiles, (rows,), device="cuda")
    sf = torch.tensor(0.7079, device="cuda")
    layout, _, _ = _require_flashinfer()
    width = world * full_heads * dim
    codes = torch.empty((rows, width // 2), dtype=torch.uint8, device="cuda")
    fp4_scales = torch.zeros((((rows + 127) // 128) * 128, width // 16), dtype=torch.uint8, device="cuda")
    with torch.inference_mode():
        decoded = _apply_gate(_unpack_heads_fp8(payload, scales), compressed.flatten(0, 1).transpose(0, 1), row_tile, gate)
        expected_q, expected_s = _nvfp4_quantize(decoded.flatten(1), sf, sfLayout=layout.layout_128x4)
        for chunk in range(chunks):
            start, stop = chunk * heads, (chunk + 1) * heads
            received = payload[:, :, start:stop].contiguous()
            if peer_rank is not None:
                from fastvideo.distributed.async_collectives import PeerSlices
                received = PeerSlices(received[peer_rank], received[1 - peer_rank].clone(), torch.empty_like(received), peer_rank)
            result = unpack_fp8_gate(received, scales[:, :, start:stop].contiguous(),
                                     compressed[:, start:stop].contiguous(), gate, row_tile, sf, head_chunk=chunk,
                                     head_chunks=chunks, fp4_output=(codes, fp4_scales))
            assert result[0].data_ptr() == codes.data_ptr()
            assert result[1].data_ptr() == fp4_scales.data_ptr()
    assert torch.equal(codes, expected_q)
    assert torch.equal(fp4_scales, expected_s)


@pytest.mark.parametrize("world,rows,heads,start", [(1, 17, 2, 0), (2, 33, 7, 128), (3, 47, 3, 0),
                                                  (2, 7430, 7, 768)])
def test_fp8_query_unpack_matches_bf16_scatter_and_native_fp4(world, rows, heads, start):
    from fastvideo.models.dits.minimax_h3_fusions.sp_exchange import unpack_fp8_tiles
    from fastvideo.models.dits.minimax_h3_vsa_fp4 import _api, _unpack_seq_fp8

    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("native FP4 conversion requires Blackwell")
    torch.manual_seed(357)
    dim, sequence = 128, world * rows - 1
    tile_rows = ((sequence + 127) // 128 + 1) * 128
    payload = torch.randn((world, 3, rows, heads, dim), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand((world, 3, rows, heads), device="cuda").clamp_min_(1e-12)
    untile = torch.randperm(tile_rows, device="cuda")[:sequence]
    inverse = torch.full((tile_rows,), -1, device="cuda", dtype=torch.int64)
    inverse[untile] = torch.arange(sequence, device="cuda")
    expected = torch.zeros((3, tile_rows, heads, dim), device="cuda", dtype=torch.bfloat16)
    expected.index_copy_(1, untile, _unpack_seq_fp8(payload, scales, sequence))
    native = _api().scale_and_quant_fp4(expected[0:1, start:].transpose(1, 2))
    output = torch.full_like(expected, float("nan"))
    actual, packed = unpack_fp8_tiles(payload, scales, inverse, output, query_start=start)
    assert actual.data_ptr() == output.data_ptr()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    for candidate, reference in zip(packed, native, strict=True):
        assert torch.equal(candidate.view(torch.uint8), reference.view(torch.uint8))
