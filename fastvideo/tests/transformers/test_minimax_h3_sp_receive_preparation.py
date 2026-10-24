# SPDX-License-Identifier: Apache-2.0
"""SP fused receives retain native quantization, pooling, padding and head order."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="requires Blackwell native FP4 conversion",
)


@pytest.mark.parametrize("world,rows,heads,tiled,start", [
    (1, 73, 2, 128, 0), (2, 121, 7, 384, 128), (3, 83, 3, 384, 0), (2, 7430, 7, 16384, 768),
    (2, 7430, 4, 16384, 768), (2, 7430, 2, 16384, 768), (2, 7430, 1, 16384, 768),
])
@pytest.mark.parametrize("peer_rank", [None, 0, 1])
@pytest.mark.parametrize("compact_queries", [False, True])
@pytest.mark.parametrize("compact_values", [False, True])
def test_receive_means_and_native_query_value_quantization_are_byte_exact(
        world, rows, heads, tiled, start, peer_rank, compact_queries, compact_values):
    if peer_rank is not None and world != 2:
        pytest.skip("direct-local payloads require two ranks")
    from attn_qat_infer import api
    from fastvideo.attention.backends.video_sparse_attn_h3 import _pool_tiles
    from fastvideo.models.dits.minimax_h3_fusions.sp_receive import unpack_pool_quantize_fp8
    from fastvideo.models.dits.minimax_h3_vsa_fp4 import _unpack_seq_fp8

    torch.manual_seed(3927)
    dim, length = 128, world * rows - 1
    payload = torch.randn((world, 3, rows, heads, dim), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(payload.shape[:-1], device="cuda") * 0.5
    payload[:, :, 0] = 0
    untile = torch.randperm(tiled, device="cuda")[:length]
    inverse = torch.full((tiled,), -1, device="cuda", dtype=torch.int64)
    inverse[untile] = torch.arange(length, device="cuda")
    valid = (inverse >= 0).view(-1, 64).sum(1).clamp_min(1).to(torch.int32)
    expected = torch.zeros((3, tiled, heads, dim), device="cuda", dtype=torch.bfloat16)
    actual = torch.full_like(expected, float("nan"))
    query_rows = min(start + 128, tiled) if compact_queries else None
    positions = (inverse >= 0).nonzero().flatten()
    prefix_inverse = torch.full_like(inverse, -1)
    prefix_inverse[positions] = torch.arange(positions.numel(), device="cuda")
    prefix_value = torch.empty((1, positions.numel(), heads, dim), device="cuda", dtype=torch.bfloat16)
    prefix = (prefix_inverse, prefix_value) if compact_values else None
    with torch.inference_mode():
        expected.index_copy_(1, untile, _unpack_seq_fp8(payload, scales, length))
        pool = torch.cat([_pool_tiles(expected[i:i + 1], valid, 64) for i in range(3)])
        q_quant = api.scale_and_quant_fp4(expected[0:1, start:].transpose(1, 2))
        v_quant = api.scale_and_quant_fp4_transpose(expected[2:3].transpose(1, 2))
        received = payload
        if peer_rank is not None:
            from fastvideo.distributed.async_collectives import PeerSlices
            received = PeerSlices(payload[peer_rank], payload[1 - peer_rank].clone(), torch.empty_like(payload), peer_rank)
            assert torch.equal(received.materialize().view(torch.uint8), payload.view(torch.uint8))
        got_pool, got_v, got_q = unpack_pool_quantize_fp8(
            received, scales, inverse, valid, actual, start, query_rows=query_rows, prefix_values=prefix)
    live_queries = tiled if query_rows is None else query_rows
    assert torch.equal(actual[0, :live_queries].view(torch.uint8), expected[0, :live_queries].view(torch.uint8))
    assert torch.isnan(actual[0, live_queries:]).all()
    assert torch.equal(actual[1].view(torch.uint8), expected[1].view(torch.uint8))
    if compact_values:
        assert torch.isnan(actual[2]).all()
        assert torch.equal(prefix_value, expected[2:3, positions])
    else:
        assert torch.equal(actual[2].view(torch.uint8), expected[2].view(torch.uint8))
    assert torch.equal(got_pool.contiguous().view(torch.uint8), pool.contiguous().view(torch.uint8))
    for candidate, reference in zip((*got_q, *got_v), (*q_quant, *v_quant), strict=True):
        assert torch.equal(candidate.view(torch.uint8), reference.view(torch.uint8))


@pytest.mark.parametrize("world,rows,heads,prefix,start,logical", [
    (1, 79, 2, 64, 0, 128), (2, 123, 3, 128, 128, 384),
    (3, 77, 7, 192, 128, 384), (2, 7430, 7, 832, 768, 16384),
    (2, 7430, 4, 832, 768, 16384), (2, 7430, 2, 832, 768, 16384), (2, 7430, 1, 832, 768, 16384),
])
@pytest.mark.parametrize("strided", [False, True])
def test_split_output_pack_matches_concat_gather_and_fp8_rounding(world, rows, heads, prefix, start, logical, strided):
    from fastvideo.models.dits.minimax_h3_fusions.sp_output_pack import pack_split_fp8
    from fastvideo.models.dits.minimax_h3_vsa_fp4 import _pack_seq_fp8

    torch.manual_seed(907)
    p = torch.randn((1, heads, prefix, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((1, heads, logical - start, 128), device="cuda", dtype=torch.bfloat16)
    if strided:
        p = p.transpose(1, 2).contiguous().transpose(1, 2)
        v = v.transpose(1, 2).contiguous().transpose(1, 2)
    untile = torch.randperm(logical, device="cuda")[:world * rows - 1]
    with torch.inference_mode():
        expected = _pack_seq_fp8(torch.cat((p, v[:, :, prefix - start:]), dim=2), untile, world, rows)
        actual = pack_split_fp8((p, v, start), untile, world, rows)
    for candidate, reference in zip(actual, expected, strict=True):
        assert torch.equal(candidate.view(torch.uint8), reference.view(torch.uint8))


@pytest.mark.parametrize("prefix", [0, 64, 128, 192])
def test_native_attention_does_not_read_omitted_query_suffix(prefix):
    """Exercise the real native kernel with a poisoned raw query suffix."""
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires consumer Blackwell sparse FP4 attention")
    from attn_qat_infer import api
    from fastvideo.models.dits import minimax_h3_vsa_fp4 as impl
    from fastvideo.models.dits.minimax_h3_fusions.sp_receive import unpack_pool_quantize_fp8

    torch.manual_seed(713)
    rows, heads, tiled = 191, 4, 384
    start = prefix // 128 * 128
    payload = torch.randn((2, 3, rows, heads, 128), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(payload.shape[:-1], device="cuda") * 0.1
    inverse = torch.arange(tiled, device="cuda", dtype=torch.int64)
    inverse[2 * rows - 1:] = -1
    valid = (inverse >= 0).view(-1, 64).sum(1).to(torch.int32)
    full = torch.empty((3, tiled, heads, 128), device="cuda", dtype=torch.bfloat16)
    compact = torch.full_like(full, float("nan"))
    with torch.inference_mode():
        unpack_pool_quantize_fp8(payload, scales, inverse, valid, full, start)
        _, prepared_value, prepared_query = unpack_pool_quantize_fp8(
            payload, scales, inverse, valid, compact, start, query_rows=min(start + 128, tiled))
        mask = torch.eye(tiled // 64, device="cuda", dtype=torch.bool).expand(1, heads, -1, -1).clone()
        mask[:, :, :prefix // 64] = True
        mask[:, :, :, :prefix // 64] = True
        lists = api.vsa_tile_mask_to_fp4_blocks(mask, 64, valid)
        kwargs = dict(logical=tiled, prefix_rows=prefix, precision="fp8",
                      options={"two_level_p": True, "smooth_k": True},
                      prepared_query=prepared_query, prepared_value=prepared_value)
        expected = impl._fp4_fine_attention(api, full[0:1], full[1:2], full[2:3], *lists, **kwargs)
        actual = impl._fp4_fine_attention(api, compact[0:1], compact[1:2], compact[2:3], *lists, **kwargs)
    assert actual.isfinite().all()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("prefix", [64, 128, 192])
@pytest.mark.parametrize("pooled_mean", [False, True])
@pytest.mark.parametrize("direct_keys", [False, True])
@pytest.mark.parametrize("peer_rank", [None, 0, 1])
def test_native_attention_accepts_direct_prefix_values_with_poisoned_raw_values(prefix, pooled_mean, direct_keys, peer_rank):
    """The real native/prefix attention must not read omitted BF16 values."""
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires consumer Blackwell sparse FP4 attention")
    from attn_qat_infer import api
    from fastvideo.models.dits import minimax_h3_vsa_fp4 as impl
    from fastvideo.models.dits.minimax_h3_fusions.sp_receive import unpack_pool_quantize_fp8
    from fastvideo.models.dits.minimax_h3_fusions.key_quant import quantize_smoothed_keys
    from fastvideo.models.dits.minimax_h3_fusions.sp_key_mean import key_mean_from_tiles

    torch.manual_seed(1729)
    rows, heads, tiled = 191, 4, 384
    start = prefix // 128 * 128
    payload = torch.randn((2, 3, rows, heads, 128), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(payload.shape[:-1], device="cuda") * 0.1
    inverse = torch.arange(tiled, device="cuda", dtype=torch.int64)
    inverse[2 * rows - 1:] = -1
    valid = (inverse >= 0).view(-1, 64).sum(1).to(torch.int32)
    full = torch.empty((3, tiled, heads, 128), device="cuda", dtype=torch.bfloat16)
    compact = torch.full_like(full, float("nan"))
    layout = impl._KeyIndexCache()
    layout.tile, layout.n_tiles = 64, tiled // 64
    received = payload
    if peer_rank is not None:
        from fastvideo.distributed.async_collectives import PeerSlices
        received = PeerSlices(payload[peer_rank], payload[1 - peer_rank], payload, peer_rank)
    with torch.inference_mode():
        index, prefix_inverse = layout.valid_keys_from_tiles(valid, api.HALF_N)
        prefix_value = torch.empty((1, index.numel(), heads, 128), device="cuda", dtype=torch.bfloat16)
        unpack_pool_quantize_fp8(payload, scales, inverse, valid, full, start)
        pool, prepared_value, prepared_query = unpack_pool_quantize_fp8(
            received, scales, inverse, valid, compact, start, query_rows=min(start + 128, tiled),
            prefix_values=(prefix_inverse, prefix_value), store_keys=not direct_keys)
        mean = key_mean_from_tiles(pool[1:2], valid, tiled) if pooled_mean else full[1:2].mean(dim=1, keepdim=True)
        expected_keys = quantize_smoothed_keys(full[1:2], full[2:3], mean, prefix_inverse, index.numel())
        keys = quantize_smoothed_keys(compact[1:2], compact[2:3], mean, prefix_inverse, index.numel(),
                                      prefix_value=prefix_value,
                                      received_keys=(received, scales, inverse) if direct_keys else None)
        if direct_keys:
            assert torch.isnan(compact[1]).all()
        assert keys[2].data_ptr() == prefix_value.data_ptr()
        for candidate, reference in zip((*keys[0], *keys[1:]), (*expected_keys[0], *expected_keys[1:]), strict=True):
            assert torch.equal(candidate.view(torch.uint8), reference.view(torch.uint8))
        assert torch.isnan(compact[2]).all()
        mask = torch.eye(tiled // 64, device="cuda", dtype=torch.bool).expand(1, heads, -1, -1).clone()
        mask[:, :, :prefix // 64] = True
        mask[:, :, :, :prefix // 64] = True
        lists = api.vsa_tile_mask_to_fp4_blocks(mask, 64, valid)
        kwargs = dict(logical=tiled, prefix_rows=prefix, precision="fp8", layout=layout,
                      options={"two_level_p": True, "smooth_k": True},
                      prepared_query=prepared_query, prepared_value=prepared_value)
        expected = impl._fp4_fine_attention(api, full[0:1], full[1:2], full[2:3], *lists,
                                             prepared_keys=expected_keys, **kwargs)
        actual = impl._fp4_fine_attention(api, compact[0:1], compact[1:2], compact[2:3], *lists,
                                           prepared_keys=keys, **kwargs)
    assert actual.isfinite().all()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
