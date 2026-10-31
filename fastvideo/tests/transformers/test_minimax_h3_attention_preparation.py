# SPDX-License-Identifier: Apache-2.0
"""Byte parity for native Blackwell attention input preparation."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="requires Blackwell native FP4 conversion",
)


@pytest.mark.parametrize("batch,heads,seq,start", [(1, 1, 128, 0), (2, 3, 512, 128), (1, 56, 1024, 384)])
def test_norm_rope_query_codes_and_suffix_scales_are_exact(batch, heads, seq, start):
    from attn_qat_infer import api
    from fastvideo.models.dits.minimax_h3_fusions.qknorm_rope import (
        fused_qknorm_rope, qknorm_rope_quantize_fp4,
    )
    torch.manual_seed(3581)
    x = torch.randn((batch, seq, heads, 128), device="cuda", dtype=torch.bfloat16)
    x[:, :3] = 0
    weight = torch.randn((128,), device="cuda", dtype=x.dtype)
    angles = torch.randn((seq, 48), device="cuda")
    angles = torch.cat((angles, angles), -1)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    with torch.inference_mode():
        expected = fused_qknorm_rope(x, weight, cos, sin, 1e-6)
        codes, scales = api.scale_and_quant_fp4(expected[:, start:].transpose(1, 2))
        actual, (actual_codes, actual_scales) = qknorm_rope_quantize_fp4(x, weight, cos, sin, 1e-6, start)
    assert torch.equal(expected, actual)
    assert torch.equal(codes, actual_codes)
    assert torch.equal(scales.view(torch.uint8), actual_scales.view(torch.uint8))
    assert actual_codes.is_contiguous() and actual_scales.is_contiguous()


@pytest.mark.parametrize("batch,heads", [(1, 3), (1, 56), (2, 3)])
@pytest.mark.parametrize("rotary_dim", [96, 128])
def test_omni_identity_gather_preserves_padded_qk_and_query_packing(batch, heads, rotary_dim):
    from fastvideo.models.dits.minimax_h3_fusions.qknorm_rope import (
        fused_qknorm_rope, gather_qknorm_rope, qknorm_rope_quantize_fp4,
    )

    torch.manual_seed(2459)
    rows = 512
    x = torch.randn((batch, rows, heads, 128), device="cuda", dtype=torch.bfloat16)
    exponents = torch.randint(-12, 8, x.shape, device=x.device)
    x = torch.ldexp(x.float(), exponents).to(x.dtype)
    x[:, 17:29] = 0  # Preserve the padded rows too, including interleaved padding.
    weight = torch.randn(128, device=x.device, dtype=x.dtype)
    cos, sin = (torch.randn((rows, rotary_dim), device=x.device, dtype=x.dtype) for _ in range(2))
    identity = torch.arange(rows, device=x.device, dtype=torch.int64)
    with torch.inference_mode():
        expected = fused_qknorm_rope(x, weight, cos, sin, 1e-6)
        actual, no_pack = gather_qknorm_rope(x, weight, cos, sin, 1e-6, identity)
        assert no_pack is None
        assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
        expected, expected_pack = qknorm_rope_quantize_fp4(x, weight, cos, sin, 1e-6, 128)
        actual, actual_pack = gather_qknorm_rope(x, weight, cos, sin, 1e-6, identity, fp4_start=128)
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    assert actual_pack is not None
    for old, new in zip(expected_pack, actual_pack, strict=True):
        assert torch.equal(old.view(torch.uint8), new.view(torch.uint8))


@pytest.mark.parametrize("tile,tiles", [(64, 1), (64, 5), (64, 288), (128, 5), (256, 3)])
@pytest.mark.parametrize("density", [0.0, 0.5, 1.0])
def test_native_block_lists_preserve_order_quadrants_and_partial_tiles(tile, tiles, density):
    from attn_qat_infer import api
    from fastvideo.models.dits.minimax_h3_fusions.block_lists import tile_mask_to_fp4_blocks

    torch.manual_seed(842)
    mask = torch.rand((2, 3, tiles, tiles), device="cuda") < density
    valid = torch.randint(0, tile + 1, (tiles,), device="cuda", dtype=torch.int32)
    valid[0] = tile
    expected = api.vsa_tile_mask_to_fp4_blocks(mask, tile, valid, validate=False)
    actual = tile_mask_to_fp4_blocks(mask, tile, valid)
    for a, b in zip(actual, expected, strict=True):
        if a is None:
            assert b is None
        else:
            assert torch.equal(a, b)


@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("prepared", ["query", "value", "both"])
def test_prepared_inputs_keep_native_key_permutation_without_prepared_keys(smooth, prepared):
    from attn_qat_infer import api
    from fastvideo.models.dits.minimax_h3_vsa_fp4 import _sparse_fp4_attention

    torch.manual_seed(918)
    q, k, v = (torch.randn((1, 256, 2, 128), device="cuda", dtype=torch.bfloat16) for _ in range(3))
    mask = torch.ones((1, 2, 4, 4), device="cuda", dtype=torch.bool)
    valid = torch.full((4,), 64, device="cuda", dtype=torch.int32)
    lists = api.vsa_tile_mask_to_fp4_blocks(mask, 64, valid, validate=False)
    with torch.inference_mode():
        expected = _sparse_fp4_attention(api, q, k, v, *lists, smooth_k=smooth)
        kwargs = {}
        if prepared in ("query", "both"):
            kwargs["prepared_query"] = api.scale_and_quant_fp4(q.transpose(1, 2))
        if prepared in ("value", "both"):
            kwargs["prepared_value"] = api.scale_and_quant_fp4_transpose(v.transpose(1, 2))
        actual = _sparse_fp4_attention(api, q, k, v, *lists, smooth_k=smooth, **kwargs)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("batch,heads,seq,strided", [(1, 1, 128, False), (2, 3, 512, True), (1, 56, 1024, False)])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize("large_range", [False, True])
def test_value_pool_and_native_transposed_quantization_are_bit_exact(
    batch, heads, seq, strided, wide, tile, large_range,
):
    from attn_qat_infer import api
    from fastvideo.attention.backends.video_sparse_attn_h3 import _pool_tiles
    from fastvideo.models.dits.minimax_h3_fusions.value_quant import _pool_and_quantize, pool_quantize_values

    torch.manual_seed(5517)
    x = torch.randn((batch, seq, heads * (2 if strided else 1), 128), device="cuda", dtype=torch.bfloat16)
    if strided:
        x = x[:, :, ::2]
    if large_range:
        exponents = torch.randint(-20, 9, x.shape, device=x.device)
        x.copy_(torch.ldexp(x.float(), exponents).to(x.dtype))
    x[:, :16] = 0
    valid = torch.randint(1, tile + 1, (seq // tile,), device="cuda", dtype=torch.int32)
    with torch.inference_mode():
        expected = _pool_tiles(x, valid, tile)
        codes, scales = api.scale_and_quant_fp4_transpose(x.transpose(1, 2))
        actual, (actual_codes, actual_scales) = pool_quantize_values(x, valid, tile_elems=tile)
        if wide:
            _pool_and_quantize[(heads * 4, seq // tile, batch)](
                x, valid, actual, actual_codes, actual_scales, *x.stride()[:3],
                seq, heads, 128, 32, True, TILE=tile, num_warps=4,
            )
    assert torch.equal(expected.contiguous().view(torch.int32), actual.contiguous().view(torch.int32))
    assert torch.equal(codes, actual_codes)
    assert torch.equal(scales.view(torch.uint8), actual_scales.view(torch.uint8))


@pytest.mark.parametrize("batch,heads", [(1, 3), (2, 4), (1, 56)])
@pytest.mark.parametrize("rotary_dim", [32, 96, 128])
def test_compact_projection_consumers_match_zero_padded_inputs(batch, heads, rotary_dim):
    from fastvideo.models.dits.minimax_h3_fusions.key_quant import quantize_smoothed_keys, pool_quantize_smoothed_keys
    from fastvideo.models.dits.minimax_h3_fusions.qknorm_rope import (
        fused_qknorm_rope, gather_qknorm_rope, qknorm_rope_quantize_fp4,
    )
    from fastvideo.models.dits.minimax_h3_fusions.value_quant import pool_quantize_values

    torch.manual_seed(7619)
    rows, packed_rows = 512, 337
    compact = torch.randn((batch, packed_rows, heads, 128), device="cuda", dtype=torch.bfloat16)
    positions = torch.randperm(rows, device="cuda")[:packed_rows]
    inverse = torch.full((rows,), -1, device="cuda", dtype=torch.int64)
    inverse[positions] = torch.arange(packed_rows, device="cuda")
    tiled = torch.zeros((batch, rows, heads, 128), device="cuda", dtype=compact.dtype)
    tiled.index_copy_(1, positions, compact)
    valid = (inverse >= 0).reshape(-1, 64).sum(-1).to(torch.int32)
    weight = torch.randn(128, device="cuda", dtype=compact.dtype)
    cos, sin = (torch.randn((rows, rotary_dim), device="cuda", dtype=compact.dtype) for _ in range(2))
    with torch.inference_mode():
        expected = fused_qknorm_rope(tiled, weight, cos, sin, 1e-6)
        actual, no_pack = gather_qknorm_rope(compact, weight, cos, sin, 1e-6, inverse)
        assert no_pack is None and torch.equal(expected, actual)
        expected, expected_pack = qknorm_rope_quantize_fp4(tiled, weight, cos, sin, 1e-6, 128)
        actual, actual_pack = gather_qknorm_rope(compact, weight, cos, sin, 1e-6, inverse, fp4_start=128)
        assert torch.equal(expected, actual)
        assert torch.equal(expected_pack[0], actual_pack[0])
        assert torch.equal(expected_pack[1].view(torch.uint8), actual_pack[1].view(torch.uint8))
        expected_pool, expected_pack = pool_quantize_values(tiled, valid)
        actual_pool, actual_pack = pool_quantize_values(compact, valid, input_index=inverse)
        assert torch.equal(expected_pool.contiguous().view(torch.int32), actual_pool.contiguous().view(torch.int32))
        assert torch.equal(expected_pack[0], actual_pack[0])
        assert torch.equal(expected_pack[1].view(torch.uint8), actual_pack[1].view(torch.uint8))
        valid_rows = (inverse >= 0).nonzero().flatten()
        prefix_inverse = torch.full_like(inverse, -1)
        prefix_inverse[valid_rows] = torch.arange(packed_rows, device="cuda")
        mean = tiled.mean(dim=1, keepdim=True)
        expected_k, expected_k8, expected_v8 = quantize_smoothed_keys(tiled, tiled, mean, prefix_inverse, packed_rows)
        actual_k, actual_k8, actual_v8 = quantize_smoothed_keys(
            tiled, compact, mean, prefix_inverse, packed_rows, value_index=inverse,
        )
        pooled_k, pooled_k8, pooled_v8, key_pool = pool_quantize_smoothed_keys(
            tiled, compact, mean, prefix_inverse, packed_rows, valid, value_index=inverse,
        )
        assert torch.equal(key_pool.contiguous().view(torch.int32), expected_pool.contiguous().view(torch.int32))
        assert torch.equal(pooled_k[0], expected_k[0])
        assert torch.equal(pooled_k[1].view(torch.uint8), expected_k[1].view(torch.uint8))
        assert torch.equal(pooled_k8, expected_k8)
        assert torch.equal(pooled_v8, expected_v8)
    assert torch.equal(expected_k[0], actual_k[0])
    assert torch.equal(expected_k[1].view(torch.uint8), actual_k[1].view(torch.uint8))
    assert torch.equal(expected_k8, actual_k8)
    assert torch.equal(expected_v8, actual_v8)


@pytest.mark.parametrize("batch,rows,heads", [(1, 128, 3), (2, 512, 4), (1, 1024, 56)])
def test_key_only_preparation_matches_separate_bf16_shift_and_native_packing(batch, rows, heads):
    from attn_qat_infer import api
    from fastvideo.models.dits.minimax_h3_fusions.key_quant import quantize_smoothed_keys_only

    torch.manual_seed(153)
    key = torch.randn((batch, rows, heads * 2, 128), device="cuda", dtype=torch.bfloat16)[:, :, ::2]
    with torch.inference_mode():
        mean = key.mean(dim=1, keepdim=True)
        expected = api.scale_and_quant_fp4_permute((key - mean).transpose(1, 2))
        actual = quantize_smoothed_keys_only(key, mean)
    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[1].view(torch.uint8), actual[1].view(torch.uint8))


def test_fused_key_only_attention_matches_original_native_entrypoint():
    from attn_qat_infer import api
    from fastvideo.models.dits.minimax_h3_vsa_fp4 import _sparse_fp4_attention

    torch.manual_seed(255)
    q, k, v = (torch.randn((1, 256, 2, 128), device="cuda", dtype=torch.bfloat16) for _ in range(3))
    mask = torch.ones((1, 2, 4, 4), device="cuda", dtype=torch.bool)
    valid = torch.full((4,), 64, device="cuda", dtype=torch.int32)
    lists = api.vsa_tile_mask_to_fp4_blocks(mask, 64, valid, validate=False)
    with torch.inference_mode():
        expected = api.sageattn_blackwell_sparse_bshd(q, k - k.mean(dim=1, keepdim=True), v, *lists,
                                                     validate=False, single_level_p_quant=False)
        actual = _sparse_fp4_attention(api, q, k, v, *lists, smooth_k=True, two_level_p=True)
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("batch,heads,seq,strided", [(1, 1, 128, False), (2, 3, 512, True), (1, 56, 1024, False)])
@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize("large_range", [False, True])
def test_key_pool_and_prefix_preparation_preserve_native_bits(batch, heads, seq, strided, tile, large_range):
    from attn_qat_infer import api
    from fastvideo.attention.backends.video_sparse_attn_h3 import _pool_tiles
    from fastvideo.models.dits.minimax_h3_fusions.key_quant import pool_quantize_smoothed_keys
    from fastvideo.models.dits.minimax_h3_vsa_fp4 import _fp8_per_token

    torch.manual_seed(5518)
    k, v = (torch.randn((batch, seq, heads * (2 if strided else 1), 128), device="cuda", dtype=torch.bfloat16)
            for _ in range(2))
    if strided:
        k, v = k[:, :, ::2], v[:, :, ::2]
    if large_range:
        exponents = torch.randint(-20, 9, k.shape, device=k.device)
        k.copy_(torch.ldexp(k.float(), exponents).to(k.dtype))
    valid = torch.randint(1, tile + 1, (seq // tile,), device=k.device, dtype=torch.int32)
    positions = (torch.arange(tile, device=k.device)[None, :] < valid[:, None]).flatten().nonzero().flatten()
    inverse = torch.full((seq,), -1, device=k.device, dtype=torch.int64)
    inverse[positions] = torch.arange(positions.numel(), device=k.device)
    k[:, inverse < 0] = 0
    v[:, inverse < 0] = 0
    with torch.inference_mode():
        mean = k.mean(dim=1, keepdim=True)
        shifted = k - mean
        expected_pool = _pool_tiles(k, valid, tile)
        expected_pack = api.scale_and_quant_fp4_permute(shifted.transpose(1, 2))
        expected_key = _fp8_per_token(shifted.index_select(1, positions))
        expected_value = v.index_select(1, positions)
        actual_pack, actual_key, actual_value, actual_pool = pool_quantize_smoothed_keys(
            k, v, mean, inverse, positions.numel(), valid, tile_elems=tile,
        )
    assert torch.equal(expected_pool.contiguous().view(torch.int32), actual_pool.contiguous().view(torch.int32))
    assert torch.equal(expected_pack[0], actual_pack[0])
    assert torch.equal(expected_pack[1].view(torch.uint8), actual_pack[1].view(torch.uint8))
    assert torch.equal(expected_key, actual_key)
    assert torch.equal(expected_value, actual_value)


@pytest.mark.parametrize("tile", [64, 128])
def test_prefix_maps_from_tile_validity_include_both_native_halves(tile):
    from fastvideo.models.dits.minimax_h3_vsa_fp4 import _KeyIndexCache

    counts = torch.tensor([tile, 0, 1, tile - 1, 63, 64], device="cuda", dtype=torch.int32)
    layout = _KeyIndexCache()
    layout.tile, layout.n_tiles = tile, counts.numel()
    expected = (torch.arange(tile, device=counts.device)[None, :] < counts[:, None]).flatten().nonzero().flatten()
    actual, inverse = layout.valid_keys_from_tiles(counts, 64)
    assert torch.equal(expected, actual)
    assert torch.equal(inverse[expected], torch.arange(expected.numel(), device=counts.device))
    assert torch.equal((inverse >= 0).nonzero().flatten(), expected)
    again, inverse_again = layout.valid_keys_from_tiles(counts, 64)
    assert actual.data_ptr() == again.data_ptr()
    assert inverse.data_ptr() == inverse_again.data_ptr()

@pytest.mark.parametrize("batch,heads,rotary_dim", [(1, 3, 32), (2, 7, 96), (1, 56, 128)])
@pytest.mark.parametrize("prefix_tiles", [1, 2, 3])
@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize("compact", [False, True])
def test_pooled_queries_match_full_materialization(batch, heads, rotary_dim, prefix_tiles, tile, compact):
    from fastvideo.models.dits.minimax_h3_fusions.qknorm_rope import gather_qknorm_rope, qknorm_rope_quantize_fp4
    from fastvideo.models.dits.minimax_h3_fusions.query_pool import prepare_pooled_queries
    from fastvideo.models.dits.minimax_h3_vsa_fp4 import _fp8_per_token, _pool_tiles

    torch.manual_seed(6521)
    rows, packed_rows = 512, 337 if compact else 512
    prefix_rows = prefix_tiles * tile
    x = torch.randn((batch, packed_rows, heads * 2, 128), device="cuda", dtype=torch.bfloat16)[:, :, ::2]
    positions = torch.randperm(rows, device="cuda")[:packed_rows]
    inverse = torch.full((rows,), -1, device="cuda", dtype=torch.int64)
    inverse[positions] = torch.arange(packed_rows, device="cuda")
    valid = (inverse >= 0).reshape(-1, tile).sum(-1).to(torch.int32)
    if not compact:
        inverse = None
        valid = torch.randint(1, tile + 1, (rows // tile,), device="cuda", dtype=torch.int32)
        # Direct padded inputs must retain all row values, including invalid rows.
        x.copy_(torch.ldexp(x.float(), torch.randint(-20, 9, x.shape, device=x.device)).to(x.dtype))
    weight = torch.randn(128, device="cuda", dtype=x.dtype)
    cos, sin = (torch.randn((rows, rotary_dim), device="cuda", dtype=x.dtype) for _ in range(2))
    with torch.inference_mode():
        if compact:
            full, packed = gather_qknorm_rope(
                x, weight, cos, sin, 1e-6, inverse, fp4_start=prefix_rows // 128 * 128,
            )
        else:
            full, packed = qknorm_rope_quantize_fp4(x, weight, cos, sin, 1e-6, prefix_rows // 128 * 128)
        expected_pool = _pool_tiles(full, valid, tile)
        prefix, actual_packed, pool = prepare_pooled_queries(
            x, weight, cos, sin, 1e-6, inverse, valid, prefix_rows, tile_elems=tile,
        )
    assert prefix.shape == (batch, prefix_rows, heads, 128)
    assert torch.equal(prefix, full[:, :prefix_rows])
    assert torch.equal(actual_packed[0], packed[0])
    assert torch.equal(actual_packed[1].view(torch.uint8), packed[1].view(torch.uint8))
    assert torch.equal(pool.contiguous().view(torch.int32), expected_pool.contiguous().view(torch.int32))
    if tile == 128:
        for split_prefix in (False, True):
            with torch.inference_mode():
                prefix8, packed8, pool8 = prepare_pooled_queries(
                    x, weight, cos, sin, 1e-6, inverse, valid, prefix_rows, tile_elems=tile,
                    prefix_fp8=True, split_prefix=split_prefix,
                )
                expected_prefix8 = _fp8_per_token(full[:, :prefix_rows])
            assert torch.equal(prefix8, expected_prefix8)
            assert torch.equal(packed8[0], packed[0])
            assert torch.equal(packed8[1].view(torch.uint8), packed[1].view(torch.uint8))
            assert torch.equal(pool8.contiguous().view(torch.int32), expected_pool.contiguous().view(torch.int32))



@pytest.mark.parametrize("batch,heads", [(1, 3), (2, 7), (1, 56)])
@pytest.mark.parametrize("large_range", [False, True])
@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize("compact", [False, True])
def test_combined_kv_preserves_separate_prefix_pool_and_native_bytes(batch, heads, large_range, tile, compact):
    from fastvideo.models.dits.minimax_h3_fusions.key_quant import pool_quantize_smoothed_keys
    from fastvideo.models.dits.minimax_h3_fusions.kv_pool import pool_quantize_prefix_kv
    from fastvideo.models.dits.minimax_h3_fusions.value_quant import pool_quantize_values

    torch.manual_seed(7713)
    rows, compact_rows = 512, 337 if compact else 512
    key = torch.randn((batch, rows, heads * 2, 128), device="cuda", dtype=torch.bfloat16)[:, :, ::2]
    value = torch.randn((batch, compact_rows, heads * 2, 128), device="cuda", dtype=torch.bfloat16)[:, :, ::2]
    if large_range:
        key.mul_(torch.exp2(torch.randint(-12, 12, key.shape, device="cuda").float()).bfloat16())
        value.mul_(torch.exp2(torch.randint(-12, 12, value.shape, device="cuda").float()).bfloat16())
    source = torch.randint(compact_rows, (rows,), device="cuda", dtype=torch.int64)
    source[torch.rand(rows, device="cuda") < 0.2] = -1
    if compact:
        key[:, source < 0] = 0
    valid_index = (source >= 0).nonzero().flatten()
    inverse = torch.full_like(source, -1)
    inverse[valid_index] = torch.arange(valid_index.numel(), device="cuda")
    valid = (source >= 0).reshape(-1, tile).sum(1).to(torch.int32)
    if not compact:
        source = None
    mean = key.mean(dim=1, keepdim=True)
    with torch.inference_mode():
        expected_key = pool_quantize_smoothed_keys(
            key, value, mean, inverse, valid_index.numel(), valid, value_index=source, tile_elems=tile,
        )
        expected_value = pool_quantize_values(value, valid, input_index=source, tile_elems=tile)
        actual_key, actual_value = pool_quantize_prefix_kv(
            key, value, mean, inverse, valid_index.numel(), valid, source, tile_elems=tile,
        )
    for actual, expected in ((actual_key[0][0], expected_key[0][0]),
                             (actual_key[0][1], expected_key[0][1]),
                             (actual_key[1], expected_key[1]), (actual_key[2], expected_key[2]),
                             (actual_key[3], expected_key[3]), (actual_value[0], expected_value[0]),
                             (actual_value[1][0], expected_value[1][0]),
                             (actual_value[1][1], expected_value[1][1])):
        assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
    assert actual_key[1].transpose(1, 2).is_contiguous()
    assert actual_key[2].transpose(1, 2).is_contiguous()


@pytest.mark.parametrize("batch", [1, 2])
def test_preconverted_prefix_queries_preserve_dense_attention(batch):
    from fastvideo.models.dits.minimax_h3_vsa_fp4 import _fp8_per_token, _prefix_dense_attention

    torch.manual_seed(4917)
    q, k, v = (torch.randn((batch, 256, 3, 128), device="cuda", dtype=torch.bfloat16) for _ in range(3))
    positions = torch.arange(211, device="cuda")
    with torch.inference_mode():
        k = k - k.mean(dim=1, keepdim=True)
        prepared = (_fp8_per_token(k[:, :211]).contiguous(), v[:, :211].contiguous())
        expected = _prefix_dense_attention(q, k, v, 128, 256, positions, "fp8",
                                            key_is_smoothed=True, prepared_kv=prepared)
        actual = _prefix_dense_attention(_fp8_per_token(q[:, :128]), k, v, 128, 256, positions, "fp8",
                                          key_is_smoothed=True, prepared_kv=prepared, query_is_fp8=True)
    assert torch.equal(actual, expected)
