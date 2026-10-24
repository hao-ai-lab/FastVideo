# SPDX-License-Identifier: Apache-2.0
"""SP transport preparation must preserve normalized BF16 values and FP8 codes/scales."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")


@pytest.mark.parametrize("rows,heads,dim,rotary,world", [
    (17, 2, 64, 48, 1),
    (33, 6, 128, 96, 2),
    (47, 9, 128, 128, 3),
    (7430, 56, 128, 96, 2),
])
def test_normalized_qkv_transport_matches_separate_bf16_rope_and_fp8_pack(rows, heads, dim, rotary, world):
    from fastvideo.models.dits.minimax_h3_fusions import fused_qknorm_rope
    from fastvideo.models.dits.minimax_h3_fusions.sp_qkv_pack import pack_normalized_qkv_fp8
    from fastvideo.models.dits.minimax_h3_vsa_fp4 import _pack_heads_fp8

    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("native FP8 conversion requires Hopper or newer")
    torch.manual_seed(823)
    query, key, value = [torch.randn((1, rows, heads, dim), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    q_weight, k_weight = [torch.randn(dim, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    theta = torch.randn((rows, rotary), device="cuda")
    cos, sin = theta.cos().bfloat16(), theta.sin().bfloat16()
    query[:, 0].zero_()
    key[:, -1].zero_()
    value[:, 0].zero_()
    with torch.no_grad():
        expected = _pack_heads_fp8(fused_qknorm_rope(query, q_weight, cos, sin, 1e-6)[0],
                                   fused_qknorm_rope(key, k_weight, cos, sin, 1e-5)[0], value[0], world)
        actual = pack_normalized_qkv_fp8(query[0], key[0], value[0], q_weight, k_weight, cos, sin, 1e-6, 1e-5, world)
    for candidate, reference in zip(actual, expected, strict=True):
        assert torch.equal(candidate.view(torch.uint8), reference.view(torch.uint8))


@pytest.mark.parametrize("rows,heads,dim,rotary,world,chunks", [
    (17, 12, 64, 48, 2, 3),
    (33, 8, 128, 96, 2, 2),
    (7430, 56, 128, 96, 2, 2),
    (7430, 56, 128, 96, 2, 4),
    (7430, 56, 128, 96, 2, 7),
    (7430, 56, 128, 96, 2, 14),
    (7430, 56, 128, 96, 2, 28),
])
def test_chunked_qkv_transport_matches_separate_pack_and_permute(rows, heads, dim, rotary, world, chunks):
    from fastvideo.models.dits.minimax_h3_fusions import fused_qknorm_rope
    from fastvideo.models.dits.minimax_h3_fusions.sp_qkv_pack import pack_normalized_qkv_fp8
    from fastvideo.models.dits.minimax_h3_vsa_fp4 import _pack_heads_fp8

    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("native FP8 conversion requires Hopper or newer")
    torch.manual_seed(324)
    q, k, v = [torch.randn((1, rows, heads, dim), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    qw, kw = [torch.randn(dim, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    theta = torch.randn((rows, rotary), device="cuda")
    cos, sin = theta.cos().bfloat16(), theta.sin().bfloat16()
    with torch.no_grad():
        p, s = _pack_heads_fp8(fused_qknorm_rope(q, qw, cos, sin, 1e-6)[0],
                               fused_qknorm_rope(k, kw, cos, sin, 1e-5)[0], v[0], world)
        expected = (p.view(world, 3, rows, chunks, heads // (world * chunks), dim).permute(3, 0, 1, 2, 4, 5).contiguous(),
                    s.view(world, 3, rows, chunks, heads // (world * chunks)).permute(3, 0, 1, 2, 4).contiguous())
        actual = pack_normalized_qkv_fp8(q[0], k[0], v[0], qw, kw, cos, sin, 1e-6, 1e-5, world, chunks)
    for candidate, reference in zip(actual, expected, strict=True):
        assert torch.equal(candidate.view(torch.uint8), reference.view(torch.uint8))


@pytest.mark.parametrize("rows,heads,rotary,world,chunks", [
    (17, 4, 32, 2, 1), (33, 8, 96, 2, 2), (47, 12, 128, 3, 2),
    (7430, 56, 96, 2, 4), (7430, 56, 96, 2, 7), (7430, 56, 96, 2, 14),
])
def test_separate_projection_packing_matches_joint_qkv_bytes(rows, heads, rotary, world, chunks):
    from fastvideo.models.dits.minimax_h3_fusions.sp_projection import allocate, pack_one, pack_value
    from fastvideo.models.dits.minimax_h3_fusions.sp_qkv_pack import pack_normalized_qkv_fp8

    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("ordered parallel QK producer requires Blackwell")
    torch.manual_seed(6039)
    q, k, v = [torch.randn((rows, heads, 128), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    qw, kw = [torch.randn(128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    theta = torch.randn((rows, rotary), device="cuda")
    cos, sin = theta.cos().bfloat16(), theta.sin().bfloat16()
    q[0].zero_()
    k[-1].zero_()
    v[0].zero_()
    with torch.no_grad():
        expected = pack_normalized_qkv_fp8(q, k, v, qw, kw, cos, sin, 1e-6, 1e-5, world, chunks)
        payload, scale = allocate(rows, heads, world, chunks, q.device)
        pack_one(q, qw, cos, sin, payload, scale, 1e-6, 0)
        pack_one(k, kw, cos, sin, payload, scale, 1e-5, 1)
        pack_value(v, payload, scale)
    actual = (payload[0], scale[0]) if chunks == 1 else (payload, scale)
    for candidate, reference in zip(actual, expected, strict=True):
        assert torch.equal(candidate.view(torch.uint8), reference.view(torch.uint8))
