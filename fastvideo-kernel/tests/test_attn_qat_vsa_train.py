# SPDX-License-Identifier: Apache-2.0
"""Block-sparse FP4 Attn-QAT (VSA tiles 64 and 128): the PyTorch specification and the Triton kernel.

CPU tests pin the reference emulator (E2M1/E4M3 rounding, list builder, quadrant
masks, online softmax, dV weights). GPU tests (any CUDA GPU with Triton; run on
GB200) compare the Triton forward and STE backward with the reference.
"""
from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

_DIR = Path(__file__).resolve().parents[1] / "python" / "fastvideo_kernel" / "triton_kernels"
_PKG = "_test_qat_vsa_kernels"


def _module(name: str):
    if _PKG not in sys.modules:
        spec = importlib.util.spec_from_loader(_PKG, loader=None, is_package=True)
        spec.submodule_search_locations.append(str(_DIR))
        sys.modules[_PKG] = importlib.util.module_from_spec(spec)
    return importlib.import_module(f"{_PKG}.{name}")


ref = _module("attn_qat_vsa_reference")
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs a CUDA GPU")


def _problem(n_tiles: int, heads: int = 2, keep: float = 0.4, seed: int = 0, device: str = "cpu",
             partial_tail: bool = True):
    """Random tile-ordered Q/K/V with a VSA-like mask: block 0 always kept, tail tile partially valid."""
    gen = torch.Generator().manual_seed(seed)
    seq_len = n_tiles * ref.BLOCK
    tile_valid = torch.full((n_tiles, ), ref.BLOCK, dtype=torch.int32)
    if partial_tail:
        tile_valid[-1] = 37
        tile_valid[n_tiles // 2] = 100
    token_valid = (torch.arange(ref.BLOCK)[None, :] < tile_valid[:, None]).reshape(-1)
    q, k, v = (torch.randn(1, heads, seq_len, 128, generator=gen) * token_valid[None, None, :, None]
               for _ in range(3))
    # RMS-normed Q/K have per-channel structure; add a channel bias so smoothing matters.
    k = k + 0.5 * torch.randn(1, heads, 1, 128, generator=gen) * token_valid[None, None, :, None]
    mask = torch.rand(1, heads, n_tiles, n_tiles, generator=gen) < keep
    mask[..., 0] = True
    q2k_idx, q2k_num, kv_valid = ref.tile128_mask_to_fp4_blocks(mask, tile_valid)
    move = lambda t: t.to(device)  # noqa: E731
    return (move(q.bfloat16()), move(k.bfloat16()), move(v.bfloat16()), move(q2k_idx), move(q2k_num),
            move(kv_valid), move(mask), move(tile_valid))


def test_e2m1_grid_and_ties_round_to_even():
    values = torch.tensor([0.0, 0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.1, 7.0, -0.75, -5.0])
    expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, -1.0, -4.0])
    assert torch.equal(ref.e2m1_rn(values), expected)


def test_e4m3_rounding_saturates_and_flushes():
    values = torch.tensor([448.0, 1000.0, 1.0625, 2.0**-10, 2.0**-9, 0.0])
    out = ref.e4m3_rn(values)
    assert out[0] == 448 and out[1] == 448
    assert out[2] == 1.0  # tie between 1.0 and 1.125 rounds to even
    assert out[3] == 0.0 and out[4] == 2.0**-9 and out[5] == 0.0


def test_fake_quant_matches_nvfp4_definition():
    x = torch.tensor([[0.0] * 15 + [6.0], [1.0] * 16]).float()
    out = ref.nvfp4_fake_quant_lastdim(x)
    assert torch.equal(out[0], x[0])  # amax 6 -> scale exactly 1
    # amax 1 -> scale e4m3(1/6) = 0.171875, 1 / 0.171875 = 5.82 -> E2M1 6 -> 1.03125
    assert torch.equal(out[1], torch.full((16, ), 1.03125))
    assert ref.nvfp4_fake_quant_lastdim(torch.zeros(1, 16)).abs().sum() == 0


def test_list_builder_is_descending_and_counts_valid_tiles():
    _, _, _, q2k_idx, q2k_num, kv_valid, mask, tile_valid = _problem(6)
    for h in range(mask.shape[1]):
        for m in range(6):
            count = int(q2k_num[0, h, m])
            listed = q2k_idx[0, h, m, :count].tolist()
            assert listed == sorted(listed, reverse=True)
            assert listed == sorted(torch.nonzero(mask[0, h, m]).flatten().tolist(), reverse=True)
            assert listed[-1] == 0
    assert kv_valid.tolist()[-2:] == [37, 0]
    assert kv_valid.numel() == 2 * tile_valid.numel()


def test_invert_block_lists_round_trips():
    _, _, _, q2k_idx, q2k_num, _, mask, _ = _problem(5, heads=3, seed=3)
    k2q_idx, k2q_slot, k2q_num = ref.invert_block_lists(q2k_idx, q2k_num, 5)
    for h in range(3):
        for n in range(5):
            entries = [(int(k2q_idx[0, h, n, i]), int(k2q_slot[0, h, n, i])) for i in range(int(k2q_num[0, h, n]))]
            assert [m for m, _ in entries] == torch.nonzero(mask[0, h, :, n]).flatten().tolist()
            for m, slot in entries:
                assert int(q2k_idx[0, h, m, slot]) == n


def test_reference_without_quantization_is_masked_softmax():
    q, k, v, q2k_idx, q2k_num, kv_valid, _, _ = _problem(4)
    out = ref.fp4_vsa_attention_reference(q, k, v, q2k_idx, q2k_num, kv_valid, quantize=False)
    token_mask = ref.dense_block_mask(q2k_idx, q2k_num, q.shape[2], kv_valid)
    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(128)
    expected = torch.softmax(scores.masked_fill(~token_mask, -math.inf), dim=-1) @ v.float()
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("two_level_p", [False, True])
def test_reference_dv_weights_reproduce_output(two_level_p):
    q, k, v, q2k_idx, q2k_num, kv_valid, _, _ = _problem(3, seed=1)
    out, weights = ref.fp4_vsa_attention_reference(q, k, v, q2k_idx, q2k_num, kv_valid, two_level_p=two_level_p,
                                                   return_dv_weights=True)
    _, _, v_hat = ref.fake_quant_qkv(q, k, v)
    torch.testing.assert_close(weights @ v_hat, out, rtol=1e-4, atol=1e-5)


def test_two_level_p_is_closer_to_exact_than_single_level():
    q, k, v, q2k_idx, q2k_num, kv_valid, _, _ = _problem(6, seed=2, keep=0.8)
    exact = ref.fp4_vsa_attention_reference(q, k, v, q2k_idx, q2k_num, kv_valid, quantize=False)
    errors = {}
    for two_level in (False, True):
        out = ref.fp4_vsa_attention_reference(q, k, v, q2k_idx, q2k_num, kv_valid, two_level_p=two_level)
        errors[two_level] = float((out - exact).norm() / exact.norm())
    assert errors[True] < errors[False]


# ----------------------------------------------------------------------------- tile 64: quadrant masks
def _problem64(n_tiles: int, heads: int = 2, keep: float = 0.4, seed: int = 0, device: str = "cpu"):
    """A 64-token VSA problem: two tiles share one 128-token kernel block."""
    gen = torch.Generator().manual_seed(seed)
    tile_valid = torch.full((n_tiles, ), 64, dtype=torch.int32)
    if n_tiles > 2:
        tile_valid[-1] = 21
        tile_valid[n_tiles // 2] = 40
    tile_mask = torch.rand(1, heads, n_tiles, n_tiles, generator=gen) < keep
    tile_mask[..., 0] = True
    q2k_idx, q2k_num, kv_valid, q2k_quad = ref.tile_mask_to_fp4_blocks(tile_mask, 64, tile_valid)
    n_blocks = q2k_idx.shape[-1]
    seq_len = n_blocks * ref.BLOCK
    token_valid = torch.zeros(seq_len, dtype=torch.bool)
    for tile in range(n_tiles):
        token_valid[tile * 64:(tile + 1) * 64] = torch.arange(64) < tile_valid[tile]
    q, k, v = (torch.randn(1, heads, seq_len, 128, generator=gen) * token_valid[None, None, :, None]
               for _ in range(3))
    k = k + 0.5 * torch.randn(1, heads, 1, 128, generator=gen) * token_valid[None, None, :, None]
    move = lambda t: t.to(device)  # noqa: E731
    return (move(q.bfloat16()), move(k.bfloat16()), move(v.bfloat16()), move(q2k_idx), move(q2k_num),
            move(kv_valid), move(q2k_quad), move(tile_mask), move(tile_valid))


def _dense_quad_mask(q2k_idx: torch.Tensor, q2k_num: torch.Tensor, q2k_quad: torch.Tensor, seq_len: int) -> torch.Tensor:
    """``[B, H, S, S]`` token mask: listed blocks intersected with their selected 64x64 quadrants."""
    batch, heads, n_blocks, _ = q2k_idx.shape
    mask = torch.zeros(batch, heads, seq_len, seq_len, dtype=torch.bool, device=q2k_idx.device)
    half = torch.arange(ref.BLOCK, device=q2k_idx.device) // ref.HALF
    for b in range(batch):
        for h in range(heads):
            for m in range(n_blocks):
                for slot in range(int(q2k_num[b, h, m])):
                    n = int(q2k_idx[b, h, m, slot])
                    quad = int(q2k_quad[b, h, m, slot])
                    qvalid = ((quad >> (half[:, None] * 2 + half[None, :])) & 1).bool()
                    mask[b, h, m * ref.BLOCK:(m + 1) * ref.BLOCK, n * ref.BLOCK:(n + 1) * ref.BLOCK] = qvalid
    return mask


def test_tile64_lists_a_block_for_any_selected_quadrant():
    _, _, _, q2k_idx, q2k_num, kv_valid, q2k_quad, tile_mask, tile_valid = _problem64(6, heads=2, seed=2)
    n_blocks = q2k_idx.shape[-1]
    half_mask = tile_mask & (tile_valid > 0)[None, None, None, :]
    for h in range(2):
        for m in range(n_blocks):
            listed = q2k_idx[0, h, m, :int(q2k_num[0, h, m])].tolist()
            assert listed == sorted(listed, reverse=True)
            expected = torch.nonzero(half_mask[0, h, 2 * m:2 * m + 2, :].any(dim=0).view(-1, 2).any(dim=1)).flatten()
            assert listed == sorted(expected.tolist(), reverse=True)
    assert kv_valid.numel() == 2 * n_blocks


def test_tile64_quadrant_bits_match_the_half_tile_mask():
    _, _, _, q2k_idx, q2k_num, _, q2k_quad, tile_mask, tile_valid = _problem64(6, heads=2, seed=3)
    half_mask = tile_mask & (tile_valid > 0)[None, None, None, :]
    weights = torch.tensor([[1, 2], [4, 8]])
    for h in range(2):
        for m in range(q2k_idx.shape[2]):
            for slot in range(int(q2k_num[0, h, m])):
                n = int(q2k_idx[0, h, m, slot])
                quads = half_mask[0, h, 2 * m:2 * m + 2, 2 * n:2 * n + 2]
                expected = int((quads.to(torch.int32) * weights).sum())
                assert int(q2k_quad[0, h, m, slot]) == expected


def test_tile128_route_is_the_general_route_without_quadrants():
    gen = torch.Generator().manual_seed(4)
    tile_mask = torch.rand(1, 2, 5, 5, generator=gen) < 0.5
    tile_valid = torch.tensor([128, 128, 77, 0, 128], dtype=torch.int32)
    q2k_idx, q2k_num, kv_valid, q2k_quad = ref.tile_mask_to_fp4_blocks(tile_mask, 128, tile_valid)
    old_idx, old_num, old_valid = ref.tile128_mask_to_fp4_blocks(tile_mask, tile_valid)
    assert q2k_quad is None
    assert torch.equal(q2k_idx, old_idx) and torch.equal(q2k_num, old_num) and torch.equal(kv_valid, old_valid)


def test_tile64_quadrant_mask_is_exact_masked_softmax():
    q, k, v, q2k_idx, q2k_num, kv_valid, q2k_quad, _, _ = _problem64(5, seed=5)
    out = ref.fp4_vsa_attention_reference(q, k, v, q2k_idx, q2k_num, kv_valid, q2k_quad, quantize=False)
    mask = _dense_quad_mask(q2k_idx, q2k_num, q2k_quad, q.shape[2])
    valid = ref.key_valid_mask(kv_valid, q2k_idx.shape[-1], q.device).reshape(-1)
    mask = mask & valid[None, None, None, :]
    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(128)
    expected = torch.softmax(scores.masked_fill(~mask, -math.inf), dim=-1) @ v.float()
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)


# ----------------------------------------------------------------------------- Triton (GPU)
def _triton():
    return _module("attn_qat_vsa_train")


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


@needs_gpu
@pytest.mark.parametrize("two_level_p,smooth_k", [(False, False), (True, True)])
def test_triton_tile64_forward_matches_reference(two_level_p, smooth_k):
    q, k, v, q2k_idx, q2k_num, kv_valid, q2k_quad, _, _ = _problem64(6, heads=2, seed=10, device="cuda")
    expected = ref.fp4_vsa_attention_reference(q, k, v, q2k_idx, q2k_num, kv_valid, q2k_quad,
                                               two_level_p=two_level_p, smooth_k=smooth_k)
    out = _triton().fp4_vsa_attn_qat(q, k, v, q2k_idx, q2k_num, kv_valid, q2k_quad=q2k_quad,
                                     two_level_p=two_level_p, smooth_k=smooth_k)
    exact = ref.fp4_vsa_attention_reference(q, k, v, q2k_idx, q2k_num, kv_valid, q2k_quad, quantize=False)
    assert _rel(out, expected) < 2e-3 < 0.1 * _rel(expected, exact)


@needs_gpu
@pytest.mark.parametrize("two_level_p,smooth_k", [(False, False), (True, False), (False, True), (True, True)])
def test_triton_forward_matches_reference(two_level_p, smooth_k):
    q, k, v, q2k_idx, q2k_num, kv_valid, _, _ = _problem(6, heads=2, seed=4, device="cuda")
    expected = ref.fp4_vsa_attention_reference(q, k, v, q2k_idx, q2k_num, kv_valid, two_level_p=two_level_p,
                                               smooth_k=smooth_k)
    out = _triton().fp4_vsa_attn_qat(q, k, v, q2k_idx, q2k_num, kv_valid, two_level_p=two_level_p,
                                     smooth_k=smooth_k)
    exact = ref.fp4_vsa_attention_reference(q, k, v, q2k_idx, q2k_num, kv_valid, quantize=False)
    # The emulator must sit far closer to the reference than FP4 sits to exact attention.
    assert _rel(out, expected) < 2e-3 < 0.1 * _rel(expected, exact)


@needs_gpu
def test_triton_forward_unquantized_matches_softmax():
    q, k, v, q2k_idx, q2k_num, kv_valid, _, _ = _problem(5, seed=5, device="cuda")
    expected = ref.fp4_vsa_attention_reference(q, k, v, q2k_idx, q2k_num, kv_valid, quantize=False)
    out = _triton().fp4_vsa_attn_qat(q, k, v, q2k_idx, q2k_num, kv_valid, quantize=False)
    assert _rel(out, expected) < 5e-3


@needs_gpu
def test_triton_first_block_floor_matches_reference():
    q, k, v, q2k_idx, q2k_num, kv_valid, _, _ = _problem(4, seed=6, device="cuda")
    expected = ref.fp4_vsa_attention_reference(q, k, v, q2k_idx, q2k_num, kv_valid, first_block_max_floor=0.0)
    out = _triton().fp4_vsa_attn_qat(q, k, v, q2k_idx, q2k_num, kv_valid, first_block_max_floor=0.0)
    assert _rel(out, expected) < 2e-3


@needs_gpu
@pytest.mark.parametrize("two_level_p", [False, True])
def test_triton_backward_matches_ste_reference(two_level_p):
    q, k, v, q2k_idx, q2k_num, kv_valid, _, _ = _problem(5, heads=2, seed=7, device="cuda")
    q, k, v = (t.clone().requires_grad_(True) for t in (q, k, v))
    out = _triton().fp4_vsa_attn_qat(q, k, v, q2k_idx, q2k_num, kv_valid, two_level_p=two_level_p)
    grad_out = torch.randn_like(out)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), grad_out)
    _, weights = ref.fp4_vsa_attention_reference(q, k, v, q2k_idx, q2k_num, kv_valid, two_level_p=two_level_p,
                                                 return_dv_weights=True)
    q_hat, k_hat, v_hat = ref.fake_quant_qkv(q.detach(), k.detach(), v.detach())
    token_mask = ref.dense_block_mask(q2k_idx, q2k_num, q.shape[2], kv_valid)
    expected = ref.ste_gradients_reference(q_hat, k_hat, v_hat, token_mask, weights, grad_out, 1 / math.sqrt(128))
    for name, got, want in zip("qkv", (dq, dk, dv), expected, strict=True):
        assert _rel(got, want) < 2e-2, name


@needs_gpu
def test_triton_dv_reaches_every_head_on_long_sequences():
    """Regression for the GB10 Attn-QAT bug: re-quantized normalized P underflowed and zeroed dV per head."""
    q, k, v, q2k_idx, q2k_num, kv_valid, _, _ = _problem(64, heads=4, keep=0.9, seed=8, device="cuda")
    v = v.clone().requires_grad_(True)
    out = _triton().fp4_vsa_attn_qat(q, k, v, q2k_idx, q2k_num, kv_valid)
    (dv, ) = torch.autograd.grad(out, (v, ), torch.ones_like(out))
    per_head = dv.float().norm(dim=(-1, -2))[0]
    assert bool((per_head > 0.1 * per_head.mean()).all()), per_head


@needs_gpu
def test_triton_fake_quant_is_bit_identical_to_reference():
    q, k, v, *_ = _problem(4, heads=3, seed=9, device="cuda")
    q = q * 37.0  # exercise large and subnormal E4M3 scales
    got = _triton().fake_quant_qkv_triton(q, k, v)
    want = ref.fake_quant_qkv(q, k, v)
    for name, a, b in zip("qkv", got, want, strict=True):
        assert torch.equal(a.float(), b.to(torch.bfloat16).float()), name
