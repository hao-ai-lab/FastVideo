#!/usr/bin/env python3
"""Correctness tests for the block-sparse SageAttention3 FP4 forward (``fwd_sparse``).

Covers the VSA tile layouts MiniMax-H3 uses: 64-token tiles carried by 64x64
quadrant masks on the kernel's 128x128 blocks (odd and even tile counts),
256-token tiles, partially valid tiles, and an exempt prefix that every query
attends. The sparse kernel is compared with a token-masked fp32 reference and
must stay at the dense FP4 kernel's own error on the same data. Also checks
that full block lists reproduce the dense kernel bit for bit and that the
sequence-major entry point matches the head-major one bit for bit.

Requires a Blackwell GPU (sm_120a) and the fp4attn_cuda / fp4quant_cuda
extensions built via ``cd fastvideo-kernel && ./build.sh``.

    pytest tests/test_attn_qat_infer_sparse.py -v
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("fp4attn_cuda", reason="ATTN_QAT_INFER FP4 kernels require a sm_120a build")

from attn_qat_infer.api import (BLOCK_M, BLOCK_N, check_sparse_block_lists, sageattn_blackwell,
                                sageattn_blackwell_sparse, sageattn_blackwell_sparse_bshd,
                                vsa_tile_mask_to_fp4_blocks)

DEVICE = torch.device("cuda")
HEAD_DIM = 128


def _tile_sizes(prefix: tuple[int, ...], video_tiles: int, tile: int, partial: dict[int, int]) -> torch.Tensor:
    """Valid tokens per tile: segment-pure prefix chunks, then video tiles (some partial)."""
    sizes = []
    for segment in prefix:
        full, rem = divmod(segment, tile)
        sizes += [tile] * full + ([rem] if rem else [])
    sizes += [partial.get(index, tile) for index in range(video_tiles)]
    return torch.tensor(sizes, dtype=torch.int32, device=DEVICE)


def _exempt_mask(n_prefix: int, n_tiles: int, sparsity: float, heads: int, gen: torch.Generator) -> torch.Tensor:
    """VSA-H3 exempt selection: prefix rows dense, prefix keys always, top-k video keys."""
    n_video = n_tiles - n_prefix
    k_video = max(1, math.ceil((1 - sparsity) * n_video))
    scores = torch.rand(1, heads, n_tiles, n_video, generator=gen, device=DEVICE)
    mask = torch.zeros(1, heads, n_tiles, n_tiles, dtype=torch.bool, device=DEVICE)
    mask.scatter_(-1, scores.topk(k_video, dim=-1).indices + n_prefix, True)
    mask[..., :n_prefix] = True
    mask[:, :, :n_prefix, :] = True
    return mask


def _case(tile: int, prefix: tuple[int, ...], video_tiles: int, partial: dict[int, int], heads: int = 4):
    gen = torch.Generator(device=DEVICE).manual_seed(tile * 1000 + video_tiles)
    sizes = _tile_sizes(prefix, video_tiles, tile, partial)
    n_tiles = sizes.numel()
    n_prefix = n_tiles - video_tiles
    rows = math.ceil(n_tiles * tile / BLOCK_M) * BLOCK_M
    valid = torch.zeros(rows, dtype=torch.bool, device=DEVICE)
    valid[:n_tiles * tile] = (torch.arange(tile, device=DEVICE)[None, :] < sizes[:, None]).reshape(-1)
    q, k, v = (torch.randn(1, heads, rows, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16, generator=gen)
               for _ in range(3))
    for x in (q, k, v):
        x[:, :, ~valid] = 0
    mask = _exempt_mask(n_prefix, n_tiles, 0.8, heads, gen)
    return q, k, v, sizes, mask, valid


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize(
    ("tile", "prefix", "video_tiles", "partial"),
    [
        (64, (77, 46), 19, {
            3: 40,
            11: 1
        }),  # odd tile count, short prefix chunks
        (64, (64, 40), 9, {}),  # even tile count
        (256, (100, 70), 4, {
            1: 200
        }),  # 256-token tiles, partial tile
    ],
)
def test_sparse_matches_masked_reference_at_fp4_floor(tile, prefix, video_tiles, partial) -> None:
    q, k, v, sizes, mask, valid = _case(tile, prefix, video_tiles, partial)
    q2k_idx, q2k_num, kv_valid, q2k_quad = vsa_tile_mask_to_fp4_blocks(mask, tile, sizes, validate=True)
    assert (q2k_quad is not None) == (tile < BLOCK_N)
    out = sageattn_blackwell_sparse(q.clone(), k.clone(), v.clone(), q2k_idx, q2k_num, kv_valid, q2k_quad)

    n_tok = sizes.numel() * tile
    token_tile = torch.arange(sizes.numel(), device=DEVICE).repeat_interleave(tile)
    token_mask = mask[:, :, token_tile][:, :, :, token_tile] & valid[None, None, None, :n_tok]
    ref = F.scaled_dot_product_attention(q[:, :, :n_tok].float(),
                                         k[:, :, :n_tok].float(),
                                         v[:, :, :n_tok].float(),
                                         attn_mask=token_mask)
    rows = valid[:n_tok]
    assert torch.isfinite(out[:, :, :n_tok][:, :, rows]).all()
    sparse_err = _rel_l2(out[:, :, :n_tok][:, :, rows], ref[:, :, rows])

    # FP4 floor: the dense kernel against dense fp32 attention on the valid tokens.
    qv, kv, vv = (x[:, :, valid].contiguous() for x in (q, k, v))
    floor = _rel_l2(sageattn_blackwell(qv, kv, vv), F.scaled_dot_product_attention(qv.float(), kv.float(), vv.float()))
    assert sparse_err <= 1.05 * floor + 1e-3, (sparse_err, floor)


def test_full_lists_reproduce_dense_bitwise() -> None:
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    q, k, v = (torch.randn(1, 4, 2048, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16, generator=gen) for _ in range(3))
    n_blocks = 2048 // BLOCK_N
    q2k_idx = torch.arange(n_blocks, device=DEVICE, dtype=torch.int32).expand(1, 4, 2048 // BLOCK_M,
                                                                              n_blocks).contiguous()
    q2k_num = torch.full((1, 4, 2048 // BLOCK_M), n_blocks, device=DEVICE, dtype=torch.int32)
    dense = sageattn_blackwell(q.clone(), k.clone(), v.clone())
    sparse = sageattn_blackwell_sparse(q.clone(), k.clone(), v.clone(), q2k_idx, q2k_num)
    assert torch.equal(dense, sparse)


def test_bshd_entry_matches_bhsd() -> None:
    q, k, v, sizes, mask, _ = _case(64, (77, 46), 19, {3: 40})
    lists = vsa_tile_mask_to_fp4_blocks(mask, 64, sizes)
    bhsd = sageattn_blackwell_sparse(q.clone(), k.clone(), v.clone(), *lists)
    bshd = sageattn_blackwell_sparse_bshd(*(x.transpose(1, 2).contiguous() for x in (q, k, v)), *lists)
    assert torch.equal(bhsd, bshd)


def test_first_visited_block_check_rejects_unanchored_rows() -> None:
    # Without the exempt prefix a query block may start from a block that masks
    # one of its 64-row halves entirely; validate=True must refuse it.
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool, device=DEVICE)
    mask[0, 0, 0, 3] = True  # query tile 0 -> key tile 3 only
    mask[0, 0, 1, 1] = True  # query tile 1 -> key tile 1 only (block 0, other quadrant)
    mask[0, 0, 2:, 2:] = True
    with pytest.raises(ValueError, match="first block"):
        vsa_tile_mask_to_fp4_blocks(mask, 64, torch.full((4, ), 64, dtype=torch.int32, device=DEVICE), validate=True)


def test_block_list_check_rejects_out_of_bounds_lists() -> None:
    idx = torch.zeros((1, 1, 2, 2), dtype=torch.int32, device=DEVICE)
    num = torch.ones((1, 1, 2), dtype=torch.int32, device=DEVICE)
    idx[..., 1] = 99  # beyond q2k_num: never read, so not checked
    check_sparse_block_lists(idx, num, 2 * BLOCK_N)
    with pytest.raises(ValueError, match="q2k_num"):
        check_sparse_block_lists(idx, torch.zeros_like(num), 2 * BLOCK_N)
    with pytest.raises(ValueError, match="q2k_idx"):
        check_sparse_block_lists(idx, num + 1, 2 * BLOCK_N)
