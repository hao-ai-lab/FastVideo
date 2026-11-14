# SPDX-License-Identifier: Apache-2.0
"""Compact VSA tile masks into the native FP4 kernel's descending block lists."""
import torch
import triton
import triton.language as tl


@triton.jit
def _block_lists(MASK, VALID, IDX, NUM, KV, QUAD,
                 MB: tl.constexpr, MH: tl.constexpr, MR: tl.constexpr, MC: tl.constexpr,
                 H: tl.constexpr, T: tl.constexpr, HT: tl.constexpr,
                 NB: tl.constexpr, BLOCK: tl.constexpr, WRITE_QUAD: tl.constexpr):
    query = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    lane = tl.arange(0, BLOCK)
    col = NB - 1 - lane
    valid_col = lane < NB
    a, b = col * 2, col * 2 + 1
    va = tl.load(VALID + a // HT, (a < T * HT) & valid_col, other=0).to(tl.int32) - a % HT * 64
    vb = tl.load(VALID + b // HT, (b < T * HT) & valid_col, other=0).to(tl.int32) - b % HT * 64
    va = tl.minimum(tl.maximum(va, 0), 64)
    vb = tl.minimum(tl.maximum(vb, 0), 64)
    q0, q1 = query * 2, query * 2 + 1
    base = batch * MB + head * MH
    m00 = tl.load(MASK + base + q0 // HT * MR + a // HT * MC,
                  (q0 < T * HT) & (a < T * HT) & valid_col, other=0)
    m01 = tl.load(MASK + base + q0 // HT * MR + b // HT * MC,
                  (q0 < T * HT) & (b < T * HT) & valid_col, other=0)
    m10 = tl.load(MASK + base + q1 // HT * MR + a // HT * MC,
                  (q1 < T * HT) & (a < T * HT) & valid_col, other=0)
    m11 = tl.load(MASK + base + q1 // HT * MR + b // HT * MC,
                  (q1 < T * HT) & (b < T * HT) & valid_col, other=0)
    # The native builder pads an odd final query half with a key-zero entry.
    m10 = tl.where(q1 >= T * HT, a == 0, m10)
    m11 = tl.where(q1 >= T * HT, b == 0, m11)
    bits = ((m00 & (va > 0)).to(tl.uint8) + 2 * (m01 & (vb > 0)).to(tl.uint8)
            + 4 * (m10 & (va > 0)).to(tl.uint8) + 8 * (m11 & (vb > 0)).to(tl.uint8))
    keep = (bits != 0) & valid_col
    pos = tl.cumsum(keep.to(tl.int32), 0) - 1
    offset = ((batch * H + head) * NB + query) * NB
    tl.store(IDX + offset + lane, 0, valid_col)
    if WRITE_QUAD:
        tl.store(QUAD + offset + lane, 0, valid_col)
    tl.debug_barrier()
    tl.store(IDX + offset + pos, col, keep)
    if WRITE_QUAD:
        tl.store(QUAD + offset + pos, bits, keep)
    tl.store(NUM + (batch * H + head) * NB + query, tl.sum(keep.to(tl.int32), 0))
    if batch == 0 and head == 0 and query == 0:
        tl.store(KV + a, va, valid_col)
        tl.store(KV + b, vb, valid_col)


def tile_mask_to_fp4_blocks(mask: torch.Tensor, tile: int, valid: torch.Tensor):
    """Match Sage's indices, counts, half validity, and optional quadrant bytes.

    Inputs come from H3's mask builder: square CUDA bool masks and contiguous
    integer validity counts, with positive tile sizes divisible by 64.
    """
    batch, heads, tiles, _ = mask.shape
    halves = tile // 64
    blocks = triton.cdiv(tiles * halves, 2)
    indices = torch.empty((batch, heads, blocks, blocks), device=mask.device, dtype=torch.int32)
    counts = torch.empty((batch, heads, blocks), device=mask.device, dtype=torch.int32)
    kv_valid = torch.empty((blocks * 2,), device=mask.device, dtype=torch.int32)
    quadrants = torch.empty_like(indices, dtype=torch.uint8) if halves == 1 else None
    _block_lists[(blocks, heads, batch)](
        mask, valid, indices, counts, kv_valid, quadrants if quadrants is not None else indices,
        *mask.stride(), heads, tiles, halves, blocks, triton.next_power_of_2(blocks), halves == 1,
        num_warps=4,
    )
    return indices, counts, kv_valid, quadrants
