# SPDX-License-Identifier: Apache-2.0
"""Parallel SP QK rows with the original four-warp RMS addition order.

The standalone head-128 normalization reduces 32 lanes in each warp first,
then combines four warp sums. Explicit XOR gathers retain that addition tree
when multiple heads share a CTA; a plain two-dimensional reduction changes it.
"""
import triton
import triton.language as tl

@triton.jit
def _normalize(X, Weight, Cos, Sin, row, head, HEADS: tl.constexpr, ROWS: tl.constexpr, DIM: tl.constexpr,
               ROTARY: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.arange(0, BLOCK)
    offset = (row[:, None] * HEADS + head[:, None]) * DIM
    mask = (row[:, None] < ROWS) & (col[None, :] < DIM)
    value = tl.load(X + offset + col[None, :], mask, other=0.0).to(tl.float32)
    squared = value * value
    indices = tl.broadcast_to(col[None, :], squared.shape)
    for shift in tl.static_range(5):
        squared = squared + tl.gather(squared, indices ^ (16 >> shift), 1)
    for shift in tl.static_range(2):
        squared = squared + tl.gather(squared, indices ^ (64 >> shift), 1)
    variance = tl.sum(tl.where(col[None, :] == 0, squared, 0.0), 1) / DIM
    inv = tl.math.rsqrt(variance + EPS)
    weight = tl.load(Weight + col, col < DIM, other=0.0).to(tl.float32)
    normalized = value * inv[:, None] * weight[None, :]
    partner = tl.where(col < ROTARY // 2, col + ROTARY // 2, col - ROTARY // 2)
    rot_mask = (row[:, None] < ROWS) & (col[None, :] < ROTARY)
    pv = tl.load(X + offset + partner[None, :], rot_mask, other=0.0).to(tl.float32)
    pw = tl.load(Weight + partner, col < ROTARY, other=0.0).to(tl.float32)
    pn = pv * inv[:, None] * pw[None, :]
    rotated = tl.where(col[None, :] < ROTARY // 2, -pn, pn)
    cosine = tl.load(Cos + row[:, None] * ROTARY + col[None, :], rot_mask, other=1.0).to(tl.float32)
    sine = tl.load(Sin + row[:, None] * ROTARY + col[None, :], rot_mask, other=0.0).to(tl.float32)
    return tl.where(col[None, :] < ROTARY, normalized * cosine + rotated * sine,
                    normalized).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _pack_qk(Q, K, QW, KW, Cos, Sin, Payload, Scale, ROWS: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr,
             ROTARY: tl.constexpr, WORLD: tl.constexpr, CHUNKS: tl.constexpr, QEPS: tl.constexpr, KEPS: tl.constexpr,
             BLOCK_ROWS: tl.constexpr, BLOCK_DIM: tl.constexpr):
    row_head = (tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)).to(tl.int64)
    row, head = row_head // HEADS, row_head % HEADS
    col = tl.arange(0, BLOCK_DIM)
    projection = tl.program_id(1)
    if projection == 0:
        values = _normalize(Q, QW, Cos, Sin, row, head, HEADS, ROWS, DIM, ROTARY, QEPS, BLOCK_DIM)
    else:
        values = _normalize(K, KW, Cos, Sin, row, head, HEADS, ROWS, DIM, ROTARY, KEPS, BLOCK_DIM)
    maximum = tl.max(tl.abs(values), 1)
    scale = tl.maximum(maximum * (1.0 / 448.0), 1e-12)
    quantized = (values / scale[:, None]).to(tl.float8e4nv)
    local_heads: tl.constexpr = HEADS // WORLD
    chunk_heads: tl.constexpr = local_heads // CHUNKS
    chunk = head % local_heads // chunk_heads
    destination = ((
        (chunk * WORLD + head // local_heads) * 3 + projection) * ROWS + row) * chunk_heads + head % chunk_heads
    tl.store(Payload + destination[:, None] * DIM + col[None, :], quantized,
             (row[:, None] < ROWS) & (col[None, :] < DIM))
    tl.store(Scale + destination, scale, row < ROWS)

