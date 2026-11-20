# SPDX-License-Identifier: Apache-2.0
"""Fuse H3 local Q/K normalization and RoPE with FP8 Ulysses packing."""

import torch
import triton
import triton.language as tl


@triton.jit
def _normalized_rotary(X, Weight, Cos, Sin, row, head, HEADS: tl.constexpr, DIM: tl.constexpr,
                        ROTARY: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    column = tl.arange(0, BLOCK)
    offset = (row * HEADS + head) * DIM
    value = tl.load(X + offset + column, column < DIM, other=0.0).to(tl.float32)
    variance = tl.sum(value * value, axis=0) / DIM
    inverse_rms = tl.math.rsqrt(variance + EPS)
    weight = tl.load(Weight + column, column < DIM, other=0.0).to(tl.float32)
    normalized = value * inverse_rms * weight
    partner = tl.where(column < ROTARY // 2, column + ROTARY // 2, column - ROTARY // 2)
    partner_value = tl.load(X + offset + partner, column < ROTARY, other=0.0).to(tl.float32)
    partner_weight = tl.load(Weight + partner, column < ROTARY, other=0.0).to(tl.float32)
    partner_normalized = partner_value * inverse_rms * partner_weight
    rotated = tl.where(column < ROTARY // 2, -partner_normalized, partner_normalized)
    cosine = tl.load(Cos + row * ROTARY + column, column < ROTARY, other=1.0).to(tl.float32)
    sine = tl.load(Sin + row * ROTARY + column, column < ROTARY, other=0.0).to(tl.float32)
    result = tl.where(column < ROTARY, normalized * cosine + rotated * sine, normalized)
    # The former standalone norm/RoPE kernel materialized BF16 here.
    return result.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _pack_normalized_qk(Q, K, QWeight, KWeight, Cos, Sin, Payload, Scale,
                         ROWS: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr,
                         ROTARY: tl.constexpr, WORLD: tl.constexpr, CHUNKS: tl.constexpr,
                         QEPS: tl.constexpr, KEPS: tl.constexpr, BLOCK: tl.constexpr, ROWS_PER_PROGRAM: tl.constexpr):
    local_heads: tl.constexpr = HEADS // WORLD
    chunk_heads: tl.constexpr = local_heads // CHUNKS
    # Preserve each head's four-warp reduction while amortizing CTA scheduling.
    for index in tl.static_range(ROWS_PER_PROGRAM):
        row_head = tl.program_id(0).to(tl.int64) * ROWS_PER_PROGRAM + index
        if ROWS * HEADS % ROWS_PER_PROGRAM == 0 or row_head < ROWS * HEADS:
            row, head = row_head // HEADS, row_head % HEADS
            projection = tl.program_id(1)
            column = tl.arange(0, BLOCK)
            if projection == 0:
                values = _normalized_rotary(Q, QWeight, Cos, Sin, row, head, HEADS, DIM, ROTARY, QEPS, BLOCK)
            else:
                values = _normalized_rotary(K, KWeight, Cos, Sin, row, head, HEADS, DIM, ROTARY, KEPS, BLOCK)
            maximum = tl.max(tl.where(column < DIM, tl.abs(values), 0.0), axis=0)
            scale = tl.maximum(maximum * (1.0 / 448.0), 1e-12)
            quantized = (values / scale).to(tl.float8e4nv)
            chunk = (head % local_heads) // chunk_heads
            destination = (((chunk * WORLD + head // local_heads) * 3 + projection) * ROWS + row) * chunk_heads + head % chunk_heads
            tl.store(Payload + destination * DIM + column, quantized, column < DIM)
            tl.store(Scale + destination, scale)



@triton.jit
def _pack_value_fp8(V, Payload, Scale, ROWS: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr,
                    WORLD: tl.constexpr, CHUNKS: tl.constexpr, BLOCK_HEADS: tl.constexpr, BLOCK_DIM: tl.constexpr):
    row_head = (tl.program_id(0) * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)).to(tl.int64)
    column = tl.arange(0, BLOCK_DIM)
    valid = (row_head[:, None] < ROWS * HEADS) & (column[None, :] < DIM)
    values = tl.load(V + row_head[:, None] * DIM + column[None, :], valid, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), axis=1) * (1.0 / 448.0), 1e-12)
    quantized = (values / scale[:, None]).to(tl.float8e4nv)
    row, head = row_head // HEADS, row_head % HEADS
    local_heads: tl.constexpr = HEADS // WORLD
    chunk_heads: tl.constexpr = local_heads // CHUNKS
    chunk = (head % local_heads) // chunk_heads
    destination = (((chunk * WORLD + head // local_heads) * 3 + 2) * ROWS + row) * chunk_heads + head % chunk_heads
    tl.store(Payload + destination[:, None] * DIM + column[None, :], quantized, valid)
    tl.store(Scale + destination, scale, row_head < ROWS * HEADS)


def pack_normalized_qkv_fp8(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                            q_weight: torch.Tensor, k_weight: torch.Tensor,
                            cos: torch.Tensor, sin: torch.Tensor, q_eps: float, k_eps: float,
                            world: int, head_chunks: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack local `[rows,H,D]` after BF16 norm/RoPE rounding.

    With multiple chunks, write `[chunks,W,3,rows,H/W/chunks,D]` directly so
    each chunk's all-to-all buffer is contiguous without another memory pass.
    """
    rows, heads, dim = query.shape
    assert query.dtype == key.dtype == value.dtype == torch.bfloat16
    assert q_weight.dtype == k_weight.dtype == cos.dtype == sin.dtype == query.dtype
    assert query.shape == key.shape == value.shape and head_chunks > 0 and heads % (world * head_chunks) == 0
    assert all(t.is_contiguous() for t in (query, key, value, q_weight, k_weight, cos, sin))
    assert q_weight.shape == k_weight.shape == (dim,)
    assert cos.shape == sin.shape and cos.shape[0] == rows and 0 < cos.shape[1] <= dim
    assert cos.shape[1] % 2 == 0
    payload = torch.empty((head_chunks, world, 3, rows, heads // (world * head_chunks), dim), device=query.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((head_chunks, world, 3, rows, heads // (world * head_chunks)), device=query.device, dtype=torch.float32)
    rows_per_program = 8 if dim == 128 and torch.cuda.get_device_capability(query.device)[0] >= 10 else 1
    if dim == 128 and torch.cuda.get_device_capability(query.device)[0] >= 10:
        from .sp_qkv_ordered import _pack_qk
        _pack_qk[(triton.cdiv(rows * heads, 8), 2)](
            query, key, q_weight, k_weight, cos, sin, payload, scales,
            rows, heads, dim, cos.shape[1], world, head_chunks, q_eps, k_eps, 8, 128, num_warps=8,
        )
    else:
        _pack_normalized_qk[(triton.cdiv(rows * heads, rows_per_program), 2)](
            query, key, q_weight, k_weight, cos, sin, payload, scales,
            rows, heads, dim, cos.shape[1], world, head_chunks, q_eps, k_eps,
            triton.next_power_of_2(dim), rows_per_program, num_warps=4,
        )
    _pack_value_fp8[(triton.cdiv(rows * heads, 16),)](
        value, payload, scales, rows, heads, dim, world, head_chunks, 16, triton.next_power_of_2(dim), num_warps=4,
    )
    return (payload[0], scales[0]) if head_chunks == 1 else (payload, scales)
