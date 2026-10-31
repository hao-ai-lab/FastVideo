# SPDX-License-Identifier: Apache-2.0
"""Exact per-projection QKV packing for overlap with two-rank SP traffic."""

import torch
import triton
import triton.language as tl

from .sp_qkv_ordered import _normalize
from .sp_qkv_pack import _pack_value_fp8


@triton.jit
def _pack_one(X, Weight, Cos, Sin, Payload, Scale, ROWS: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr,
              ROTARY: tl.constexpr, WORLD: tl.constexpr, CHUNKS: tl.constexpr, EPS: tl.constexpr,
              PROJECTION: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_DIM: tl.constexpr):
    row_head = (tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)).to(tl.int64)
    row, head = row_head // HEADS, row_head % HEADS
    col = tl.arange(0, BLOCK_DIM)
    values = _normalize(X, Weight, Cos, Sin, row, head, HEADS, ROWS, DIM, ROTARY, EPS, BLOCK_DIM)
    maximum = tl.max(tl.abs(values), 1)
    scale = tl.maximum(maximum * (1.0 / 448.0), 1e-12)
    quantized = (values / scale[:, None]).to(tl.float8e4nv)
    local_heads: tl.constexpr = HEADS // WORLD
    chunk_heads: tl.constexpr = local_heads // CHUNKS
    chunk = head % local_heads // chunk_heads
    destination = ((
        (chunk * WORLD + head // local_heads) * 3 + PROJECTION) * ROWS + row) * chunk_heads + head % chunk_heads
    tl.store(Payload + destination[:, None] * DIM + col[None, :], quantized,
             (row[:, None] < ROWS) & (col[None, :] < DIM))
    tl.store(Scale + destination, scale, row < ROWS)


def allocate(rows, heads, world, chunks, device) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.empty((chunks, world, 3, rows, heads // (world * chunks), 128),
                          device=device,
                          dtype=torch.float8_e4m3fn)
    return payload, torch.empty(payload.shape[:-1], device=device, dtype=torch.float32)


def pack_one(x, weight, cosine, sine, payload, scale, epsilon, projection) -> None:
    chunks, world, _, rows, local_heads, dim = payload.shape
    heads = local_heads * world * chunks
    _pack_one[(triton.cdiv(rows * heads, 8), )](
        x,
        weight,
        cosine,
        sine,
        payload,
        scale,
        rows,
        heads,
        dim,
        cosine.shape[1],
        world,
        chunks,
        epsilon,
        projection,
        8,
        128,
        num_warps=8,
    )


def pack_value(x, payload, scale) -> None:
    chunks, world, _, rows, local_heads, dim = payload.shape
    heads = local_heads * world * chunks
    _pack_value_fp8[(triton.cdiv(rows * heads, 16), )](
        x,
        payload,
        scale,
        rows,
        heads,
        dim,
        world,
        chunks,
        16,
        128,
        num_warps=4,
    )


def project_and_exchange(attn, hidden, rotary, group, chunks, pre_quantized):
    """Start Q traffic before K's GEMM, and K traffic before V's GEMM.

    The caller checks that the three NVFP4 projections and gate can share
    packed input. Normalization, RoPE and FP8 division preserve the ordinary
    packer's arithmetic and reduction order. Nothing is reused across calls.
    """
    from fastvideo.distributed.projection_exchange import ProjectionExchange

    linears = (attn.to_q, attn.to_k, attn.to_v)
    pre = pre_quantized if pre_quantized is not None else linears[0].quant_method.quantize_input(hidden)
    rows, heads = hidden.shape[1], attn.num_attention_heads
    cosine, sine = (item.to(hidden.dtype) for item in rotary)
    payload, scale = allocate(rows, heads, 2, chunks, hidden.device)
    exchange = ProjectionExchange(payload, scale, group)
    for projection, linear in enumerate(linears):
        value = linear.quant_method.apply(linear, hidden, linear.bias, pre_quantized=pre)
        if projection == 2:
            pack_value(value, payload, scale)
        else:
            norm = attn.norm_q if projection == 0 else attn.norm_k
            pack_one(value, norm.weight, cosine, sine, payload, scale, norm.eps, projection)
        exchange.start(projection)
    return exchange.packets(), pre
