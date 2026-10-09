# SPDX-License-Identifier: Apache-2.0
"""Fused VSA64 tile scatter and BSHD->BHSD layout conversion for SM100 inference."""

import torch
import triton
import triton.language as tl


@triton.jit
def _tile_to_bhsd_kernel(
    x_ptr,
    source_index_ptr,
    output_ptr,
    sequence,
    padded_sequence,
    heads,
    head_dim: tl.constexpr,
    stride_batch,
    stride_sequence,
    stride_head,
    stride_dim,
    block: tl.constexpr,
):
    pid0 = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    linear = pid0 * block + tl.arange(0, block).to(tl.int64)
    batch = batch_head // heads
    head = batch_head % heads
    padded_pos = linear // head_dim
    dim = linear % head_dim
    valid = linear < padded_sequence * head_dim
    source_pos = tl.load(source_index_ptr + padded_pos, mask=valid, other=-1).to(tl.int64)
    x_offset = (batch * stride_batch + source_pos * stride_sequence + head * stride_head + dim * stride_dim)
    value = tl.load(x_ptr + x_offset, mask=valid & (source_pos >= 0) & (source_pos < sequence), other=0)
    out_offset = batch_head * padded_sequence * head_dim + linear
    tl.store(output_ptr + out_offset, value, mask=valid)


def tile_to_bhsd(
    x: torch.Tensor,
    source_index: torch.Tensor,
    out: torch.Tensor,
    sequence: int,
    padded_sequence: int,
    heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Scatter-tile ``x`` [B, S, H, D] into ``out`` [B, H, padded_S, D] in BHSD order."""
    block = 1024
    grid = (triton.cdiv(padded_sequence * head_dim, block), x.shape[0] * heads)
    _tile_to_bhsd_kernel[grid](
        x,
        source_index,
        out,
        sequence,
        padded_sequence,
        heads,
        head_dim,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        block,
        num_warps=4,
    )
    return out
