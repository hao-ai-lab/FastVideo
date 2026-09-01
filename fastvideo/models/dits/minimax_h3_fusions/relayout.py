# SPDX-License-Identifier: Apache-2.0
"""Compile-safe MiniMax-H3 Ulysses relayout kernels adapted from Sol-Engine."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _pack_qkv_kernel(
        out_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        total_elements,
        rows,
        heads_local,
        head_dim,
        stride_q_row,
        stride_q_head,
        stride_k_row,
        stride_k_head,
        stride_v_row,
        stride_v_head,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total_elements
        dim = offsets % head_dim
        head_slot = offsets // head_dim
        local_head = head_slot % heads_local
        row_slot = head_slot // heads_local
        row = row_slot % rows
        destination = row_slot // rows
        global_head = destination * heads_local + local_head

        q = tl.load(q_ptr + row * stride_q_row + global_head * stride_q_head + dim, mask=mask)
        k = tl.load(k_ptr + row * stride_k_row + global_head * stride_k_head + dim, mask=mask)
        v = tl.load(v_ptr + row * stride_v_row + global_head * stride_v_head + dim, mask=mask)
        base = head_slot * (3 * head_dim) + dim
        tl.store(out_ptr + base, q, mask=mask)
        tl.store(out_ptr + base + head_dim, k, mask=mask)
        tl.store(out_ptr + base + 2 * head_dim, v, mask=mask)

    @triton.jit
    def _merge_heads_kernel(out_ptr, x_ptr, total_elements, world, rows, inner, BLOCK: tl.constexpr):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total_elements
        tail = offsets % inner
        slot = offsets // inner
        source = slot % world
        row = slot // world
        src = (source * rows + row) * inner + tail
        tl.store(out_ptr + offsets, tl.load(x_ptr + src, mask=mask), mask=mask)

    @torch.library.triton_op("fastvideo::minimax_h3_pack_qkv", mutates_args={})
    def _pack_qkv_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, world: int) -> torch.Tensor:
        rows, heads, head_dim = q.shape
        heads_local = heads // world
        out = torch.empty((world, rows, heads_local, 3 * head_dim), dtype=q.dtype, device=q.device)
        total = rows * heads * head_dim
        torch.library.wrap_triton(_pack_qkv_kernel)[(triton.cdiv(total, 1024), )](
            out,
            q,
            k,
            v,
            total,
            rows,
            heads_local,
            head_dim,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            BLOCK=1024,
            num_warps=8,
        )
        return out

    @torch.library.triton_op("fastvideo::minimax_h3_merge_heads", mutates_args={})
    def _merge_heads_op(x: torch.Tensor) -> torch.Tensor:
        world, rows, heads_local, head_dim = x.shape
        out = torch.empty((rows, world, heads_local, head_dim), dtype=x.dtype, device=x.device)
        total = out.numel()
        torch.library.wrap_triton(_merge_heads_kernel)[(triton.cdiv(total, 1024), )](
            out,
            x,
            total,
            world,
            rows,
            heads_local * head_dim,
            BLOCK=1024,
            num_warps=8,
        )
        return out


def pack_qkv_destination_major(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, world: int) -> torch.Tensor:
    """Move three ``(rows, heads, dim)`` tensors into destination-major QKV in one pass."""
    if not HAVE_TRITON:
        raise RuntimeError("MiniMax-H3 packed sequence parallelism requires Triton")
    if world < 1:
        raise ValueError(f"sequence parallel world size must be positive, got {world}")
    if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have matching (rows, heads, head_dim) shapes")
    if q.device != k.device or q.device != v.device:
        raise ValueError(f"q, k, and v must be on one device, got {q.device}, {k.device}, and {v.device}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"q, k, and v must have one dtype, got {q.dtype}, {k.dtype}, and {v.dtype}")
    if not q.is_cuda:
        raise ValueError("MiniMax-H3 packed QKV relayout requires CUDA tensors")
    if q.shape[1] % world:
        raise ValueError(f"heads ({q.shape[1]}) must be divisible by sequence parallel size ({world})")
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError("q, k, and v must be contiguous in head_dim")
    return _pack_qkv_op(q, k, v, world)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Move source-major all-to-all output back to row-major head order in one pass."""
    if not HAVE_TRITON:
        raise RuntimeError("MiniMax-H3 packed sequence parallelism requires Triton")
    if x.ndim != 4 or not x.is_contiguous():
        raise ValueError("packed all-to-all output must be a contiguous 4D tensor")
    if not x.is_cuda:
        raise ValueError("MiniMax-H3 packed head relayout requires a CUDA tensor")
    world, rows, heads_local, head_dim = x.shape
    return _merge_heads_op(x).reshape(rows, world * heads_local, head_dim)


__all__ = ["HAVE_TRITON", "merge_heads", "pack_qkv_destination_major"]
