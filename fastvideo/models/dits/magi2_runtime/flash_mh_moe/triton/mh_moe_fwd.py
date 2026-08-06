# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""Triton kernel for multi-head MoE forward (inference).

Multi-head MoE forward: gather → GEMM(gate,up) → SwiGLU7 → GEMM(down) → scatter.
Supports deterministic mode (sequential scatter_back) for bit-exact reproducibility.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

__all__ = ["mh_moe_fwd_func"]

SWIGLU7_ALPHA = tl.constexpr(1.702)
SWIGLU7_LIMIT = tl.constexpr(7.0)
SWIGLU7_BIAS = tl.constexpr(1.0)


@triton.jit
def _swiglu7_fwd(gate, up, out_dtype: tl.constexpr):
    gate_clamped = tl.minimum(gate, SWIGLU7_LIMIT)
    up_clamped = tl.maximum(tl.minimum(up, SWIGLU7_LIMIT), -SWIGLU7_LIMIT)
    sig = tl.sigmoid(SWIGLU7_ALPHA * gate_clamped)
    swish = gate_clamped * sig
    return (swish * (up_clamped + SWIGLU7_BIAS)).to(out_dtype)


@triton.jit
def _binary_search_expert_id(
    cu_expert_tile_counts_ptr,
    tile_id,
    NUM_EXPERTS: tl.constexpr,
    LOG2_NUM_EXPERTS: tl.constexpr,
):
    """Map a flattened output tile to the expert that owns the tile."""
    lo = 0
    hi = NUM_EXPERTS
    for _ in tl.static_range(0, LOG2_NUM_EXPERTS + 1):
        mid = (lo + hi + 1) // 2
        below = tl.load(cu_expert_tile_counts_ptr + mid) <= tile_id
        lo = tl.where(below, mid, lo)
        hi = tl.where(below, hi, mid - 1)
    return lo


@triton.jit
def _mh_moe_fwd_kernel(
    x_ptr,
    wg_ptr,
    wu_ptr,
    wd_ptr,
    y_ptr,
    gather_ids_ptr,
    probs_ptr,
    expert_offsets_ptr,
    cu_expert_tile_counts_ptr,
    stride_x_s,
    stride_x_h,
    stride_x_dh,
    stride_wg_ne,
    stride_wg_dh,
    stride_wg_de,
    stride_wu_ne,
    stride_wu_dh,
    stride_wu_de,
    stride_wd_ne,
    stride_wd_de,
    stride_wd_dh,
    stride_y_s,
    stride_y_h,
    stride_y_dh,
    D_HEAD: tl.constexpr,
    D_EXPERT: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    LOG2_NUM_EXPERTS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_DH: tl.constexpr,
    BLOCK_DE: tl.constexpr,
    ACC_DTYPE: tl.constexpr = tl.float32,
    DETERMINISTIC: tl.constexpr = False,
):
    """Run the fused gather, expert SwiGLU7 projections, and output scatter."""
    tile_id = tl.program_id(0)

    total_tiles = tl.load(cu_expert_tile_counts_ptr + NUM_EXPERTS)
    if tile_id >= total_tiles:
        return

    expert_id = _binary_search_expert_id(
        cu_expert_tile_counts_ptr, tile_id, NUM_EXPERTS, LOG2_NUM_EXPERTS
    )
    expert_id_i64 = expert_id.to(tl.int64)
    head_id = expert_id // (NUM_EXPERTS // NUM_HEADS)

    tile_in_expert = tile_id - tl.load(cu_expert_tile_counts_ptr + expert_id)
    token_start = tl.load(expert_offsets_ptr + expert_id) + tile_in_expert * BLOCK_T
    expert_end = tl.load(expert_offsets_ptr + expert_id + 1)
    n = tl.minimum(token_start + BLOCK_T, expert_end) - token_start

    block_dh_offs = tl.arange(0, BLOCK_DH)
    block_de_offs = tl.arange(0, BLOCK_DE)
    block_t_offs = tl.arange(0, BLOCK_T)
    dh_offs = tl.arange(0, D_HEAD)

    block_token_offs = token_start + block_t_offs
    block_token_mask = block_t_offs < n

    gather_ids = tl.load(
        gather_ids_ptr + block_token_offs, mask=block_token_mask, other=0
    )
    probs = tl.load(probs_ptr + block_token_offs, mask=block_token_mask, other=0.0)
    x_offs = gather_ids * stride_x_s + head_id * stride_x_h

    y_acc = tl.zeros([BLOCK_T, D_HEAD], dtype=ACC_DTYPE)
    for dexpert_start in tl.range(0, D_EXPERT, BLOCK_DE):
        dexpert_block_offs = dexpert_start + block_de_offs

        gate_acc = tl.zeros([BLOCK_T, BLOCK_DE], dtype=ACC_DTYPE)
        up_acc = tl.zeros([BLOCK_T, BLOCK_DE], dtype=ACC_DTYPE)
        for dhead_start in tl.static_range(0, D_HEAD, BLOCK_DH):
            dhead_block_offs = dhead_start + block_dh_offs

            x_block = tl.load(
                x_ptr + x_offs[:, None] + dhead_block_offs[None, :] * stride_x_dh,
                mask=block_token_mask[:, None],
                other=0.0,
            )
            w_gate = tl.load(
                wg_ptr
                + expert_id_i64 * stride_wg_ne
                + dhead_block_offs[:, None] * stride_wg_dh
                + dexpert_block_offs[None, :] * stride_wg_de
            )
            w_up = tl.load(
                wu_ptr
                + expert_id_i64 * stride_wu_ne
                + dhead_block_offs[:, None] * stride_wu_dh
                + dexpert_block_offs[None, :] * stride_wu_de
            )

            gate_acc += tl.dot(x_block, w_gate)
            up_acc += tl.dot(x_block, w_up)

        hidden = _swiglu7_fwd(gate_acc, up_acc, wd_ptr.dtype.element_ty)

        w_down = tl.load(
            wd_ptr
            + expert_id_i64 * stride_wd_ne
            + dexpert_block_offs[:, None] * stride_wd_de
            + dh_offs[None, :] * stride_wd_dh
        )

        y_acc += tl.dot(hidden, w_down)

    y_acc = y_acc * probs[:, None]

    if DETERMINISTIC:
        # Store to flat (T, D_HEAD) buffer; host scatter_back accumulates deterministically.
        y_ptrs = (
            y_ptr
            + block_token_offs[:, None] * stride_y_s
            + dh_offs[None, :] * stride_y_dh
        )
        tl.store(
            y_ptrs,
            y_acc.to(y_ptr.dtype.element_ty),
            mask=block_token_mask[:, None],
        )
    else:
        # Scatter-reduce by atomic add into (S, H, D_HEAD).
        y_offs = gather_ids * stride_y_s + head_id * stride_y_h
        y_ptrs = y_ptr + y_offs[:, None] + dh_offs[None, :] * stride_y_dh
        tl.atomic_add(
            y_ptrs,
            y_acc.to(y_ptr.dtype.element_ty),
            mask=block_token_mask[:, None],
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Host-side scatter_back (deterministic path)
# ═══════════════════════════════════════════════════════════════════════════


def _scatter_back(
    y_sorted: torch.Tensor,
    ref_y: torch.Tensor,
    gather_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    num_flatten_experts: int,
    num_heads: int,
) -> torch.Tensor:
    """Scatter prob-weighted expert outputs back to (S, H, D) deterministically."""
    D = ref_y.size(2)
    num_experts_per_head = num_flatten_experts // num_heads

    y = torch.zeros_like(ref_y).view(-1, D)  # (S*H, D)

    expert_seqlens = expert_offsets.diff()
    head_id_values = torch.arange(num_flatten_experts, device=gather_ids.device) // num_experts_per_head
    head_ids = torch.repeat_interleave(head_id_values, expert_seqlens)

    if num_heads == 1:
        scatter_idx = gather_ids.long()
    else:
        scatter_idx = gather_ids.long() * num_heads + head_ids.long()

    scatter_idx = scatter_idx.unsqueeze(-1).expand_as(y_sorted)
    y.scatter_add_(0, scatter_idx, y_sorted.to(y.dtype))

    return y.view_as(ref_y)


# ═══════════════════════════════════════════════════════════════════════════
#  Public forward function
# ═══════════════════════════════════════════════════════════════════════════


def _select_block_config() -> tuple[int, int, int, int, int]:
    """(BLOCK_T, BLOCK_DH, BLOCK_DE, NUM_STAGES, NUM_WARPS) for d_head=256."""
    major, _ = torch.cuda.get_device_capability()
    if major >= 10:  # Blackwell
        return (128, 64, 32, 2, 8)
    return (128, 64, 32, 2, 8)  # Hopper


def mh_moe_fwd_func(
    x: torch.Tensor,
    gather_ids: torch.Tensor,
    probs: torch.Tensor,
    expert_offsets: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    deterministic: bool = False,
) -> torch.Tensor:
    """Triton multi-head MoE forward: gather → gate/up GEMM → SwiGLU7 → down GEMM → scatter.

    When ``deterministic=True``, the kernel writes per-expert outputs to a flat buffer
    and the host performs a sequential scatter_back for bit-exact reproducibility.

    Args:
        x: (S, H, d_head)
        gather_ids: (T,) int32
        probs: (T,) float32
        expert_offsets: (H*E+1,) int32/int64
        W_gate: (H*E, d_head, d_expert)
        W_up: (H*E, d_head, d_expert)
        W_down: (H*E, d_expert, d_head)
        deterministic: use sequential scatter for reproducibility
    """
    T = gather_ids.size(0)
    if T == 0:
        return torch.zeros_like(x)

    num_heads = x.size(1)
    d_head, d_expert = x.size(2), W_down.size(1)

    BLOCK_T, BLOCK_DH, BLOCK_DE, NUM_STAGES, NUM_WARPS = _select_block_config()
    assert d_head % BLOCK_DH == 0, f"d_head={d_head} must be divisible by {BLOCK_DH}"
    assert d_expert % BLOCK_DE == 0, f"d_expert={d_expert} must be divisible by {BLOCK_DE}"

    if deterministic:
        y = torch.empty(T, 1, d_head, device=x.device, dtype=x.dtype)
    else:
        y = torch.zeros_like(x)

    num_flatten_experts = expert_offsets.size(0) - 1
    num_flatten_experts_log2 = max(1, math.ceil(math.log2(max(num_flatten_experts, 1) + 1)))

    expert_token_counts = expert_offsets.diff()
    expert_tile_counts = (expert_token_counts + BLOCK_T - 1) // BLOCK_T
    cu_expert_tile_counts = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=expert_offsets.device),
        torch.cumsum(expert_tile_counts, dim=0, dtype=torch.int32),
    ])

    grid_size = (T + BLOCK_T - 1) // BLOCK_T + num_flatten_experts

    _mh_moe_fwd_kernel[(grid_size,)](
        x_ptr=x,
        wg_ptr=W_gate,
        wu_ptr=W_up,
        wd_ptr=W_down,
        y_ptr=y,
        gather_ids_ptr=gather_ids,
        probs_ptr=probs,
        expert_offsets_ptr=expert_offsets,
        cu_expert_tile_counts_ptr=cu_expert_tile_counts,
        stride_x_s=x.stride(0),
        stride_x_h=x.stride(1),
        stride_x_dh=x.stride(2),
        stride_wg_ne=W_gate.stride(0),
        stride_wg_dh=W_gate.stride(1),
        stride_wg_de=W_gate.stride(2),
        stride_wu_ne=W_up.stride(0),
        stride_wu_dh=W_up.stride(1),
        stride_wu_de=W_up.stride(2),
        stride_wd_ne=W_down.stride(0),
        stride_wd_de=W_down.stride(1),
        stride_wd_dh=W_down.stride(2),
        stride_y_s=y.stride(0),
        stride_y_h=y.stride(1),
        stride_y_dh=y.stride(2),
        D_HEAD=d_head,
        D_EXPERT=d_expert,
        NUM_HEADS=num_heads,
        NUM_EXPERTS=num_flatten_experts,
        LOG2_NUM_EXPERTS=num_flatten_experts_log2,
        BLOCK_T=BLOCK_T,
        BLOCK_DH=BLOCK_DH,
        BLOCK_DE=BLOCK_DE,
        ACC_DTYPE=tl.float32,
        DETERMINISTIC=deterministic,
        num_stages=NUM_STAGES,
        num_warps=NUM_WARPS,
    )

    if deterministic:
        y = _scatter_back(
            y_sorted=y.view(T, d_head),
            ref_y=x,
            gather_ids=gather_ids,
            expert_offsets=expert_offsets,
            num_flatten_experts=num_flatten_experts,
            num_heads=num_heads,
        )

    return y
