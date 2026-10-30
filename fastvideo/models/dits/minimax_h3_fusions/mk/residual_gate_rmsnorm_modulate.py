# SPDX-License-Identifier: Apache-2.0
"""MotionKernel H3 kernel: Fused H3 residual gate + RMSNorm + AdaLN modulation.

Found with MotionKernel (agent-driven kernel search). Tuned on GB200 / sm_100 for the
MiniMax-H3 inference shapes; opt-in through ``FASTVIDEO_MINIMAX_H3_FUSIONS=mk``.

Chunked gated residual + RMSNorm + AdaLN modulation (H3 post-attention).
"""

import torch
import triton
import triton.language as tl

H3_NORM_EPS = 1e-5

_BLOCK = 1024
_NUM_WARPS = 4
_NUM_STAGES = 3


@triton.jit
def _residual_gate_rmsnorm_modulate_kernel(
    hidden_out_ptr,
    normed_out_ptr,
    residual_ptr,
    branch_ptr,
    gate_ptr,
    weight_ptr,
    scale_ptr,
    shift_ptr,
    index_ptr,
    n_cols,
    n_index,
    eps,
    stride_gate,
    stride_scale,
    stride_shift,
    BLOCK: tl.constexpr,
    IDENTITY_INDEX: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    base = row * n_cols
    if IDENTITY_INDEX:
        table_row = tl.load(index_ptr + row).to(tl.int64)
    else:
        table_row = tl.load(index_ptr + (row % n_index)).to(tl.int64)
    gbase = table_row * stride_gate
    sbase = table_row * stride_scale
    hbase = table_row * stride_shift

    # Pass 1: gated residual, store it, accumulate the row sum of squares.
    acc = tl.zeros((BLOCK, ), tl.float32)
    for c in range(0, n_cols, BLOCK):
        cur = c + cols
        m = cur < n_cols
        off = base + cur
        residual = tl.load(residual_ptr + off, mask=m, other=0.0).to(tl.float32)
        branch = tl.load(branch_ptr + off, mask=m, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + gbase + cur, mask=m, other=0.0).to(tl.float32)
        hidden = residual + gate * branch
        tl.store(
            hidden_out_ptr + off,
            hidden.to(hidden_out_ptr.dtype.element_ty),
            mask=m,
        )
        acc += hidden * hidden

    rstd = tl.math.rsqrt(tl.sum(acc, axis=0) / n_cols + eps)

    # Pass 2: normalize + modulate. ``hidden`` is re-read from L2 (it was just
    # written) so no extra HBM traffic is generated.
    for c in range(0, n_cols, BLOCK):
        cur = c + cols
        m = cur < n_cols
        off = base + cur
        hidden = tl.load(hidden_out_ptr + off, mask=m, other=0.0, cache_modifier=".cg").to(tl.float32)
        weight = tl.load(weight_ptr + cur, mask=m, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + sbase + cur, mask=m, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + hbase + cur, mask=m, other=0.0).to(tl.float32)
        output = hidden * rstd * weight * (1.0 + scale) + shift
        tl.store(
            normed_out_ptr + off,
            output.to(normed_out_ptr.dtype.element_ty),
            mask=m,
        )


def _row_addressable(table: torch.Tensor) -> torch.Tensor:
    return table if table.stride(-1) == 1 else table.contiguous()


def residual_gate_rmsnorm_modulate(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(hidden, normed)`` exactly as FastVideo's optimized profile does."""
    if not residual.is_cuda:
        raise ValueError("h3_residual_gate_rmsnorm_modulate requires CUDA tensors")
    if branch.shape != residual.shape or branch.dtype != residual.dtype:
        raise ValueError("branch must match residual shape and dtype")
    hidden_size = residual.shape[-1]
    if index.ndim != 1 or index.numel() != residual.shape[-2]:
        raise ValueError("index must hold one table row per token")
    flat_residual = residual.reshape(-1, hidden_size).contiguous()
    flat_branch = branch.reshape(-1, hidden_size).contiguous()
    weight = weight.contiguous()
    gate = _row_addressable(gate)
    scale = _row_addressable(scale)
    shift = _row_addressable(shift)
    index = index.contiguous()
    hidden = torch.empty_like(flat_residual)
    modulated = torch.empty_like(flat_residual)
    _residual_gate_rmsnorm_modulate_kernel[(flat_residual.shape[0], )](
        hidden,
        modulated,
        flat_residual,
        flat_branch,
        gate,
        weight,
        scale,
        shift,
        index,
        hidden_size,
        index.numel(),
        H3_NORM_EPS,
        gate.stride(0),
        scale.stride(0),
        shift.stride(0),
        IDENTITY_INDEX=index.numel() == flat_residual.shape[0],
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return hidden.view_as(residual), modulated.view_as(residual)
