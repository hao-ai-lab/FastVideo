# SPDX-License-Identifier: Apache-2.0
"""Exact residual modulation fused with NVFP4 activation preparation."""
import torch
import triton
import triton.language as tl

from .modulation import (
    _num_warps,
    _require_forward_only,
    _require_triton_cuda,
    _row_addressable,
    _validate_contract,
    _validate_residual_branch,
)
from .swiglu_quant import _rcp as rcp


@triton.jit
def _modulate_quant(
    hidden_out_ptr,
    Q, S, SF,
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
    stride_input_row,
    stride_gate_row,
    stride_scale_row,
    stride_shift_row,
    BLOCK: tl.constexpr,
    WIDTH: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    ROUND_RESIDUAL: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    input_offsets = row * stride_input_row + cols
    table_row = tl.load(index_ptr + row % n_index).to(tl.int64)

    hidden = tl.load(residual_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    if ADD_RESIDUAL:
        branch = tl.load(branch_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(
            gate_ptr + table_row * stride_gate_row + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        hidden = hidden + gate * branch
        tl.store(hidden_out_ptr + input_offsets, hidden.to(hidden_out_ptr.dtype.element_ty), mask=mask)

    if ROUND_RESIDUAL:
        hidden = hidden.to(tl.bfloat16).to(tl.float32)
    variance = tl.sum(hidden * hidden, axis=0) / n_cols
    normed = hidden * tl.math.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(
        scale_ptr + table_row * stride_scale_row + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    shift = tl.load(
        shift_ptr + table_row * stride_shift_row + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    output = normed * weight * (1.0 + scale) + shift
    y = tl.where(mask, output.to(tl.bfloat16).to(tl.float32), 0.0)
    grouped = tl.reshape(y, (BLOCK // 16, 16))
    amax = tl.max(tl.abs(grouped), 1)
    global_sf = tl.load(SF)
    sf8 = (global_sf * (amax * rcp(6.0))).to(tl.float8e4nv)
    inv = tl.where(amax != 0, rcp(sf8.to(tl.float32) * rcp(global_sf)), 0.0)
    quant = tl.reshape(grouped * inv[:, None], (BLOCK // 2, 2))
    lo, hi = tl.split(quant)
    packed = tl.inline_asm_elementwise('''{
        .reg .b8 q;
        cvt.rn.satfinite.e2m1x2.f32 q, $1, $2;
        mov.b32 $0, {q,q,q,q};
    }''', constraints="=r,f,f", args=[hi, lo], dtype=tl.uint8, is_pure=True, pack=1)
    qc = tl.arange(0, BLOCK // 2)
    tl.store(Q + row * (WIDTH // 2) + qc, packed, qc < WIDTH // 2)
    group = tl.arange(0, BLOCK // 16)
    scale_tile = (row // 128) * (WIDTH // 64) + group // 4
    si = ((scale_tile * 32 + row % 32) * 4 + (row // 32) % 4) * 4 + group % 4
    tl.store(S + si, sf8.to(tl.uint8, bitcast=True), group < WIDTH // 16)


def residual_modulate_quantize_nvfp4(
    residual: torch.Tensor, branch: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor,
    scale: torch.Tensor, shift: torch.Tensor, index: torch.Tensor, eps: float,
    global_scale: torch.Tensor,
    *, round_residual_before_norm: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return the updated residual and packed FP4 modulated activation.

    The FP32 residual participates in RMSNorm before its BF16 output cast,
    matching the existing fused residual/modulation kernel exactly. The
    modulated activation is rounded to BF16 before FP4 preparation.
    Across transformer blocks, round the residual to BF16 before RMSNorm too,
    preserving the separate block output's rounding boundary.
    """
    _validate_residual_branch(residual, branch)
    _validate_contract(residual, weight, (gate, scale, shift), index, eps)
    _require_forward_only(residual, branch, gate, weight, scale, shift)
    _require_triton_cuda(residual)
    width = residual.shape[-1]
    if residual.dtype != torch.bfloat16 or width % 64:
        raise ValueError("NVFP4 modulation requires BF16 with a width divisible by 64")
    if (global_scale.numel() != 1 or global_scale.dtype != torch.float32
            or global_scale.device != residual.device):
        raise ValueError("NVFP4 modulation requires a scalar FP32 scale on the input device")
    flat = residual.reshape(-1, width).contiguous()
    branch = branch.reshape(-1, width).contiguous()
    gate, scale, shift = (_row_addressable(t) for t in (gate, scale, shift))
    rows = flat.shape[0]
    hidden = torch.empty_like(flat)
    codes = torch.empty((rows, width // 2), dtype=torch.uint8, device=residual.device)
    scales = torch.zeros((triton.cdiv(rows, 128) * 128, width // 16), dtype=torch.uint8,
                         device=residual.device)
    block = triton.next_power_of_2(width)
    _modulate_quant[(rows,)](
        hidden, codes, scales, global_scale, flat, branch, gate, weight.contiguous(),
        scale, shift, index.contiguous(), width, index.numel(), eps, flat.stride(0),
        gate.stride(0), scale.stride(0), shift.stride(0), BLOCK=block, WIDTH=width,
        ADD_RESIDUAL=True, ROUND_RESIDUAL=round_residual_before_norm, num_warps=_num_warps(block),
    )
    return hidden.view_as(residual), (codes, scales, global_scale)


def modulate_quantize_nvfp4(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
    index: torch.Tensor, eps: float, global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm and row-indexed modulation followed by exact BF16-to-NVFP4 preparation."""
    _validate_contract(x, weight, (scale, shift), index, eps)
    _require_forward_only(x, weight, scale, shift)
    _require_triton_cuda(x)
    width = x.shape[-1]
    if x.dtype != torch.bfloat16 or width % 64:
        raise ValueError("NVFP4 modulation requires BF16 with a width divisible by 64")
    if global_scale.numel() != 1 or global_scale.dtype != torch.float32 or global_scale.device != x.device:
        raise ValueError("NVFP4 modulation requires a scalar FP32 scale on the input device")
    flat = x.reshape(-1, width).contiguous()
    scale, shift = (_row_addressable(t) for t in (scale, shift))
    rows = flat.shape[0]
    codes = torch.empty((rows, width // 2), dtype=torch.uint8, device=x.device)
    scales = torch.zeros((triton.cdiv(rows, 128) * 128, width // 16), dtype=torch.uint8, device=x.device)
    block = triton.next_power_of_2(width)
    _modulate_quant[(rows,)](
        flat, codes, scales, global_scale, flat, flat, scale, weight.contiguous(),
        scale, shift, index.contiguous(), width, index.numel(), eps, flat.stride(0),
        scale.stride(0), scale.stride(0), shift.stride(0), BLOCK=block, WIDTH=width,
        ADD_RESIDUAL=False, num_warps=_num_warps(block),
    )
    return codes, scales, global_scale
