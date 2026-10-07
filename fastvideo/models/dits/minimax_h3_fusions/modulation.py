# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 RMSNorm and row-indexed modulation fusions."""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - depends on the runtime image
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR: ImportError | None = exc
else:
    _TRITON_IMPORT_ERROR = None


__all__ = [
    "fused_residual_gate_rmsnorm_modulate",
    "fused_rmsnorm_modulate",
]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_rmsnorm_modulate_kernel = None
_residual_gate_rmsnorm_modulate_kernel = None


if triton is not None:

    @triton.jit
    def _rmsnorm_modulate_kernel(
        out_ptr,
        x_ptr,
        weight_ptr,
        scale_ptr,
        shift_ptr,
        index_ptr,
        n_cols,
        n_index,
        eps,
        stride_x_row,
        stride_scale_row,
        stride_shift_row,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x_offsets = row * stride_x_row + cols
        table_row = tl.load(index_ptr + row % n_index).to(tl.int64)

        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / n_cols
        normed = x * tl.math.rsqrt(variance + eps)
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
        tl.store(out_ptr + x_offsets, output.to(out_ptr.dtype.element_ty), mask=mask)

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
        stride_input_row,
        stride_gate_row,
        stride_scale_row,
        stride_shift_row,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        input_offsets = row * stride_input_row + cols
        table_row = tl.load(index_ptr + row % n_index).to(tl.int64)

        residual = tl.load(residual_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        branch = tl.load(branch_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(
            gate_ptr + table_row * stride_gate_row + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        hidden = residual + gate * branch
        tl.store(hidden_out_ptr + input_offsets, hidden.to(hidden_out_ptr.dtype.element_ty), mask=mask)

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
        tl.store(normed_out_ptr + input_offsets, output.to(normed_out_ptr.dtype.element_ty), mask=mask)


def _validate_contract(
    x: torch.Tensor,
    weight: torch.Tensor,
    tables: tuple[torch.Tensor, ...],
    index: torch.Tensor,
    eps: float,
) -> None:
    if x.ndim < 2:
        raise ValueError(f"x must have shape (..., sequence_length, hidden_size), got {tuple(x.shape)}.")
    if x.numel() == 0 or x.shape[-1] == 0:
        raise ValueError("x must not be empty.")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"x must use float16, bfloat16, or float32, got {x.dtype}.")
    hidden_size = x.shape[-1]
    sequence_length = x.shape[-2]
    if weight.shape != (hidden_size, ):
        raise ValueError(f"weight must have shape ({hidden_size},), got {tuple(weight.shape)}.")
    if weight.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"weight must use float16, bfloat16, or float32, got {weight.dtype}.")
    if index.ndim != 1 or index.numel() != sequence_length:
        raise ValueError(
            f"index must have shape ({sequence_length},) so it can wrap over batch rows, got {tuple(index.shape)}."
        )
    if index.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"index must use int32 or int64, got {index.dtype}.")
    if not isinstance(eps, (float, int)) or isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be a positive finite number, got {eps!r}.")

    table_rows = tables[0].shape[0] if tables and tables[0].ndim == 2 else None
    for name, table in zip(("gate", "scale", "shift")[-len(tables):], tables, strict=True):
        if table.ndim != 2 or table.shape[1] != hidden_size:
            raise ValueError(f"{name} must have shape (table_rows, {hidden_size}), got {tuple(table.shape)}.")
        if table.shape[0] == 0 or table.shape[0] != table_rows:
            raise ValueError("all modulation tables must have the same non-zero row count.")
        if table.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{name} must use float16, bfloat16, or float32, got {table.dtype}.")

    tensors = (x, weight, *tables, index)
    if any(tensor.device != x.device for tensor in tensors[1:]):
        raise ValueError("x, weight, modulation tables, and index must be on the same device.")


def _validate_residual_branch(residual: torch.Tensor, branch: torch.Tensor) -> None:
    if branch.shape != residual.shape:
        raise ValueError(f"branch must match residual shape {tuple(residual.shape)}, got {tuple(branch.shape)}.")
    if branch.dtype != residual.dtype:
        raise TypeError(f"branch dtype must match residual dtype {residual.dtype}, got {branch.dtype}.")
    if branch.device != residual.device:
        raise ValueError("branch and residual must be on the same device.")


def _require_triton_cuda(x: torch.Tensor) -> None:
    if triton is None:
        detail = f": {_TRITON_IMPORT_ERROR}" if _TRITON_IMPORT_ERROR is not None else ""
        raise RuntimeError(f"MiniMax H3 modulation fusion requires Triton{detail}.")
    if x.device.type != "cuda":
        raise RuntimeError(f"MiniMax H3 modulation fusion requires CUDA tensors, got device {x.device}.")


def _require_forward_only(*tensors: torch.Tensor) -> None:
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError("MiniMax H3 modulation fusion is forward-only and does not support autograd.")


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _num_warps(block_size: int) -> int:
    if block_size >= 8192:
        return 16
    if block_size >= 2048:
        return 8
    return 4


def _row_addressable(table: torch.Tensor) -> torch.Tensor:
    return table if table.stride(-1) == 1 else table.contiguous()


def _fused_rmsnorm_modulate_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    _validate_contract(x, weight, (scale, shift), index, eps)
    _require_forward_only(x, weight, scale, shift)
    _require_triton_cuda(x)

    hidden_size = x.shape[-1]
    flat_x = x.reshape(-1, hidden_size).contiguous()
    weight = weight.contiguous()
    scale = _row_addressable(scale)
    shift = _row_addressable(shift)
    index = index.contiguous()
    output = torch.empty_like(flat_x)
    block_size = _next_power_of_two(hidden_size)
    _rmsnorm_modulate_kernel[(flat_x.shape[0], )](
        output,
        flat_x,
        weight,
        scale,
        shift,
        index,
        hidden_size,
        index.numel(),
        eps,
        flat_x.stride(0),
        scale.stride(0),
        shift.stride(0),
        BLOCK=block_size,
        num_warps=_num_warps(block_size),
    )
    return output.view_as(x)


def _fused_residual_gate_rmsnorm_modulate_impl(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_residual_branch(residual, branch)
    _validate_contract(residual, weight, (gate, scale, shift), index, eps)
    _require_forward_only(residual, branch, gate, weight, scale, shift)
    _require_triton_cuda(residual)

    hidden_size = residual.shape[-1]
    flat_residual = residual.reshape(-1, hidden_size).contiguous()
    flat_branch = branch.reshape(-1, hidden_size).contiguous()
    weight = weight.contiguous()
    gate = _row_addressable(gate)
    scale = _row_addressable(scale)
    shift = _row_addressable(shift)
    index = index.contiguous()
    hidden = torch.empty_like(flat_residual)
    modulated = torch.empty_like(flat_residual)
    block_size = _next_power_of_two(hidden_size)
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
        eps,
        flat_residual.stride(0),
        gate.stride(0),
        scale.stride(0),
        shift.stride(0),
        BLOCK=block_size,
        num_warps=_num_warps(block_size),
    )
    return hidden.view_as(residual), modulated.view_as(residual)


# Dynamo must see the Triton launches as opaque nodes.  In eager mode the
# public wrappers below keep their strict, actionable input validation; while
# compiling, these custom-op boundaries avoid tracing into Triton's launcher
# and let the surrounding H3 block remain one full graph.
@torch.library.custom_op(
    "fastvideo::_minimax_h3_rmsnorm_modulate",
    mutates_args=(),
    device_types="cuda",
)
def _fused_rmsnorm_modulate_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return _fused_rmsnorm_modulate_impl(x, weight, scale, shift, index, eps)


@torch.library.register_fake("fastvideo::_minimax_h3_rmsnorm_modulate")
def _fused_rmsnorm_modulate_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del weight, scale, shift, index, eps
    return x.new_empty(x.shape)


@torch.library.custom_op(
    "fastvideo::_minimax_h3_residual_gate_rmsnorm_modulate",
    mutates_args=(),
    device_types="cuda",
)
def _fused_residual_gate_rmsnorm_modulate_op(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _fused_residual_gate_rmsnorm_modulate_impl(
        residual,
        branch,
        gate,
        weight,
        scale,
        shift,
        index,
        eps,
    )


@torch.library.register_fake("fastvideo::_minimax_h3_residual_gate_rmsnorm_modulate")
def _fused_residual_gate_rmsnorm_modulate_fake(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del branch, gate, weight, scale, shift, index, eps
    return residual.new_empty(residual.shape), residual.new_empty(residual.shape)


def fused_rmsnorm_modulate(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run RMSNorm and row-indexed modulation in one strict Triton kernel.

    ``index`` values must lie in ``[0, table_rows)``. Unlike eager
    ``index_select``, the kernel does not raise on out-of-range values (a
    device-side bounds check would synchronize); callers are safe by
    construction (``timestep_indices * 3 + token_tags``, SP pads with 0).
    """
    _require_forward_only(x, weight, scale, shift)
    if torch.compiler.is_compiling():
        return torch.ops.fastvideo._minimax_h3_rmsnorm_modulate(x, weight, scale, shift, index, eps)
    return _fused_rmsnorm_modulate_impl(x, weight, scale, shift, index, eps)


def fused_residual_gate_rmsnorm_modulate(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse residual update, row-indexed gate, RMSNorm, and modulation.

    ``index`` values must lie in ``[0, table_rows)``; see
    :func:`fused_rmsnorm_modulate` for why the wrapper does not check them.
    """
    _require_forward_only(residual, branch, gate, weight, scale, shift)
    if torch.compiler.is_compiling():
        return torch.ops.fastvideo._minimax_h3_residual_gate_rmsnorm_modulate(
            residual,
            branch,
            gate,
            weight,
            scale,
            shift,
            index,
            eps,
        )
    return _fused_residual_gate_rmsnorm_modulate_impl(
        residual,
        branch,
        gate,
        weight,
        scale,
        shift,
        index,
        eps,
    )
