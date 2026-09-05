# SPDX-License-Identifier: Apache-2.0
"""MXFP8 block quantization and linear operations for Blackwell inference."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from quack.mx_utils import to_blocked, to_mx

MXFP8_BLOCK_SIZE = 32
_BLOCKS_PER_PROGRAM = 16


def _validate_mxfp8_matrix(matrix: torch.Tensor) -> None:
    """Validate one row-major matrix before MXFP8 quantization."""
    if matrix.ndim != 2:
        raise ValueError(f"MXFP8 quantization requires a 2D tensor, got shape {tuple(matrix.shape)}.")
    if matrix.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"MXFP8 quantization requires BF16 or FP32 input, got {matrix.dtype}.")
    if matrix.shape[1] % MXFP8_BLOCK_SIZE:
        raise ValueError(f"MXFP8 reduction dimension must be divisible by {MXFP8_BLOCK_SIZE}, got {matrix.shape[1]}.")


@torch.compile(dynamic=True, fullgraph=True)
def _quantize_mxfp8_weight_blockwise(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values, natural_scales = to_mx(matrix.contiguous(), MXFP8_BLOCK_SIZE)
    return values, to_blocked(natural_scales)


def quantize_mxfp8_weight_blockwise(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Prequantize a weight with Quack and return hardware-blocked scales."""
    _validate_mxfp8_matrix(matrix)
    return _quantize_mxfp8_weight_blockwise(matrix)


@triton.jit
def _store_mxfp8_blocks(
    matrix,
    quantized_ptr,
    scale_ptr,
    row,
    first_block,
    row_count,
    column_count,
    scale_column_count,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    """Quantize row blocks and store scales in cuBLAS's 32x4x4 layout."""
    element_offsets = tl.arange(0, BLOCKS_PER_PROGRAM * 32)
    columns = first_block * 32 + element_offsets
    valid_elements = (row < row_count) & (columns < column_count)
    matrix = tl.reshape(matrix, (BLOCKS_PER_PROGRAM, 32))

    # FLOOR scaling encodes exponent(max_abs) - 8 as one E8M0 byte per 32 values.
    max_abs = tl.max(tl.abs(matrix), axis=1)
    exponent = (max_abs.to(tl.int32, bitcast=True) >> 23) & 0xFF
    scale_biased = tl.maximum(tl.minimum(exponent - 8, 255), 0)
    scale_biased = tl.where(max_abs != max_abs, 255, scale_biased)
    scale_bits = scale_biased.to(tl.int32) << 23
    scale = tl.maximum(scale_bits.to(tl.float32, bitcast=True), 1.1754943508222875e-38)

    scaled_matrix = tl.maximum(tl.minimum(matrix / scale[:, None], 448.0), -448.0)
    tl.store(
        quantized_ptr + row * column_count + columns,
        tl.reshape(scaled_matrix.to(tl.float8e4nv), (BLOCKS_PER_PROGRAM * 32, )),
        mask=valid_elements,
    )

    blocks = first_block + tl.arange(0, BLOCKS_PER_PROGRAM)
    # Map natural [row, K / 32] coordinates into cuBLAS's padded 128x4 tiles.
    row_block = row // 128
    row_in_block = row % 128
    column_block = blocks // 4
    column_in_block = blocks % 4
    scale_offset = ((row_block * (scale_column_count // 4) + column_block) * 512 + (row_in_block % 32) * 16 +
                    (row_in_block // 32) * 4 + column_in_block)
    tl.store(scale_ptr + scale_offset, scale_biased.to(tl.uint8), mask=blocks < scale_column_count)


@triton.jit
def _quantize_mxfp8_kernel(
    matrix_ptr,
    quantized_ptr,
    scale_ptr,
    row_count,
    column_count,
    scale_column_count,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    """Load BF16 activation blocks for direct MXFP8 quantization."""
    row = tl.program_id(0)
    first_block = tl.program_id(1) * BLOCKS_PER_PROGRAM
    element_offsets = tl.arange(0, BLOCKS_PER_PROGRAM * 32)
    columns = first_block * 32 + element_offsets
    matrix = tl.load(
        matrix_ptr + row * column_count + columns,
        mask=(row < row_count) & (columns < column_count),
        other=0.0,
    ).to(tl.float32)
    _store_mxfp8_blocks(
        matrix,
        quantized_ptr,
        scale_ptr,
        row,
        first_block,
        row_count,
        column_count,
        scale_column_count,
        BLOCKS_PER_PROGRAM,
    )


@triton.jit
def _swiglu_quantize_mxfp8_kernel(
    preactivation_ptr,
    quantized_ptr,
    scale_ptr,
    row_count,
    column_count,
    scale_column_count,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    """Apply value-first SwiGLU and directly quantize its BF16 result."""
    row = tl.program_id(0)
    first_block = tl.program_id(1) * BLOCKS_PER_PROGRAM
    element_offsets = tl.arange(0, BLOCKS_PER_PROGRAM * 32)
    columns = first_block * 32 + element_offsets
    valid_elements = (row < row_count) & (columns < column_count)
    packed_row_offset = row * (2 * column_count)
    values = tl.load(
        preactivation_ptr + packed_row_offset + columns,
        mask=valid_elements,
        other=0.0,
    ).to(tl.float32)
    gates = tl.load(
        preactivation_ptr + packed_row_offset + column_count + columns,
        mask=valid_elements,
        other=0.0,
    ).to(tl.float32)
    postactivation = (values * gates * tl.sigmoid(gates)).to(tl.bfloat16).to(tl.float32)
    _store_mxfp8_blocks(
        postactivation,
        quantized_ptr,
        scale_ptr,
        row,
        first_block,
        row_count,
        column_count,
        scale_column_count,
        BLOCKS_PER_PROGRAM,
    )


def _allocate_mxfp8_outputs(matrix: torch.Tensor, column_count: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Allocate contiguous values and a complete padded blocked-scale buffer."""
    row_count = matrix.shape[0]
    scale_column_count = triton.cdiv(column_count // MXFP8_BLOCK_SIZE, 4) * 4
    blocked_scale_size = triton.cdiv(row_count, 128) * 128 * scale_column_count
    quantized = torch.empty((row_count, column_count), dtype=torch.float8_e4m3fn, device=matrix.device)
    scale_storage = torch.empty(blocked_scale_size, dtype=torch.uint8, device=matrix.device)
    return quantized, scale_storage.view(torch.float8_e8m0fnu), scale_column_count


def quantize_mxfp8_blockwise(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activation rows and write hardware-blocked scales directly."""
    _validate_mxfp8_matrix(matrix)
    matrix = matrix.contiguous()
    row_count, column_count = matrix.shape
    quantized, blocked_scales, scale_column_count = _allocate_mxfp8_outputs(matrix, column_count)
    grid = (triton.cdiv(row_count, 128) * 128, triton.cdiv(scale_column_count, _BLOCKS_PER_PROGRAM))
    _quantize_mxfp8_kernel[grid](
        matrix,
        quantized,
        blocked_scales.view(torch.uint8),
        row_count,
        column_count,
        scale_column_count,
        BLOCKS_PER_PROGRAM=_BLOCKS_PER_PROGRAM,
        num_warps=8,
    )
    return quantized, blocked_scales


def swiglu_quantize_mxfp8_blockwise(preactivation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply H3 value-first SwiGLU and quantize the BF16 result to MXFP8."""
    if preactivation.ndim != 2:
        raise ValueError(f"MXFP8 SwiGLU requires a 2D tensor, got shape {tuple(preactivation.shape)}.")
    if preactivation.dtype != torch.bfloat16:
        raise TypeError(f"MXFP8 SwiGLU requires BF16 input, got {preactivation.dtype}.")
    if preactivation.shape[1] % (2 * MXFP8_BLOCK_SIZE):
        raise ValueError("MXFP8 SwiGLU requires each packed half to be divisible by "
                         f"{MXFP8_BLOCK_SIZE}, got packed width {preactivation.shape[1]}.")
    preactivation = preactivation.contiguous()
    row_count = preactivation.shape[0]
    column_count = preactivation.shape[1] // 2
    quantized, blocked_scales, scale_column_count = _allocate_mxfp8_outputs(preactivation, column_count)
    grid = (triton.cdiv(row_count, 128) * 128, triton.cdiv(scale_column_count, _BLOCKS_PER_PROGRAM))
    _swiglu_quantize_mxfp8_kernel[grid](
        preactivation,
        quantized,
        blocked_scales.view(torch.uint8),
        row_count,
        column_count,
        scale_column_count,
        BLOCKS_PER_PROGRAM=_BLOCKS_PER_PROGRAM,
        num_warps=8,
    )
    return quantized, blocked_scales


def mxfp8_scaled_mm(
    activation_values: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Multiply two MXFP8 matrices and return a BF16 matrix."""
    return F.scaled_mm(
        mat_a=activation_values,
        mat_b=weight_values.mT,
        scale_a=activation_scales,
        scale_recipe_a=F.ScalingType.BlockWise1x32,
        scale_b=weight_scales,
        scale_recipe_b=F.ScalingType.BlockWise1x32,
        swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
        bias=bias,
        output_dtype=torch.bfloat16,
    )


def _resolve_merged_linear(linear: torch.nn.Module) -> torch.nn.Module:
    """Return a linear whose weight includes every active inference adapter."""
    base_layer = getattr(linear, "base_layer", linear)
    if base_layer is linear:
        return base_layer
    if not getattr(linear, "merged", False) and not getattr(linear, "disable_lora", False):
        raise RuntimeError("MXFP8 feed-forward requires active LoRA weights to be merged before inference.")
    return base_layer


def mxfp8_swiglu_feed_forward(
    hidden_states: torch.Tensor,
    fc_in: torch.nn.Module,
    fc_out: torch.nn.Module,
) -> torch.Tensor:
    """Run the MiniMax-H3 feed-forward network with MXFP8 GEMMs."""
    fc_in_base = _resolve_merged_linear(fc_in)
    fc_out_base = _resolve_merged_linear(fc_out)
    fc_in_method = fc_in_base.quant_method
    fc_out_method = fc_out_base.quant_method

    preactivation = fc_in_method.apply(fc_in_base, hidden_states, fc_in_base.bias)
    output_shape = (*hidden_states.shape[:-1], fc_out_base.output_size)
    preactivation_2d = preactivation.reshape(-1, preactivation.shape[-1])
    activation_values, activation_scales = swiglu_quantize_mxfp8_blockwise(preactivation_2d)
    output_2d = fc_out_method.apply_quantized(
        fc_out_base,
        activation_values,
        activation_scales,
        fc_out_base.bias,
    )
    return output_2d.reshape(output_shape)


__all__ = [
    "MXFP8_BLOCK_SIZE",
    "mxfp8_scaled_mm",
    "mxfp8_swiglu_feed_forward",
    "quantize_mxfp8_blockwise",
    "quantize_mxfp8_weight_blockwise",
    "swiglu_quantize_mxfp8_blockwise",
]
