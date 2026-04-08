#!/usr/bin/env python3
"""
Debug script to understand the scale computation issue.
"""

import torch
import triton
import triton.language as tl
from nvfp4_utils import _compute_quant_and_scale, _compute_dequant

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@triton.jit
def debug_scale_kernel(
    src_ptr,
    scale_ptr,
    s_dec_ptr,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    stride_src_outer,
    stride_src_quant,
    outer_dim,
    quant_dim,
):
    """Debug kernel to print scale values."""
    outer_idx = tl.program_id(0)
    quant_idx = tl.program_id(1)
    
    start_outer = outer_idx * BLOCK_SIZE_OUT_DIM
    start_quant = quant_idx * BLOCK_SIZE_QUANT_DIM
    
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None]
    offs_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :]
    
    mask_outer = (start_outer + offs_outer) < outer_dim
    mask_quant = (start_quant + offs_quant) < quant_dim
    full_mask = mask_outer & mask_quant
    
    src_offsets = (start_outer + offs_outer) * stride_src_outer + (start_quant + offs_quant) * stride_src_quant
    src_tensor = tl.load(src_ptr + src_offsets, mask=full_mask, other=0.0)
    
    valid_mask = tl.full(src_tensor.shape, 1, tl.int1)
    quantized_tensor, scale_tensor, s_dec = _compute_quant_and_scale(
        src_tensor=src_tensor,
        valid_src_mask=valid_mask,
        mx_tensor_dtype=tl.float8e4nv
    )
    
    # Store scale and s_dec for inspection
    # scale_tensor has shape [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE] where BLOCK_SIZE_QUANT_MX_SCALE = BLOCK_SIZE_QUANT_DIM // 16
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // 16
    scale_offsets = (start_outer + offs_outer) * stride_src_outer + (start_quant // 16) * stride_src_quant
    tl.store(scale_ptr + scale_offsets, scale_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE])[:, 0].to(tl.float32), mask=mask_outer)
    tl.store(s_dec_ptr + outer_idx, s_dec)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        exit(1)
    
    # Test with all ones
    x = torch.ones(128, 64, device=DEVICE, dtype=torch.float16)
    x_2d = x.view(-1, x.shape[-1])
    outer_dim, quant_dim = x_2d.shape
    
    scale_output = torch.zeros(outer_dim, quant_dim // 16, device=DEVICE, dtype=torch.float32)
    s_dec_output = torch.zeros(outer_dim // 128, device=DEVICE, dtype=torch.float32)
    
    grid = (
        triton.cdiv(outer_dim, 128),
        triton.cdiv(quant_dim, 64),
    )
    
    debug_scale_kernel[grid](
        src_ptr=x_2d,
        scale_ptr=scale_output,
        s_dec_ptr=s_dec_output,
        BLOCK_SIZE_OUT_DIM=128,
        BLOCK_SIZE_QUANT_DIM=64,
        stride_src_outer=x_2d.stride(0),
        stride_src_quant=x_2d.stride(1),
        outer_dim=outer_dim,
        quant_dim=quant_dim,
    )
    
    print(f"Scale values (first block): {scale_output[0, :4]}")
    print(f"s_dec value: {s_dec_output[0]}")
    print(f"Expected s_dec: {1.0 / (6 * 448 / 1.0)} = {1.0 / 2688}")
    print(f"Expected scale: {(1.0 / 6) * 2688} = {2688 / 6} = {448}")


