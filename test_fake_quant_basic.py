#!/usr/bin/env python3
"""
Basic tests for the fake_quantize function.
Tests core functionality with simple inputs and edge cases.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple
from qat_attn import fake_quantize
from nvfp4_utils import _downcast_to_mxfp, _upcast_from_mxfp, MXFP_BLOCK_SIZE, _compute_quant_and_scale, _compute_dequant

# Import get_fake_quant from test_quantization_precision
try:
    from test_quantization_precision import get_fake_quant
except ImportError:
    # Fallback: define get_fake_quant here if import fails
    from flashinfer import SfLayout, nvfp4_quantize, e2m1_and_ufp8sf_scale_to_float
    def get_fake_quant(x: torch.Tensor):
        orig_shape = x.shape
        orig_dtype = x.dtype
        device = x.device
        x = x.view(-1, x.shape[-1])
        x_global_sf = (448 * 6) / x.float().abs().nan_to_num().max()
        x_fp4, x_scale = nvfp4_quantize(x, x_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        x_dequant = e2m1_and_ufp8sf_scale_to_float(x_fp4, x_scale, 1 / x_global_sf)
        val = x_dequant.view(orig_shape).to(orig_dtype).to(device)
        return val

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_similarity(tensor1, tensor2):
    """
    Compute cosine similarity between two tensors.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor (same shape as tensor1)
    
    Returns:
        Cosine similarity value (scalar)
    """
    # Flatten tensors for computation
    t1_flat = tensor1.flatten().float()
    t2_flat = tensor2.flatten().float()
    
    # Compute cosine similarity: (A · B) / (||A|| * ||B||)
    dot_product = torch.dot(t1_flat, t2_flat)
    norm1 = torch.norm(t1_flat)
    norm2 = torch.norm(t2_flat)
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cos_sim = dot_product / (norm1 * norm2)
    return cos_sim.item()


@triton.jit
def fake_quantize_kernel(
    src_ptr,
    dst_ptr,
    high_prec_ptr,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    dst_dtype: tl.constexpr,
    mx_tensor_dtype: tl.constexpr,
    stride_src_outer,
    stride_src_quant,
    stride_dst_outer,
    stride_dst_quant,
    stride_high_prec_outer,
    stride_high_prec_quant,
    outer_dim,
    quant_dim,
):
    """
    Kernel wrapper to test fake_quantize function.
    """
    outer_idx = tl.program_id(0)
    quant_idx = tl.program_id(1)
    
    # Compute offsets
    start_outer = outer_idx * BLOCK_SIZE_OUT_DIM
    start_quant = quant_idx * BLOCK_SIZE_QUANT_DIM
    
    # Create offset arrays
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None]
    offs_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :]
    
    # Create masks for valid elements
    mask_outer = (start_outer + offs_outer) < outer_dim
    mask_quant = (start_quant + offs_quant) < quant_dim
    full_mask = mask_outer & mask_quant
    
    # Load source tensor
    src_offsets = (start_outer + offs_outer) * stride_src_outer + (start_quant + offs_quant) * stride_src_quant
    src_tensor = tl.load(src_ptr + src_offsets, mask=full_mask, other=0.0)
    
    # Call fake_quantize
    quantized_tensor, high_prec_tensor = fake_quantize(
        src_tensor=src_tensor,
        BLOCK_SIZE_OUT_DIM=BLOCK_SIZE_OUT_DIM,
        BLOCK_SIZE_QUANT_DIM=BLOCK_SIZE_QUANT_DIM,
        dst_dtype=dst_dtype,
        mx_tensor_dtype=mx_tensor_dtype
    )
    
    # Store results
    dst_offsets = (start_outer + offs_outer) * stride_dst_outer + (start_quant + offs_quant) * stride_dst_quant
    high_prec_offsets = (start_outer + offs_outer) * stride_high_prec_outer + (start_quant + offs_quant) * stride_high_prec_quant
    
    tl.store(dst_ptr + dst_offsets, quantized_tensor, mask=full_mask)
    tl.store(high_prec_ptr + high_prec_offsets, high_prec_tensor, mask=full_mask)


def call_fake_quantize(
    x: torch.Tensor,
    BLOCK_SIZE_OUT_DIM: int = 128,
    BLOCK_SIZE_QUANT_DIM: int = 128,
    dst_dtype: torch.dtype = None,
    mx_tensor_dtype = tl.float8e4nv
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Call fake_quantize on a tensor.
    
    Args:
        x: Input tensor (2D or can be reshaped to 2D)
        BLOCK_SIZE_OUT_DIM: Block size for outer dimension
        BLOCK_SIZE_QUANT_DIM: Block size for quantization dimension (must be multiple of 16)
        dst_dtype: Output dtype (defaults to input dtype)
        mx_tensor_dtype: Quantization dtype (tl.float8e4nv or tl.uint8 for fp4)
    
    Returns:
        Tuple of (quantized_tensor, high_precision_tensor)
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert BLOCK_SIZE_QUANT_DIM % 16 == 0, f"BLOCK_SIZE_QUANT_DIM must be multiple of 16, got {BLOCK_SIZE_QUANT_DIM}"
    
    orig_shape = x.shape
    orig_dtype = x.dtype
    
    # Reshape to 2D
    x_2d = x.view(-1, x.shape[-1])
    outer_dim, quant_dim = x_2d.shape
    
    if dst_dtype is None:
        dst_dtype = orig_dtype
    
    # Map torch dtype to triton dtype
    dtype_map = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
    }
    triton_dst_dtype = dtype_map.get(dst_dtype, tl.float16)
    
    # Allocate outputs
    quantized_output = torch.empty_like(x_2d, dtype=dst_dtype)
    high_prec_output = torch.empty_like(x_2d, dtype=torch.float32)
    
    # Launch kernel
    grid = (
        triton.cdiv(outer_dim, BLOCK_SIZE_OUT_DIM),
        triton.cdiv(quant_dim, BLOCK_SIZE_QUANT_DIM),
    )
    
    fake_quantize_kernel[grid](
        src_ptr=x_2d,
        dst_ptr=quantized_output,
        high_prec_ptr=high_prec_output,
        BLOCK_SIZE_OUT_DIM=BLOCK_SIZE_OUT_DIM,
        BLOCK_SIZE_QUANT_DIM=BLOCK_SIZE_QUANT_DIM,
        dst_dtype=triton_dst_dtype,
        mx_tensor_dtype=mx_tensor_dtype,
        stride_src_outer=x_2d.stride(0),
        stride_src_quant=x_2d.stride(1),
        stride_dst_outer=quantized_output.stride(0),
        stride_dst_quant=quantized_output.stride(1),
        stride_high_prec_outer=high_prec_output.stride(0),
        stride_high_prec_quant=high_prec_output.stride(1),
        outer_dim=outer_dim,
        quant_dim=quant_dim,
    )
    
    return quantized_output.view(orig_shape).to(dst_dtype), high_prec_output.view(orig_shape)


def test_basic_functionality():
    """Test basic functionality of fake_quantize."""
    print("Test 1: Basic functionality")
    
    torch.manual_seed(42)
    x = torch.randn(128, 64, device=DEVICE, dtype=torch.float16)
    
    quantized, high_prec = call_fake_quantize(x, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=64)
    
    # Check shapes
    assert quantized.shape == x.shape, f"Shape mismatch: {quantized.shape} vs {x.shape}"
    assert high_prec.shape == x.shape, f"Shape mismatch: {high_prec.shape} vs {x.shape}"
    
    # Check that quantization happened (values should be different)
    diff = (quantized.float() - x.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    assert max_diff > 0, "Quantization should change values"
    
    # Compute cosine similarity between quantized and original
    cos_sim = cosine_similarity(quantized, x)
    
    # Check that high_prec matches original
    high_prec_diff = (high_prec - x.float()).abs().max().item()
    assert high_prec_diff < 1e-5, f"High precision tensor should match original, got diff: {high_prec_diff}"
    
    print(f"  ✓ Shape check passed")
    print(f"  ✓ Quantization occurred (max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}, cosine sim: {cos_sim:.6f})")
    print(f"  ✓ High precision tensor matches original (diff: {high_prec_diff:.6e})")


def test_different_shapes():
    """Test fake_quantize with different tensor shapes."""
    print("\nTest 2: Different shapes")
    
    torch.manual_seed(42)
    test_shapes = [
        (64, 64),
        (128, 128),
        (256, 128),
        (128, 256),
        (32, 64),
    ]
    
    for shape in test_shapes:
        x = torch.randn(shape, device=DEVICE, dtype=torch.float16)
        
        # Use block sizes that divide evenly
        block_out = min(128, shape[0])
        block_quant = min(128, shape[1])
        
        # Ensure block_quant is multiple of 16
        block_quant = (block_quant // 16) * 16
        if block_quant == 0:
            block_quant = 16
        
        quantized, high_prec = call_fake_quantize(
            x,
            BLOCK_SIZE_OUT_DIM=block_out,
            BLOCK_SIZE_QUANT_DIM=block_quant
        )
        
        assert quantized.shape == x.shape, f"Shape mismatch for {shape}"
        assert high_prec.shape == x.shape, f"Shape mismatch for {shape}"
        
        print(f"  ✓ Shape {shape} passed")


def test_different_dtypes():
    """Test fake_quantize with different input dtypes."""
    print("\nTest 3: Different dtypes")
    
    torch.manual_seed(42)
    shape = (128, 64)
    dtypes = [torch.float16, torch.bfloat16, torch.float32]
    
    for dtype in dtypes:
        x = torch.randn(shape, device=DEVICE, dtype=dtype)
        
        quantized, high_prec = call_fake_quantize(
            x,
            BLOCK_SIZE_OUT_DIM=128,
            BLOCK_SIZE_QUANT_DIM=64,
            dst_dtype=dtype
        )
        
        assert quantized.dtype == dtype, f"Dtype mismatch: {quantized.dtype} vs {dtype}"
        assert quantized.shape == x.shape, f"Shape mismatch for dtype {dtype}"
        
        print(f"  ✓ Dtype {dtype} passed")


def test_edge_cases():
    """Test fake_quantize with edge cases."""
    print("\nTest 4: Edge cases")
    
    # Test with zeros
    print("  Testing zeros...")
    x_zeros = torch.zeros(128, 64, device=DEVICE, dtype=torch.float16)
    quantized, high_prec = call_fake_quantize(x_zeros, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=64)
    assert quantized.shape == x_zeros.shape
    # Quantized zeros should still be close to zero
    assert quantized.abs().max().item() < 1e-3, "Quantized zeros should be close to zero"
    print("    ✓ Zeros passed")
    
    # Test with ones (FAIL)
    print("  Testing ones...")
    x_ones = torch.ones(128, 64, device=DEVICE, dtype=torch.float16)
    print(f"x_ones: {x_ones}")
    quantized, high_prec = call_fake_quantize(x_ones, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=64)
    print(f"quantized: {quantized}")
    print(f"high_prec: {high_prec}")
    assert quantized.shape == x_ones.shape
    # Quantized ones should be close to 1.0
    diff = (quantized - x_ones).abs().max().item()
    assert diff < 0.1, f"Quantized ones should be close to 1.0, got max diff: {diff}"
    print(f"    ✓ Ones passed (max diff: {diff:.6f})")
    
    # Test with small values
    print("  Testing small values...")
    x_small = torch.randn(128, 64, device=DEVICE, dtype=torch.float16) * 0.001
    quantized, high_prec = call_fake_quantize(x_small, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=64)
    assert quantized.shape == x_small.shape
    print("    ✓ Small values passed")
    
    # Test with large values
    print("  Testing large values...")
    x_large = torch.randn(128, 64, device=DEVICE, dtype=torch.float16) * 100.0
    quantized, high_prec = call_fake_quantize(x_large, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=64)
    assert quantized.shape == x_large.shape
    print("    ✓ Large values passed")


def test_fp4_quantization():
    """Test fake_quantize with FP4 quantization (uint8)."""
    print("\nTest 5: FP4 quantization (uint8)")
    
    torch.manual_seed(42)
    x = torch.randn(128, 64, device=DEVICE, dtype=torch.float16)
    
    quantized, high_prec = call_fake_quantize(
        x,
        BLOCK_SIZE_OUT_DIM=128,
        BLOCK_SIZE_QUANT_DIM=64,
        mx_tensor_dtype=tl.uint8  # FP4 uses uint8
    )
    
    assert quantized.shape == x.shape
    assert high_prec.shape == x.shape
    
    # Check that quantization happened
    diff = (quantized.float() - x.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    assert max_diff > 0, "Quantization should change values"
    
    # Compute cosine similarity between quantized and original
    cos_sim = cosine_similarity(quantized, x)
    
    print(f"  ✓ FP4 quantization passed (max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}, cosine sim: {cos_sim:.6f})")



def test_finite_values():
    """Test that fake_quantize produces finite values."""
    print("\nTest 7: Finite values check")
    
    torch.manual_seed(42)
    x = torch.randn(128, 64, device=DEVICE, dtype=torch.float16)
    
    quantized, high_prec = call_fake_quantize(x, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=64)
    
    # Check for NaN and Inf
    assert not torch.isnan(quantized).any(), "Quantized tensor contains NaN"
    assert not torch.isinf(quantized).any(), "Quantized tensor contains Inf"
    assert not torch.isnan(high_prec).any(), "High precision tensor contains NaN"
    assert not torch.isinf(high_prec).any(), "High precision tensor contains Inf"
    
    print("  ✓ All values are finite")


def test_different_block_sizes():
    """Test fake_quantize with different block sizes."""
    print("\nTest 8: Different block sizes")
    
    torch.manual_seed(42)
    x = torch.randn(256, 128, device=DEVICE, dtype=torch.float16)
    
    block_configs = [
        (64, 64),
        (128, 64),
        (64, 128),
        (128, 128),
    ]
    
    for block_out, block_quant in block_configs:
        quantized, high_prec = call_fake_quantize(
            x,
            BLOCK_SIZE_OUT_DIM=block_out,
            BLOCK_SIZE_QUANT_DIM=block_quant
        )
        
        assert quantized.shape == x.shape
        assert high_prec.shape == x.shape
        
        print(f"  ✓ Block size ({block_out}, {block_quant}) passed")


# ============================================================================
# MXFP Kernel Tests
# ============================================================================

@triton.jit
def mxfp_fake_quantize_kernel(
    src_ptr,
    dst_ptr,
    high_prec_ptr,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    dst_dtype: tl.constexpr,
    mx_tensor_dtype: tl.constexpr,
    stride_src_outer,
    stride_src_quant,
    stride_dst_outer,
    stride_dst_quant,
    stride_high_prec_outer,
    stride_high_prec_quant,
    outer_dim,
    quant_dim,
):
    """
    Kernel wrapper to test MXFP fake_quantize using _downcast_to_mxfp and _upcast_from_mxfp.
    """
    outer_idx = tl.program_id(0)
    quant_idx = tl.program_id(1)
    
    # Determine if we are dealing with fp4 types
    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8
    K_DIVISOR: tl.constexpr = 2 if is_fp4 else 1
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // MXFP_BLOCK_SIZE
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR
    
    # Allocate temporary storage for quantized tensor and scale
    # These need to be stored in shared memory or we need to allocate them
    # For simplicity, we'll compute quantize and dequantize in the same kernel
    
    # Compute offsets
    start_outer = outer_idx * BLOCK_SIZE_OUT_DIM
    start_quant = quant_idx * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_idx * BLOCK_SIZE_QUANT_MX_SCALE
    start_mx_quant = quant_idx * BLOCK_SIZE_QUANT_MX_TENSOR
    
    # Create offset arrays
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None]
    offs_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :]
    offs_mx_scale_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :]
    offs_mx_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :]
    
    # Create masks for valid elements
    mask_outer = (start_outer + offs_outer) < outer_dim
    mask_quant = (start_quant + offs_quant) < quant_dim
    full_mask = mask_outer & mask_quant
    
    # Load source tensor
    src_offsets = (start_outer + offs_outer) * stride_src_outer + (start_quant + offs_quant) * stride_src_quant
    src_tensor = tl.load(src_ptr + src_offsets, mask=full_mask, other=0.0)
    
    # Store high precision tensor (original)
    high_prec_offsets = (start_outer + offs_outer) * stride_high_prec_outer + (start_quant + offs_quant) * stride_high_prec_quant
    tl.store(high_prec_ptr + high_prec_offsets, src_tensor.to(tl.float32), mask=full_mask)
    
    # Quantize using _compute_quant_and_scale (same as fake_quantize uses internally)
    valid_mask = tl.full(src_tensor.shape, 1, tl.int1)
    quantized_tensor, scale_tensor, s_dec = _compute_quant_and_scale(
        src_tensor=src_tensor,
        valid_src_mask=valid_mask,
        mx_tensor_dtype=mx_tensor_dtype
    )
    
    # Dequantize using _compute_dequant (same as fake_quantize uses internally)
    dequantized_tensor = _compute_dequant(
        mx_tensor=quantized_tensor,
        scale=scale_tensor,
        s_dec=s_dec,
        BLOCK_SIZE_OUT_DIM=BLOCK_SIZE_OUT_DIM,
        BLOCK_SIZE_QUANT_DIM=BLOCK_SIZE_QUANT_DIM,
        dst_dtype=dst_dtype
    )
    
    # Store result
    dst_offsets = (start_outer + offs_outer) * stride_dst_outer + (start_quant + offs_quant) * stride_dst_quant
    tl.store(dst_ptr + dst_offsets, dequantized_tensor, mask=full_mask)


def call_mxfp_fake_quantize(
    x: torch.Tensor,
    BLOCK_SIZE_OUT_DIM: int = 128,
    BLOCK_SIZE_QUANT_DIM: int = 128,
    dst_dtype: torch.dtype = None,
    mx_tensor_dtype = tl.uint8  # Default to FP4 (uint8)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Call MXFP fake_quantize on a tensor using _downcast_to_mxfp and _upcast_from_mxfp logic.
    
    Args:
        x: Input tensor (2D or can be reshaped to 2D)
        BLOCK_SIZE_OUT_DIM: Block size for outer dimension
        BLOCK_SIZE_QUANT_DIM: Block size for quantization dimension (must be multiple of 16)
        dst_dtype: Output dtype (defaults to input dtype)
        mx_tensor_dtype: Quantization dtype (tl.uint8 for fp4 or tl.float8e4nv for fp8)
    
    Returns:
        Tuple of (quantized_tensor, high_precision_tensor)
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert BLOCK_SIZE_QUANT_DIM % 16 == 0, f"BLOCK_SIZE_QUANT_DIM must be multiple of 16, got {BLOCK_SIZE_QUANT_DIM}"
    
    orig_shape = x.shape
    orig_dtype = x.dtype
    
    # Reshape to 2D
    x_2d = x.view(-1, x.shape[-1])
    outer_dim, quant_dim = x_2d.shape
    
    if dst_dtype is None:
        dst_dtype = orig_dtype
    
    # Map torch dtype to triton dtype
    dtype_map = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
    }
    triton_dst_dtype = dtype_map.get(dst_dtype, tl.float16)
    
    # Allocate outputs
    quantized_output = torch.empty_like(x_2d, dtype=dst_dtype)
    high_prec_output = torch.empty_like(x_2d, dtype=torch.float32)
    
    # Launch kernel
    grid = (
        triton.cdiv(outer_dim, BLOCK_SIZE_OUT_DIM),
        triton.cdiv(quant_dim, BLOCK_SIZE_QUANT_DIM),
    )
    
    mxfp_fake_quantize_kernel[grid](
        src_ptr=x_2d,
        dst_ptr=quantized_output,
        high_prec_ptr=high_prec_output,
        BLOCK_SIZE_OUT_DIM=BLOCK_SIZE_OUT_DIM,
        BLOCK_SIZE_QUANT_DIM=BLOCK_SIZE_QUANT_DIM,
        dst_dtype=triton_dst_dtype,
        mx_tensor_dtype=mx_tensor_dtype,
        stride_src_outer=x_2d.stride(0),
        stride_src_quant=x_2d.stride(1),
        stride_dst_outer=quantized_output.stride(0),
        stride_dst_quant=quantized_output.stride(1),
        stride_high_prec_outer=high_prec_output.stride(0),
        stride_high_prec_quant=high_prec_output.stride(1),
        outer_dim=outer_dim,
        quant_dim=quant_dim,
    )
    
    return quantized_output.view(orig_shape).to(dst_dtype), high_prec_output.view(orig_shape)


def test_mxfp_basic_functionality():
    """Test basic functionality of MXFP fake_quantize."""
    print("\n" + "=" * 60)
    print("MXFP Kernel Tests")
    print("=" * 60)
    print("\nTest 1: Basic functionality (MXFP)")
    
    torch.manual_seed(42)
    x = torch.randn(128, 64, device=DEVICE, dtype=torch.float16)
    
    quantized, high_prec = call_mxfp_fake_quantize(x, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=64)
    
    # Check shapes
    assert quantized.shape == x.shape, f"Shape mismatch: {quantized.shape} vs {x.shape}"
    assert high_prec.shape == x.shape, f"Shape mismatch: {high_prec.shape} vs {x.shape}"
    
    # Check that quantization happened (values should be different)
    diff = (quantized.float() - x.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    assert max_diff > 0, "Quantization should change values"
    
    # Compute cosine similarity between quantized and original
    cos_sim = cosine_similarity(quantized, x)
    
    # Check that high_prec matches original
    high_prec_diff = (high_prec - x.float()).abs().max().item()
    assert high_prec_diff < 1e-5, f"High precision tensor should match original, got diff: {high_prec_diff}"
    
    print(f"  ✓ Shape check passed")
    print(f"  ✓ Quantization occurred (max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}, cosine sim: {cos_sim:.6f})")
    print(f"  ✓ High precision tensor matches original (diff: {high_prec_diff:.6e})")


def test_mxfp_different_shapes():
    """Test MXFP fake_quantize with different tensor shapes."""
    print("\nTest 2: Different shapes (MXFP)")
    
    torch.manual_seed(42)
    test_shapes = [
        (64, 64),
        (128, 128),
        (256, 128),
        (128, 256),
        (32, 64),
    ]
    
    for shape in test_shapes:
        x = torch.randn(shape, device=DEVICE, dtype=torch.float16)
        
        # Use block sizes that divide evenly
        block_out = min(128, shape[0])
        block_quant = min(128, shape[1])
        
        # Ensure block_quant is multiple of 16
        block_quant = (block_quant // 16) * 16
        if block_quant == 0:
            block_quant = 16
        
        quantized, high_prec = call_mxfp_fake_quantize(
            x,
            BLOCK_SIZE_OUT_DIM=block_out,
            BLOCK_SIZE_QUANT_DIM=block_quant
        )
        
        assert quantized.shape == x.shape, f"Shape mismatch for {shape}"
        assert high_prec.shape == x.shape, f"Shape mismatch for {shape}"
        
        print(f"  ✓ Shape {shape} passed")


def test_mxfp_different_dtypes():
    """Test MXFP fake_quantize with different input dtypes."""
    print("\nTest 3: Different dtypes (MXFP)")
    
    torch.manual_seed(42)
    shape = (128, 64)
    dtypes = [torch.float16, torch.bfloat16, torch.float32]
    
    for dtype in dtypes:
        x = torch.randn(shape, device=DEVICE, dtype=dtype)
        
        quantized, high_prec = call_mxfp_fake_quantize(
            x,
            BLOCK_SIZE_OUT_DIM=128,
            BLOCK_SIZE_QUANT_DIM=64,
            dst_dtype=dtype
        )
        
        assert quantized.dtype == dtype, f"Dtype mismatch: {quantized.dtype} vs {dtype}"
        assert quantized.shape == x.shape, f"Shape mismatch for dtype {dtype}"
        
        print(f"  ✓ Dtype {dtype} passed")


def test_mxfp_edge_cases():
    """Test MXFP fake_quantize with edge cases."""
    print("\nTest 4: Edge cases (MXFP)")
    
    # Test with zeros
    print("  Testing zeros...")
    x_zeros = torch.zeros(128, 64, device=DEVICE, dtype=torch.float16)
    quantized, high_prec = call_mxfp_fake_quantize(x_zeros, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=64)
    assert quantized.shape == x_zeros.shape
    # Quantized zeros should still be close to zero
    assert quantized.abs().max().item() < 1e-3, "Quantized zeros should be close to zero"
    print("    ✓ Zeros passed")
    
    # Test with ones
    print("  Testing ones...")
    x_ones = torch.ones(128, 64, device=DEVICE, dtype=torch.float16)
    quantized, high_prec = call_mxfp_fake_quantize(x_ones, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=64)
    assert quantized.shape == x_ones.shape
    # Quantized ones should be close to 1.0
    diff = (quantized - x_ones).abs().max().item()
    assert diff < 0.1, f"Quantized ones should be close to 1.0, got max diff: {diff}"
    print(f"    ✓ Ones passed (max diff: {diff:.6f})")
    
    # Test with small values
    print("  Testing small values...")
    x_small = torch.randn(128, 64, device=DEVICE, dtype=torch.float16) * 0.001
    quantized, high_prec = call_mxfp_fake_quantize(x_small, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=64)
    assert quantized.shape == x_small.shape
    print("    ✓ Small values passed")
    
    # Test with large values
    print("  Testing large values...")
    x_large = torch.randn(128, 64, device=DEVICE, dtype=torch.float16) * 100.0
    quantized, high_prec = call_mxfp_fake_quantize(x_large, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=64)
    assert quantized.shape == x_large.shape
    print("    ✓ Large values passed")


def test_mxfp_fp4_quantization():
    """Test MXFP fake_quantize with FP4 quantization (uint8)."""
    print("\nTest 5: FP4 quantization (MXFP, uint8)")
    
    torch.manual_seed(42)
    x = torch.randn(128, 64, device=DEVICE, dtype=torch.float16)
    
    quantized, high_prec = call_mxfp_fake_quantize(
        x,
        BLOCK_SIZE_OUT_DIM=128,
        BLOCK_SIZE_QUANT_DIM=64,
        mx_tensor_dtype=tl.uint8  # FP4 uses uint8
    )
    
    assert quantized.shape == x.shape
    assert high_prec.shape == x.shape
    
    # Check that quantization happened
    diff = (quantized.float() - x.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    assert max_diff > 0, "Quantization should change values"
    
    # Compute cosine similarity between quantized and original
    cos_sim = cosine_similarity(quantized, x)
    
    print(f"  ✓ FP4 quantization passed (max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}, cosine sim: {cos_sim:.6f})")


def test_mxfp_fp8_quantization():
    """Test MXFP fake_quantize with FP8 quantization (float8e4nv)."""
    print("\nTest 6: FP8 quantization (MXFP, float8e4nv)")
    
    torch.manual_seed(42)
    x = torch.randn(128, 64, device=DEVICE, dtype=torch.float16)
    
    quantized, high_prec = call_mxfp_fake_quantize(
        x,
        BLOCK_SIZE_OUT_DIM=128,
        BLOCK_SIZE_QUANT_DIM=64,
        mx_tensor_dtype=tl.float8e4nv  # FP8 uses float8e4nv
    )
    
    assert quantized.shape == x.shape
    assert high_prec.shape == x.shape
    
    # Check that quantization happened
    diff = (quantized.float() - x.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    assert max_diff > 0, "Quantization should change values"
    
    # Compute cosine similarity between quantized and original
    cos_sim = cosine_similarity(quantized, x)
    
    print(f"  ✓ FP8 quantization passed (max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}, cosine sim: {cos_sim:.6f})")


def test_mxfp_finite_values():
    """Test that MXFP fake_quantize produces finite values."""
    print("\nTest 7: Finite values check (MXFP)")
    
    torch.manual_seed(42)
    x = torch.randn(128, 64, device=DEVICE, dtype=torch.float16)
    
    quantized, high_prec = call_mxfp_fake_quantize(x, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=64)
    
    # Check for NaN and Inf
    assert not torch.isnan(quantized).any(), "Quantized tensor contains NaN"
    assert not torch.isinf(quantized).any(), "Quantized tensor contains Inf"
    assert not torch.isnan(high_prec).any(), "High precision tensor contains NaN"
    assert not torch.isinf(high_prec).any(), "High precision tensor contains Inf"
    
    print("  ✓ All values are finite")


def test_mxfp_different_block_sizes():
    """Test MXFP fake_quantize with different block sizes."""
    print("\nTest 8: Different block sizes (MXFP)")
    
    torch.manual_seed(42)
    x = torch.randn(256, 128, device=DEVICE, dtype=torch.float16)
    
    block_configs = [
        (64, 64),
        (128, 64),
        (64, 128),
        (128, 128),
    ]
    
    for block_out, block_quant in block_configs:
        quantized, high_prec = call_mxfp_fake_quantize(
            x,
            BLOCK_SIZE_OUT_DIM=block_out,
            BLOCK_SIZE_QUANT_DIM=block_quant
        )
        
        assert quantized.shape == x.shape
        assert high_prec.shape == x.shape
        
        print(f"  ✓ Block size ({block_out}, {block_quant}) passed")


def test_get_fake_quant_metrics():
    """Test get_fake_quant function with cosine similarity, mean_diff, and max_diff metrics."""
    print("\n" + "=" * 60)
    print("Testing get_fake_quant with cosine similarity, mean_diff, and max_diff")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    device = torch.device("cuda")
    torch.manual_seed(42)
    
    # Test with different shapes and dtypes
    test_cases = [
        ((128, 64), torch.bfloat16, "2D bfloat16"),
        ((256, 128), torch.bfloat16, "2D bfloat16 large"),
        ((2, 8, 128), torch.bfloat16, "3D bfloat16"),
        ((128, 64), torch.float16, "2D float16"),
        ((1, 8, 256, 128), torch.bfloat16, "4D bfloat16"),
    ]
    
    for shape, dtype, name in test_cases:
        print(f"\nTest: {name} - Shape: {shape}")
        x = torch.randn(*shape, device=device, dtype=dtype)
        x_fq = get_fake_quant(x)
        
        # Check shapes match
        assert x_fq.shape == x.shape, f"Shape mismatch: {x_fq.shape} vs {x.shape}"
        assert x_fq.dtype == x.dtype, f"Dtype mismatch: {x_fq.dtype} vs {x.dtype}"
        
        # Compute metrics
        diff = (x_fq.float() - x.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        cos_sim = cosine_similarity(x_fq, x)
        
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        
        # Basic sanity checks
        assert max_diff < 10.0, f"Max difference too large: {max_diff}"
        assert cos_sim > 0.5, f"Cosine similarity too low: {cos_sim}"
    
    print("\n✓ All metric tests passed!")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. These tests require CUDA.")
        exit(1)
    
    print("=" * 60)
    print("Running basic tests for fake_quantize function")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print()
    
    try:
        test_basic_functionality()
        test_different_shapes()
        test_different_dtypes()
        test_edge_cases()
        test_fp4_quantization()
        test_finite_values()
        
        # Test get_fake_quant with metrics
        test_get_fake_quant_metrics()
        
        # Run MXFP kernel tests
        # test_mxfp_basic_functionality()
        # test_mxfp_different_shapes()
        # test_mxfp_different_dtypes()
        # test_mxfp_edge_cases()
        # test_mxfp_fp4_quantization()
        # test_mxfp_fp8_quantization()
        # test_mxfp_finite_values()
        # test_mxfp_different_block_sizes()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


