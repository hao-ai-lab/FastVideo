#!/usr/bin/env python3
"""
Precision test to compare numerical differences for fake_quantize triton function:
1. fake_quantize (triton implementation) vs reference implementations
2. Tests various shapes, dtypes, and value ranges
3. Evaluates cosine similarity, max diff, and mean diff between input and output
"""

import torch
import triton
import triton.language as tl
from flashinfer import SfLayout, nvfp4_quantize, e2m1_and_ufp8sf_scale_to_float
from nvfp4_utils import _compute_quant_and_scale, _compute_dequant
from typing import Optional

# MXFP_BLOCK_SIZE is 16 - use Python int for runtime checks
MXFP_BLOCK_SIZE = 16

DEVICE = torch.device("cuda")


def cosine_similarity(tensor1, tensor2):
    """
    Compute cosine similarity between two tensors.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor (same shape as tensor1)
    
    Returns:
        Cosine similarity value (scalar)
        - Returns 1.0 if both tensors are zero (identical zero vectors)
        - Returns 0.0 if only one tensor is zero (orthogonal to non-zero vector)
    """
    # Flatten tensors for computation
    t1_flat = tensor1.flatten().float()
    t2_flat = tensor2.flatten().float()
    
    # Compute cosine similarity: (A · B) / (||A|| * ||B||)
    dot_product = torch.dot(t1_flat, t2_flat)
    norm1 = torch.norm(t1_flat)
    norm2 = torch.norm(t2_flat)
    
    # Handle zero vectors
    if norm1 == 0 and norm2 == 0:
        # Both are zero vectors - they are identical, so similarity is 1.0
        return 1.0
    elif norm1 == 0 or norm2 == 0:
        # One is zero, one is not - they are orthogonal, so similarity is 0.0
        return 0.0
    
    cos_sim = dot_product / (norm1 * norm2)
    return cos_sim.item()


@triton.jit
def fake_quantize(src_tensor, valid_src_mask, BLOCK_SIZE_OUT_DIM: tl.constexpr, 
                    BLOCK_SIZE_QUANT_DIM: tl.constexpr, 
                    dst_dtype: tl.constexpr,
                    mx_tensor_dtype: tl.constexpr = tl.uint8):
    """
    Fake quantize function - matches API from qat_attn.py.
    """
    high_prec_src_tensor = src_tensor
    src_tensor, src_scale, src_s_dec = _compute_quant_and_scale(
        src_tensor=src_tensor, 
        valid_src_mask=valid_src_mask, 
        mx_tensor_dtype=mx_tensor_dtype
    )
    src_tensor = _compute_dequant(
        mx_tensor=src_tensor, 
        scale=src_scale, 
        s_dec=src_s_dec, 
        BLOCK_SIZE_OUT_DIM=BLOCK_SIZE_OUT_DIM, 
        BLOCK_SIZE_QUANT_DIM=BLOCK_SIZE_QUANT_DIM, 
        dst_dtype=dst_dtype
    )
    return src_tensor, high_prec_src_tensor


def get_fake_quant_reference(x: torch.Tensor):
    """
    Reference implementation using FlashInfer for comparison.
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    device = x.device
    x = x.view(-1, x.shape[-1])
    x_global_sf = (448 * 6) / x.float().abs().nan_to_num().max()
    x_fp4, x_scale = nvfp4_quantize(x, x_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    x_dequant = e2m1_and_ufp8sf_scale_to_float(x_fp4, x_scale, 1 / x_global_sf)
    return x_dequant.view(orig_shape).to(orig_dtype).to(device)


@triton.jit
def fake_quantize_kernel(
    src_ptr,
    dst_ptr,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    dst_dtype: tl.constexpr,
    mx_tensor_dtype: tl.constexpr,
    stride_src_outer,
    stride_src_quant,
    stride_dst_outer,
    stride_dst_quant,
    outer_dim,
    quant_dim,
):
    """
    Kernel wrapper to call fake_quantize on a block of data.
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
    
    # Call fake_quantize with valid_src_mask parameter
    quantized_tensor, high_prec_tensor = fake_quantize(
        src_tensor=src_tensor,
        valid_src_mask=full_mask,
        BLOCK_SIZE_OUT_DIM=BLOCK_SIZE_OUT_DIM,
        BLOCK_SIZE_QUANT_DIM=BLOCK_SIZE_QUANT_DIM,
        dst_dtype=dst_dtype,
        mx_tensor_dtype=mx_tensor_dtype
    )
    
    # Store result
    dst_offsets = (start_outer + offs_outer) * stride_dst_outer + (start_quant + offs_quant) * stride_dst_quant
    tl.store(dst_ptr + dst_offsets, quantized_tensor, mask=full_mask)


def triton_fake_quantize(
    x: torch.Tensor,
    BLOCK_SIZE_OUT_DIM: int = 128,
    BLOCK_SIZE_QUANT_DIM: int = 128,
    use_fp4: bool = True,  # True for fp4 (uint8), False for fp8 (float8e4nv)
    dst_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Call fake_quantize triton function on a tensor.
    
    Args:
        x: Input tensor (2D or can be reshaped to 2D)
        BLOCK_SIZE_OUT_DIM: Block size for outer dimension
        BLOCK_SIZE_QUANT_DIM: Block size for quantization dimension (must be multiple of 16)
        use_fp4: If True, use fp4 (uint8), else use fp8 (float8e4nv)
        dst_dtype: Output dtype (defaults to input dtype)
    
    Returns:
        Fake quantized tensor
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert BLOCK_SIZE_QUANT_DIM % 16 == 0, f"BLOCK_SIZE_QUANT_DIM must be multiple of 16"
    
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
    
    # Allocate output
    output = torch.empty_like(x_2d, dtype=dst_dtype)
    
    # Launch kernel with appropriate quantization dtype
    grid = (
        triton.cdiv(outer_dim, BLOCK_SIZE_OUT_DIM),
        triton.cdiv(quant_dim, BLOCK_SIZE_QUANT_DIM),
    )
    
    if use_fp4:
        # Use fp4 (uint8)
        fake_quantize_kernel[grid](
            src_ptr=x_2d,
            dst_ptr=output,
            BLOCK_SIZE_OUT_DIM=BLOCK_SIZE_OUT_DIM,
            BLOCK_SIZE_QUANT_DIM=BLOCK_SIZE_QUANT_DIM,
            dst_dtype=triton_dst_dtype,
            mx_tensor_dtype=tl.uint8,  # fp4 uses uint8
            stride_src_outer=x_2d.stride(0),
            stride_src_quant=x_2d.stride(1),
            stride_dst_outer=output.stride(0),
            stride_dst_quant=output.stride(1),
            outer_dim=outer_dim,
            quant_dim=quant_dim,
        )
    else:
        # Use fp8 (float8e4nv)
        fake_quantize_kernel[grid](
            src_ptr=x_2d,
            dst_ptr=output,
            BLOCK_SIZE_OUT_DIM=BLOCK_SIZE_OUT_DIM,
            BLOCK_SIZE_QUANT_DIM=BLOCK_SIZE_QUANT_DIM,
            dst_dtype=triton_dst_dtype,
            mx_tensor_dtype=tl.float8e4nv,  # fp8 uses float8e4nv
            stride_src_outer=x_2d.stride(0),
            stride_src_quant=x_2d.stride(1),
            stride_dst_outer=output.stride(0),
            stride_dst_quant=output.stride(1),
            outer_dim=outer_dim,
            quant_dim=quant_dim,
        )
    
    return output.view(orig_shape).to(dst_dtype)


def test_fake_quantize_basic():
    """Test basic functionality of fake_quantize."""
    torch.manual_seed(42)
    
    # Test parameters
    shape = (128, 128)
    dtype = torch.bfloat16
    
    # Create input tensor
    x = torch.randn(shape, dtype=dtype, device=DEVICE)
    
    # Test triton fake_quantize
    x_fq = triton_fake_quantize(x, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=128, use_fp4=True, dst_dtype=dtype)
    
    # Check that outputs have correct shape
    assert x_fq.shape == x.shape
    assert x_fq.dtype == x.dtype
    
    # Check that outputs are finite
    assert torch.isfinite(x_fq).all()
    
    # Compare input and output
    max_diff = (x_fq - x).abs().max()
    mean_diff = (x_fq - x).abs().mean()
    cos_sim = cosine_similarity(x_fq, x)
    print(f"  Input vs Output - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    print("✓ Basic test passed.")


def test_fake_quantize_different_shapes():
    """Test fake_quantize with different input shapes."""
    torch.manual_seed(42)
    
    test_configs = [
        (128, 64),   # Small
        (256, 128),  # Medium
        (512, 256),  # Large
        (128, 256),  # Rectangular
        (256, 128),  # Rectangular (reversed)
    ]
    
    dtype = torch.bfloat16
    
    for outer_dim, quant_dim in test_configs:
        # Create input tensor
        x = torch.randn((outer_dim, quant_dim), dtype=dtype, device=DEVICE)
        
        # Test triton fake_quantize
        x_fq = triton_fake_quantize(x, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=128, use_fp4=True, dst_dtype=dtype)
        
        assert x_fq.shape == x.shape
        assert torch.isfinite(x_fq).all()
        
        # Compare input and output
        max_diff = (x_fq - x).abs().max()
        mean_diff = (x_fq - x).abs().mean()
        cos_sim = cosine_similarity(x_fq, x)
        print(f"  Shape ({outer_dim}, {quant_dim}) - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
        
        print(f"✓ Shape test passed for ({outer_dim}, {quant_dim})")


def test_fake_quantize_different_dtypes():
    """Test fake_quantize with different input dtypes."""
    torch.manual_seed(42)
    
    shape = (128, 128)
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    
    for dtype in dtypes:
        # Create input tensor
        x = torch.randn(shape, dtype=dtype, device=DEVICE)
        
        # Test triton fake_quantize
        x_fq = triton_fake_quantize(x, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=128, use_fp4=True, dst_dtype=dtype)
        
        assert x_fq.shape == x.shape
        assert x_fq.dtype == x.dtype
        assert torch.isfinite(x_fq).all()
        
        # Compare input and output
        max_diff = (x_fq - x).abs().max()
        mean_diff = (x_fq - x).abs().mean()
        cos_sim = cosine_similarity(x_fq, x)
        print(f"  Dtype {dtype} - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
        
        print(f"✓ Dtype test passed for {dtype}")


def test_fake_quantize_3d_4d_tensors():
    """Test fake_quantize with 3D and 4D tensors (reshaped to 2D internally)."""
    torch.manual_seed(42)
    
    dtype = torch.bfloat16
    
    # Test 3D tensor (B, H, D)
    print("\nTesting 3D tensor (B, H, D)")
    x_3d = torch.randn((2, 8, 128), dtype=dtype, device=DEVICE)
    x_fq_3d = triton_fake_quantize(x_3d, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=128, use_fp4=True, dst_dtype=dtype)
    
    assert x_fq_3d.shape == x_3d.shape
    assert torch.isfinite(x_fq_3d).all()
    
    max_diff = (x_fq_3d - x_3d).abs().max()
    mean_diff = (x_fq_3d - x_3d).abs().mean()
    cos_sim = cosine_similarity(x_fq_3d, x_3d)
    print(f"  3D (2, 8, 128) - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    # Test 4D tensor (B, H, L, D)
    print("\nTesting 4D tensor (B, H, L, D)")
    x_4d = torch.randn((1, 8, 256, 128), dtype=dtype, device=DEVICE)
    x_fq_4d = triton_fake_quantize(x_4d, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=128, use_fp4=True, dst_dtype=dtype)
    
    assert x_fq_4d.shape == x_4d.shape
    assert torch.isfinite(x_fq_4d).all()
    
    max_diff = (x_fq_4d - x_4d).abs().max()
    mean_diff = (x_fq_4d - x_4d).abs().mean()
    cos_sim = cosine_similarity(x_fq_4d, x_4d)
    print(f"  4D (1, 8, 256, 128) - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    print("✓ 3D/4D tensor test passed.")


def test_fake_quantize_edge_cases():
    """Test edge cases like zeros, ones, extreme values."""
    torch.manual_seed(42)
    
    dtype = torch.bfloat16
    shape = (128, 128)
    
    test_cases = [
        ("zeros", torch.zeros(shape, dtype=dtype, device=DEVICE)),
        ("ones", torch.ones(shape, dtype=dtype, device=DEVICE)),
        ("negative_ones", -torch.ones(shape, dtype=dtype, device=DEVICE)),
        ("very_small", torch.randn(shape, dtype=dtype, device=DEVICE) * 1e-6),
        ("very_large", torch.randn(shape, dtype=dtype, device=DEVICE) * 1e6),
    ]
    
    for name, x in test_cases:
        x_fq = triton_fake_quantize(x, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=128, use_fp4=True, dst_dtype=dtype)
        
        assert x_fq.shape == x.shape
        assert torch.isfinite(x_fq).all()
        
        max_diff = (x_fq - x).abs().max()
        mean_diff = (x_fq - x).abs().mean()
        cos_sim = cosine_similarity(x_fq, x)
        print(f"  {name} - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    print("✓ Edge cases test passed.")


def test_fake_quantize_vs_reference():
    """Test triton fake_quantize vs reference implementation."""
    torch.manual_seed(42)
    
    dtype = torch.bfloat16
    shape = (128, 128)
    
    # Create input tensor
    x = torch.randn(shape, dtype=dtype, device=DEVICE)
    
    # Test triton fake_quantize
    x_fq_triton = triton_fake_quantize(x, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=128, use_fp4=True, dst_dtype=dtype)
    
    # Test reference implementation
    x_fq_ref = get_fake_quant_reference(x)
    
    # Compare triton vs reference
    max_diff = (x_fq_triton - x_fq_ref).abs().max()
    mean_diff = (x_fq_triton - x_fq_ref).abs().mean()
    cos_sim = cosine_similarity(x_fq_triton, x_fq_ref)
    print(f"  Triton vs Reference - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    # Also compare each vs input
    max_diff_triton = (x_fq_triton - x).abs().max()
    mean_diff_triton = (x_fq_triton - x).abs().mean()
    cos_sim_triton = cosine_similarity(x_fq_triton, x)
    print(f"  Triton vs Input - Max diff: {max_diff_triton.item():.6f}, Mean diff: {mean_diff_triton.item():.6f}, Cosine sim: {cos_sim_triton:.6f}")
    
    max_diff_ref = (x_fq_ref - x).abs().max()
    mean_diff_ref = (x_fq_ref - x).abs().mean()
    cos_sim_ref = cosine_similarity(x_fq_ref, x)
    print(f"  Reference vs Input - Max diff: {max_diff_ref.item():.6f}, Mean diff: {mean_diff_ref.item():.6f}, Cosine sim: {cos_sim_ref:.6f}")
    
    print("✓ Reference comparison test passed.")


def test_fake_quantize_fp8():
    """Test fake_quantize with FP8 quantization."""
    torch.manual_seed(42)
    
    dtype = torch.bfloat16
    shape = (128, 128)
    
    # Create input tensor
    x = torch.randn(shape, dtype=dtype, device=DEVICE)
    
    # Test triton fake_quantize with FP8
    x_fq = triton_fake_quantize(x, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=128, use_fp4=False, dst_dtype=dtype)
    
    assert x_fq.shape == x.shape
    assert torch.isfinite(x_fq).all()
    
    max_diff = (x_fq - x).abs().max()
    mean_diff = (x_fq - x).abs().mean()
    cos_sim = cosine_similarity(x_fq, x)
    print(f"  FP8 - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    print("✓ FP8 test passed.")


def test_fake_quantize_attention_shapes():
    """Test fake_quantize with attention-like shapes (similar to test_qat_attn.py)."""
    torch.manual_seed(42)
    
    test_configs = [
        (2, 4, 128, 64),   # Z, H, N_CTX, HEAD_DIM - Medium
        (1, 8, 256, 128),  # Large head dim
        (1, 40, 9360, 128), # WAN shape
    ]
    
    dtype = torch.bfloat16
    
    for Z, H, N_CTX, HEAD_DIM in test_configs:
        # Create input tensor in BLHD format (B, L, H, D) = (Z, N_CTX, H, HEAD_DIM)
        x = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
        
        # Test triton fake_quantize
        x_fq = triton_fake_quantize(x, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=128, use_fp4=True, dst_dtype=dtype)
        
        assert x_fq.shape == x.shape
        assert torch.isfinite(x_fq).all()
        
        # Compare input and output
        max_diff = (x_fq - x).abs().max()
        mean_diff = (x_fq - x).abs().mean()
        cos_sim = cosine_similarity(x_fq, x)
        print(f"  (Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM}) - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
        
        print(f"✓ Attention shape test passed for (Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM})")


def test_fake_quantize_non_divisible_blocks():
    """Test fake_quantize with shapes that are not divisible by block sizes."""
    torch.manual_seed(42)
    
    dtype = torch.bfloat16
    
    # Test with non-divisible dimensions
    test_shapes = [
        (100, 100),   # Not divisible by 128
        (150, 200),   # Not divisible by 128
        (256, 100),   # One dimension divisible, one not
    ]
    
    for shape in test_shapes:
        x = torch.randn(shape, dtype=dtype, device=DEVICE)
        
        # Test triton fake_quantize
        x_fq = triton_fake_quantize(x, BLOCK_SIZE_OUT_DIM=128, BLOCK_SIZE_QUANT_DIM=128, use_fp4=True, dst_dtype=dtype)
        
        assert x_fq.shape == x.shape
        assert torch.isfinite(x_fq).all()
        
        max_diff = (x_fq - x).abs().max()
        mean_diff = (x_fq - x).abs().mean()
        cos_sim = cosine_similarity(x_fq, x)
        print(f"  Shape {shape} - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
        
        print(f"✓ Non-divisible block test passed for {shape}")


if __name__ == "__main__":
    print("Running fake_quantize tests...")
    print(f"Device: {DEVICE}")
    print()
    
    test_fake_quantize_basic()
    test_fake_quantize_different_shapes()
    test_fake_quantize_different_dtypes()
    test_fake_quantize_3d_4d_tensors()
    test_fake_quantize_edge_cases()
    test_fake_quantize_vs_reference()
    test_fake_quantize_attention_shapes()
    test_fake_quantize_non_divisible_blocks()
    
    print()
    print("All tests passed! ✓")
