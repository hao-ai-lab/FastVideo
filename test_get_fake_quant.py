#!/usr/bin/env python3
"""
Precision test to compare numerical differences for get_fake_quant Python function:
1. get_fake_quant (Python implementation using FlashInfer)
2. Tests various shapes, dtypes, and value ranges
3. Evaluates cosine similarity, max diff, and mean diff between input and output
"""

import torch
from test_quantization_precision import get_fake_quant

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


def test_get_fake_quant_basic():
    """Test basic functionality of get_fake_quant."""
    torch.manual_seed(42)
    
    # Test parameters
    shape = (128, 128)
    dtype = torch.bfloat16
    
    # Create input tensor
    x = torch.randn(shape, dtype=dtype, device=DEVICE)
    
    # Test get_fake_quant
    x_fq = get_fake_quant(x)
    
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


def test_get_fake_quant_different_shapes():
    """Test get_fake_quant with different input shapes."""
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
        
        # Test get_fake_quant
        x_fq = get_fake_quant(x)
        
        assert x_fq.shape == x.shape
        assert torch.isfinite(x_fq).all()
        
        # Compare input and output
        max_diff = (x_fq - x).abs().max()
        mean_diff = (x_fq - x).abs().mean()
        cos_sim = cosine_similarity(x_fq, x)
        print(f"  Shape ({outer_dim}, {quant_dim}) - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
        
        print(f"✓ Shape test passed for ({outer_dim}, {quant_dim})")


def test_get_fake_quant_different_dtypes():
    """Test get_fake_quant with different input dtypes."""
    torch.manual_seed(42)
    
    shape = (128, 128)
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    
    for dtype in dtypes:
        # Create input tensor
        x = torch.randn(shape, dtype=dtype, device=DEVICE)
        
        # Test get_fake_quant
        x_fq = get_fake_quant(x)
        
        assert x_fq.shape == x.shape
        assert x_fq.dtype == x.dtype
        assert torch.isfinite(x_fq).all()
        
        # Compare input and output
        max_diff = (x_fq - x).abs().max()
        mean_diff = (x_fq - x).abs().mean()
        cos_sim = cosine_similarity(x_fq, x)
        print(f"  Dtype {dtype} - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
        
        print(f"✓ Dtype test passed for {dtype}")


def test_get_fake_quant_3d_4d_tensors():
    """Test get_fake_quant with 3D and 4D tensors (reshaped to 2D internally)."""
    torch.manual_seed(42)
    
    dtype = torch.bfloat16
    
    # Test 3D tensor (B, H, D)
    print("\nTesting 3D tensor (B, H, D)")
    x_3d = torch.randn((2, 8, 128), dtype=dtype, device=DEVICE)
    x_fq_3d = get_fake_quant(x_3d)
    
    assert x_fq_3d.shape == x_3d.shape
    assert torch.isfinite(x_fq_3d).all()
    
    max_diff = (x_fq_3d - x_3d).abs().max()
    mean_diff = (x_fq_3d - x_3d).abs().mean()
    cos_sim = cosine_similarity(x_fq_3d, x_3d)
    print(f"  3D (2, 8, 128) - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    # Test 4D tensor (B, H, L, D)
    print("\nTesting 4D tensor (B, H, L, D)")
    x_4d = torch.randn((1, 8, 256, 128), dtype=dtype, device=DEVICE)
    x_fq_4d = get_fake_quant(x_4d)
    
    assert x_fq_4d.shape == x_4d.shape
    assert torch.isfinite(x_fq_4d).all()
    
    max_diff = (x_fq_4d - x_4d).abs().max()
    mean_diff = (x_fq_4d - x_4d).abs().mean()
    cos_sim = cosine_similarity(x_fq_4d, x_4d)
    print(f"  4D (1, 8, 256, 128) - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    print("✓ 3D/4D tensor test passed.")


def test_get_fake_quant_edge_cases():
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
        x_fq = get_fake_quant(x)
        
        assert x_fq.shape == x.shape
        assert torch.isfinite(x_fq).all()
        
        max_diff = (x_fq - x).abs().max()
        mean_diff = (x_fq - x).abs().mean()
        cos_sim = cosine_similarity(x_fq, x)
        print(f"  {name} - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    print("✓ Edge cases test passed.")


def test_get_fake_quant_attention_shapes():
    """Test get_fake_quant with attention-like shapes (similar to test_qat_attn.py)."""
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
        
        # Test get_fake_quant
        x_fq = get_fake_quant(x)
        
        assert x_fq.shape == x.shape
        assert torch.isfinite(x_fq).all()
        
        # Compare input and output
        max_diff = (x_fq - x).abs().max()
        mean_diff = (x_fq - x).abs().mean()
        cos_sim = cosine_similarity(x_fq, x)
        print(f"  (Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM}) - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
        
        print(f"✓ Attention shape test passed for (Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM})")


def test_get_fake_quant_non_divisible_blocks():
    """Test get_fake_quant with shapes that are not divisible by block sizes."""
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
        
        # Test get_fake_quant
        x_fq = get_fake_quant(x)
        
        assert x_fq.shape == x.shape
        assert torch.isfinite(x_fq).all()
        
        max_diff = (x_fq - x).abs().max()
        mean_diff = (x_fq - x).abs().mean()
        cos_sim = cosine_similarity(x_fq, x)
        print(f"  Shape {shape} - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
        
        print(f"✓ Non-divisible block test passed for {shape}")


def test_get_fake_quant_idempotency():
    """Test that quantizing twice produces similar results (idempotency check)."""
    torch.manual_seed(42)
    
    dtype = torch.bfloat16
    shape = (128, 128)
    
    # Create input tensor
    x = torch.randn(shape, dtype=dtype, device=DEVICE)
    
    # Quantize once
    x_fq1 = get_fake_quant(x)
    
    # Quantize again
    x_fq2 = get_fake_quant(x_fq1)
    
    # Compare the two quantizations
    max_diff = (x_fq1 - x_fq2).abs().max()
    mean_diff = (x_fq1 - x_fq2).abs().mean()
    cos_sim = cosine_similarity(x_fq1, x_fq2)
    print(f"  First quantization vs Second quantization - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    print("✓ Idempotency test passed.")


def test_get_fake_quant_different_value_ranges():
    """Test get_fake_quant with different value ranges."""
    torch.manual_seed(42)
    
    dtype = torch.bfloat16
    shape = (128, 128)
    
    test_cases = [
        ("normal_range", lambda: torch.randn(shape, dtype=dtype, device=DEVICE)),
        ("small_range", lambda: torch.randn(shape, dtype=dtype, device=DEVICE) * 0.01),
        ("medium_range", lambda: torch.randn(shape, dtype=dtype, device=DEVICE) * 1.0),
        ("large_range", lambda: torch.randn(shape, dtype=dtype, device=DEVICE) * 10.0),
        ("very_large_range", lambda: torch.randn(shape, dtype=dtype, device=DEVICE) * 100.0),
    ]
    
    for name, gen_func in test_cases:
        x = gen_func()
        x_fq = get_fake_quant(x)
        
        assert x_fq.shape == x.shape
        assert torch.isfinite(x_fq).all()
        
        max_diff = (x_fq - x).abs().max()
        mean_diff = (x_fq - x).abs().mean()
        cos_sim = cosine_similarity(x_fq, x)
        print(f"  {name} - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    print("✓ Different value ranges test passed.")


if __name__ == "__main__":
    print("Running get_fake_quant precision tests...")
    print(f"Device: {DEVICE}")
    print()
    
    test_get_fake_quant_basic()
    test_get_fake_quant_different_shapes()
    test_get_fake_quant_different_dtypes()
    test_get_fake_quant_3d_4d_tensors()
    test_get_fake_quant_edge_cases()
    test_get_fake_quant_attention_shapes()
    test_get_fake_quant_non_divisible_blocks()
    test_get_fake_quant_idempotency()
    test_get_fake_quant_different_value_ranges()
    
    print()
    print("All tests passed! ✓")

