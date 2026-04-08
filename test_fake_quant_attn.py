#!/usr/bin/env python3
"""
Test script for fake quantization of attention computation using get_fake_quant method.
Tests attention computation with fake-quantized Q, K, V tensors and compares with reference.
"""

import torch
import torch.nn.functional as F
from math import sqrt
from test_quantization_precision import get_fake_quant
import sys
import os
from contextlib import contextmanager


# Global flag to control verbosity
VERBOSE = False


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout."""
    if not VERBOSE:
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
    else:
        yield


def scaled_dot_product_attention_reference(q, k, v, causal=False, scale=None):
    """
    Reference implementation of scaled dot-product attention.
    
    Args:
        q: Query tensor [..., seq_len_q, head_dim]
        k: Key tensor [..., seq_len_kv, head_dim]
        v: Value tensor [..., seq_len_kv, head_dim]
        causal: Whether to apply causal masking
        scale: Scale factor (default: 1/sqrt(head_dim))
    
    Returns:
        Output tensor [..., seq_len_q, head_dim]
    """
    if scale is None:
        scale = 1.0 / sqrt(q.shape[-1])
    
    # Compute attention scores: Q @ K^T
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Apply causal mask if needed
    if causal:
        seq_len = scores.shape[-1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device, dtype=scores.dtype), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        scores = scores + causal_mask
    
    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply attention weights to values
    output = torch.matmul(attn_weights, v)
    
    return output


def attention_with_fake_quant(q, k, v, causal=False, scale=None, quantize_qkv=True):
    """
    Compute attention with fake quantization applied to Q, K, V tensors.
    
    Args:
        q: Query tensor [..., seq_len_q, head_dim]
        k: Key tensor [..., seq_len_kv, head_dim]
        v: Value tensor [..., seq_len_kv, head_dim]
        causal: Whether to apply causal masking
        scale: Scale factor (default: 1/sqrt(head_dim))
        quantize_qkv: Whether to apply fake quantization to Q, K, V
    
    Returns:
        Output tensor [..., seq_len_q, head_dim]
    """
    if quantize_qkv:
        # Apply fake quantization to Q, K, V
        if VERBOSE:
            print("  Applying fake quantization to Q, K, V...")
        with suppress_stdout():
            q = get_fake_quant(q)
            k = get_fake_quant(k)
            v = get_fake_quant(v)
    
    # Use PyTorch's scaled_dot_product_attention
    return F.scaled_dot_product_attention(
        q, k, v,
        is_causal=causal,
        scale=scale
    )


def test_attention_fake_quant_basic():
    """Test basic attention computation with fake quantization."""
    print("\n" + "="*60)
    print("Test 1: Basic Attention with Fake Quantization")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    device = "cuda"
    dtype = torch.bfloat16
    
    # Test parameters
    batch_size = 2
    num_heads = 4
    seq_len = 128
    head_dim = 64
    
    # Create input tensors
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    
    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    print(f"Input dtype: {dtype}")
    print(f"Q stats: min={q.min().item():.4f}, max={q.max().item():.4f}, mean={q.mean().item():.4f}")
    
    # Reference attention (no quantization)
    print("\nComputing reference attention (no quantization)...")
    with torch.no_grad():
        ref_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    
    # Attention with fake quantization
    print("\nComputing attention with fake quantization...")
    with torch.no_grad():
        quant_output = attention_with_fake_quant(q, k, v, causal=False, quantize_qkv=True)
    
    # Compare results
    diff = torch.abs(ref_output - quant_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    relative_diff = (diff / (torch.abs(ref_output) + 1e-8)).max().item()
    
    print(f"\nResults:")
    print(f"  Reference output stats: min={ref_output.min().item():.4f}, max={ref_output.max().item():.4f}, mean={ref_output.mean().item():.4f}")
    print(f"  Quantized output stats: min={quant_output.min().item():.4f}, max={quant_output.max().item():.4f}, mean={quant_output.mean().item():.4f}")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    print(f"  Max relative difference: {relative_diff:.6f}")
    
    # Check that outputs are finite
    assert torch.isfinite(ref_output).all(), "Reference output contains NaN/Inf"
    assert torch.isfinite(quant_output).all(), "Quantized output contains NaN/Inf"
    
    print("✓ Basic test passed!")
    return max_diff, mean_diff, relative_diff


def test_attention_fake_quant_causal():
    """Test causal attention with fake quantization."""
    print("\n" + "="*60)
    print("Test 2: Causal Attention with Fake Quantization")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    device = "cuda"
    dtype = torch.bfloat16
    
    batch_size = 1
    num_heads = 2
    seq_len = 64
    head_dim = 32
    
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    
    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    
    # Reference causal attention
    print("\nComputing reference causal attention...")
    with torch.no_grad():
        ref_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    # Causal attention with fake quantization
    print("\nComputing causal attention with fake quantization...")
    with torch.no_grad():
        quant_output = attention_with_fake_quant(q, k, v, causal=True, quantize_qkv=True)
    
    # Compare results
    diff = torch.abs(ref_output - quant_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nResults:")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    
    assert torch.isfinite(ref_output).all()
    assert torch.isfinite(quant_output).all()
    
    print("✓ Causal attention test passed!")
    return max_diff, mean_diff


def test_attention_fake_quant_different_shapes():
    """Test attention with different tensor shapes."""
    print("\n" + "="*60)
    print("Test 3: Different Shapes")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    device = "cuda"
    dtype = torch.bfloat16
    
    test_configs = [
        (1, 1, 32, 16, "small"),
        (2, 4, 128, 64, "medium"),
        (1, 8, 256, 128, "large_seq"),
        (2, 4, 64, 256, "large_head_dim"),
    ]
    
    results = []
    
    for batch_size, num_heads, seq_len, head_dim, name in test_configs:
        print(f"\nTesting {name} configuration: B={batch_size}, H={num_heads}, L={seq_len}, D={head_dim}")
        
        torch.manual_seed(42)
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        
        with torch.no_grad():
            ref_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            quant_output = attention_with_fake_quant(q, k, v, causal=False, quantize_qkv=True)
        
        diff = torch.abs(ref_output - quant_output)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        
        assert torch.isfinite(ref_output).all()
        assert torch.isfinite(quant_output).all()
        assert ref_output.shape == quant_output.shape
        
        results.append((name, max_diff, mean_diff))
    
    print("\n✓ All shape tests passed!")
    return results


def test_attention_fake_quant_gradient():
    """Test that gradients can flow through fake-quantized attention."""
    print("\n" + "="*60)
    print("Test 4: Gradient Flow")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    device = "cuda"
    dtype = torch.bfloat16
    
    batch_size = 1
    num_heads = 2
    seq_len = 32
    head_dim = 32
    
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
    
    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    print("Note: get_fake_quant uses no_grad internally, so gradients won't flow through quantization")
    
    # Forward pass with fake quantization
    # Note: get_fake_quant uses no_grad, so we'll test that it doesn't crash
    with torch.no_grad():
        q_quant = get_fake_quant(q)
        k_quant = get_fake_quant(k)
        v_quant = get_fake_quant(v)
    
    # Now compute attention with quantized tensors (requires_grad=False)
    q_quant.requires_grad_(True)
    k_quant.requires_grad_(True)
    v_quant.requires_grad_(True)
    
    output = F.scaled_dot_product_attention(q_quant, k_quant, v_quant, is_causal=False)
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    # Check that gradients exist
    assert q_quant.grad is not None
    assert k_quant.grad is not None
    assert v_quant.grad is not None
    
    print("✓ Gradient flow test passed!")
    print(f"  Gradient shapes: Q={q_quant.grad.shape}, K={k_quant.grad.shape}, V={v_quant.grad.shape}")


def test_attention_fake_quant_precision():
    """Test precision comparison between quantized and non-quantized attention."""
    print("\n" + "="*60)
    print("Test 5: Precision Analysis")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    device = "cuda"
    dtype = torch.bfloat16
    
    batch_size = 2
    num_heads = 4
    seq_len = 128
    head_dim = 64
    
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    
    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    
    # Compute reference
    with torch.no_grad():
        ref_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    
    # Compute with quantization
    with torch.no_grad():
        quant_output = attention_with_fake_quant(q, k, v, causal=False, quantize_qkv=True)
    
    # Detailed analysis
    diff = torch.abs(ref_output - quant_output)
    
    print("\nPrecision Analysis:")
    print(f"  Output shape: {ref_output.shape}")
    print(f"  Total elements: {ref_output.numel()}")
    print(f"  Max absolute error: {diff.max().item():.6f}")
    print(f"  Mean absolute error: {diff.mean().item():.6f}")
    print(f"  Median absolute error: {diff.median().item():.6f}")
    print(f"  Std of errors: {diff.std().item():.6f}")
    
    # Relative error
    relative_diff = diff / (torch.abs(ref_output) + 1e-8)
    print(f"  Max relative error: {relative_diff.max().item():.6f}")
    print(f"  Mean relative error: {relative_diff.mean().item():.6f}")
    
    # Error distribution (convert to float32 for quantile computation)
    error_percentiles = torch.quantile(diff.float(), torch.tensor([0.5, 0.9, 0.95, 0.99], device=device, dtype=torch.float32))
    print(f"  Error percentiles (50th, 90th, 95th, 99th): {error_percentiles.cpu().tolist()}")
    
    print("\n✓ Precision analysis completed!")


def main():
    """Run all tests."""
    global VERBOSE
    
    # Check for verbose flag
    VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv
    
    print("="*60)
    print("Fake Quantization Attention Test Suite")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. These tests require CUDA.")
        sys.exit(1)
    
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch version: {torch.__version__}")
    if VERBOSE:
        print("Verbose mode: ON (will show quantization details)")
    print()
    
    try:
        # Run tests
        test_attention_fake_quant_basic()
        test_attention_fake_quant_causal()
        test_attention_fake_quant_different_shapes()
        test_attention_fake_quant_gradient()
        test_attention_fake_quant_precision()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


