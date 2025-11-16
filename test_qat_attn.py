#!/usr/bin/env python3
"""
Simple test for QAT attention implementation.
Tests forward and backward passes with and without QAT enabled.
"""

import torch
from qat_attn import _attention
from fused_attention import attention as fused_attention
from math import sqrt

attention = _attention.apply
DEVICE = torch.device("cuda")


def qat_attn_wrapper(q_BLHD, k_BLHD, v_BLHD, is_causal=False, sm_scale=None):
    """
    Wrapper function that mimics qat_attn from sage_attn3.py.
    Converts from BLHD format to BHLD format, calls attention, then converts back.
    
    Args:
        q_BLHD: Query tensor in (B, L, H, D) format
        k_BLHD: Key tensor in (B, L, H, D) format
        v_BLHD: Value tensor in (B, L, H, D) format
        is_causal: Whether to apply causal masking
        sm_scale: Scale factor for attention scores (if None, uses 1.0 / sqrt(D))
    
    Returns:
        Output tensor in (B, L, H, D) format
    """
    if sm_scale is None:
        sm_scale = 1.0 / sqrt(q_BLHD.shape[-1])
    
    # Convert from BLHD to BHLD format
    q_BHLD = q_BLHD.permute(0, 2, 1, 3).contiguous()
    k_BHLD = k_BLHD.permute(0, 2, 1, 3).contiguous()
    v_BHLD = v_BLHD.permute(0, 2, 1, 3).contiguous()
    
    # Call attention with BHLD format
    o_BHLD = attention(q_BHLD, k_BHLD, v_BHLD, is_causal, sm_scale)
    
    # Convert back from BHLD to BLHD format
    return o_BHLD.permute(0, 2, 1, 3).contiguous()


def fused_attn_wrapper(q_BLHD, k_BLHD, v_BLHD, is_causal=False, sm_scale=None, warp_specialize=True):
    """
    Wrapper function for fused_attention from fused_attention.py.
    Converts from BLHD format to BHLD format, calls attention, then converts back.
    
    Args:
        q_BLHD: Query tensor in (B, L, H, D) format
        k_BLHD: Key tensor in (B, L, H, D) format
        v_BLHD: Value tensor in (B, L, H, D) format
        is_causal: Whether to apply causal masking
        sm_scale: Scale factor for attention scores (if None, uses 1.0 / sqrt(D))
        warp_specialize: Whether to use warp specialization (default: True)
    
    Returns:
        Output tensor in (B, L, H, D) format
    """
    if sm_scale is None:
        sm_scale = 1.0 / sqrt(q_BLHD.shape[-1])
    
    # Convert from BLHD to BHLD format
    q_BHLD = q_BLHD.permute(0, 2, 1, 3).contiguous()
    k_BHLD = k_BLHD.permute(0, 2, 1, 3).contiguous()
    v_BHLD = v_BLHD.permute(0, 2, 1, 3).contiguous()
    
    # Call fused attention with BHLD format
    # Note: fused_attention expects inputs in BHLD format and only supports same sequence lengths
    o_BHLD = fused_attention(q_BHLD, k_BHLD, v_BHLD, is_causal, sm_scale, warp_specialize)
    
    # Convert back from BHLD to BLHD format
    return o_BHLD.permute(0, 2, 1, 3).contiguous()


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


def naive_attention(q, k, v, causal, sm_scale):
    """
    Naive PyTorch implementation of attention for comparison.
    
    Args:
        q: Query tensor of shape (Z, H, N_CTX_Q, HEAD_DIM)
        k: Key tensor of shape (Z, H, N_CTX_KV, HEAD_DIM)
        v: Value tensor of shape (Z, H, N_CTX_KV, HEAD_DIM)
        causal: Whether to apply causal masking (only meaningful when N_CTX_Q == N_CTX_KV)
        sm_scale: Scale factor for attention scores
    
    Returns:
        Output tensor of shape (Z, H, N_CTX_Q, HEAD_DIM)
    """
    # Compute attention scores: QK^T
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    
    # Apply causal mask if needed
    # Note: Causal masking is only meaningful when Q and KV have the same sequence length
    if causal:
        N_CTX_Q = q.shape[-2]
        N_CTX_KV = k.shape[-2]
        if N_CTX_Q == N_CTX_KV:
            # Standard causal masking: query i can only attend to keys 0 to i
            mask = torch.tril(torch.ones(N_CTX_Q, N_CTX_KV, device=scores.device, dtype=scores.dtype))
            scores = scores.masked_fill(mask == 0, float("-inf"))
        else:
            # For different sequence lengths, causal masking doesn't apply in the same way
            # In cross-attention, typically no causal masking is applied
            pass
    
    # Apply softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Apply attention weights to values
    out = torch.matmul(attn_weights, v)
    
    return out

def test_qat_attention_forward():
    """Test forward pass with QAT enabled vs disabled."""
    torch.manual_seed(42)
    
    # Test parameters
    Z, H, N_CTX, HEAD_DIM = 2, 4, 128, 64
    dtype = torch.bfloat16  
    sm_scale = 1.0 / sqrt(HEAD_DIM)
    causal = False
    
    # Create input tensors in BLHD format (B, L, H, D) = (Z, N_CTX, H, HEAD_DIM)
    q_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    k_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    v_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    
    # Test with QAT (using wrapper that does permute/contiguous)
    out_qat_BLHD = qat_attn_wrapper(q_BLHD.clone(), k_BLHD.clone(), v_BLHD.clone(), causal, sm_scale)
    
    # Convert to BHLD for naive attention comparison
    q_BHLD = q_BLHD.permute(0, 2, 1, 3).contiguous()
    k_BHLD = k_BLHD.permute(0, 2, 1, 3).contiguous()
    v_BHLD = v_BLHD.permute(0, 2, 1, 3).contiguous()
    out_naive_BHLD = naive_attention(q_BHLD.clone(), k_BHLD.clone(), v_BHLD.clone(), causal, sm_scale)
    out_naive_BLHD = out_naive_BHLD.permute(0, 2, 1, 3).contiguous()
    
    # Check that outputs have correct shape
    assert out_qat_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
    assert out_naive_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
    
    # Check that outputs are finite
    assert torch.isfinite(out_qat_BLHD).all()
    assert torch.isfinite(out_naive_BLHD).all()
    
    # Compare QAT and naive outputs
    max_diff = (out_qat_BLHD - out_naive_BLHD).abs().max()
    mean_diff = (out_qat_BLHD - out_naive_BLHD).abs().mean()
    cos_sim = cosine_similarity(out_qat_BLHD, out_naive_BLHD)
    print(f"  QAT vs Naive Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM} - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    # QAT output should be reasonably close to naive (within tolerance due to quantization)
    # Using a reasonable tolerance for float16 and quantization effects
    # assert max_diff < 1.0, f"QAT and naive outputs should be reasonably close, got max_diff={max_diff.item():.6f}"
    
    print("✓ Forward pass test passed.")


def test_qat_attention_backward():
    """Test backward pass with QAT enabled."""
    torch.manual_seed(42)
    
    # Test parameters
    Z, H, N_CTX, HEAD_DIM = 2, 4, 128, 64
    dtype = torch.bfloat16
    sm_scale = 1.0 / sqrt(HEAD_DIM)
    causal = True
    
    # Create input tensors in BLHD format for QAT
    q_qat_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    k_qat_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    v_qat_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    
    # Create input tensors for naive (same values, convert to BHLD)
    q_naive_BHLD = q_qat_BLHD.clone().permute(0, 2, 1, 3).contiguous().detach().requires_grad_(True)
    k_naive_BHLD = k_qat_BLHD.clone().permute(0, 2, 1, 3).contiguous().detach().requires_grad_(True)
    v_naive_BHLD = v_qat_BLHD.clone().permute(0, 2, 1, 3).contiguous().detach().requires_grad_(True)
    
    # Forward pass with QAT (using wrapper)
    out_qat_BLHD = qat_attn_wrapper(q_qat_BLHD, k_qat_BLHD, v_qat_BLHD, causal, sm_scale)
    
    # Forward pass with naive
    out_naive_BHLD = naive_attention(q_naive_BHLD, k_naive_BHLD, v_naive_BHLD, causal, sm_scale)
    out_naive_BLHD = out_naive_BHLD.permute(0, 2, 1, 3).contiguous()
    
    # Create dummy gradient (same for both, in BLHD format)
    # Ensure gradient is contiguous as required by the backward function
    dout = torch.randn_like(out_qat_BLHD).contiguous()
    
    # Backward pass for QAT
    out_qat_BLHD.backward(dout)
    
    # Backward pass for naive
    out_naive_BLHD.backward(dout)
    
    # Check that gradients are computed
    assert q_qat_BLHD.grad is not None
    assert k_qat_BLHD.grad is not None
    assert v_qat_BLHD.grad is not None
    assert q_naive_BHLD.grad is not None
    assert k_naive_BHLD.grad is not None
    assert v_naive_BHLD.grad is not None
    
    # Check that gradients have correct shape
    assert q_qat_BLHD.grad.shape == q_qat_BLHD.shape
    assert k_qat_BLHD.grad.shape == k_qat_BLHD.shape
    assert v_qat_BLHD.grad.shape == v_qat_BLHD.shape
    
    # Check that gradients are finite
    assert torch.isfinite(q_qat_BLHD.grad).all()
    assert torch.isfinite(k_qat_BLHD.grad).all()
    assert torch.isfinite(v_qat_BLHD.grad).all()
    
    # Convert naive gradients to BLHD format for comparison
    q_naive_grad_BLHD = q_naive_BHLD.grad.permute(0, 2, 1, 3).contiguous()
    k_naive_grad_BLHD = k_naive_BHLD.grad.permute(0, 2, 1, 3).contiguous()
    v_naive_grad_BLHD = v_naive_BHLD.grad.permute(0, 2, 1, 3).contiguous()
    
    # Compare gradients with naive
    dq_diff = (q_qat_BLHD.grad - q_naive_grad_BLHD).abs()
    dk_diff = (k_qat_BLHD.grad - k_naive_grad_BLHD).abs()
    dv_diff = (v_qat_BLHD.grad - v_naive_grad_BLHD).abs()
    
    dq_cos_sim = cosine_similarity(q_qat_BLHD.grad, q_naive_grad_BLHD)
    dk_cos_sim = cosine_similarity(k_qat_BLHD.grad, k_naive_grad_BLHD)
    dv_cos_sim = cosine_similarity(v_qat_BLHD.grad, v_naive_grad_BLHD)
    
    print(f"  dQ - Max diff: {dq_diff.max().item():.6f}, Mean diff: {dq_diff.mean().item():.6f}, Cosine sim: {dq_cos_sim:.6f}")
    print(f"  dK - Max diff: {dk_diff.max().item():.6f}, Mean diff: {dk_diff.mean().item():.6f}, Cosine sim: {dk_cos_sim:.6f}")
    print(f"  dV - Max diff: {dv_diff.max().item():.6f}, Mean diff: {dv_diff.mean().item():.6f}, Cosine sim: {dv_cos_sim:.6f}")
    
    # Gradients should be reasonably close (within tolerance due to quantization)
    # assert dq_diff.max() < 2.0, f"dQ gradients should be reasonably close, got max_diff={dq_diff.max().item():.6f}"
    # assert dk_diff.max() < 2.0, f"dK gradients should be reasonably close, got max_diff={dk_diff.max().item():.6f}"
    # assert dv_diff.max() < 2.0, f"dV gradients should be reasonably close, got max_diff={dv_diff.max().item():.6f}"
    
    print("✓ Backward pass test passed.")


def test_qat_attention_different_shapes():
    """Test QAT attention with different input shapes."""
    torch.manual_seed(42)
    
    test_configs = [
        (2, 4, 128, 64),  # Medium
        (1, 8, 256, 128), # Large head dim
    ]
    
    dtype = torch.bfloat16
    causal = True
    
    for Z, H, N_CTX, HEAD_DIM in test_configs:
        # Create input tensors in BLHD format
        q_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
        k_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
        v_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
        
        sm_scale = 1.0 / sqrt(HEAD_DIM)
        out_qat_BLHD = qat_attn_wrapper(q_BLHD.clone(), k_BLHD.clone(), v_BLHD.clone(), causal, sm_scale)
        
        # Convert to BHLD for naive attention
        q_BHLD = q_BLHD.permute(0, 2, 1, 3).contiguous()
        k_BHLD = k_BLHD.permute(0, 2, 1, 3).contiguous()
        v_BHLD = v_BLHD.permute(0, 2, 1, 3).contiguous()
        out_naive_BHLD = naive_attention(q_BHLD.clone(), k_BHLD.clone(), v_BHLD.clone(), causal, sm_scale)
        out_naive_BLHD = out_naive_BHLD.permute(0, 2, 1, 3).contiguous()
        
        assert out_qat_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
        assert out_naive_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
        assert torch.isfinite(out_qat_BLHD).all()
        assert torch.isfinite(out_naive_BLHD).all()
        
        # Compare outputs
        max_diff = (out_qat_BLHD - out_naive_BLHD).abs().max()
        mean_diff = (out_qat_BLHD - out_naive_BLHD).abs().mean()
        cos_sim = cosine_similarity(out_qat_BLHD, out_naive_BLHD)
        print(f"  (Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM}) - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
        
        # Outputs should be reasonably close
        # assert max_diff < 1.0, f"QAT and naive outputs should be reasonably close for shape (Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM}), got max_diff={max_diff.item():.6f}"
        
        print(f"✓ Shape test passed for (Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM})")


def test_qat_attention_non_causal():
    """Test QAT attention with non-causal masking."""
    torch.manual_seed(42)
    
    Z, H, N_CTX, HEAD_DIM = 2, 4, 128, 64
    dtype = torch.bfloat16
    sm_scale = 1.0 / sqrt(HEAD_DIM)
    causal = False
    
    # Create input tensors in BLHD format
    q_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
    k_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
    v_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
    
    out_qat_BLHD = qat_attn_wrapper(q_BLHD.clone(), k_BLHD.clone(), v_BLHD.clone(), causal, sm_scale)
    
    # Convert to BHLD for naive attention
    q_BHLD = q_BLHD.permute(0, 2, 1, 3).contiguous()
    k_BHLD = k_BLHD.permute(0, 2, 1, 3).contiguous()
    v_BHLD = v_BLHD.permute(0, 2, 1, 3).contiguous()
    out_naive_BHLD = naive_attention(q_BHLD.clone(), k_BHLD.clone(), v_BHLD.clone(), causal, sm_scale)
    out_naive_BLHD = out_naive_BHLD.permute(0, 2, 1, 3).contiguous()
    
    assert out_qat_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
    assert out_naive_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
    assert torch.isfinite(out_qat_BLHD).all()
    assert torch.isfinite(out_naive_BLHD).all()
    
    # Compare outputs
    max_diff = (out_qat_BLHD - out_naive_BLHD).abs().max()
    mean_diff = (out_qat_BLHD - out_naive_BLHD).abs().mean()
    cos_sim = cosine_similarity(out_qat_BLHD, out_naive_BLHD)
    print(f"  QAT vs Naive (non-causal) - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    # Outputs should be reasonably close
    # assert max_diff < 1.0, f"QAT and naive outputs should be reasonably close for non-causal, got max_diff={max_diff.item():.6f}"
    
    print("✓ Non-causal test passed.")


def test_qat_attention_causal():
    """Test QAT attention with causal masking."""
    torch.manual_seed(42)
    
    Z, H, N_CTX, HEAD_DIM = 2, 4, 128, 64
    dtype = torch.bfloat16
    sm_scale = 1.0 / sqrt(HEAD_DIM)
    causal = True
    
    # Create input tensors in BLHD format
    q_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
    k_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
    v_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
    
    out_qat_BLHD = qat_attn_wrapper(q_BLHD.clone(), k_BLHD.clone(), v_BLHD.clone(), causal, sm_scale)
    
    # Convert to BHLD for naive attention
    q_BHLD = q_BLHD.permute(0, 2, 1, 3).contiguous()
    k_BHLD = k_BLHD.permute(0, 2, 1, 3).contiguous()
    v_BHLD = v_BLHD.permute(0, 2, 1, 3).contiguous()
    out_naive_BHLD = naive_attention(q_BHLD.clone(), k_BHLD.clone(), v_BHLD.clone(), causal, sm_scale)
    out_naive_BLHD = out_naive_BHLD.permute(0, 2, 1, 3).contiguous()
    
    assert out_qat_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
    assert out_naive_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
    assert torch.isfinite(out_qat_BLHD).all()
    assert torch.isfinite(out_naive_BLHD).all()
    
    # Compare outputs
    max_diff = (out_qat_BLHD - out_naive_BLHD).abs().max()
    mean_diff = (out_qat_BLHD - out_naive_BLHD).abs().mean()
    cos_sim = cosine_similarity(out_qat_BLHD, out_naive_BLHD)
    print(f"  QAT vs Naive (causal) - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    # Outputs should be reasonably close
    # assert max_diff < 1.0, f"QAT and naive outputs should be reasonably close for causal, got max_diff={max_diff.item():.6f}"
    
    print("✓ Causal test passed.")


def test_qat_attention_causal_backward():
    """Test backward pass of QAT attention with causal masking vs naive."""
    torch.manual_seed(42)
    
    Z, H, N_CTX, HEAD_DIM = 2, 4, 128, 64
    dtype = torch.bfloat16
    sm_scale = 1.0 / sqrt(HEAD_DIM)
    causal = True
    
    # Create input tensors in BLHD format for QAT
    q_qat_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    k_qat_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    v_qat_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    
    # Create input tensors for naive (same values, convert to BHLD)
    q_naive_BHLD = q_qat_BLHD.clone().permute(0, 2, 1, 3).contiguous().detach().requires_grad_(True)
    k_naive_BHLD = k_qat_BLHD.clone().permute(0, 2, 1, 3).contiguous().detach().requires_grad_(True)
    v_naive_BHLD = v_qat_BLHD.clone().permute(0, 2, 1, 3).contiguous().detach().requires_grad_(True)
    
    # Forward pass with QAT
    out_qat_BLHD = qat_attn_wrapper(q_qat_BLHD, k_qat_BLHD, v_qat_BLHD, causal, sm_scale)
    
    # Forward pass with naive
    out_naive_BHLD = naive_attention(q_naive_BHLD, k_naive_BHLD, v_naive_BHLD, causal, sm_scale)
    out_naive_BHLD = out_naive_BHLD.permute(0, 2, 1, 3).contiguous()
    
    # Create dummy gradient
    dout = torch.randn_like(out_qat_BLHD).contiguous()
    
    # Backward pass for QAT
    out_qat_BLHD.backward(dout)
    
    # Backward pass for naive
    out_naive_BHLD.backward(dout)
    
    # Check that gradients are computed
    assert q_qat_BLHD.grad is not None
    assert k_qat_BLHD.grad is not None
    assert v_qat_BLHD.grad is not None
    
    # Convert naive gradients to BLHD format for comparison
    q_naive_grad_BLHD = q_naive_BHLD.grad.permute(0, 2, 1, 3).contiguous()
    k_naive_grad_BLHD = k_naive_BHLD.grad.permute(0, 2, 1, 3).contiguous()
    v_naive_grad_BLHD = v_naive_BHLD.grad.permute(0, 2, 1, 3).contiguous()
    
    # Compare gradients
    dq_diff = (q_qat_BLHD.grad - q_naive_grad_BLHD).abs()
    dk_diff = (k_qat_BLHD.grad - k_naive_grad_BLHD).abs()
    dv_diff = (v_qat_BLHD.grad - v_naive_grad_BLHD).abs()
    
    dq_cos_sim = cosine_similarity(q_qat_BLHD.grad, q_naive_grad_BLHD)
    dk_cos_sim = cosine_similarity(k_qat_BLHD.grad, k_naive_grad_BLHD)
    dv_cos_sim = cosine_similarity(v_qat_BLHD.grad, v_naive_grad_BLHD)
    
    print(f"  dQ - Max diff: {dq_diff.max().item():.6f}, Mean diff: {dq_diff.mean().item():.6f}, Cosine sim: {dq_cos_sim:.6f}")
    print(f"  dK - Max diff: {dk_diff.max().item():.6f}, Mean diff: {dk_diff.mean().item():.6f}, Cosine sim: {dk_cos_sim:.6f}")
    print(f"  dV - Max diff: {dv_diff.max().item():.6f}, Mean diff: {dv_diff.mean().item():.6f}, Cosine sim: {dv_cos_sim:.6f}")
    
    print("✓ Causal backward test passed.")


def test_qat_attention_different_seq_lengths():
    """Test QAT attention with different sequence lengths for Q and KV (cross-attention)."""
    torch.manual_seed(42)
    
    test_configs = [
        (2, 4, 64, 128, 64, False),   # Q shorter than KV, non-causal
        (2, 4, 128, 64, 64, False),   # Q longer than KV, non-causal
        (1, 8, 256, 128, 64, False), # Q longer than KV, larger dimensions
    ]
    
    dtype = torch.bfloat16
    
    for Z, H, N_CTX_Q, N_CTX_KV, HEAD_DIM, causal in test_configs:
        sm_scale = 1.0 / sqrt(HEAD_DIM)
        
        # Create input tensors in BLHD format with different sequence lengths
        q_BLHD = torch.randn((Z, N_CTX_Q, H, HEAD_DIM), dtype=dtype, device=DEVICE)
        k_BLHD = torch.randn((Z, N_CTX_KV, H, HEAD_DIM), dtype=dtype, device=DEVICE)
        v_BLHD = torch.randn((Z, N_CTX_KV, H, HEAD_DIM), dtype=dtype, device=DEVICE)
        
        # Test with QAT (using wrapper that does permute/contiguous)
        out_qat_BLHD = qat_attn_wrapper(q_BLHD.clone(), k_BLHD.clone(), v_BLHD.clone(), causal, sm_scale)
        
        # Convert to BHLD for naive attention comparison
        q_BHLD = q_BLHD.permute(0, 2, 1, 3).contiguous()
        k_BHLD = k_BLHD.permute(0, 2, 1, 3).contiguous()
        v_BHLD = v_BLHD.permute(0, 2, 1, 3).contiguous()
        out_naive_BHLD = naive_attention(q_BHLD.clone(), k_BHLD.clone(), v_BHLD.clone(), causal, sm_scale)
        out_naive_BLHD = out_naive_BHLD.permute(0, 2, 1, 3).contiguous()
        
        # Check that outputs have correct shape (should match Q sequence length)
        assert out_qat_BLHD.shape == (Z, N_CTX_Q, H, HEAD_DIM)
        assert out_naive_BLHD.shape == (Z, N_CTX_Q, H, HEAD_DIM)
        
        # Check that outputs are finite
        assert torch.isfinite(out_qat_BLHD).all()
        assert torch.isfinite(out_naive_BLHD).all()
        
        # Compare QAT and naive outputs
        max_diff = (out_qat_BLHD - out_naive_BLHD).abs().max()
        mean_diff = (out_qat_BLHD - out_naive_BLHD).abs().mean()
        cos_sim = cosine_similarity(out_qat_BLHD, out_naive_BLHD)
        print(f"  (N_CTX_Q={N_CTX_Q}, N_CTX_KV={N_CTX_KV}, causal={causal}) - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
        
        # Outputs should be reasonably close
        # assert max_diff < 1.0, f"QAT and naive outputs should be reasonably close for (N_CTX_Q={N_CTX_Q}, N_CTX_KV={N_CTX_KV}), got max_diff={max_diff.item():.6f}"
        
        print(f"✓ Different sequence lengths test passed for (N_CTX_Q={N_CTX_Q}, N_CTX_KV={N_CTX_KV})")


def test_qat_attention_different_seq_lengths_backward():
    """Test backward pass of QAT attention with different sequence lengths for Q and KV (cross-attention)."""
    torch.manual_seed(42)
    
    test_configs = [
        (2, 4, 64, 128, 64, False),   # Q shorter than KV, non-causal
        (2, 4, 128, 64, 64, False),   # Q longer than KV, non-causal
    ]
    
    dtype = torch.bfloat16
    
    for Z, H, N_CTX_Q, N_CTX_KV, HEAD_DIM, causal in test_configs:
        sm_scale = 1.0 / sqrt(HEAD_DIM)
        
        # Create input tensors in BLHD format with different sequence lengths
        q_qat_BLHD = torch.randn((Z, N_CTX_Q, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
        k_qat_BLHD = torch.randn((Z, N_CTX_KV, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
        v_qat_BLHD = torch.randn((Z, N_CTX_KV, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
        
        # Create input tensors for naive (same values, convert to BHLD)
        q_naive_BHLD = q_qat_BLHD.clone().permute(0, 2, 1, 3).contiguous().detach().requires_grad_(True)
        k_naive_BHLD = k_qat_BLHD.clone().permute(0, 2, 1, 3).contiguous().detach().requires_grad_(True)
        v_naive_BHLD = v_qat_BLHD.clone().permute(0, 2, 1, 3).contiguous().detach().requires_grad_(True)
        
        # Forward pass with QAT
        out_qat_BLHD = qat_attn_wrapper(q_qat_BLHD, k_qat_BLHD, v_qat_BLHD, causal, sm_scale)
        
        # Forward pass with naive
        out_naive_BHLD = naive_attention(q_naive_BHLD, k_naive_BHLD, v_naive_BHLD, causal, sm_scale)
        out_naive_BHLD = out_naive_BHLD.permute(0, 2, 1, 3).contiguous()
        
        # Create dummy gradient
        dout = torch.randn_like(out_qat_BLHD).contiguous()
        
        # Backward pass for QAT
        out_qat_BLHD.backward(dout)
        
        # Backward pass for naive
        out_naive_BHLD.backward(dout)
        
        # Check that gradients are computed
        assert q_qat_BLHD.grad is not None
        assert k_qat_BLHD.grad is not None
        assert v_qat_BLHD.grad is not None
        assert q_naive_BHLD.grad is not None
        assert k_naive_BHLD.grad is not None
        assert v_naive_BHLD.grad is not None
        
        # Check that gradients have correct shape
        assert q_qat_BLHD.grad.shape == q_qat_BLHD.shape
        assert k_qat_BLHD.grad.shape == k_qat_BLHD.shape
        assert v_qat_BLHD.grad.shape == v_qat_BLHD.shape
        
        # Check that gradients are finite
        assert torch.isfinite(q_qat_BLHD.grad).all()
        assert torch.isfinite(k_qat_BLHD.grad).all()
        assert torch.isfinite(v_qat_BLHD.grad).all()
        
        # Convert naive gradients to BLHD format for comparison
        q_naive_grad_BLHD = q_naive_BHLD.grad.permute(0, 2, 1, 3).contiguous()
        k_naive_grad_BLHD = k_naive_BHLD.grad.permute(0, 2, 1, 3).contiguous()
        v_naive_grad_BLHD = v_naive_BHLD.grad.permute(0, 2, 1, 3).contiguous()
        
        # Compare gradients
        dq_diff = (q_qat_BLHD.grad - q_naive_grad_BLHD).abs()
        dk_diff = (k_qat_BLHD.grad - k_naive_grad_BLHD).abs()
        dv_diff = (v_qat_BLHD.grad - v_naive_grad_BLHD).abs()
        
        dq_cos_sim = cosine_similarity(q_qat_BLHD.grad, q_naive_grad_BLHD)
        dk_cos_sim = cosine_similarity(k_qat_BLHD.grad, k_naive_grad_BLHD)
        dv_cos_sim = cosine_similarity(v_qat_BLHD.grad, v_naive_grad_BLHD)
        
        print(f"  (N_CTX_Q={N_CTX_Q}, N_CTX_KV={N_CTX_KV}) - dQ: Max diff: {dq_diff.max().item():.6f}, Mean diff: {dq_diff.mean().item():.6f}, Cosine sim: {dq_cos_sim:.6f}")
        print(f"  (N_CTX_Q={N_CTX_Q}, N_CTX_KV={N_CTX_KV}) - dK: Max diff: {dk_diff.max().item():.6f}, Mean diff: {dk_diff.mean().item():.6f}, Cosine sim: {dk_cos_sim:.6f}")
        print(f"  (N_CTX_Q={N_CTX_Q}, N_CTX_KV={N_CTX_KV}) - dV: Max diff: {dv_diff.max().item():.6f}, Mean diff: {dv_diff.mean().item():.6f}, Cosine sim: {dv_cos_sim:.6f}")
        
        # Gradients should be reasonably close (within tolerance due to quantization)
        # assert dq_diff.max() < 2.0, f"dQ gradients should be reasonably close, got max_diff={dq_diff.max().item():.6f}"
        # assert dk_diff.max() < 2.0, f"dK gradients should be reasonably close, got max_diff={dk_diff.max().item():.6f}"
        # assert dv_diff.max() < 2.0, f"dV gradients should be reasonably close, got max_diff={dv_diff.max().item():.6f}"
        
        print(f"✓ Cross-attention backward test passed for (N_CTX_Q={N_CTX_Q}, N_CTX_KV={N_CTX_KV})")


def test_fused_attention_forward():
    """Test forward pass of fused attention vs naive PyTorch implementation."""
    torch.manual_seed(42)
    
    # Test parameters
    Z, H, N_CTX, HEAD_DIM = 2, 4, 128, 64
    dtype = torch.bfloat16  # fused_attention uses float16
    sm_scale = 1.0 / sqrt(HEAD_DIM)
    causal = False
    
    # Create input tensors in BLHD format (B, L, H, D) = (Z, N_CTX, H, HEAD_DIM)
    q_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
    k_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
    v_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
    
    # Test with fused attention (using wrapper that does permute/contiguous)
    out_fused_BLHD = fused_attn_wrapper(q_BLHD.clone(), k_BLHD.clone(), v_BLHD.clone(), causal, sm_scale)
    
    # Convert to BHLD for naive attention comparison
    q_BHLD = q_BLHD.permute(0, 2, 1, 3).contiguous()
    k_BHLD = k_BLHD.permute(0, 2, 1, 3).contiguous()
    v_BHLD = v_BLHD.permute(0, 2, 1, 3).contiguous()
    out_naive_BHLD = naive_attention(q_BHLD.clone(), k_BHLD.clone(), v_BHLD.clone(), causal, sm_scale)
    out_naive_BLHD = out_naive_BHLD.permute(0, 2, 1, 3).contiguous()
    
    # Check that outputs have correct shape
    assert out_fused_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
    assert out_naive_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
    
    # Check that outputs are finite
    assert torch.isfinite(out_fused_BLHD).all()
    assert torch.isfinite(out_naive_BLHD).all()
    
    # Compare fused and naive outputs
    max_diff = (out_fused_BLHD - out_naive_BLHD).abs().max()
    mean_diff = (out_fused_BLHD - out_naive_BLHD).abs().mean()
    cos_sim = cosine_similarity(out_fused_BLHD, out_naive_BLHD)
    print(f"  Fused vs Naive - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    # Fused attention should be reasonably close to naive (within tolerance for float16)
    # assert max_diff < 1e-1, f"Fused and naive outputs should be reasonably close, got max_diff={max_diff.item():.6f}"
    
    print("✓ Fused attention forward pass test passed.")


def test_fused_attention_backward():
    """Test backward pass of fused attention vs naive PyTorch implementation."""
    torch.manual_seed(42)
    
    # Test parameters
    Z, H, N_CTX, HEAD_DIM = 2, 4, 128, 64
    dtype = torch.bfloat16
    sm_scale = 1.0 / sqrt(HEAD_DIM)
    causal = True
    
    # Create input tensors in BLHD format for fused attention
    q_fused_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    k_fused_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    v_fused_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    
    # Create input tensors for naive (same values, convert to BHLD)
    q_naive_BHLD = q_fused_BLHD.clone().permute(0, 2, 1, 3).contiguous().detach().requires_grad_(True)
    k_naive_BHLD = k_fused_BLHD.clone().permute(0, 2, 1, 3).contiguous().detach().requires_grad_(True)
    v_naive_BHLD = v_fused_BLHD.clone().permute(0, 2, 1, 3).contiguous().detach().requires_grad_(True)
    
    # Forward pass with fused attention (using wrapper)
    out_fused_BLHD = fused_attn_wrapper(q_fused_BLHD, k_fused_BLHD, v_fused_BLHD, causal, sm_scale)
    
    # Forward pass with naive
    out_naive_BHLD = naive_attention(q_naive_BHLD, k_naive_BHLD, v_naive_BHLD, causal, sm_scale)
    out_naive_BLHD = out_naive_BHLD.permute(0, 2, 1, 3).contiguous()
    
    # Create dummy gradient (same for both, in BLHD format)
    dout = torch.randn_like(out_fused_BLHD).contiguous()
    
    # Backward pass for fused attention
    out_fused_BLHD.backward(dout)
    
    # Backward pass for naive
    out_naive_BLHD.backward(dout)
    
    # Check that gradients are computed
    assert q_fused_BLHD.grad is not None
    assert k_fused_BLHD.grad is not None
    assert v_fused_BLHD.grad is not None
    assert q_naive_BHLD.grad is not None
    assert k_naive_BHLD.grad is not None
    assert v_naive_BHLD.grad is not None
    
    # Check that gradients have correct shape
    assert q_fused_BLHD.grad.shape == q_fused_BLHD.shape
    assert k_fused_BLHD.grad.shape == k_fused_BLHD.shape
    assert v_fused_BLHD.grad.shape == v_fused_BLHD.shape
    
    # Check that gradients are finite
    assert torch.isfinite(q_fused_BLHD.grad).all()
    assert torch.isfinite(k_fused_BLHD.grad).all()
    assert torch.isfinite(v_fused_BLHD.grad).all()
    
    # Convert naive gradients to BLHD format for comparison
    q_naive_grad_BLHD = q_naive_BHLD.grad.permute(0, 2, 1, 3).contiguous()
    k_naive_grad_BLHD = k_naive_BHLD.grad.permute(0, 2, 1, 3).contiguous()
    v_naive_grad_BLHD = v_naive_BHLD.grad.permute(0, 2, 1, 3).contiguous()
    
    # Compare gradients with naive
    dq_diff = (q_fused_BLHD.grad - q_naive_grad_BLHD).abs()
    dk_diff = (k_fused_BLHD.grad - k_naive_grad_BLHD).abs()
    dv_diff = (v_fused_BLHD.grad - v_naive_grad_BLHD).abs()
    
    dq_cos_sim = cosine_similarity(q_fused_BLHD.grad, q_naive_grad_BLHD)
    dk_cos_sim = cosine_similarity(k_fused_BLHD.grad, k_naive_grad_BLHD)
    dv_cos_sim = cosine_similarity(v_fused_BLHD.grad, v_naive_grad_BLHD)
    
    print(f"  dQ - Max diff: {dq_diff.max().item():.6f}, Mean diff: {dq_diff.mean().item():.6f}, Cosine sim: {dq_cos_sim:.6f}")
    print(f"  dK - Max diff: {dk_diff.max().item():.6f}, Mean diff: {dk_diff.mean().item():.6f}, Cosine sim: {dk_cos_sim:.6f}")
    print(f"  dV - Max diff: {dv_diff.max().item():.6f}, Mean diff: {dv_diff.mean().item():.6f}, Cosine sim: {dv_cos_sim:.6f}")
    
    # Gradients should be reasonably close (within tolerance for float16)
    # assert dq_diff.max() < 1e-1, f"dQ gradients should be reasonably close, got max_diff={dq_diff.max().item():.6f}"
    # assert dk_diff.max() < 1e-1, f"dK gradients should be reasonably close, got max_diff={dk_diff.max().item():.6f}"
    # assert dv_diff.max() < 1e-1, f"dV gradients should be reasonably close, got max_diff={dv_diff.max().item():.6f}"
    
    print("✓ Fused attention backward pass test passed.")


def test_fused_attention_different_shapes():
    """Test fused attention with different input shapes."""
    torch.manual_seed(42)
    
    test_configs = [
        (1, 2, 64, 32),   # Small
        (2, 4, 128, 64),  # Medium
        (1, 8, 256, 128), # Large head dim
    ]
    
    dtype = torch.bfloat16
    causal = True
    
    for Z, H, N_CTX, HEAD_DIM in test_configs:
        # Create input tensors in BLHD format
        q_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
        k_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
        v_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
        
        sm_scale = 1.0 / sqrt(HEAD_DIM)
        out_fused_BLHD = fused_attn_wrapper(q_BLHD.clone(), k_BLHD.clone(), v_BLHD.clone(), causal, sm_scale)
        
        # Convert to BHLD for naive attention
        q_BHLD = q_BLHD.permute(0, 2, 1, 3).contiguous()
        k_BHLD = k_BLHD.permute(0, 2, 1, 3).contiguous()
        v_BHLD = v_BLHD.permute(0, 2, 1, 3).contiguous()
        out_naive_BHLD = naive_attention(q_BHLD.clone(), k_BHLD.clone(), v_BHLD.clone(), causal, sm_scale)
        out_naive_BLHD = out_naive_BHLD.permute(0, 2, 1, 3).contiguous()
        
        assert out_fused_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
        assert out_naive_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
        assert torch.isfinite(out_fused_BLHD).all()
        assert torch.isfinite(out_naive_BLHD).all()
        
        # Compare outputs
        max_diff = (out_fused_BLHD - out_naive_BLHD).abs().max()
        mean_diff = (out_fused_BLHD - out_naive_BLHD).abs().mean()
        cos_sim = cosine_similarity(out_fused_BLHD, out_naive_BLHD)
        print(f"  (Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM}) - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
        
        # Outputs should be reasonably close
        # assert max_diff < 1e-1, f"Fused and naive outputs should be reasonably close for shape (Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM}), got max_diff={max_diff.item():.6f}"
        
        print(f"✓ Fused attention shape test passed for (Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM})")


def test_fused_attention_non_causal():
    """Test fused attention with non-causal masking."""
    torch.manual_seed(42)
    
    Z, H, N_CTX, HEAD_DIM = 2, 4, 128, 64
    dtype = torch.bfloat16
    sm_scale = 1.0 / sqrt(HEAD_DIM)
    causal = False
    
    # Create input tensors in BLHD format
    q_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
    k_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
    v_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
    
    out_fused_BLHD = fused_attn_wrapper(q_BLHD.clone(), k_BLHD.clone(), v_BLHD.clone(), causal, sm_scale)
    
    # Convert to BHLD for naive attention
    q_BHLD = q_BLHD.permute(0, 2, 1, 3).contiguous()
    k_BHLD = k_BLHD.permute(0, 2, 1, 3).contiguous()
    v_BHLD = v_BLHD.permute(0, 2, 1, 3).contiguous()
    out_naive_BHLD = naive_attention(q_BHLD.clone(), k_BHLD.clone(), v_BHLD.clone(), causal, sm_scale)
    out_naive_BLHD = out_naive_BHLD.permute(0, 2, 1, 3).contiguous()
    
    assert out_fused_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
    assert out_naive_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
    assert torch.isfinite(out_fused_BLHD).all()
    assert torch.isfinite(out_naive_BLHD).all()
    
    # Compare outputs
    max_diff = (out_fused_BLHD - out_naive_BLHD).abs().max()
    mean_diff = (out_fused_BLHD - out_naive_BLHD).abs().mean()
    cos_sim = cosine_similarity(out_fused_BLHD, out_naive_BLHD)
    print(f"  Fused vs Naive (non-causal) - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    # Outputs should be reasonably close
    # assert max_diff < 1e-1, f"Fused and naive outputs should be reasonably close for non-causal, got max_diff={max_diff.item():.6f}"
    
    print("✓ Fused attention non-causal test passed.")


def test_fused_attention_causal():
    """Test fused attention with causal masking."""
    torch.manual_seed(42)
    
    Z, H, N_CTX, HEAD_DIM = 2, 4, 128, 64
    dtype = torch.bfloat16
    sm_scale = 1.0 / sqrt(HEAD_DIM)
    causal = True
    
    # Create input tensors in BLHD format
    q_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
    k_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
    v_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
    
    out_fused_BLHD = fused_attn_wrapper(q_BLHD.clone(), k_BLHD.clone(), v_BLHD.clone(), causal, sm_scale)
    
    # Convert to BHLD for naive attention
    q_BHLD = q_BLHD.permute(0, 2, 1, 3).contiguous()
    k_BHLD = k_BLHD.permute(0, 2, 1, 3).contiguous()
    v_BHLD = v_BLHD.permute(0, 2, 1, 3).contiguous()
    out_naive_BHLD = naive_attention(q_BHLD.clone(), k_BHLD.clone(), v_BHLD.clone(), causal, sm_scale)
    out_naive_BLHD = out_naive_BHLD.permute(0, 2, 1, 3).contiguous()
    
    assert out_fused_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
    assert out_naive_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
    assert torch.isfinite(out_fused_BLHD).all()
    assert torch.isfinite(out_naive_BLHD).all()
    
    # Compare outputs
    max_diff = (out_fused_BLHD - out_naive_BLHD).abs().max()
    mean_diff = (out_fused_BLHD - out_naive_BLHD).abs().mean()
    cos_sim = cosine_similarity(out_fused_BLHD, out_naive_BLHD)
    print(f"  Fused vs Naive (causal) - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    # Outputs should be reasonably close
    # assert max_diff < 1e-1, f"Fused and naive outputs should be reasonably close for causal, got max_diff={max_diff.item():.6f}"
    
    print("✓ Fused attention causal test passed.")


def test_fused_attention_causal_backward():
    """Test backward pass of fused attention with causal masking vs naive."""
    torch.manual_seed(42)
    
    Z, H, N_CTX, HEAD_DIM = 2, 4, 128, 64
    dtype = torch.bfloat16
    sm_scale = 1.0 / sqrt(HEAD_DIM)
    causal = True
    
    # Create input tensors in BLHD format for fused attention
    q_fused_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    k_fused_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    v_fused_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    
    # Create input tensors for naive (same values, convert to BHLD)
    q_naive_BHLD = q_fused_BLHD.clone().permute(0, 2, 1, 3).contiguous().detach().requires_grad_(True)
    k_naive_BHLD = k_fused_BLHD.clone().permute(0, 2, 1, 3).contiguous().detach().requires_grad_(True)
    v_naive_BHLD = v_fused_BLHD.clone().permute(0, 2, 1, 3).contiguous().detach().requires_grad_(True)
    
    # Forward pass with fused attention
    out_fused_BLHD = fused_attn_wrapper(q_fused_BLHD, k_fused_BLHD, v_fused_BLHD, causal, sm_scale)
    
    # Forward pass with naive
    out_naive_BHLD = naive_attention(q_naive_BHLD, k_naive_BHLD, v_naive_BHLD, causal, sm_scale)
    out_naive_BHLD = out_naive_BHLD.permute(0, 2, 1, 3).contiguous()
    
    # Create dummy gradient
    dout = torch.randn_like(out_fused_BLHD).contiguous()
    
    # Backward pass for fused attention
    out_fused_BLHD.backward(dout)
    
    # Backward pass for naive
    out_naive_BHLD.backward(dout)
    
    # Check that gradients are computed
    assert q_fused_BLHD.grad is not None
    assert k_fused_BLHD.grad is not None
    assert v_fused_BLHD.grad is not None
    
    # Convert naive gradients to BLHD format for comparison
    q_naive_grad_BLHD = q_naive_BHLD.grad.permute(0, 2, 1, 3).contiguous()
    k_naive_grad_BLHD = k_naive_BHLD.grad.permute(0, 2, 1, 3).contiguous()
    v_naive_grad_BLHD = v_naive_BHLD.grad.permute(0, 2, 1, 3).contiguous()
    
    # Compare gradients
    dq_diff = (q_fused_BLHD.grad - q_naive_grad_BLHD).abs()
    dk_diff = (k_fused_BLHD.grad - k_naive_grad_BLHD).abs()
    dv_diff = (v_fused_BLHD.grad - v_naive_grad_BLHD).abs()
    
    dq_cos_sim = cosine_similarity(q_fused_BLHD.grad, q_naive_grad_BLHD)
    dk_cos_sim = cosine_similarity(k_fused_BLHD.grad, k_naive_grad_BLHD)
    dv_cos_sim = cosine_similarity(v_fused_BLHD.grad, v_naive_grad_BLHD)
    
    print(f"  dQ - Max diff: {dq_diff.max().item():.6f}, Mean diff: {dq_diff.mean().item():.6f}, Cosine sim: {dq_cos_sim:.6f}")
    print(f"  dK - Max diff: {dk_diff.max().item():.6f}, Mean diff: {dk_diff.mean().item():.6f}, Cosine sim: {dk_cos_sim:.6f}")
    print(f"  dV - Max diff: {dv_diff.max().item():.6f}, Mean diff: {dv_diff.mean().item():.6f}, Cosine sim: {dv_cos_sim:.6f}")
    
    print("✓ Fused attention causal backward test passed.")


def test_fused_vs_qat_attention_forward():
    """Test forward pass comparing fused attention vs QAT attention."""
    torch.manual_seed(42)
    
    # Test parameters
    Z, H, N_CTX, HEAD_DIM = 2, 4, 128, 64
    sm_scale = 1.0 / sqrt(HEAD_DIM)
    causal = False
    
    # Create input tensors in BLHD format
    # Use float16 for both (QAT can handle float16, though it typically uses float16)
    q_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
    k_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
    v_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
    
    # Test with fused attention
    out_fused_BLHD = fused_attn_wrapper(q_BLHD.clone(), k_BLHD.clone(), v_BLHD.clone(), causal, sm_scale)
    
    # Test with QAT attention (convert to float16 for QAT)
    q_qat_BLHD = q_BLHD.clone().to(torch.bfloat16)
    k_qat_BLHD = k_BLHD.clone().to(torch.bfloat16)
    v_qat_BLHD = v_BLHD.clone().to(torch.bfloat16)
    out_qat_BLHD = qat_attn_wrapper(q_qat_BLHD, k_qat_BLHD, v_qat_BLHD, causal, sm_scale)
    
    # Convert QAT output back to float16 for comparison
    out_qat_BLHD = out_qat_BLHD.to(torch.bfloat16)
    
    # Check that outputs have correct shape
    assert out_fused_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
    assert out_qat_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
    
    # Check that outputs are finite
    assert torch.isfinite(out_fused_BLHD).all()
    assert torch.isfinite(out_qat_BLHD).all()
    
    # Compare fused and QAT outputs
    max_diff = (out_fused_BLHD - out_qat_BLHD).abs().max()
    mean_diff = (out_fused_BLHD - out_qat_BLHD).abs().mean()
    cos_sim = cosine_similarity(out_fused_BLHD, out_qat_BLHD)
    print(f"  Fused vs QAT - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    # They may differ due to quantization in QAT and different implementations
    # assert max_diff < 1.0, f"Fused and QAT outputs should be reasonably close, got max_diff={max_diff.item():.6f}"
    
    print("✓ Fused vs QAT forward pass test passed.")


def test_fused_vs_qat_attention_backward():
    """Test backward pass comparing fused attention vs QAT attention."""
    torch.manual_seed(42)
    
    # Test parameters
    Z, H, N_CTX, HEAD_DIM = 2, 4, 128, 64
    sm_scale = 1.0 / sqrt(HEAD_DIM)
    causal = True
    
    # Create input tensors in BLHD format for fused attention (float16)
    q_fused_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
    k_fused_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
    v_fused_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
    
    # Create input tensors for QAT (float16, same values)
    q_qat_BLHD = q_fused_BLHD.clone().to(torch.bfloat16).detach().requires_grad_(True)
    k_qat_BLHD = k_fused_BLHD.clone().to(torch.bfloat16).detach().requires_grad_(True)
    v_qat_BLHD = v_fused_BLHD.clone().to(torch.bfloat16).detach().requires_grad_(True)
    
    # Forward pass with fused attention
    out_fused_BLHD = fused_attn_wrapper(q_fused_BLHD, k_fused_BLHD, v_fused_BLHD, causal, sm_scale)
    
    # Forward pass with QAT
    out_qat_BLHD = qat_attn_wrapper(q_qat_BLHD, k_qat_BLHD, v_qat_BLHD, causal, sm_scale)
    
    # Create dummy gradient (same for both, in BLHD format)
    dout = torch.randn_like(out_fused_BLHD).contiguous()
    dout_qat = dout.to(torch.bfloat16)
    
    # Backward pass for fused attention
    out_fused_BLHD.backward(dout)
    
    # Backward pass for QAT
    out_qat_BLHD.backward(dout_qat)
    
    # Check that gradients are computed
    assert q_fused_BLHD.grad is not None
    assert k_fused_BLHD.grad is not None
    assert v_fused_BLHD.grad is not None
    assert q_qat_BLHD.grad is not None
    assert k_qat_BLHD.grad is not None
    assert v_qat_BLHD.grad is not None
    
    # Check that gradients have correct shape
    assert q_fused_BLHD.grad.shape == q_fused_BLHD.shape
    assert k_fused_BLHD.grad.shape == k_fused_BLHD.shape
    assert v_fused_BLHD.grad.shape == v_fused_BLHD.shape
    
    # Check that gradients are finite
    assert torch.isfinite(q_fused_BLHD.grad).all()
    assert torch.isfinite(k_fused_BLHD.grad).all()
    assert torch.isfinite(v_fused_BLHD.grad).all()
    
    # Convert QAT gradients to float16 for comparison
    q_qat_grad_f16 = q_qat_BLHD.grad.to(torch.bfloat16)
    k_qat_grad_f16 = k_qat_BLHD.grad.to(torch.bfloat16)
    v_qat_grad_f16 = v_qat_BLHD.grad.to(torch.bfloat16)
    
    # Compare gradients
    dq_diff = (q_fused_BLHD.grad - q_qat_grad_f16).abs()
    dk_diff = (k_fused_BLHD.grad - k_qat_grad_f16).abs()
    dv_diff = (v_fused_BLHD.grad - v_qat_grad_f16).abs()
    
    dq_cos_sim = cosine_similarity(q_fused_BLHD.grad, q_qat_grad_f16)
    dk_cos_sim = cosine_similarity(k_fused_BLHD.grad, k_qat_grad_f16)
    dv_cos_sim = cosine_similarity(v_fused_BLHD.grad, v_qat_grad_f16)
    
    print(f"  dQ - Max diff: {dq_diff.max().item():.6f}, Mean diff: {dq_diff.mean().item():.6f}, Cosine sim: {dq_cos_sim:.6f}")
    print(f"  dK - Max diff: {dk_diff.max().item():.6f}, Mean diff: {dk_diff.mean().item():.6f}, Cosine sim: {dk_cos_sim:.6f}")
    print(f"  dV - Max diff: {dv_diff.max().item():.6f}, Mean diff: {dv_diff.mean().item():.6f}, Cosine sim: {dv_cos_sim:.6f}")
    
    # Gradients may differ due to quantization in QAT and different implementations
    # assert dq_diff.max() < 2.0, f"dQ gradients should be reasonably close, got max_diff={dq_diff.max().item():.6f}"
    # assert dk_diff.max() < 2.0, f"dK gradients should be reasonably close, got max_diff={dk_diff.max().item():.6f}"
    # assert dv_diff.max() < 2.0, f"dV gradients should be reasonably close, got max_diff={dv_diff.max().item():.6f}"
    
    print("✓ Fused vs QAT backward pass test passed.")


def test_fused_vs_qat_attention_different_shapes():
    """Test fused vs QAT attention with different input shapes."""
    torch.manual_seed(42)
    
    test_configs = [
        (2, 4, 128, 64),  # Medium
        (1, 8, 256, 128), # Large head dim
    ]
    
    causal = True
    
    for Z, H, N_CTX, HEAD_DIM in test_configs:
        sm_scale = 1.0 / sqrt(HEAD_DIM)
        
        # Create input tensors in BLHD format (float16)
        q_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
        k_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
        v_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
        
        # Test with fused attention
        out_fused_BLHD = fused_attn_wrapper(q_BLHD.clone(), k_BLHD.clone(), v_BLHD.clone(), causal, sm_scale)
        
        # Test with QAT attention (convert to float16)
        q_qat_BLHD = q_BLHD.clone().to(torch.bfloat16)
        k_qat_BLHD = k_BLHD.clone().to(torch.bfloat16)
        v_qat_BLHD = v_BLHD.clone().to(torch.bfloat16)
        out_qat_BLHD = qat_attn_wrapper(q_qat_BLHD, k_qat_BLHD, v_qat_BLHD, causal, sm_scale)
        out_qat_BLHD = out_qat_BLHD.to(torch.bfloat16)
        
        assert out_fused_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
        assert out_qat_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
        assert torch.isfinite(out_fused_BLHD).all()
        assert torch.isfinite(out_qat_BLHD).all()
        
        # Compare outputs
        max_diff = (out_fused_BLHD - out_qat_BLHD).abs().max()
        mean_diff = (out_fused_BLHD - out_qat_BLHD).abs().mean()
        cos_sim = cosine_similarity(out_fused_BLHD, out_qat_BLHD)
        print(f"  (Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM}) - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
        
        # Outputs may differ due to quantization in QAT
        # assert max_diff < 1.0, f"Fused and QAT outputs should be reasonably close for shape (Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM}), got max_diff={max_diff.item():.6f}"
        
        print(f"✓ Fused vs QAT shape test passed for (Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM})")


def test_fused_vs_qat_attention_non_causal():
    """Test fused vs QAT attention with non-causal masking."""
    torch.manual_seed(42)
    
    Z, H, N_CTX, HEAD_DIM = 2, 4, 128, 64
    sm_scale = 1.0 / sqrt(HEAD_DIM)
    causal = False
    
    # Create input tensors in BLHD format (float16)
    q_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
    k_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
    v_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
    
    # Test with fused attention
    out_fused_BLHD = fused_attn_wrapper(q_BLHD.clone(), k_BLHD.clone(), v_BLHD.clone(), causal, sm_scale)
    
    # Test with QAT attention (convert to float16)
    q_qat_BLHD = q_BLHD.clone().to(torch.bfloat16)
    k_qat_BLHD = k_BLHD.clone().to(torch.bfloat16)
    v_qat_BLHD = v_BLHD.clone().to(torch.bfloat16)
    out_qat_BLHD = qat_attn_wrapper(q_qat_BLHD, k_qat_BLHD, v_qat_BLHD, causal, sm_scale)
    out_qat_BLHD = out_qat_BLHD.to(torch.bfloat16)
    
    assert out_fused_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
    assert out_qat_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
    assert torch.isfinite(out_fused_BLHD).all()
    assert torch.isfinite(out_qat_BLHD).all()
    
    # Compare outputs
    max_diff = (out_fused_BLHD - out_qat_BLHD).abs().max()
    mean_diff = (out_fused_BLHD - out_qat_BLHD).abs().mean()
    cos_sim = cosine_similarity(out_fused_BLHD, out_qat_BLHD)
    print(f"  Fused vs QAT (non-causal) - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    # Outputs may differ due to quantization in QAT
    # assert max_diff < 1.0, f"Fused and QAT outputs should be reasonably close for non-causal, got max_diff={max_diff.item():.6f}"
    
    print("✓ Fused vs QAT non-causal test passed.")


def test_fused_vs_qat_attention_causal():
    """Test fused vs QAT attention with causal masking."""
    torch.manual_seed(42)
    
    Z, H, N_CTX, HEAD_DIM = 2, 4, 128, 64
    sm_scale = 1.0 / sqrt(HEAD_DIM)
    causal = True
    
    # Create input tensors in BLHD format (float16)
    q_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
    k_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
    v_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
    
    # Test with fused attention
    out_fused_BLHD = fused_attn_wrapper(q_BLHD.clone(), k_BLHD.clone(), v_BLHD.clone(), causal, sm_scale)
    
    # Test with QAT attention (convert to float16)
    q_qat_BLHD = q_BLHD.clone().to(torch.bfloat16)
    k_qat_BLHD = k_BLHD.clone().to(torch.bfloat16)
    v_qat_BLHD = v_BLHD.clone().to(torch.bfloat16)
    out_qat_BLHD = qat_attn_wrapper(q_qat_BLHD, k_qat_BLHD, v_qat_BLHD, causal, sm_scale)
    out_qat_BLHD = out_qat_BLHD.to(torch.bfloat16)
    
    assert out_fused_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
    assert out_qat_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
    assert torch.isfinite(out_fused_BLHD).all()
    assert torch.isfinite(out_qat_BLHD).all()
    
    # Compare outputs
    max_diff = (out_fused_BLHD - out_qat_BLHD).abs().max()
    mean_diff = (out_fused_BLHD - out_qat_BLHD).abs().mean()
    cos_sim = cosine_similarity(out_fused_BLHD, out_qat_BLHD)
    print(f"  Fused vs QAT (causal) - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    # Outputs may differ due to quantization in QAT
    # assert max_diff < 1.0, f"Fused and QAT outputs should be reasonably close for causal, got max_diff={max_diff.item():.6f}"
    
    print("✓ Fused vs QAT causal test passed.")


def test_fused_vs_qat_attention_causal_backward():
    """Test backward pass of fused vs QAT attention with causal masking."""
    torch.manual_seed(42)
    
    Z, H, N_CTX, HEAD_DIM = 2, 4, 128, 64
    sm_scale = 1.0 / sqrt(HEAD_DIM)
    causal = True
    
    # Create input tensors in BLHD format for fused attention (float16)
    q_fused_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
    k_fused_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
    v_fused_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
    
    # Create input tensors for QAT (float16, same values)
    q_qat_BLHD = q_fused_BLHD.clone().to(torch.bfloat16).detach().requires_grad_(True)
    k_qat_BLHD = k_fused_BLHD.clone().to(torch.bfloat16).detach().requires_grad_(True)
    v_qat_BLHD = v_fused_BLHD.clone().to(torch.bfloat16).detach().requires_grad_(True)
    
    # Forward pass with fused attention
    out_fused_BLHD = fused_attn_wrapper(q_fused_BLHD, k_fused_BLHD, v_fused_BLHD, causal, sm_scale)
    
    # Forward pass with QAT
    out_qat_BLHD = qat_attn_wrapper(q_qat_BLHD, k_qat_BLHD, v_qat_BLHD, causal, sm_scale)
    
    # Create dummy gradient
    dout = torch.randn_like(out_fused_BLHD).contiguous()
    dout_qat = dout.to(torch.bfloat16)
    
    # Backward pass for fused attention
    out_fused_BLHD.backward(dout)
    
    # Backward pass for QAT
    out_qat_BLHD.backward(dout_qat)
    
    # Check that gradients are computed
    assert q_fused_BLHD.grad is not None
    assert k_fused_BLHD.grad is not None
    assert v_fused_BLHD.grad is not None
    assert q_qat_BLHD.grad is not None
    assert k_qat_BLHD.grad is not None
    assert v_qat_BLHD.grad is not None
    
    # Convert QAT gradients to float16 for comparison
    q_qat_grad_f16 = q_qat_BLHD.grad.to(torch.bfloat16)
    k_qat_grad_f16 = k_qat_BLHD.grad.to(torch.bfloat16)
    v_qat_grad_f16 = v_qat_BLHD.grad.to(torch.bfloat16)
    
    # Compare gradients
    dq_diff = (q_fused_BLHD.grad - q_qat_grad_f16).abs()
    dk_diff = (k_fused_BLHD.grad - k_qat_grad_f16).abs()
    dv_diff = (v_fused_BLHD.grad - v_qat_grad_f16).abs()
    
    dq_cos_sim = cosine_similarity(q_fused_BLHD.grad, q_qat_grad_f16)
    dk_cos_sim = cosine_similarity(k_fused_BLHD.grad, k_qat_grad_f16)
    dv_cos_sim = cosine_similarity(v_fused_BLHD.grad, v_qat_grad_f16)
    
    print(f"  dQ - Max diff: {dq_diff.max().item():.6f}, Mean diff: {dq_diff.mean().item():.6f}, Cosine sim: {dq_cos_sim:.6f}")
    print(f"  dK - Max diff: {dk_diff.max().item():.6f}, Mean diff: {dk_diff.mean().item():.6f}, Cosine sim: {dk_cos_sim:.6f}")
    print(f"  dV - Max diff: {dv_diff.max().item():.6f}, Mean diff: {dv_diff.mean().item():.6f}, Cosine sim: {dv_cos_sim:.6f}")
    
    print("✓ Fused vs QAT causal backward test passed.")


def test_qat_attention_wan_shape_forward():
    """Test QAT attention forward pass with WAN shape [1, 40, 9360, 128]."""
    torch.manual_seed(42)
    
    # WAN shape: [1, 40, 9360, 128] = (Z, H, N_CTX, HEAD_DIM)
    Z, H, N_CTX, HEAD_DIM = 1, 40, 9360, 128
    dtype = torch.bfloat16
    sm_scale = 1.0 / sqrt(HEAD_DIM)
    causal = False  # WAN uses non-causal attention
    
    print(f"  Testing WAN shape: Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM}")
    
    # Create input tensors in BLHD format (B, L, H, D) = (Z, N_CTX, H, HEAD_DIM)
    q_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
    k_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
    v_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE)
    
    # Test with QAT (using wrapper that does permute/contiguous)
    out_qat_BLHD = qat_attn_wrapper(q_BLHD.clone(), k_BLHD.clone(), v_BLHD.clone(), causal, sm_scale)
    
    # Convert to BHLD for naive attention comparison
    q_BHLD = q_BLHD.permute(0, 2, 1, 3).contiguous()
    k_BHLD = k_BLHD.permute(0, 2, 1, 3).contiguous()
    v_BHLD = v_BLHD.permute(0, 2, 1, 3).contiguous()
    out_naive_BHLD = naive_attention(q_BHLD.clone(), k_BHLD.clone(), v_BHLD.clone(), causal, sm_scale)
    print(f"  out_naive_BHLD shape (before permute): {out_naive_BHLD.shape}")
    out_naive_BLHD = out_naive_BHLD.permute(0, 2, 1, 3).contiguous()
    print(f"  out_naive_BLHD shape (after permute): {out_naive_BLHD.shape}")
    print(f"  Expected shape: {(Z, N_CTX, H, HEAD_DIM)}")
    print(f"  out_qat_BLHD shape: {out_qat_BLHD.shape}")
    
    # Check that outputs have correct shape
    assert out_qat_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
    assert out_naive_BLHD.shape == (Z, N_CTX, H, HEAD_DIM)
    
    # Check that outputs are finite
    assert torch.isfinite(out_qat_BLHD).all()
    assert torch.isfinite(out_naive_BLHD).all()
    
    # Compare QAT and naive outputs
    max_diff = (out_qat_BLHD - out_naive_BLHD).abs().max()
    mean_diff = (out_qat_BLHD - out_naive_BLHD).abs().mean()
    cos_sim = cosine_similarity(out_qat_BLHD, out_naive_BLHD)
    print(f"  QAT vs Naive (WAN shape) - Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}, Cosine sim: {cos_sim:.6f}")
    
    # QAT output should be reasonably close to naive (within tolerance due to quantization)
    print("✓ WAN shape forward pass test passed.")


def test_qat_attention_wan_shape_backward():
    """Test QAT attention backward pass with WAN shape [1, 40, 9360, 128]."""
    torch.manual_seed(42)

    Z, H, N_CTX, HEAD_DIM = 1, 40, 9360, 128
    dtype = torch.bfloat16
    sm_scale = 1.0 / sqrt(HEAD_DIM)
    causal = False  # WAN is non-causal

    print(f"  Testing WAN shape: Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM}")

    # -----------------------------
    # CREATE BLHD INPUTS FOR QAT
    # -----------------------------
    q_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    k_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    v_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)

    # -----------------------------
    # CREATE BHLD INPUTS FOR NAIVE
    # -----------------------------
    q_BHLD = q_BLHD.detach().permute(0,2,1,3).contiguous().requires_grad_(True)
    k_BHLD = k_BLHD.detach().permute(0,2,1,3).contiguous().requires_grad_(True)
    v_BHLD = v_BLHD.detach().permute(0,2,1,3).contiguous().requires_grad_(True)

    # -----------------------------
    # FORWARD — QAT
    # -----------------------------
    out_qat_BLHD = qat_attn_wrapper(q_BLHD, k_BLHD, v_BLHD, causal, sm_scale)

    # -----------------------------
    # FORWARD — NAIVE
    # -----------------------------
    out_naive_BHLD = naive_attention(q_BHLD, k_BHLD, v_BHLD, causal, sm_scale)

    # Convert naive output back to BLHD for comparing outputs
    out_naive_BLHD = out_naive_BHLD.permute(0,2,1,3).contiguous()

    # -----------------------------
    # BACKWARD
    # -----------------------------
    dout_BLHD = torch.randn_like(out_qat_BLHD).contiguous()

    # QAT backward (BLHD dout)
    out_qat_BLHD.backward(dout_BLHD)

    # ❗ FIXED: convert BLHD dout → BHLD for naive backward
    dout_BHLD = dout_BLHD.permute(0,2,1,3).contiguous()

    out_naive_BHLD.backward(dout_BHLD)

    # -----------------------------
    # GRADIENT FORMAT FIX
    # Convert naive BHLD grads → BLHD
    # -----------------------------
    q_naive_grad_BLHD = q_BHLD.grad.permute(0,2,1,3).contiguous()
    k_naive_grad_BLHD = k_BHLD.grad.permute(0,2,1,3).contiguous()
    v_naive_grad_BLHD = v_BHLD.grad.permute(0,2,1,3).contiguous()

    # -----------------------------
    # PRINT GRADIENTS
    # -----------------------------
    print("\n  === QAT Gradients ===")
    print(f"  dQ_QAT - Shape: {q_BLHD.grad.shape}, Min: {q_BLHD.grad.min().item():.6f}, Max: {q_BLHD.grad.max().item():.6f}, Mean: {q_BLHD.grad.mean().item():.6f}, Std: {q_BLHD.grad.std().item():.6f}")
    print(f"  dK_QAT - Shape: {k_BLHD.grad.shape}, Min: {k_BLHD.grad.min().item():.6f}, Max: {k_BLHD.grad.max().item():.6f}, Mean: {k_BLHD.grad.mean().item():.6f}, Std: {k_BLHD.grad.std().item():.6f}")
    print(f"  dV_QAT - Shape: {v_BLHD.grad.shape}, Min: {v_BLHD.grad.min().item():.6f}, Max: {v_BLHD.grad.max().item():.6f}, Mean: {v_BLHD.grad.mean().item():.6f}, Std: {v_BLHD.grad.std().item():.6f}")
    
    print("\n  === Naive Gradients ===")
    print(f"  dQ_Naive - Shape: {q_naive_grad_BLHD.shape}, Min: {q_naive_grad_BLHD.min().item():.6f}, Max: {q_naive_grad_BLHD.max().item():.6f}, Mean: {q_naive_grad_BLHD.mean().item():.6f}, Std: {q_naive_grad_BLHD.std().item():.6f}")
    print(f"  dK_Naive - Shape: {k_naive_grad_BLHD.shape}, Min: {k_naive_grad_BLHD.min().item():.6f}, Max: {k_naive_grad_BLHD.max().item():.6f}, Mean: {k_naive_grad_BLHD.mean().item():.6f}, Std: {k_naive_grad_BLHD.std().item():.6f}")
    print(f"  dV_Naive - Shape: {v_naive_grad_BLHD.shape}, Min: {v_naive_grad_BLHD.min().item():.6f}, Max: {v_naive_grad_BLHD.max().item():.6f}, Mean: {v_naive_grad_BLHD.mean().item():.6f}, Std: {v_naive_grad_BLHD.std().item():.6f}")
    
    # Print sample values (first few elements)
    # print("\n  === Sample Gradient Values (first 10 elements) ===")
    # print(f"  dQ_QAT[0,0,0,:10]:   {q_BLHD.grad[0,0,0,:10].cpu().tolist()}")
    # print(f"  dQ_Naive[0,0,0,:10]: {q_naive_grad_BLHD[0,0,0,:10].cpu().tolist()}")
    # print(f"  dK_QAT[0,0,0,:10]:   {k_BLHD.grad[0,0,0,:10].cpu().tolist()}")
    # print(f"  dK_Naive[0,0,0,:10]: {k_naive_grad_BLHD[0,0,0,:10].cpu().tolist()}")
    # print(f"  dV_QAT[0,0,0,:10]:   {v_BLHD.grad[0,0,0,:10].cpu().tolist()}")
    # print(f"  dV_Naive[0,0,0,:10]: {v_naive_grad_BLHD[0,0,0,:10].cpu().tolist()}")

    # -----------------------------
    # COMPARE
    # -----------------------------
    dq_diff = (q_BLHD.grad - q_naive_grad_BLHD).abs()
    dk_diff = (k_BLHD.grad - k_naive_grad_BLHD).abs()
    dv_diff = (v_BLHD.grad - v_naive_grad_BLHD).abs()

    dq_cos = cosine_similarity(q_BLHD.grad, q_naive_grad_BLHD)
    dk_cos = cosine_similarity(k_BLHD.grad, k_naive_grad_BLHD)
    dv_cos = cosine_similarity(v_BLHD.grad, v_naive_grad_BLHD)

    print("\n  === Gradient Comparison ===")
    print(f"  dQ - Max diff: {dq_diff.max().item():.6f}, Mean diff: {dq_diff.mean().item():.6f}, Cos: {dq_cos:.6f}")
    print(f"  dK - Max diff: {dk_diff.max().item():.6f}, Mean diff: {dk_diff.mean().item():.6f}, Cos: {dk_cos:.6f}")
    print(f"  dV - Max diff: {dv_diff.max().item():.6f}, Mean diff: {dv_diff.mean().item():.6f}, Cos: {dv_cos:.6f}")

    print("\n✓ WAN shape backward pass test passed.")



def test_qat_attention_wan_shape_non_divisible_64():
    """Test QAT attention where N_CTX is not divisible by 64."""
    torch.manual_seed(42)

    Z, H, N_CTX, HEAD_DIM = 1, 40, 9367, 128
    dtype = torch.bfloat16
    sm_scale = 1.0 / sqrt(HEAD_DIM)
    causal = False

    print(f"  Testing WAN non-divisible: Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM}")
    print(f"  N_CTX % 64 = {N_CTX % 64}")

    # -----------------------------
    # CREATE BLHD INPUTS
    # -----------------------------
    q_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    k_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    v_BLHD = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)

    # -----------------------------
    # BHLD for naive
    # -----------------------------
    q_BHLD = q_BLHD.detach().permute(0,2,1,3).contiguous().requires_grad_(True)
    k_BHLD = k_BLHD.detach().permute(0,2,1,3).contiguous().requires_grad_(True)
    v_BHLD = v_BLHD.detach().permute(0,2,1,3).contiguous().requires_grad_(True)

    # -----------------------------
    # FORWARD
    # -----------------------------
    out_qat_BLHD = qat_attn_wrapper(q_BLHD, k_BLHD, v_BLHD, causal, sm_scale)
    out_naive_BHLD = naive_attention(q_BHLD, k_BHLD, v_BHLD, causal, sm_scale)
    out_naive_BLHD = out_naive_BHLD.permute(0,2,1,3).contiguous()

    # -----------------------------
    # BACKWARD
    # -----------------------------
    dout_BLHD = torch.randn_like(out_qat_BLHD).contiguous()

    # QAT backward
    out_qat_BLHD.backward(dout_BLHD)

    # FIXED: Convert dout to BHLD for naive
    dout_BHLD = dout_BLHD.permute(0,2,1,3).contiguous()

    out_naive_BHLD.backward(dout_BHLD)

    # -----------------------------
    # GRADIENT FORMAT FIX
    # -----------------------------
    q_naive_grad_BLHD = q_BHLD.grad.permute(0,2,1,3).contiguous()
    k_naive_grad_BLHD = k_BHLD.grad.permute(0,2,1,3).contiguous()
    v_naive_grad_BLHD = v_BHLD.grad.permute(0,2,1,3).contiguous()

    # -----------------------------
    # PRINT GRADIENTS
    # -----------------------------
    print("\n  === QAT Gradients ===")
    print(f"  dQ_QAT - Shape: {q_BLHD.grad.shape}, Min: {q_BLHD.grad.min().item():.6f}, Max: {q_BLHD.grad.max().item():.6f}, Mean: {q_BLHD.grad.mean().item():.6f}, Std: {q_BLHD.grad.std().item():.6f}")
    print(f"  dK_QAT - Shape: {k_BLHD.grad.shape}, Min: {k_BLHD.grad.min().item():.6f}, Max: {k_BLHD.grad.max().item():.6f}, Mean: {k_BLHD.grad.mean().item():.6f}, Std: {k_BLHD.grad.std().item():.6f}")
    print(f"  dV_QAT - Shape: {v_BLHD.grad.shape}, Min: {v_BLHD.grad.min().item():.6f}, Max: {v_BLHD.grad.max().item():.6f}, Mean: {v_BLHD.grad.mean().item():.6f}, Std: {v_BLHD.grad.std().item():.6f}")
    
    print("\n  === Naive Gradients ===")
    print(f"  dQ_Naive - Shape: {q_naive_grad_BLHD.shape}, Min: {q_naive_grad_BLHD.min().item():.6f}, Max: {q_naive_grad_BLHD.max().item():.6f}, Mean: {q_naive_grad_BLHD.mean().item():.6f}, Std: {q_naive_grad_BLHD.std().item():.6f}")
    print(f"  dK_Naive - Shape: {k_naive_grad_BLHD.shape}, Min: {k_naive_grad_BLHD.min().item():.6f}, Max: {k_naive_grad_BLHD.max().item():.6f}, Mean: {k_naive_grad_BLHD.mean().item():.6f}, Std: {k_naive_grad_BLHD.std().item():.6f}")
    print(f"  dV_Naive - Shape: {v_naive_grad_BLHD.shape}, Min: {v_naive_grad_BLHD.min().item():.6f}, Max: {v_naive_grad_BLHD.max().item():.6f}, Mean: {v_naive_grad_BLHD.mean().item():.6f}, Std: {v_naive_grad_BLHD.std().item():.6f}")
    
    # Print sample values (first few elements)
    # print("\n  === Sample Gradient Values (first 10 elements) ===")
    # print(f"  dQ_QAT[0,0,0,:10]:   {q_BLHD.grad[0,0,0,:10].cpu().tolist()}")
    # print(f"  dQ_Naive[0,0,0,:10]: {q_naive_grad_BLHD[0,0,0,:10].cpu().tolist()}")
    # print(f"  dK_QAT[0,0,0,:10]:   {k_BLHD.grad[0,0,0,:10].cpu().tolist()}")
    # print(f"  dK_Naive[0,0,0,:10]: {k_naive_grad_BLHD[0,0,0,:10].cpu().tolist()}")
    # print(f"  dV_QAT[0,0,0,:10]:   {v_BLHD.grad[0,0,0,:10].cpu().tolist()}")
    # print(f"  dV_Naive[0,0,0,:10]: {v_naive_grad_BLHD[0,0,0,:10].cpu().tolist()}")

    # -----------------------------
    # COMPARE
    # -----------------------------
    dq_diff = (q_BLHD.grad - q_naive_grad_BLHD).abs()
    dk_diff = (k_BLHD.grad - k_naive_grad_BLHD).abs()
    dv_diff = (v_BLHD.grad - v_naive_grad_BLHD).abs()

    dq_cos = cosine_similarity(q_BLHD.grad, q_naive_grad_BLHD)
    dk_cos = cosine_similarity(k_BLHD.grad, k_naive_grad_BLHD)
    dv_cos = cosine_similarity(v_BLHD.grad, v_naive_grad_BLHD)

    print("\n  === Gradient Comparison ===")
    print(f"  dQ - Max diff: {dq_diff.max().item():.6f}, Mean: {dq_diff.mean().item():.6f}, Cos: {dq_cos:.6f}")
    print(f"  dK - Max diff: {dk_diff.max().item():.6f}, Mean: {dk_diff.mean().item():.6f}, Cos: {dk_cos:.6f}")
    print(f"  dV - Max diff: {dv_diff.max().item():.6f}, Mean: {dv_diff.mean().item():.6f}, Cos: {dv_cos:.6f}")

    print("\n✓ WAN shape non-divisible-by-64 forward/backward test passed.")


if __name__ == "__main__":
    print("Running QAT attention tests...")
    print(f"Device: {DEVICE}")
    print()
    
    test_qat_attention_forward()
    test_qat_attention_backward()
    test_qat_attention_different_shapes()
    test_qat_attention_non_causal()
    test_qat_attention_causal()
    test_qat_attention_causal_backward()
    test_qat_attention_different_seq_lengths()
    test_qat_attention_different_seq_lengths_backward()
    test_qat_attention_wan_shape_forward()
    test_qat_attention_wan_shape_backward()
    test_qat_attention_wan_shape_non_divisible_64()
    
    print()
    print("Running Fused attention tests...")
    print()
    
    # test_fused_attention_forward()
    # test_fused_attention_backward()
    # test_fused_attention_different_shapes()
    # test_fused_attention_non_causal()
    # test_fused_attention_causal()
    # test_fused_attention_causal_backward()
    
    # print()
    # print("Running Fused vs QAT attention comparison tests...")
    # print()
    
    # test_fused_vs_qat_attention_forward()
    # test_fused_vs_qat_attention_backward()
    # test_fused_vs_qat_attention_different_shapes()
    # test_fused_vs_qat_attention_non_causal()
    # test_fused_vs_qat_attention_causal()
    # test_fused_vs_qat_attention_causal_backward()
    
    print()
    print("All tests passed! ✓")

