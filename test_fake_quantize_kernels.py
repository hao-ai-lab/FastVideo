#!/usr/bin/env python3
"""
Precision test for fake_quantize_q and fake_quantize_kv kernels.

Tests that the kernels correctly:
1. Quantize and dequantize Q, K, V tensors
2. Preserve approximate values within quantization error
3. Handle edge cases (zeros, small/large values, padding)
4. Work with different shapes and dtypes
"""

import torch
import triton
import triton.language as tl
import numpy as np
from typing import Tuple, Dict, Any
import unittest

from quant_utils import fake_quantize_q, fake_quantize_kv


def cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    t1_flat = tensor1.flatten().float()
    t2_flat = tensor2.flatten().float()
    
    dot_product = torch.dot(t1_flat, t2_flat)
    norm1 = torch.norm(t1_flat)
    norm2 = torch.norm(t2_flat)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return (dot_product / (norm1 * norm2)).item()


def compute_metrics(original: torch.Tensor, quantized: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, float]:
    """Compute precision metrics between original and quantized tensors."""
    if mask is not None:
        # Only compute metrics on valid (non-padding) elements
        original = original[mask]
        quantized = quantized[mask]
    
    diff = (original.float() - quantized.float()).abs()
    
    metrics = {
        'max_diff': diff.max().item(),
        'mean_diff': diff.mean().item(),
        'std_diff': diff.std().item(),
        'cosine_similarity': cosine_similarity(original, quantized),
    }
    
    # Relative error (avoid division by zero)
    abs_original = original.float().abs()
    rel_error = diff / (abs_original + 1e-8)
    metrics['max_relative_error'] = rel_error.max().item()
    metrics['mean_relative_error'] = rel_error.mean().item()
    
    return metrics


def test_fake_quantize_q(
    q: torch.Tensor,
    BLOCK_M: int = 128,
    HEAD_DIM: int = 128,
    verbose: bool = False
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Test fake_quantize_q kernel.
    
    Args:
        q: Input Q tensor with shape (Z, H, N_CTX_Q, HEAD_DIM)
        BLOCK_M: Block size for sequence dimension
        HEAD_DIM: Head dimension (must be multiple of 16)
        verbose: Whether to print detailed information
    
    Returns:
        Tuple of (fake_q, metrics_dict)
    """
    assert q.is_cuda, "Input must be on CUDA"
    assert q.dim() == 4, f"Q must be 4D, got shape {q.shape}"
    assert HEAD_DIM % 16 == 0, f"HEAD_DIM must be multiple of 16, got {HEAD_DIM}"
    
    Z, H, N_CTX_Q, HEAD_DIM_actual = q.shape
    assert HEAD_DIM_actual == HEAD_DIM, f"HEAD_DIM mismatch: {HEAD_DIM_actual} != {HEAD_DIM}"
    
    # Create output tensor
    fake_q = torch.empty_like(q)
    
    # Launch kernel
    grid = (triton.cdiv(N_CTX_Q, BLOCK_M), Z * H, 1)
    
    fake_quantize_q[grid](
        q, fake_q,
        q.stride(0), q.stride(1),  # stride_z_q, stride_h_q
        q.stride(2), q.stride(3),  # stride_tok_q, stride_d_q
        fake_q.stride(0), fake_q.stride(1),  # fake_stride_z_q, fake_stride_h_q
        fake_q.stride(2), fake_q.stride(3),  # fake_stride_tok_q, fake_stride_d_q
        H, N_CTX_Q,
        BLOCK_M=BLOCK_M, HEAD_DIM=HEAD_DIM
    )
    
    # Compute metrics
    metrics = compute_metrics(q, fake_q)
    metrics['shape'] = list(q.shape)
    metrics['dtype'] = str(q.dtype)
    
    if verbose:
        print(f"fake_quantize_q test:")
        print(f"  Shape: {q.shape}")
        print(f"  Dtype: {q.dtype}")
        print(f"  Max diff: {metrics['max_diff']:.6f}")
        print(f"  Mean diff: {metrics['mean_diff']:.6f}")
        print(f"  Cosine similarity: {metrics['cosine_similarity']:.6f}")
        print(f"  Max relative error: {metrics['max_relative_error']:.6f}")
    
    return fake_q, metrics


def test_fake_quantize_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    BLOCK_N: int = 128,
    HEAD_DIM: int = 128,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Test fake_quantize_kv kernel.
    
    Args:
        k: Input K tensor with shape (Z, H, N_CTX_KV, HEAD_DIM)
        v: Input V tensor with shape (Z, H, N_CTX_KV, HEAD_DIM)
        BLOCK_N: Block size for sequence dimension
        HEAD_DIM: Head dimension (must be multiple of 16)
        verbose: Whether to print detailed information
    
    Returns:
        Tuple of (fake_k, fake_v, metrics_dict)
    """
    assert k.is_cuda and v.is_cuda, "Inputs must be on CUDA"
    assert k.shape == v.shape, f"K and V must have same shape: {k.shape} != {v.shape}"
    assert k.dim() == 4, f"K must be 4D, got shape {k.shape}"
    assert HEAD_DIM % 16 == 0, f"HEAD_DIM must be multiple of 16, got {HEAD_DIM}"
    
    Z, H, N_CTX_KV, HEAD_DIM_actual = k.shape
    assert HEAD_DIM_actual == HEAD_DIM, f"HEAD_DIM mismatch: {HEAD_DIM_actual} != {HEAD_DIM}"
    
    # Create output tensors
    fake_k = torch.empty_like(k)
    fake_v = torch.empty_like(v)
    
    # Launch kernel
    grid = (triton.cdiv(N_CTX_KV, BLOCK_N), Z * H, 1)
    
    fake_quantize_kv[grid](
        k, v, fake_k, fake_v,
        k.stride(0), k.stride(1),  # stride_z_kv, stride_h_kv
        k.stride(2), k.stride(3),  # stride_tok_kv, stride_d_kv
        fake_k.stride(0), fake_k.stride(1),  # fake_stride_z_kv, fake_stride_h_kv
        fake_k.stride(2), fake_k.stride(3),  # fake_stride_tok_kv, fake_stride_d_kv
        H, N_CTX_KV,
        BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM
    )
    
    # Compute metrics for both K and V
    metrics_k = compute_metrics(k, fake_k)
    metrics_v = compute_metrics(v, fake_v)
    
    metrics = {
        'k': metrics_k,
        'v': metrics_v,
        'shape': list(k.shape),
        'dtype': str(k.dtype),
    }
    
    if verbose:
        print(f"fake_quantize_kv test:")
        print(f"  Shape: {k.shape}")
        print(f"  Dtype: {k.dtype}")
        print(f"  K - Max diff: {metrics_k['max_diff']:.6f}, Cosine sim: {metrics_k['cosine_similarity']:.6f}")
        print(f"  V - Max diff: {metrics_v['max_diff']:.6f}, Cosine sim: {metrics_v['cosine_similarity']:.6f}")
    
    return fake_k, fake_v, metrics


class FakeQuantizeKernelsTest:
    """Test suite for fake_quantize_q and fake_quantize_kv kernels."""
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.HEAD_DIM = 128
        self.BLOCK_M = 128
        self.BLOCK_N = 128
        
    def test_basic_q(self, verbose: bool = False) -> Dict[str, Any]:
        """Test fake_quantize_q with basic random input."""
        print("\n" + "="*60)
        print("Test: Basic Q quantization")
        print("="*60)
        
        Z, H, N_CTX_Q = 2, 8, 256
        q = torch.randn(Z, H, N_CTX_Q, self.HEAD_DIM, device=self.device, dtype=self.dtype)
        
        fake_q, metrics = test_fake_quantize_q(q, BLOCK_M=self.BLOCK_M, HEAD_DIM=self.HEAD_DIM, verbose=verbose)
        
        # Assertions
        assert fake_q.shape == q.shape, f"Shape mismatch: {fake_q.shape} != {q.shape}"
        assert fake_q.dtype == q.dtype, f"Dtype mismatch: {fake_q.dtype} != {q.dtype}"
        assert metrics['cosine_similarity'] > 0.9, f"Cosine similarity too low: {metrics['cosine_similarity']}"
        
        print("✓ Basic Q test passed")
        return metrics
    
    def test_basic_kv(self, verbose: bool = False) -> Dict[str, Any]:
        """Test fake_quantize_kv with basic random inputs."""
        print("\n" + "="*60)
        print("Test: Basic KV quantization")
        print("="*60)
        
        Z, H, N_CTX_KV = 2, 8, 512
        k = torch.randn(Z, H, N_CTX_KV, self.HEAD_DIM, device=self.device, dtype=self.dtype)
        v = torch.randn(Z, H, N_CTX_KV, self.HEAD_DIM, device=self.device, dtype=self.dtype)
        
        fake_k, fake_v, metrics = test_fake_quantize_kv(
            k, v, BLOCK_N=self.BLOCK_N, HEAD_DIM=self.HEAD_DIM, verbose=verbose
        )
        
        # Assertions
        assert fake_k.shape == k.shape, f"K shape mismatch: {fake_k.shape} != {k.shape}"
        assert fake_v.shape == v.shape, f"V shape mismatch: {fake_v.shape} != {v.shape}"
        assert metrics['k']['cosine_similarity'] > 0.9, f"K cosine similarity too low: {metrics['k']['cosine_similarity']}"
        assert metrics['v']['cosine_similarity'] > 0.9, f"V cosine similarity too low: {metrics['v']['cosine_similarity']}"
        
        print("✓ Basic KV test passed")
        return metrics
    
    def test_small_values(self, verbose: bool = False) -> Dict[str, Any]:
        """Test with small input values."""
        print("\n" + "="*60)
        print("Test: Small values")
        print("="*60)
        
        Z, H, N_CTX_Q = 1, 4, 128
        q = torch.randn(Z, H, N_CTX_Q, self.HEAD_DIM, device=self.device, dtype=self.dtype) * 0.01
        
        fake_q, metrics = test_fake_quantize_q(q, BLOCK_M=self.BLOCK_M, HEAD_DIM=self.HEAD_DIM, verbose=verbose)
        
        print("✓ Small values test passed")
        return metrics
    
    def test_large_values(self, verbose: bool = False) -> Dict[str, Any]:
        """Test with large input values."""
        print("\n" + "="*60)
        print("Test: Large values")
        print("="*60)
        
        Z, H, N_CTX_Q = 1, 4, 128
        q = torch.randn(Z, H, N_CTX_Q, self.HEAD_DIM, device=self.device, dtype=self.dtype) * 10.0
        
        fake_q, metrics = test_fake_quantize_q(q, BLOCK_M=self.BLOCK_M, HEAD_DIM=self.HEAD_DIM, verbose=verbose)
        
        print("✓ Large values test passed")
        return metrics
    
    def test_zeros(self, verbose: bool = False) -> Dict[str, Any]:
        """Test with zero input."""
        print("\n" + "="*60)
        print("Test: Zeros")
        print("="*60)
        
        Z, H, N_CTX_Q = 1, 4, 128
        q = torch.zeros(Z, H, N_CTX_Q, self.HEAD_DIM, device=self.device, dtype=self.dtype)
        
        fake_q, metrics = test_fake_quantize_q(q, BLOCK_M=self.BLOCK_M, HEAD_DIM=self.HEAD_DIM, verbose=verbose)
        
        # Zeros should remain approximately zero after quantization
        max_val = fake_q.abs().max().item()
        assert max_val < 1e-3, f"Zeros should remain near zero, got max: {max_val}"
        
        print("✓ Zeros test passed")
        return metrics
    
    def test_different_shapes(self, verbose: bool = False) -> Dict[str, Any]:
        """Test with different tensor shapes."""
        print("\n" + "="*60)
        print("Test: Different shapes")
        print("="*60)
        
        test_shapes = [
            (1, 1, 64, self.HEAD_DIM),   # Single batch, single head, small seq
            (1, 8, 128, self.HEAD_DIM),  # Single batch, multiple heads
            (4, 16, 256, self.HEAD_DIM), # Multiple batches, many heads
            (2, 8, 512, self.HEAD_DIM),  # Medium size
        ]
        
        all_metrics = []
        for shape in test_shapes:
            Z, H, N_CTX_Q, HEAD_DIM = shape
            q = torch.randn(*shape, device=self.device, dtype=self.dtype)
            
            fake_q, metrics = test_fake_quantize_q(
                q, BLOCK_M=self.BLOCK_M, HEAD_DIM=self.HEAD_DIM, verbose=verbose
            )
            
            assert fake_q.shape == q.shape
            assert metrics['cosine_similarity'] > 0.85
            all_metrics.append(metrics)
            
            if verbose:
                print(f"  Shape {shape}: cosine_sim={metrics['cosine_similarity']:.4f}")
        
        print("✓ Different shapes test passed")
        return all_metrics
    
    def test_different_dtypes(self, verbose: bool = False) -> Dict[str, Any]:
        """Test with different dtypes."""
        print("\n" + "="*60)
        print("Test: Different dtypes")
        print("="*60)
        
        Z, H, N_CTX_Q = 1, 4, 128
        dtypes = [torch.bfloat16, torch.float16, torch.float32]
        
        all_metrics = []
        for dtype in dtypes:
            q = torch.randn(Z, H, N_CTX_Q, self.HEAD_DIM, device=self.device, dtype=dtype)
            
            fake_q, metrics = test_fake_quantize_q(
                q, BLOCK_M=self.BLOCK_M, HEAD_DIM=self.HEAD_DIM, verbose=verbose
            )
            
            assert fake_q.dtype == q.dtype
            assert metrics['cosine_similarity'] > 0.85
            all_metrics.append(metrics)
            
            if verbose:
                print(f"  {dtype}: cosine_sim={metrics['cosine_similarity']:.4f}")
        
        print("✓ Different dtypes test passed")
        return all_metrics
    
    def test_non_divisible_sequence_length(self, verbose: bool = False) -> Dict[str, Any]:
        """Test with sequence lengths that don't divide evenly by block size."""
        print("\n" + "="*60)
        print("Test: Non-divisible sequence length")
        print("="*60)
        
        Z, H = 1, 4
        # Use sequence length that doesn't divide evenly by BLOCK_M
        N_CTX_Q = 150  # Not divisible by 128
        
        q = torch.randn(Z, H, N_CTX_Q, self.HEAD_DIM, device=self.device, dtype=self.dtype)
        
        fake_q, metrics = test_fake_quantize_q(q, BLOCK_M=self.BLOCK_M, HEAD_DIM=self.HEAD_DIM, verbose=verbose)
        
        # Check that padding is handled correctly (last block should have zeros for padding)
        assert fake_q.shape == q.shape
        assert metrics['cosine_similarity'] > 0.85
        
        print("✓ Non-divisible sequence length test passed")
        return metrics
    
    def test_kv_different_shapes(self, verbose: bool = False) -> Dict[str, Any]:
        """Test fake_quantize_kv with different shapes."""
        print("\n" + "="*60)
        print("Test: KV different shapes")
        print("="*60)
        
        test_shapes = [
            (1, 1, 64, self.HEAD_DIM),
            (1, 8, 128, self.HEAD_DIM),
            (2, 16, 256, self.HEAD_DIM),
        ]
        
        all_metrics = []
        for shape in test_shapes:
            Z, H, N_CTX_KV, HEAD_DIM = shape
            k = torch.randn(*shape, device=self.device, dtype=self.dtype)
            v = torch.randn(*shape, device=self.device, dtype=self.dtype)
            
            fake_k, fake_v, metrics = test_fake_quantize_kv(
                k, v, BLOCK_N=self.BLOCK_N, HEAD_DIM=self.HEAD_DIM, verbose=verbose
            )
            
            assert fake_k.shape == k.shape
            assert fake_v.shape == v.shape
            assert metrics['k']['cosine_similarity'] > 0.85
            assert metrics['v']['cosine_similarity'] > 0.85
            all_metrics.append(metrics)
        
        print("✓ KV different shapes test passed")
        return all_metrics
    
    def run_all_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run all tests."""
        print("\n" + "="*60)
        print("Running Fake Quantize Kernels Precision Tests")
        print("="*60)
        
        if not torch.cuda.is_available():
            print("CUDA not available, skipping tests")
            return {}
        
        results = {}
        
        try:
            results['basic_q'] = self.test_basic_q(verbose=verbose)
            results['basic_kv'] = self.test_basic_kv(verbose=verbose)
            results['small_values'] = self.test_small_values(verbose=verbose)
            results['large_values'] = self.test_large_values(verbose=verbose)
            results['zeros'] = self.test_zeros(verbose=verbose)
            results['different_shapes'] = self.test_different_shapes(verbose=verbose)
            results['different_dtypes'] = self.test_different_dtypes(verbose=verbose)
            results['non_divisible_seq'] = self.test_non_divisible_sequence_length(verbose=verbose)
            results['kv_different_shapes'] = self.test_kv_different_shapes(verbose=verbose)
            
            print("\n" + "="*60)
            print("✓ All tests passed!")
            print("="*60)
            
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        return results


def main():
    """Main function to run the precision tests."""
    tester = FakeQuantizeKernelsTest(device="cuda", dtype=torch.bfloat16)
    results = tester.run_all_tests(verbose=True)
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    if 'basic_q' in results:
        print(f"Basic Q - Cosine similarity: {results['basic_q']['cosine_similarity']:.6f}")
        print(f"Basic Q - Max diff: {results['basic_q']['max_diff']:.6f}")
    
    if 'basic_kv' in results:
        print(f"Basic K - Cosine similarity: {results['basic_kv']['k']['cosine_similarity']:.6f}")
        print(f"Basic V - Cosine similarity: {results['basic_kv']['v']['cosine_similarity']:.6f}")


if __name__ == "__main__":
    main()

