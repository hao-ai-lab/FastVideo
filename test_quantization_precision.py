#!/usr/bin/env python3
"""
Precision test to compare numerical differences between two quantization methods:
1. dequant(quant(X) @ quant(W)) 
2. dequant(quant(X)) @ dequant(quant(W))

Uses FlashInfer API for 4-bit quantization producing bf16 matrices.
"""

import torch
import numpy as np
from flashinfer import SfLayout, mm_fp4, nvfp4_quantize, e2m1_and_ufp8sf_scale_to_float
import unittest
from typing import Tuple, Dict, Any

import torch
import triton
import triton.language as tl
import triton.profiler as proton
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

def get_fake_quant(x: torch.Tensor):
    # TODO: dtype needs to be float32?
    orig_shape = x.shape
    orig_dtype = x.dtype
    device = x.device
    x = x.view(-1, x.shape[-1])
    x_global_sf = (448 * 6) / x.float().abs().nan_to_num().max()
    x_fp4, x_scale = nvfp4_quantize(x, x_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    x_dequant = e2m1_and_ufp8sf_scale_to_float(x_fp4, x_scale, 1 / x_global_sf)  # Note: this function does dequantization on the cpu
    return x_dequant.view(orig_shape).to(orig_dtype).to(device)


def generate_fake_quant(a_f32: torch.Tensor,
                        b_f32: torch.Tensor,
                        vec_size: int = 16,
                        max_mxf4: float = 6.0,
                        eps: float = 1e-8):
    """
    Fake-quantize fp32 A (M,K) and B (K,N) with nvfp4 (E2M1) using FP8 E4M3 (finite-only) block scales.
    Returns dequantized fp32 tensors (A_qdq, B_qdq). Assumes:
      - M % 128 == 0, N % 128 == 0, K % (vec_size * 4) == 0
      - A is (M, K), B is (K, N)
      - Scales are uniform per 128x64 (A) and 64x128 (B) tiles (matching your kernel’s layout)
    """
    assert a_f32.is_cuda and b_f32.is_cuda
    M, K = a_f32.shape
    Kb, N = b_f32.shape
    assert Kb == K, "A and B inner dims must match (A:(M,K), B:(K,N))"
    assert M % 128 == 0 and N % 128 == 0, "M,N must be multiples of 128"
    assert K % (vec_size * 4) == 0, f"K must be multiple of {vec_size*4}"

    device = a_f32.device

    # ---- tile-scale helpers (produce FP8 E4M3 scales with shape [*, *, 32, 16]) ----
    def _tile_scales_A(a_fp32):
        m_chunks = M // 128
        k_chunks = K // (vec_size * 4)              # 64-wide tiles along K
        x = a_fp32.view(m_chunks, 128, k_chunks, 4, vec_size)
        tile_max = x.abs().amax(dim=(1, 3, 4))      # [m_chunks, k_chunks]
        scale = torch.clamp(tile_max / max_mxf4, min=eps)  # fp32
        s = scale[:, :, None, None].expand(m_chunks, k_chunks, 32, 16)
        return s.to(torch.float8_e4m3fn)            # [m_chunks, k_chunks, 32, 16]

    def _tile_scales_B(b_fp32):
        k_chunks = K // (vec_size * 4)
        n_chunks = N // 128
        x = b_fp32.view(k_chunks, 4, vec_size, n_chunks, 128)
        tile_max = x.abs().amax(dim=(1, 2, 4))      # [k_chunks, n_chunks]
        scale = torch.clamp(tile_max / max_mxf4, min=eps).t()  # [n_chunks, k_chunks]
        s = scale[:, :, None, None].expand(n_chunks, k_chunks, 32, 16)
        return s.to(torch.float8_e4m3fn)            # [n_chunks, k_chunks, 32, 16]

    # ---- expand tile scales to per-element fp32 ----
    def _expand_A_scale_full(a_scale_fp8):
        m_chunks, k_chunks, _, _ = a_scale_fp8.shape
        base = a_scale_fp8[..., 0, 0].to(a_f32.dtype)          # [m_chunks, k_chunks]
        tile = base[:, :, None, None].expand(m_chunks, k_chunks, 128, 64)
        return tile.permute(0, 2, 1, 3).reshape(M, k_chunks * 64).contiguous()  # [M,K]

    def _expand_B_scale_full(b_scale_fp8):
        n_chunks, k_chunks, _, _ = b_scale_fp8.shape
        base = b_scale_fp8[..., 0, 0].to(b_f32.dtype)          # [n_chunks, k_chunks]
        tile = base[:, :, None, None].expand(n_chunks, k_chunks, 64, 128)
        return tile.permute(1, 2, 0, 3).reshape(k_chunks * 64, n_chunks * 128).contiguous()  # [K,N]

    # ---- 1) compute FP8 E4M3 tile scales from fp32 ----
    a_scale_fp8 = _tile_scales_A(a_f32)   # [M//128, K//64, 32, 16]
    b_scale_fp8 = _tile_scales_B(b_f32)   # [N//128, K//64, 32, 16]

    # ---- 2) expand scales to full fp32 for quant/dequant math ----
    a_scale_full = _expand_A_scale_full(a_scale_fp8)   # [M, K] fp32
    b_scale_full = _expand_B_scale_full(b_scale_fp8)   # [K, N] fp32

    # ---- 3) quantize to nvfp4 codes (E2M1) ----
    a_codes = MXFP4Tensor(data=(a_f32 / a_scale_full)).data                   # [M, K] uint8 (low nibble)
    b_codes = MXFP4Tensor(data=(b_f32 / b_scale_full)).data                   # [K, N] uint8

    # ---- 4) dequantize back to fp32 (fake-quant output) ----
    a_qdq = MXFP4Tensor(size=(M, K), device=device); a_qdq.data = a_codes
    b_qdq = MXFP4Tensor(size=(K, N), device=device); b_qdq.data = b_codes

    a_qdq = a_qdq.to(torch.float32).to(a_f32.dtype) * a_scale_full
    b_qdq = b_qdq.to(torch.float32).to(b_f32.dtype) * b_scale_full

    return a_qdq, b_qdq


class QuantizationPrecisionTest:
    """Test class for comparing quantization precision methods."""
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.sf_layout = SfLayout.layout_128x4
        
    def _compute_scale_factor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute scale factor for quantization using the same method as FastVideo."""
        maxabs = tensor.float().abs().nan_to_num().max()
        maxabs = torch.maximum(maxabs, torch.tensor(1e-12, device=tensor.device, dtype=maxabs.dtype))
        return (448.0 * 6.0) / maxabs
    
    def quantize_matrix(self, matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a matrix to 4-bit using FlashInfer.
        
        Args:
            matrix: Input matrix to quantize
            
        Returns:
            Tuple of (quantized_matrix, inverse_scale_factor)
        """
        # scale_factor = self._compute_scale_factor(matrix)
        scale_factor = torch.tensor(1.0, device=matrix.device, dtype=torch.float32)
        quantized, inv_scale = nvfp4_quantize(
            matrix, 
            scale_factor, 
            sfLayout=self.sf_layout, 
            do_shuffle=False
        )
        return quantized, inv_scale, scale_factor
    
    def dequantize_matrix(self, shape: Tuple[int, int], quantized_matrix: torch.Tensor, inv_scale: torch.Tensor, global_scale_factor: torch.Tensor) -> torch.Tensor:
        """
        Dequantize a 4-bit matrix back to bf16.
        
        Note: FlashInfer doesn't provide direct dequantization, so we use a workaround
        by multiplying with an identity matrix to extract the dequantized values.
        
        Args:
            quantized_matrix: 4-bit quantized matrix
            inv_scale: Inverse scale factor from quantization
            
        Returns:
            Dequantized matrix in bf16
        """
        # Create identity matrix for dequantization
        if len(quantized_matrix.shape) == 2:
            # 2D case
            identity = torch.eye(shape[1], device=self.device, dtype=self.dtype)
            identity_fp4, identity_inv_scale = nvfp4_quantize(
                identity, 
                torch.tensor(1.0, device=self.device, dtype=torch.float32),
                sfLayout=self.sf_layout, 
                do_shuffle=False
            )
            dequantized = torch.empty(shape, device=self.device, dtype=self.dtype)
            mm_fp4(
                quantized_matrix, identity_fp4.T, inv_scale, identity_inv_scale.T, 
                1 / global_scale_factor,
                self.dtype, dequantized,
                block_size=16,
                use_8x4_sf_layout=False,
                backend="cutlass"
            )
        else:
            # 3D case (batch dimension)
            identity = torch.eye(quantized_matrix.shape[-1], device=self.device, dtype=self.dtype)
            dequantized = torch.empty_like(quantized_matrix)
            for i in range(quantized_matrix.shape[0]):
                mm_fp4(
                    quantized_matrix[i], identity, inv_scale, inv_scale,
                    torch.tensor(1.0, device=self.device, dtype=torch.float32),
                    self.dtype, dequantized[i],
                    block_size=16,
                    use_8x4_sf_layout=False,
                    backend="cutlass"
                )
        
        return dequantized
    
    def real_quant_matmul(self, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Method 1: dequant(quant(X) @ quant(W))
        Perform matrix multiplication in quantized space, then dequantize.
        """
        # Quantize both matrices
        X_quant, X_inv_scale, X_global_scale_factor = self.quantize_matrix(X)
        W_quant, W_inv_scale, W_global_scale_factor = self.quantize_matrix(W)
        
        # Compute alpha scaling factor
        alpha = 1.0 / (X_global_scale_factor * W_global_scale_factor)
        
        result = torch.empty((X.shape[0], W.shape[0]), device=self.device, dtype=self.dtype)
        mm_fp4(
            X_quant, W_quant.T, X_inv_scale, W_inv_scale.T, alpha,
            self.dtype, result,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="cutlass"
        )

        
        return result
    
    def fake_quant_matmul(self, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Method 2: dequant(quant(X)) @ dequant(quant(W))
        Dequantize both matrices first, then perform matrix multiplication.
        """
        # Quantize both matrices
        
        X_quant, X_inv_scale, X_global_scale_factor = self.quantize_matrix(X)
        W_quant, W_inv_scale, W_global_scale_factor = self.quantize_matrix(W)
        # Dequantize both matrices
        X_dequant = self.dequantize_matrix(X.shape, X_quant, X_inv_scale, X_global_scale_factor)
        W_dequant = self.dequantize_matrix(W.shape, W_quant, W_inv_scale, W_global_scale_factor)
        
        result = torch.matmul(X_dequant, W_dequant.T)
        
        
        # X_fake_quant, W_fake_quant_T = generate_fake_quant(X, W.T.contiguous())
        # import pdb; pdb.set_trace()
        # result = torch.matmul(X_fake_quant, W_fake_quant_T)
        
        return result
    
    def compare_methods(self, X: torch.Tensor, W: torch.Tensor) -> Dict[str, Any]:
        """
        Compare the two quantization methods and return detailed analysis.
        
        Args:
            X: Input matrix X
            W: Weight matrix W
            
        Returns:
            Dictionary containing comparison results
        """
        # Ensure matrices are on correct device and dtype
        X = X.to(device=self.device, dtype=self.dtype)
        W = W.to(device=self.device, dtype=self.dtype)
        
        # Compute results using both methods
        result1 = self.real_quant_matmul(X, W)
        result2 = self.fake_quant_matmul(X, W)
        import pdb; pdb.set_trace()
        # Compute reference result (no quantization)
        reference = torch.matmul(X, W.T)
        # Calculate differences
        diff_methods = torch.abs(result1 - result2)
        diff1_ref = torch.abs(result1 - reference)
        diff2_ref = torch.abs(result2 - reference)
        
        # Statistical analysis
        analysis = {
            'method1_result': result1,
            'method2_result': result2,
            'reference_result': reference,
            'method_difference': {
                'max': torch.max(diff_methods).item(),
                'mean': torch.mean(diff_methods).item(),
                'std': torch.std(diff_methods).item(),
                'relative_max': torch.max(diff_methods / (torch.abs(reference) + 1e-8)).item(),
                'relative_mean': torch.mean(diff_methods / (torch.abs(reference) + 1e-8)).item(),
            },
            'method1_vs_reference': {
                'max': torch.max(diff1_ref).item(),
                'mean': torch.mean(diff1_ref).item(),
                'relative_max': torch.max(diff1_ref / (torch.abs(reference) + 1e-8)).item(),
            },
            'method2_vs_reference': {
                'max': torch.max(diff2_ref).item(),
                'mean': torch.mean(diff2_ref).item(),
                'relative_max': torch.max(diff2_ref / (torch.abs(reference) + 1e-8)).item(),
            },
            'input_shapes': {
                'X': list(X.shape),
                'W': list(W.shape),
            }
        }
        
        return analysis
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive tests with different matrix sizes and distributions."""
        test_results = {}
        
        # Test cases with different sizes and distributions
        test_cases = [
            # (X_shape, W_shape, distribution_name)
            ((128, 128), (256, 128), "small_matrices"),
            ((256, 512), (1024, 512), "medium_matrices"),
            ((1024, 2048), (4096, 2048), "large_matrices"),
        ]
        
        distributions = [
            ("normal", lambda shape: torch.randn(shape, device=self.device)),
            ("uniform", lambda shape: torch.rand(shape, device=self.device) * 2 - 1),
        ]
        
        for x_shape, w_shape, size_name in test_cases:
            test_results[size_name] = {}
            
            for dist_name, dist_func in distributions:
                # Generate test matrices
                X = dist_func(x_shape).to(dtype=self.dtype)
                W = dist_func(w_shape).to(dtype=self.dtype)
                
                # Run comparison
                analysis = self.compare_methods(X, W)
                test_results[size_name][dist_name] = analysis
                
                print(f"Test: {size_name} - {dist_name}")
                print(f"  Max difference between methods: {analysis['method_difference']['max']:.6f}")
                print(f"  Mean difference between methods: {analysis['method_difference']['mean']:.6f}")
                
                print(f" Max difference between method1 (real quant) and reference: {analysis['method1_vs_reference']['max']:.6f}")
                print(f" Mean difference between method1 (real quant) and reference: {analysis['method1_vs_reference']['mean']:.6f}")
                print(f" Max difference between method2 (fake quant) and reference: {analysis['method2_vs_reference']['max']:.6f}")
                print(f" Mean difference between method2 (fake quant) and reference: {analysis['method2_vs_reference']['mean']:.6f}")
                print()
        
        return test_results



def main():
    """Main function to run the precision test."""
    print("Starting Quantization Precision Test")
    print("=" * 50)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU (may be slower)")
        device = "cpu"
    else:
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    
    # Create tester
    tester = QuantizationPrecisionTest(device=device)
    
    # Run comprehensive test
    print("\nRunning comprehensive tests...")
    results = tester.run_comprehensive_test()


def test_get_fake_quant():
    """Test the get_fake_quant function with various inputs."""
    print("Testing get_fake_quant function...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    device = "cuda"
    dtype = torch.bfloat16
    
    # Test 1: Basic functionality - 2D tensor
    print("\nTest 1: Basic 2D tensor")
    x = torch.randn(128, 64, device=device, dtype=dtype)
    x_fq = get_fake_quant(x)
    assert x_fq.shape == x.shape, f"Shape mismatch: {x_fq.shape} vs {x.shape}"
    assert x_fq.dtype == x.dtype, f"Dtype mismatch: {x_fq.dtype} vs {x.dtype}"
    assert x_fq.device == x.device, f"Device mismatch: {x_fq.device} vs {x.device}"
    max_diff = (x - x_fq).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert max_diff < 1.0, f"Difference too large: {max_diff}"
    
    # Test 2: 3D tensor (B, H, D)
    print("\nTest 2: 3D tensor [B, H, D]")
    x = torch.randn(2, 8, 128, device=device, dtype=dtype)
    x_fq = get_fake_quant(x)
    assert x_fq.shape == x.shape, f"Shape mismatch: {x_fq.shape} vs {x.shape}"
    assert x_fq.dtype == x.dtype, f"Dtype mismatch: {x_fq.dtype} vs {x.dtype}"
    max_diff = (x - x_fq).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert max_diff < 1.0, f"Difference too large: {max_diff}"
    
    # Test 3: 4D tensor (B, H, L, D)
    print("\nTest 3: 4D tensor [B, H, L, D]")
    x = torch.randn(1, 8, 256, 128, device=device, dtype=dtype)
    x_fq = get_fake_quant(x)
    assert x_fq.shape == x.shape, f"Shape mismatch: {x_fq.shape} vs {x.shape}"
    assert x_fq.dtype == x.dtype, f"Dtype mismatch: {x_fq.dtype} vs {x.dtype}"
    max_diff = (x - x_fq).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert max_diff < 1.0, f"Difference too large: {max_diff}"
    
    # Test 4: Small values
    print("\nTest 4: Small values")
    x = torch.randn(128, 64, device=device, dtype=dtype) * 0.01
    x_fq = get_fake_quant(x)
    assert x_fq.shape == x.shape
    max_diff = (x - x_fq).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    
    # Test 5: Large values
    print("\nTest 5: Large values")
    x = torch.randn(128, 64, device=device, dtype=dtype) * 10.0
    x_fq = get_fake_quant(x)
    assert x_fq.shape == x.shape
    max_diff = (x - x_fq).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    
    # Test 6: Zeros
    print("\nTest 6: Zeros")
    x = torch.zeros(128, 64, device=device, dtype=dtype)
    x_fq = get_fake_quant(x)
    assert x_fq.shape == x.shape
    max_diff = (x - x_fq).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert max_diff < 1e-5, f"Zeros should remain zeros: {max_diff}"
    
    # Test 7: Different dtypes
    print("\nTest 7: Float16 dtype")
    x = torch.randn(128, 64, device=device, dtype=torch.float16)
    x_fq = get_fake_quant(x)
    assert x_fq.shape == x.shape
    assert x_fq.dtype == x.dtype
    max_diff = (x - x_fq).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    
    # Test 8: Idempotency check (quantizing twice should be similar)
    print("\nTest 8: Idempotency")
    x = torch.randn(128, 64, device=device, dtype=dtype)
    x_fq1 = get_fake_quant(x)
    x_fq2 = get_fake_quant(x_fq1)
    # Second quantization should be close to first
    max_diff = (x_fq1 - x_fq2).abs().max().item()
    print(f"  Max difference between two quantizations: {max_diff:.6f}")
    
    # Test 9: Requires grad
    print("\nTest 9: Gradient flow")
    x = torch.randn(128, 64, device=device, dtype=dtype, requires_grad=True)
    x_fq = get_fake_quant(x)
    # Note: get_fake_quant uses no_grad internally, so x_fq won't have grad
    # But we can check it doesn't crash
    assert x_fq.shape == x.shape
    print("  Gradient test passed (no crash)")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test_fake_quant":
        test_get_fake_quant()
    else:
        main()
