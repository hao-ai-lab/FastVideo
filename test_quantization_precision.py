#!/usr/bin/env python3
"""
Precision test to compare numerical differences between two quantization methods:
1. dequant(quant(X) @ quant(W)) 
2. dequant(quant(X)) @ dequant(quant(W))

Uses FlashInfer API for 4-bit quantization producing bf16 matrices.
"""

import torch
import numpy as np
from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
import unittest
from typing import Tuple, Dict, Any


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
        scale_factor = self._compute_scale_factor(matrix)
        quantized, inv_scale = nvfp4_quantize(
            matrix, 
            scale_factor, 
            sfLayout=self.sf_layout, 
            do_shuffle=False
        )
        return quantized, inv_scale
    
    def dequantize_matrix(self, quantized_matrix: torch.Tensor, inv_scale: torch.Tensor) -> torch.Tensor:
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
            identity = torch.eye(quantized_matrix.shape[1], device=self.device, dtype=self.dtype)
            dequantized = torch.empty_like(quantized_matrix)
            mm_fp4(
                quantized_matrix, identity, inv_scale, inv_scale, 
                torch.tensor(1.0, device=self.device, dtype=torch.float32),
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
    
    def method1_quantized_matmul(self, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Method 1: dequant(quant(X) @ quant(W))
        Perform matrix multiplication in quantized space, then dequantize.
        """
        # Quantize both matrices
        X_quant, X_inv_scale = self.quantize_matrix(X)
        W_quant, W_inv_scale = self.quantize_matrix(W)
        
        # Compute alpha scaling factor
        alpha = 1.0 / (X_inv_scale * W_inv_scale)
        
        # Perform quantized matrix multiplication
        if len(X.shape) == 2:
            # 2D case
            result = torch.empty((X.shape[0], W.shape[0]), device=self.device, dtype=self.dtype)
            mm_fp4(
                X_quant, W_quant.T, X_inv_scale, W_inv_scale.T, alpha,
                self.dtype, result,
                block_size=16,
                use_8x4_sf_layout=False,
                backend="cutlass"
            )
        else:
            # 3D case (batch dimension)
            result = torch.empty((X.shape[0], X.shape[1], W.shape[0]), device=self.device, dtype=self.dtype)
            for i in range(X.shape[0]):
                mm_fp4(
                    X_quant[i], W_quant.T, X_inv_scale, W_inv_scale.T, alpha,
                    self.dtype, result[i],
                    block_size=16,
                    use_8x4_sf_layout=False,
                    backend="cutlass"
                )
        
        return result
    
    def method2_dequantized_matmul(self, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Method 2: dequant(quant(X)) @ dequant(quant(W))
        Dequantize both matrices first, then perform matrix multiplication.
        """
        # Quantize both matrices
        X_quant, X_inv_scale = self.quantize_matrix(X)
        W_quant, W_inv_scale = self.quantize_matrix(W)
        
        # Dequantize both matrices
        X_dequant = self.dequantize_matrix(X_quant, X_inv_scale)
        W_dequant = self.dequantize_matrix(W_quant, W_inv_scale)
        
        # Perform standard matrix multiplication
        if len(X.shape) == 2:
            result = torch.matmul(X_dequant, W_dequant)
        else:
            # 3D case (batch dimension)
            result = torch.matmul(X_dequant, W_dequant)
        
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
        result1 = self.method1_quantized_matmul(X, W)
        result2 = self.method2_dequantized_matmul(X, W)
        
        # Compute reference result (no quantization)
        reference = torch.matmul(X, W)
        
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
            ((64, 128), (128, 256), "small_matrices"),
            ((256, 512), (512, 1024), "medium_matrices"),
            ((1024, 2048), (2048, 4096), "large_matrices"),
            ((32, 64), (64, 128), "small_square"),
            ((128, 256), (256, 128), "rectangular"),
        ]
        
        distributions = [
            ("normal", lambda shape: torch.randn(shape, device=self.device)),
            ("uniform", lambda shape: torch.rand(shape, device=self.device) * 2 - 1),
            ("small_values", lambda shape: torch.randn(shape, device=self.device) * 0.1),
            ("large_values", lambda shape: torch.randn(shape, device=self.device) * 10),
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
                print(f"  Relative max difference: {analysis['method_difference']['relative_max']:.6f}")
                print()
        
        return test_results


class TestQuantizationPrecision(unittest.TestCase):
    """Unit tests for quantization precision comparison."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tester = QuantizationPrecisionTest()
        
    def test_basic_functionality(self):
        """Test basic functionality of both methods."""
        # Create simple test matrices
        X = torch.randn(32, 64, device=self.tester.device, dtype=self.tester.dtype)
        W = torch.randn(64, 128, device=self.tester.device, dtype=self.tester.dtype)
        
        # Test both methods work without errors
        result1 = self.tester.method1_quantized_matmul(X, W)
        result2 = self.tester.method2_dequantized_matmul(X, W)
        
        # Check shapes match
        self.assertEqual(result1.shape, result2.shape)
        self.assertEqual(result1.shape, (32, 128))
        
        # Check results are finite
        self.assertTrue(torch.isfinite(result1).all())
        self.assertTrue(torch.isfinite(result2).all())
    
    def test_precision_threshold(self):
        """Test that precision differences are within acceptable thresholds."""
        X = torch.randn(128, 256, device=self.tester.device, dtype=self.tester.dtype)
        W = torch.randn(256, 512, device=self.tester.device, dtype=self.tester.dtype)
        
        analysis = self.tester.compare_methods(X, W)
        
        # Check that differences are reasonable (less than 1% relative error)
        self.assertLess(analysis['method_difference']['relative_max'], 0.01)
        
        # Check that both methods produce reasonable results compared to reference
        self.assertLess(analysis['method1_vs_reference']['relative_max'], 0.1)
        self.assertLess(analysis['method2_vs_reference']['relative_max'], 0.1)
    
    def test_edge_cases(self):
        """Test edge cases like very small and very large values."""
        # Test with very small values
        X_small = torch.randn(32, 64, device=self.tester.device, dtype=self.tester.dtype) * 1e-6
        W_small = torch.randn(64, 128, device=self.tester.device, dtype=self.tester.dtype) * 1e-6
        
        analysis_small = self.tester.compare_methods(X_small, W_small)
        self.assertTrue(torch.isfinite(analysis_small['method1_result']).all())
        self.assertTrue(torch.isfinite(analysis_small['method2_result']).all())
        
        # Test with very large values
        X_large = torch.randn(32, 64, device=self.tester.device, dtype=self.tester.dtype) * 1e6
        W_large = torch.randn(64, 128, device=self.tester.device, dtype=self.tester.dtype) * 1e6
        
        analysis_large = self.tester.compare_methods(X_large, W_large)
        self.assertTrue(torch.isfinite(analysis_large['method1_result']).all())
        self.assertTrue(torch.isfinite(analysis_large['method2_result']).all())


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
    
    # Summary statistics
    print("\nSummary Statistics:")
    print("=" * 30)
    
    all_max_diffs = []
    all_mean_diffs = []
    
    for size_name, size_results in results.items():
        for dist_name, analysis in size_results.items():
            max_diff = analysis['method_difference']['max']
            mean_diff = analysis['method_difference']['mean']
            all_max_diffs.append(max_diff)
            all_mean_diffs.append(mean_diff)
    
    print(f"Overall max difference: {max(all_max_diffs):.6f}")
    print(f"Overall mean difference: {np.mean(all_mean_diffs):.6f}")
    print(f"Overall std difference: {np.std(all_mean_diffs):.6f}")
    
    # Run unit tests
    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()
