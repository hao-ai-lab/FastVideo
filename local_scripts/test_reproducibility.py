#!/usr/bin/env python3
"""
Test script to verify reproducibility of random operations in distillation pipeline.
This script can be used to debug and ensure that two different codebases produce
the same random values given the same input and seed.
"""

import torch
import numpy as np
import random
from typing import Dict, Any, Tuple

def set_deterministic_seed(seed: int):
    """Set all random seeds for deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_test_generators(seed: int) -> Tuple[torch.Generator, torch.Generator]:
    """Create test generators with fixed seeds."""
    noise_generator = torch.Generator(device="cpu").manual_seed(seed)
    validation_generator = torch.Generator(device="cpu").manual_seed(42)
    return noise_generator, validation_generator

def test_random_operations(seed: int, step: int, shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
    """Test all random operations used in the distillation pipeline."""
    set_deterministic_seed(seed)
    
    # Create generators
    step_seed = seed + step
    noise_generator, validation_generator = create_test_generators(step_seed)
    
    results = {}
    
    # Test 1: torch.randn with generator (like in _student_forward)
    results['noise_1'] = torch.randn(shape, device='cpu', dtype=torch.float32, generator=noise_generator)
    
    # Test 2: torch.randint with generator (like in _student_forward)
    results['index_1'] = torch.randint(0, 10, (2, 3), device='cpu', dtype=torch.long, generator=noise_generator)
    
    # Test 3: torch.randn for noise generation (like in _compute_dmd_loss)
    results['noise_2'] = torch.randn(shape, device='cpu', dtype=torch.float32, generator=noise_generator)
    
    # Test 4: torch.randint for timestep (like in _compute_dmd_loss)
    results['timestep_1'] = torch.randint(0, 1000, (2, 3), device='cpu', dtype=torch.long, generator=noise_generator)
    
    # Test 5: torch.randint for critic timestep (like in _critic_forward_and_compute_loss)
    results['timestep_2'] = torch.randint(0, 1000, (2, 3), device='cpu', dtype=torch.long, generator=noise_generator)
    
    # Test 6: torch.randn for critic noise (like in _critic_forward_and_compute_loss)
    results['noise_3'] = torch.randn(shape, device='cpu', dtype=torch.float32, generator=noise_generator)
    
    # Test 7: torch.randn for validation (like in dmd_inference)
    results['validation_noise'] = torch.randn(shape, device='cpu', dtype=torch.float32, generator=validation_generator)
    
    return results

def compare_results(results1: Dict[str, torch.Tensor], results2: Dict[str, torch.Tensor]) -> bool:
    """Compare two sets of results to check if they are identical."""
    if set(results1.keys()) != set(results2.keys()):
        print("Different keys found!")
        return False
    
    for key in results1.keys():
        if not torch.allclose(results1[key], results2[key], atol=1e-6):
            print(f"Mismatch found in {key}:")
            print(f"  Result 1: {results1[key]}")
            print(f"  Result 2: {results2[key]}")
            return False
    
    return True

def main():
    """Main test function."""
    print("Testing reproducibility of random operations...")
    
    # Test parameters
    seed = 42
    step = 1
    shape = (2, 3, 4, 5, 6)  # Example shape for video latents
    
    # Run test twice with same parameters
    print(f"Running test with seed={seed}, step={step}")
    
    results1 = test_random_operations(seed, step, shape)
    results2 = test_random_operations(seed, step, shape)
    
    # Compare results
    if compare_results(results1, results2):
        print("✅ SUCCESS: All random operations are reproducible!")
        
        # Print some sample values for debugging
        print("\nSample values from first run:")
        for key, value in results1.items():
            print(f"  {key}: {value.flatten()[:5]}...")  # Show first 5 values
    else:
        print("❌ FAILURE: Random operations are not reproducible!")
    
    # Test with different steps
    print(f"\nTesting with different steps (seed={seed})...")
    results_step1 = test_random_operations(seed, 1, shape)
    results_step2 = test_random_operations(seed, 2, shape)
    
    if not compare_results(results_step1, results_step2):
        print("✅ SUCCESS: Different steps produce different results!")
    else:
        print("❌ FAILURE: Different steps should produce different results!")

if __name__ == "__main__":
    main() 