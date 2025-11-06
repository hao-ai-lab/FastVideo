#!/usr/bin/env python3
"""
Compare outputs between wrapper and native LongCat implementations.
"""

import torch
from fastvideo import VideoGenerator
import numpy as np

def compare_models():
    print("=" * 80)
    print("LongCat Implementation Comparison")
    print("=" * 80)
    
    # Same seed for both
    seed = 42
    prompt = "A cat"
    
    print("\n[1/2] Testing WRAPPER (working) implementation...")
    wrapper_gen = VideoGenerator.from_pretrained("weights/longcat-for-fastvideo")
    torch.manual_seed(seed)
    wrapper_result = wrapper_gen.generate_video(
        prompt=prompt,
        num_inference_steps=2,
        guidance_scale=4.0,
        height=256,
        width=384,
        num_frames=9,
        seed=seed,
    )
    wrapper_frames = wrapper_result['frames']
    wrapper_arr = np.array(wrapper_frames)
    print(f"  Output shape: {wrapper_arr.shape}")
    print(f"  Value range: [{wrapper_arr.min():.3f}, {wrapper_arr.max():.3f}]")
    print(f"  Mean: {wrapper_arr.mean():.3f}, Std: {wrapper_arr.std():.3f}")
    
    print("\n[2/2] Testing NATIVE implementation...")
    native_gen = VideoGenerator.from_pretrained("weights/longcat-native")
    torch.manual_seed(seed)
    native_result = native_gen.generate_video(
        prompt=prompt,
        num_inference_steps=2,
        guidance_scale=4.0,
        height=256,
        width=384,
        num_frames=9,
        seed=seed,
    )
    native_frames = native_result['frames']
    native_arr = np.array(native_frames)
    print(f"  Output shape: {native_arr.shape}")
    print(f"  Value range: [{native_arr.min():.3f}, {native_arr.max():.3f}]")
    print(f"  Mean: {native_arr.mean():.3f}, Std: {native_arr.std():.3f}")
    
    print("\n" + "=" * 80)
    print("Comparison:")
    print("=" * 80)
    
    # Compare statistics
    diff = np.abs(wrapper_arr - native_arr).mean()
    print(f"  Mean absolute difference: {diff:.3f}")
    
    # Check if native output looks like noise (high std, mean around 0.5)
    if native_arr.std() > 0.3 and 0.4 < native_arr.mean() < 0.6:
        print("  ⚠️  WARNING: Native output appears to be noise!")
        print("     (High std + mean around 0.5 suggests random values)")
    
    if wrapper_arr.std() < 0.2:
        print("  ✓ Wrapper output looks reasonable (low std)")
    
    print("\n" + "=" * 80)
    
if __name__ == "__main__":
    compare_models()


