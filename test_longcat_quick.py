#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Quick sanity check for LongCat native implementation.

This performs a fast test with minimal steps to verify:
1. Model loads successfully
2. Forward pass completes without errors
3. Output shape is correct

Usage:
    python test_longcat_quick.py
"""

import torch
from fastvideo import VideoGenerator


def quick_test():
    """Quick sanity check with minimal inference steps."""
    
    print("=" * 60)
    print("LongCat Native - Quick Sanity Check")
    print("=" * 60)
    print()
    
    # Load model
    print("Loading native model...")
    generator = VideoGenerator.from_pretrained(
        "weights/longcat-native",
        num_gpus=1,
    )
    print(f"✓ Model loaded: {generator.pipeline.dit.__class__.__name__}")
    print()
    
    # Quick generation test (minimal steps)
    print("Testing inference (2 steps, small resolution)...")
    torch.manual_seed(42)
    
    video = generator.generate_video(
        prompt="A cat",
        num_inference_steps=2,  # Minimal for speed
        guidance_scale=4.0,
        height=256,  # Small for speed
        width=384,
        num_frames=9,  # Minimal frames
        seed=42,
    )
    
    print(f"✓ Generation successful!")
    print(f"  Output shape: {video.shape}")
    print(f"  Expected: (9, 256, 384, 3)")
    
    # Verify shape
    assert video.shape == (9, 256, 384, 3), f"Unexpected shape: {video.shape}"
    print()
    
    # Memory stats
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Peak GPU memory: {peak_memory_gb:.2f} GB")
        print()
    
    print("=" * 60)
    print("✓ All checks passed!")
    print("=" * 60)
    print()
    print("Native LongCat implementation is working correctly.")
    print("Ready for full inference with test_longcat_native_inference.py")
    print()


if __name__ == "__main__":
    quick_test()

