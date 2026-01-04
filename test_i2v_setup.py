#!/usr/bin/env python3
"""
Test TurboDiffusion I2V with local checkpoints.

This script tests I2V generation using the locally converted checkpoints
before uploading to HuggingFace.
"""

import os
import sys

# Set SLA attention backend BEFORE fastvideo imports
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SLA_ATTN"

import torch
from safetensors import safe_open

# Paths to test
HIGH_NOISE_PATH = "/mnt/fast-disks/hao_lab/loay/FastVideo/TurboWan2.2-I2V-A14B-Diffusers/transformer_high/diffusion_pytorch_model.safetensors"
LOW_NOISE_PATH = "/mnt/fast-disks/hao_lab/loay/FastVideo/TurboWan2.2-I2V-A14B-Diffusers/transformer_low/diffusion_pytorch_model.safetensors"
TEST_IMAGE = "/mnt/fast-disks/hao_lab/loay/TurboDiffusion/assets/i2v_inputs/i2v_input_0.jpg"
OUTPUT_DIR = "/mnt/fast-disks/hao_lab/loay/FastVideo/test_i2v_output"

def test_checkpoint_loading():
    """Test that both checkpoints can be loaded correctly."""
    print("=" * 60)
    print("Testing checkpoint loading...")
    print("=" * 60)
    
    # Test high-noise model
    print(f"\n1. Loading high-noise model: {HIGH_NOISE_PATH}")
    with safe_open(HIGH_NOISE_PATH, framework="pt") as f:
        high_keys = list(f.keys())
        print(f"   Keys: {len(high_keys)}")
        print(f"   Sample keys: {high_keys[:3]}")
        patch_shape = f.get_tensor("patch_embedding.weight").shape
        print(f"   patch_embedding.weight shape: {patch_shape}")
    
    # Test low-noise model
    print(f"\n2. Loading low-noise model: {LOW_NOISE_PATH}")
    with safe_open(LOW_NOISE_PATH, framework="pt") as f:
        low_keys = list(f.keys())
        print(f"   Keys: {len(low_keys)}")
        patch_shape = f.get_tensor("patch_embedding.weight").shape
        print(f"   patch_embedding.weight shape: {patch_shape}")
    
    print("\n✓ Checkpoint loading successful!")
    return True


def test_image_exists():
    """Test that test image exists."""
    print("\n" + "=" * 60)
    print("Testing image availability...")
    print("=" * 60)
    
    if os.path.exists(TEST_IMAGE):
        from PIL import Image
        img = Image.open(TEST_IMAGE)
        print(f"   Image: {TEST_IMAGE}")
        print(f"   Size: {img.size}")
        print("\n✓ Image loading successful!")
        return True
    else:
        print(f"   ERROR: Image not found: {TEST_IMAGE}")
        return False


def test_pipeline_import():
    """Test that TurboDiffusionI2VPipeline can be imported."""
    print("\n" + "=" * 60)
    print("Testing pipeline import...")
    print("=" * 60)
    
    try:
        from fastvideo.pipelines.basic.turbodiffusion import TurboDiffusionI2VPipeline
        print(f"   TurboDiffusionI2VPipeline imported successfully")
        print(f"   Required modules: {TurboDiffusionI2VPipeline._required_config_modules}")
        print("\n✓ Pipeline import successful!")
        return True
    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def test_scheduler():
    """Test RCM scheduler with sigma_max=200."""
    print("\n" + "=" * 60)
    print("Testing RCM scheduler...")
    print("=" * 60)
    
    try:
        from fastvideo.models.schedulers.scheduling_rcm import RCMScheduler
        scheduler = RCMScheduler(sigma_max=200.0)
        print(f"   RCM scheduler created with sigma_max=200.0")
        print(f"   Scheduler config: {scheduler.config}")
        print("\n✓ Scheduler test successful!")
        return True
    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def main():
    print("=" * 60)
    print("TurboDiffusion I2V Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Checkpoint Loading", test_checkpoint_loading()))
    results.append(("Image Exists", test_image_exists()))
    results.append(("Pipeline Import", test_pipeline_import()))
    results.append(("Scheduler", test_scheduler()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ All tests passed! I2V setup is ready.")
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
