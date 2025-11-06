#!/usr/bin/env python3
"""
Test script for native LongCat model.

This script tests the native LongCat implementation with random inputs
to verify that the model loads and forward pass works.
"""

import os
import torch

# Set up distributed environment variables
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29506"

from fastvideo.models.dits.longcat import LongCatTransformer3DModel
from fastvideo.configs.models.dits import LongCatVideoConfig
from fastvideo.distributed import (
    maybe_init_distributed_environment_and_model_parallel,
    cleanup_dist_env_and_memory,
)
from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


def test_model_instantiation():
    """Test that the model can be instantiated."""
    print("=" * 60)
    print("Testing Native LongCat Model Instantiation")
    print("=" * 60)
    
    # Create config with SMALLER size for testing
    config = LongCatVideoConfig()
    config.arch_config.depth = 2  # Just 2 blocks instead of 48
    config.arch_config.hidden_size = 512  # Smaller hidden size
    config.arch_config.num_attention_heads = 8
    config.arch_config.attention_head_dim = 64
    config.arch_config.caption_channels = 512
    
    print(f"\nModel Configuration (REDUCED FOR TESTING):")
    print(f"  Hidden size: {config.arch_config.hidden_size}")
    print(f"  Depth: {config.arch_config.depth}")
    print(f"  Num heads: {config.arch_config.num_attention_heads}")
    print(f"  Patch size: {config.arch_config.patch_size}")
    
    # Instantiate model
    print("\nInstantiating model...")
    model = LongCatTransformer3DModel(config, hf_config={})
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    
    print("✓ Model instantiated successfully!\n")
    return model


def test_forward_pass():
    """Test forward pass with random inputs."""
    print("=" * 60)
    print("Testing Native LongCat Forward Pass")
    print("=" * 60)
    
    # Create model with SMALLER config for testing
    config = LongCatVideoConfig()
    config.arch_config.depth = 2  # Just 2 blocks
    config.arch_config.hidden_size = 512
    config.arch_config.num_attention_heads = 8
    config.arch_config.attention_head_dim = 64
    config.arch_config.caption_channels = 512  # Match text encoder size
    
    model = LongCatTransformer3DModel(config, hf_config={})
    model.eval()
    model = model.cuda()  # Move to GPU
    
    # Create random inputs - SMALLER for testing
    batch_size = 1
    T, H, W = 5, 16, 28  # Much smaller: ~5 frames, lower resolution
    
    print(f"\nInput shapes:")
    print(f"  Batch size: {batch_size}")
    print(f"  Latent shape: [{batch_size}, 16, {T}, {H}, {W}]")
    print(f"  Text shape: [{batch_size}, 128, 512]")  # Smaller text
    
    hidden_states = torch.randn(batch_size, 16, T, H, W).cuda()
    encoder_hidden_states = torch.randn(batch_size, 128, 512).cuda()  # Smaller text
    timestep = torch.randint(0, 1000, (batch_size,)).cuda()
    
    print("\nRunning forward pass...")
    
    # Create forward batch
    forward_batch = ForwardBatch(data_type="t2v")
    
    with torch.no_grad():
        try:
            with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=forward_batch):
                output = model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                )
            
            print(f"✓ Forward pass successful!")
            print(f"  Output shape: {list(output.shape)}")
            print(f"  Output dtype: {output.dtype}")
            print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            
            # Check for NaN or Inf
            if torch.isnan(output).any():
                print("  ⚠️  Warning: Output contains NaN values!")
                return False
            if torch.isinf(output).any():
                print("  ⚠️  Warning: Output contains Inf values!")
                return False
            
            print("  ✓ No NaN or Inf values detected")
            return True
            
        except Exception as e:
            print(f"✗ Forward pass failed with error:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_with_attention_mask():
    """Test forward pass with attention mask for variable-length text."""
    print("\n" + "=" * 60)
    print("Testing Variable-Length Text Handling")
    print("=" * 60)
    
    config = LongCatVideoConfig()
    config.arch_config.depth = 2
    config.arch_config.hidden_size = 512
    config.arch_config.num_attention_heads = 8
    config.arch_config.attention_head_dim = 64
    config.arch_config.caption_channels = 512
    
    model = LongCatTransformer3DModel(config, hf_config={})
    model.eval()
    model = model.cuda()
    
    batch_size = 2
    T, H, W = 5, 16, 28  # Smaller for testing
    
    max_seq_len = 128
    seq_len_1 = 100
    seq_len_2 = 80
    
    print(f"\nTesting with batch_size={batch_size}")
    print(f"  Sample 1: {seq_len_1} valid tokens")
    print(f"  Sample 2: {seq_len_2} valid tokens")
    
    hidden_states = torch.randn(batch_size, 16, T, H, W).cuda()
    encoder_hidden_states = torch.randn(batch_size, max_seq_len, 512).cuda()
    timestep = torch.randint(0, 1000, (batch_size,)).cuda()
    
    # Create attention mask with different lengths
    encoder_attention_mask = torch.zeros(batch_size, max_seq_len).cuda()
    encoder_attention_mask[0, :seq_len_1] = 1  # First sample
    encoder_attention_mask[1, :seq_len_2] = 1  # Second sample
    
    print("\nRunning forward pass with variable-length text...")
    
    # Create forward batch
    forward_batch = ForwardBatch(data_type="t2v")
    
    with torch.no_grad():
        try:
            with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=forward_batch):
                output = model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    encoder_attention_mask=encoder_attention_mask,
                )
            
            print(f"✓ Variable-length forward pass successful!")
            print(f"  Output shape: {list(output.shape)}")
            
            if torch.isnan(output).any() or torch.isinf(output).any():
                print("  ⚠️  Warning: Output contains NaN or Inf values!")
                return False
            
            print("  ✓ No NaN or Inf values detected")
            return True
            
        except Exception as e:
            print(f"✗ Forward pass failed with error:")
            print(f"  {type(e).__name__}: {e}")
            return False


def main():
    print("\n" + "=" * 60)
    print("NATIVE LONGCAT MODEL TEST SUITE")
    print("=" * 60 + "\n")
    
    # Initialize distributed environment
    print("Initializing distributed environment...")
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    print("✓ Distributed environment initialized\n")
    
    # Run tests
    test_results = []
    
    # Test 1: Instantiation
    try:
        model = test_model_instantiation()
        test_results.append(("Instantiation", True))
    except Exception as e:
        print(f"✗ Instantiation failed: {e}")
        test_results.append(("Instantiation", False))
        return
    
    # Test 2: Forward pass
    success = test_forward_pass()
    test_results.append(("Forward Pass", success))
    
    # Test 3: Variable-length text
    success = test_with_attention_mask()
    test_results.append(("Variable-Length Text", success))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, success in test_results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {test_name:.<40} {status}")
    
    all_passed = all(success for _, success in test_results)
    
    if all_passed:
        print("\n✓ All tests passed!")
        print("\nNext steps:")
        print("  1. Convert weights using:")
        print("     python scripts/checkpoint_conversion/longcat_native_weights_converter.py \\")
        print("       --source weights/longcat-for-fastvideo/transformer \\")
        print("       --output weights/longcat-native/transformer \\")
        print("       --validate")
        print("  2. Copy other components and create model_index.json")
        print("  3. Test full inference with VideoGenerator")
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure cleanup happens even if test fails
        try:
            cleanup_dist_env_and_memory()
        except:
            pass

