"""
Debug script to check if weights are loaded correctly in the native LongCat model.
Compare weights between original and converted models.
"""

import torch
from safetensors.torch import load_file
from pathlib import Path
import numpy as np
import json

def compare_weights():
    print("="*80)
    print("Weight Loading Verification")
    print("="*80)
    
    # Load original weights (sharded)
    original_dir = Path("/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video/dit")
    converted_path = Path("/mnt/fast-disks/hao_lab/shao/FastVideo/weights/longcat-native/transformer/model.safetensors")
    
    print("\n[1] Loading original weights (sharded)...")
    # Load the index to know which keys are in which file
    with open(original_dir / "diffusion_pytorch_model.safetensors.index.json") as f:
        index = json.load(f)
    
    # Load all shards
    original_weights = {}
    weight_map = index["weight_map"]
    unique_files = set(weight_map.values())
    
    for shard_file in sorted(unique_files):
        shard_path = original_dir / shard_file
        print(f"  Loading {shard_file}...")
        shard_weights = load_file(shard_path)
        original_weights.update(shard_weights)
    
    print(f"  Original model keys: {len(original_weights)}")
    
    print("\n[2] Loading converted weights...")
    converted_weights = load_file(converted_path)
    print(f"  Converted model keys: {len(converted_weights)}")
    
    # Check if all weights were converted
    print("\n[3] Checking weight conversion...")
    
    # Sample some important weights to verify
    test_keys_mapping = [
        # Original -> Expected converted
        ("pos_embedder.proj.weight", "pos_embedder.proj.weight"),
        ("t_embedder.mlp.0.weight", "t_embedder.mlp.0.weight"),
        ("transformer_blocks.0.adaLN_modulation.1.weight", "transformer_blocks.0.adaln_linear_1.weight"),
        ("transformer_blocks.0.attn.qkv.weight", "transformer_blocks.0.self_attn.qkv.weight"),
        ("transformer_blocks.0.attn.proj.weight", "transformer_blocks.0.self_attn.proj.weight"),
        ("transformer_blocks.0.mlp.fc1.weight", "transformer_blocks.0.ffn.fc1.weight"),
        ("final_layer.adaLN_modulation.1.weight", "final_layer.adaln_linear.weight"),
        ("final_layer.linear.weight", "final_layer.proj.weight"),
    ]
    
    print("\n  Checking key mappings:")
    for orig_key, expected_conv_key in test_keys_mapping:
        if orig_key in original_weights:
            if expected_conv_key in converted_weights:
                orig_shape = original_weights[orig_key].shape
                conv_shape = converted_weights[expected_conv_key].shape
                
                # Check if shapes match
                if orig_shape == conv_shape:
                    # Check if values match
                    orig_val = original_weights[orig_key]
                    conv_val = converted_weights[expected_conv_key]
                    
                    max_diff = torch.abs(orig_val - conv_val).max().item()
                    mean_diff = torch.abs(orig_val - conv_val).mean().item()
                    
                    status = "✓" if max_diff < 1e-6 else "✗"
                    print(f"  {status} {orig_key}")
                    print(f"      -> {expected_conv_key}")
                    print(f"      Shape: {orig_shape}, Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
                else:
                    print(f"  ✗ {orig_key}")
                    print(f"      -> {expected_conv_key}")
                    print(f"      Shape mismatch! Orig: {orig_shape}, Conv: {conv_shape}")
            else:
                print(f"  ✗ {orig_key}")
                print(f"      -> {expected_conv_key} NOT FOUND in converted weights")
        else:
            print(f"  ? {orig_key} not in original weights")
    
    # Check for any unconverted keys
    print("\n[4] Checking for unconverted original keys...")
    original_key_patterns = set()
    for key in original_weights.keys():
        # Extract pattern (remove layer numbers)
        pattern = key
        for i in range(100):
            pattern = pattern.replace(f".{i}.", ".N.")
        original_key_patterns.add(pattern)
    
    print(f"\n  Original key patterns (first 10):")
    for i, pattern in enumerate(sorted(original_key_patterns)[:10]):
        print(f"    {pattern}")
    
    print("\n[5] Statistics of converted weights...")
    for key, value in list(converted_weights.items())[:5]:
        print(f"  {key}:")
        print(f"    Shape: {value.shape}")
        print(f"    Mean: {value.float().mean().item():.6f}, Std: {value.float().std().item():.6f}")
        print(f"    Min: {value.float().min().item():.6f}, Max: {value.float().max().item():.6f}")

if __name__ == "__main__":
    compare_weights()

