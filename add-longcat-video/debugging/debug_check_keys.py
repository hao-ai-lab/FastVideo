"""
Check the actual key names in both original and converted models.
"""

import torch
from safetensors.torch import load_file
from pathlib import Path
import json

def check_keys():
    print("="*80)
    print("Key Name Analysis")
    print("="*80)
    
    # Load original weights (sharded)
    original_dir = Path("/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video/dit")
    converted_path = Path("/mnt/fast-disks/hao_lab/shao/FastVideo/weights/longcat-native/transformer/model.safetensors")
    
    print("\n[1] Loading original weights...")
    with open(original_dir / "diffusion_pytorch_model.safetensors.index.json") as f:
        index = json.load(f)
    
    original_weights = {}
    weight_map = index["weight_map"]
    unique_files = set(weight_map.values())
    
    for shard_file in sorted(unique_files):
        shard_path = original_dir / shard_file
        shard_weights = load_file(shard_path)
        original_weights.update(shard_weights)
    
    print(f"  Original keys: {len(original_weights)}")
    
    print("\n[2] Loading converted weights...")
    converted_weights = load_file(converted_path)
    print(f"  Converted keys: {len(converted_weights)}")
    
    # Show first 20 keys of each
    print("\n[3] First 20 original keys:")
    for i, key in enumerate(sorted(original_weights.keys())[:20]):
        print(f"  {key}")
    
    print("\n[4] First 20 converted keys:")
    for i, key in enumerate(sorted(converted_weights.keys())[:20]):
        print(f"  {key}")
    
    # Check for duplicates or issues
    print("\n[5] Looking for duplicate patterns in converted keys...")
    key_patterns = {}
    for key in converted_weights.keys():
        # Extract pattern
        pattern = key
        for i in range(100):
            pattern = pattern.replace(f".{i}.", ".N.")
        if pattern in key_patterns:
            key_patterns[pattern].append(key)
        else:
            key_patterns[pattern] = [key]
    
    # Show patterns with unusual counts
    print("\n  Patterns with unusual key counts:")
    for pattern, keys in sorted(key_patterns.items()):
        if len(keys) > 50:  # More than 48 blocks would be unusual
            print(f"  {pattern}: {len(keys)} keys")
            if len(keys) < 10:
                for key in keys[:10]:
                    print(f"    - {key}")
    
    # Check if any keys from original are missing in converted
    print("\n[6] Checking conversion mapping...")
    
    # Map some original keys to expected converted keys manually
    test_mappings = [
        ("blocks.0.adaLN_modulation.1.weight", ["blocks.0.adaln_linear_1.weight"]),
        ("blocks.0.attn.qkv.weight", ["blocks.0.self_attn.qkv.weight"]),
        ("blocks.0.attn.proj.weight", ["blocks.0.self_attn.proj.weight"]),
        ("blocks.0.mlp.fc1.weight", ["blocks.0.ffn.fc1.weight"]),
        ("t_embedder.mlp.0.weight", ["t_embedder.mlp.0.weight"]),
        ("pos_embedder.proj.weight", ["pos_embedder.proj.weight"]),
        ("final_layer.adaLN_modulation.1.weight", ["final_layer.adaln_linear.weight"]),
    ]
    
    for orig_key, possible_conv_keys in test_mappings:
        if orig_key in original_weights:
            found = False
            for conv_key in possible_conv_keys:
                if conv_key in converted_weights:
                    found = True
                    # Compare
                    orig_val = original_weights[orig_key]
                    conv_val = converted_weights[conv_key]
                    if orig_val.shape == conv_val.shape:
                        max_diff = torch.abs(orig_val - conv_val).max().item()
                        status = "✓" if max_diff < 1e-6 else f"✗ (diff: {max_diff:.2e})"
                    else:
                        status = f"✗ (shape: {orig_val.shape} vs {conv_val.shape})"
                    print(f"  {status} {orig_key} -> {conv_key}")
                    break
            if not found:
                print(f"  ✗ {orig_key} -> NOT FOUND (expected: {possible_conv_keys})")
        else:
            print(f"  ? {orig_key} not in original")

if __name__ == "__main__":
    check_keys()


