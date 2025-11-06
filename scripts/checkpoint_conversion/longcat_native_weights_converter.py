#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Convert LongCat wrapper weights to native FastVideo format.

This script converts weights from the Phase 1 wrapper implementation
to the Phase 2 native implementation. Key transformations:

1. Split fused QKV projections (self-attention)
2. Split fused KV projections (cross-attention)
3. Rename parameters to match native implementation

Usage:
    python scripts/checkpoint_conversion/longcat_native_weights_converter.py \
        --source weights/longcat-for-fastvideo/transformer \
        --output weights/longcat-native/transformer \
        --validate
"""

import argparse
import glob
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def split_qkv(qkv_weight: torch.Tensor, qkv_bias: torch.Tensor | None = None):
    """
    Split fused QKV projection into separate Q, K, V.
    
    Args:
        qkv_weight: [3*dim, dim]
        qkv_bias: [3*dim] or None
    
    Returns:
        (q_weight, k_weight, v_weight), (q_bias, k_bias, v_bias)
    """
    dim = qkv_weight.shape[0] // 3
    q, k, v = torch.chunk(qkv_weight, 3, dim=0)
    
    if qkv_bias is not None:
        q_bias, k_bias, v_bias = torch.chunk(qkv_bias, 3, dim=0)
    else:
        q_bias = k_bias = v_bias = None
    
    return (q, k, v), (q_bias, k_bias, v_bias)


def split_kv(kv_weight: torch.Tensor, kv_bias: torch.Tensor | None = None):
    """
    Split fused KV projection into separate K, V.
    
    Args:
        kv_weight: [2*dim, dim]
        kv_bias: [2*dim] or None
    
    Returns:
        (k_weight, v_weight), (k_bias, v_bias)
    """
    dim = kv_weight.shape[0] // 2
    k, v = torch.chunk(kv_weight, 2, dim=0)
    
    if kv_bias is not None:
        k_bias, v_bias = torch.chunk(kv_bias, 2, dim=0)
    else:
        k_bias = v_bias = None
    
    return (k, v), (k_bias, v_bias)


def convert_weights(source_weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert LongCat wrapper weights to native FastVideo format.
    
    Main transformations:
    1. Split fused QKV projections (self-attention)
    2. Split fused KV projections (cross-attention)
    3. Rename parameters according to mapping
    """
    converted = OrderedDict()
    processed_keys = set()
    
    print("Converting weights...")
    
    for key, value in tqdm(source_weights.items()):
        if key in processed_keys:
            continue
        
        # === Embedders ===
        if key.startswith("x_embedder."):
            new_key = key.replace("x_embedder.", "patch_embed.")
            converted[new_key] = value
        
        elif key.startswith("t_embedder.mlp.0."):
            new_key = key.replace("t_embedder.mlp.0.", "time_embedder.linear_1.")
            converted[new_key] = value
        
        elif key.startswith("t_embedder.mlp.2."):
            new_key = key.replace("t_embedder.mlp.2.", "time_embedder.linear_2.")
            converted[new_key] = value
        
        elif key.startswith("y_embedder.y_proj.0."):
            new_key = key.replace("y_embedder.y_proj.0.", "caption_embedder.linear_1.")
            converted[new_key] = value
        
        elif key.startswith("y_embedder.y_proj.2."):
            new_key = key.replace("y_embedder.y_proj.2.", "caption_embedder.linear_2.")
            converted[new_key] = value
        
        # === Self-Attention QKV Splitting ===
        elif ".attn.qkv." in key:
            # Extract block index
            block_idx = key.split(".")[1]
            param_type = "weight" if "weight" in key else "bias"
            
            # Get corresponding bias if this is weight
            qkv_weight = value
            qkv_bias_key = key.replace(".weight", ".bias")
            qkv_bias = source_weights.get(qkv_bias_key)
            
            # Split QKV
            (q, k, v), (q_bias, k_bias, v_bias) = split_qkv(qkv_weight, qkv_bias)
            
            # Store split weights
            converted[f"blocks.{block_idx}.self_attn.to_q.{param_type}"] = q
            converted[f"blocks.{block_idx}.self_attn.to_k.{param_type}"] = k
            converted[f"blocks.{block_idx}.self_attn.to_v.{param_type}"] = v
            
            # Mark both weight and bias as processed
            processed_keys.add(key)
            if qkv_bias is not None:
                processed_keys.add(qkv_bias_key)
        
        # === Self-Attention Output Projection ===
        elif ".attn.proj." in key:
            new_key = key.replace(".attn.proj.", ".self_attn.to_out.")
            converted[new_key] = value
        
        # === Self-Attention Normalization ===
        elif ".attn.q_norm." in key or ".attn.k_norm." in key:
            new_key = key.replace(".attn.", ".self_attn.")
            converted[new_key] = value
        
        # === Cross-Attention Q Projection ===
        elif ".cross_attn.q_linear." in key:
            new_key = key.replace(".cross_attn.q_linear.", ".cross_attn.to_q.")
            converted[new_key] = value
        
        # === Cross-Attention KV Splitting ===
        elif ".cross_attn.kv_linear." in key:
            # Extract block index
            block_idx = key.split(".")[1]
            param_type = "weight" if "weight" in key else "bias"
            
            # Get corresponding bias if this is weight
            kv_weight = value
            kv_bias_key = key.replace(".weight", ".bias")
            kv_bias = source_weights.get(kv_bias_key)
            
            # Split KV
            (k, v), (k_bias, v_bias) = split_kv(kv_weight, kv_bias)
            
            # Store split weights
            converted[f"blocks.{block_idx}.cross_attn.to_k.{param_type}"] = k
            converted[f"blocks.{block_idx}.cross_attn.to_v.{param_type}"] = v
            
            # Mark both weight and bias as processed
            processed_keys.add(key)
            if kv_bias is not None:
                processed_keys.add(kv_bias_key)
        
        # === Cross-Attention Output Projection ===
        elif ".cross_attn.proj." in key:
            new_key = key.replace(".cross_attn.proj.", ".cross_attn.to_out.")
            converted[new_key] = value
        
        # === Cross-Attention Normalization ===
        elif ".cross_attn.q_norm." in key or ".cross_attn.k_norm." in key:
            converted[key] = value  # Keep same name
        
        # === Transformer Block AdaLN ===
        elif ".adaLN_modulation.1." in key:
            new_key = key.replace(".adaLN_modulation.1.", ".adaln_linear_1.")
            converted[new_key] = value
        
        # === Transformer Block Normalization ===
        elif ".mod_norm_attn." in key:
            new_key = key.replace(".mod_norm_attn.", ".norm_attn.")
            converted[new_key] = value
        
        elif ".mod_norm_ffn." in key:
            new_key = key.replace(".mod_norm_ffn.", ".norm_ffn.")
            converted[new_key] = value
        
        elif ".pre_crs_attn_norm." in key:
            new_key = key.replace(".pre_crs_attn_norm.", ".norm_cross.")
            converted[new_key] = value
        
        # === FFN (SwiGLU) - Keep names same ===
        elif ".ffn.w1." in key or ".ffn.w2." in key or ".ffn.w3." in key:
            converted[key] = value
        
        # === Final Layer ===
        elif key.startswith("final_layer.adaLN_modulation.1."):
            new_key = key.replace("final_layer.adaLN_modulation.1.", "final_layer.adaln_linear.")
            converted[new_key] = value
        
        elif key.startswith("final_layer.norm_final."):
            new_key = key.replace("final_layer.norm_final.", "final_layer.norm.")
            converted[new_key] = value
        
        elif key.startswith("final_layer.linear."):
            new_key = key.replace("final_layer.linear.", "final_layer.proj.")
            converted[new_key] = value
        
        else:
            # Unknown key - keep as-is and warn
            print(f"  ⚠️  Unknown key: {key}")
            converted[key] = value
    
    return converted


def validate_conversion(original: dict, converted: dict):
    """
    Validate that conversion preserved all parameters correctly.
    """
    print("\nValidating conversion...")
    
    # Count parameters
    orig_count = sum(p.numel() for p in original.values())
    conv_count = sum(p.numel() for p in converted.values())
    
    print(f"  Original parameters: {orig_count:,}")
    print(f"  Converted parameters: {conv_count:,}")
    
    if orig_count != conv_count:
        print(f"  ❌ Parameter count mismatch!")
        return False
    
    print(f"  ✓ Parameter count matches")
    
    # Verify QKV/KV splits
    print("\n  Verifying QKV/KV splits...")
    num_blocks = 48
    
    for i in range(num_blocks):
        # Check self-attention QKV split
        orig_qkv_weight = original.get(f"blocks.{i}.attn.qkv.weight")
        if orig_qkv_weight is not None:
            conv_q = converted[f"blocks.{i}.self_attn.to_q.weight"]
            conv_k = converted[f"blocks.{i}.self_attn.to_k.weight"]
            conv_v = converted[f"blocks.{i}.self_attn.to_v.weight"]
            
            # Reconstruct and compare
            reconstructed = torch.cat([conv_q, conv_k, conv_v], dim=0)
            
            if not torch.allclose(orig_qkv_weight, reconstructed):
                print(f"    ❌ QKV weight mismatch in block {i}")
                return False
        
        # Check self-attention QKV bias split
        orig_qkv_bias = original.get(f"blocks.{i}.attn.qkv.bias")
        if orig_qkv_bias is not None:
            conv_q_bias = converted[f"blocks.{i}.self_attn.to_q.bias"]
            conv_k_bias = converted[f"blocks.{i}.self_attn.to_k.bias"]
            conv_v_bias = converted[f"blocks.{i}.self_attn.to_v.bias"]
            
            reconstructed = torch.cat([conv_q_bias, conv_k_bias, conv_v_bias], dim=0)
            
            if not torch.allclose(orig_qkv_bias, reconstructed):
                print(f"    ❌ QKV bias mismatch in block {i}")
                return False
        
        # Check cross-attention KV split
        orig_kv_weight = original.get(f"blocks.{i}.cross_attn.kv_linear.weight")
        if orig_kv_weight is not None:
            conv_k = converted[f"blocks.{i}.cross_attn.to_k.weight"]
            conv_v = converted[f"blocks.{i}.cross_attn.to_v.weight"]
            
            reconstructed = torch.cat([conv_k, conv_v], dim=0)
            
            if not torch.allclose(orig_kv_weight, reconstructed):
                print(f"    ❌ KV weight mismatch in block {i}")
                return False
        
        # Check cross-attention KV bias split
        orig_kv_bias = original.get(f"blocks.{i}.cross_attn.kv_linear.bias")
        if orig_kv_bias is not None:
            conv_k_bias = converted[f"blocks.{i}.cross_attn.to_k.bias"]
            conv_v_bias = converted[f"blocks.{i}.cross_attn.to_v.bias"]
            
            reconstructed = torch.cat([conv_k_bias, conv_v_bias], dim=0)
            
            if not torch.allclose(orig_kv_bias, reconstructed):
                print(f"    ❌ KV bias mismatch in block {i}")
                return False
    
    print(f"  ✓ All splits verified successfully")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert LongCat weights to native FastVideo format"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source weights (longcat-for-fastvideo/transformer/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output converted weights",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after conversion",
    )
    
    args = parser.parse_args()
    
    # Load source weights
    print(f"Loading weights from {args.source}...")
    source_path = Path(args.source)
    
    # Load all shards
    shard_files = sorted(glob.glob(str(source_path / "*.safetensors")))
    if not shard_files:
        print(f"Error: No safetensors files found in {source_path}")
        return
    
    print(f"Found {len(shard_files)} shard(s)")
    
    source_weights = {}
    for shard_file in shard_files:
        print(f"  Loading {Path(shard_file).name}...")
        source_weights.update(load_file(shard_file))
    
    print(f"Loaded {len(source_weights)} parameters")
    
    # Convert
    converted_weights = convert_weights(source_weights)
    
    print(f"\nConverted to {len(converted_weights)} parameters")
    
    # Validate if requested
    if args.validate:
        if not validate_conversion(source_weights, converted_weights):
            print("\n❌ Validation failed!")
            return
        print("\n✓ Validation passed!")
    
    # Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / "model.safetensors"
    print(f"\nSaving to {output_file}...")
    save_file(converted_weights, str(output_file))
    
    print("\n✅ Conversion complete!")
    print("\n✓ Native weights ready for use!")
    print("  Next steps:")
    print("  1. Copy other components (vae, text_encoder, etc.) to output directory")
    print("  2. Create model_index.json pointing to LongCatTransformer3DModel")
    print("  3. Test loading with VideoGenerator.from_pretrained()")
    
    # Print size info
    size_gb = output_file.stat().st_size / (1024**3)
    print(f"\nOutput size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()

