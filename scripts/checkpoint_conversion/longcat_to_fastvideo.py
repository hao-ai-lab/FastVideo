#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Convert LongCat weights to FastVideo native format.

This script performs a complete conversion from original LongCat weights
to FastVideo native implementation in a single step:

1. Converts transformer weights (with QKV/KV splitting)
2. Copies other components (VAE, text encoder, tokenizer, scheduler)
3. Updates config files to point to native model

Usage:
    python scripts/checkpoint_conversion/longcat_to_fastvideo.py \
        --source /path/to/LongCat-Video/weights/LongCat-Video \
        --output weights/longcat-native \
        --validate
"""

import argparse
import glob
import json
import shutil
from pathlib import Path
from collections import OrderedDict

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def split_qkv(qkv_weight: torch.Tensor, qkv_bias: torch.Tensor | None = None):
    """Split fused QKV projection into separate Q, K, V."""
    dim = qkv_weight.shape[0] // 3
    q, k, v = torch.chunk(qkv_weight, 3, dim=0)
    
    if qkv_bias is not None:
        q_bias, k_bias, v_bias = torch.chunk(qkv_bias, 3, dim=0)
    else:
        q_bias = k_bias = v_bias = None
    
    return (q, k, v), (q_bias, k_bias, v_bias)


def split_kv(kv_weight: torch.Tensor, kv_bias: torch.Tensor | None = None):
    """Split fused KV projection into separate K, V."""
    dim = kv_weight.shape[0] // 2
    k, v = torch.chunk(kv_weight, 2, dim=0)
    
    if kv_bias is not None:
        k_bias, v_bias = torch.chunk(kv_bias, 2, dim=0)
    else:
        k_bias = v_bias = None
    
    return (k, v), (k_bias, v_bias)


def convert_transformer_weights(source_weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert LongCat transformer weights to native FastVideo format.
    
    Main transformations:
    1. Split fused QKV projections (self-attention)
    2. Split fused KV projections (cross-attention)
    3. Rename parameters according to mapping
    """
    converted = OrderedDict()
    processed_keys = set()
    
    print("  Converting transformer weights...")
    
    for key, value in tqdm(source_weights.items(), desc="  Processing parameters"):
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
        elif ".attn.qkv.weight" in key:
            block_idx = key.split(".")[1]
            qkv_weight = value
            qkv_bias_key = key.replace(".weight", ".bias")
            qkv_bias = source_weights.get(qkv_bias_key)
            
            (q, k, v), (q_bias, k_bias, v_bias) = split_qkv(qkv_weight, qkv_bias)
            
            converted[f"blocks.{block_idx}.self_attn.to_q.weight"] = q
            converted[f"blocks.{block_idx}.self_attn.to_k.weight"] = k
            converted[f"blocks.{block_idx}.self_attn.to_v.weight"] = v
            
            if q_bias is not None:
                converted[f"blocks.{block_idx}.self_attn.to_q.bias"] = q_bias
                converted[f"blocks.{block_idx}.self_attn.to_k.bias"] = k_bias
                converted[f"blocks.{block_idx}.self_attn.to_v.bias"] = v_bias
            
            processed_keys.add(key)
            if qkv_bias is not None:
                processed_keys.add(qkv_bias_key)
        
        elif ".attn.qkv.bias" in key:
            continue
        
        elif ".attn.proj." in key:
            new_key = key.replace(".attn.proj.", ".self_attn.to_out.")
            converted[new_key] = value
        
        elif ".attn.q_norm." in key or ".attn.k_norm." in key:
            new_key = key.replace(".attn.", ".self_attn.")
            converted[new_key] = value
        
        # === Cross-Attention ===
        elif ".cross_attn.q_linear." in key:
            new_key = key.replace(".cross_attn.q_linear.", ".cross_attn.to_q.")
            converted[new_key] = value
        
        elif ".cross_attn.kv_linear.weight" in key:
            block_idx = key.split(".")[1]
            kv_weight = value
            kv_bias_key = key.replace(".weight", ".bias")
            kv_bias = source_weights.get(kv_bias_key)
            
            (k, v), (k_bias, v_bias) = split_kv(kv_weight, kv_bias)
            
            converted[f"blocks.{block_idx}.cross_attn.to_k.weight"] = k
            converted[f"blocks.{block_idx}.cross_attn.to_v.weight"] = v
            
            if k_bias is not None:
                converted[f"blocks.{block_idx}.cross_attn.to_k.bias"] = k_bias
                converted[f"blocks.{block_idx}.cross_attn.to_v.bias"] = v_bias
            
            processed_keys.add(key)
            if kv_bias is not None:
                processed_keys.add(kv_bias_key)
        
        elif ".cross_attn.kv_linear.bias" in key:
            continue
        
        elif ".cross_attn.proj." in key:
            new_key = key.replace(".cross_attn.proj.", ".cross_attn.to_out.")
            converted[new_key] = value
        
        elif ".cross_attn.q_norm." in key or ".cross_attn.k_norm." in key:
            converted[key] = value
        
        # === Final Layer (must come BEFORE general transformer block patterns) ===
        elif key.startswith("final_layer.adaLN_modulation.1."):
            new_key = key.replace("final_layer.adaLN_modulation.1.", "final_layer.adaln_linear.")
            converted[new_key] = value
        
        # === Transformer Block AdaLN ===
        elif ".adaLN_modulation.1." in key:
            new_key = key.replace(".adaLN_modulation.1.", ".adaln_linear_1.")
            converted[new_key] = value
        
        # === Transformer Block Normalization ===
        elif ".mod_norm_attn." in key or ".mod_norm_ffn." in key:
            continue
        
        elif ".pre_crs_attn_norm.weight" in key:
            new_key = key.replace(".pre_crs_attn_norm.", ".norm_cross.")
            converted[new_key] = value
        
        elif ".pre_crs_attn_norm.bias" in key:
            new_key = key.replace(".pre_crs_attn_norm.", ".norm_cross.")
            converted[new_key] = value
        
        # === FFN (SwiGLU) ===
        elif ".ffn.w1." in key or ".ffn.w2." in key or ".ffn.w3." in key:
            converted[key] = value
        
        elif key.startswith("final_layer.norm_final."):
            continue
        
        elif key.startswith("final_layer.linear."):
            new_key = key.replace("final_layer.linear.", "final_layer.proj.")
            converted[new_key] = value
        
        else:
            print(f"    ⚠️  Unknown key: {key}")
            converted[key] = value
    
    return converted


def validate_conversion(original: dict, converted: dict) -> bool:
    """Validate that conversion preserved all parameters correctly."""
    print("\n  Validating conversion...")
    
    orig_count = sum(p.numel() for p in original.values())
    conv_count = sum(p.numel() for p in converted.values())
    
    dropped_count = 0
    for key, value in original.items():
        if ".mod_norm_attn." in key or ".mod_norm_ffn." in key:
            dropped_count += value.numel()
        elif "final_layer.norm_final." in key:
            dropped_count += value.numel()
    
    expected_conv_count = orig_count - dropped_count
    
    print(f"    Original parameters: {orig_count:,}")
    print(f"    Converted parameters: {conv_count:,}")
    print(f"    Dropped parameters (norms without params): {dropped_count:,}")
    
    if conv_count != expected_conv_count:
        print(f"    ⚠️  Parameter count mismatch!")
        return False
    
    print(f"    ✓ Parameter count matches")
    
    # Verify QKV/KV splits
    print("\n  Verifying QKV/KV splits...")
    num_blocks = 48
    
    for i in range(num_blocks):
        orig_qkv_weight = original.get(f"blocks.{i}.attn.qkv.weight")
        if orig_qkv_weight is not None:
            conv_q = converted[f"blocks.{i}.self_attn.to_q.weight"]
            conv_k = converted[f"blocks.{i}.self_attn.to_k.weight"]
            conv_v = converted[f"blocks.{i}.self_attn.to_v.weight"]
            reconstructed = torch.cat([conv_q, conv_k, conv_v], dim=0)
            if not torch.allclose(orig_qkv_weight, reconstructed):
                print(f"    ❌ QKV weight mismatch in block {i}")
                return False
        
        orig_kv_weight = original.get(f"blocks.{i}.cross_attn.kv_linear.weight")
        if orig_kv_weight is not None:
            conv_k = converted[f"blocks.{i}.cross_attn.to_k.weight"]
            conv_v = converted[f"blocks.{i}.cross_attn.to_v.weight"]
            reconstructed = torch.cat([conv_k, conv_v], dim=0)
            if not torch.allclose(orig_kv_weight, reconstructed):
                print(f"    ❌ KV weight mismatch in block {i}")
                return False
    
    print(f"    ✓ All splits verified successfully")
    return True


def copy_component(source_dir: Path, output_dir: Path, component: str, mapping: dict = None) -> bool:
    """Copy a component directory, optionally with name mapping."""
    source_name = mapping.get(component, component) if mapping else component
    source_path = source_dir / source_name
    
    if source_path.exists():
        output_path = output_dir / component
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(source_path, output_path)
        print(f"  ✓ {component} copied")
        return True
    else:
        print(f"  ⚠️  {component} not found, skipping")
        return False


def create_model_index():
    """Create model_index.json for FastVideo native model."""
    return {
        "_class_name": "LongCatPipeline",
        "_diffusers_version": "0.32.0",
        "workload_type": "video-generation",
        "tokenizer": ["transformers", "AutoTokenizer"],
        "text_encoder": ["transformers", "UMT5EncoderModel"],
        "vae": ["diffusers", "AutoencoderKLWan"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "transformer": ["diffusers", "LongCatTransformer3DModel"]  # Native model
    }


def update_transformer_config(transformer_dir: Path):
    """Update transformer config.json to point to native model."""
    config_path = transformer_dir / "config.json"
    if not config_path.exists():
        print("  ⚠️  Transformer config not found, skipping")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if '_class_name' in config:
        old_class = config['_class_name']
        config['_class_name'] = 'LongCatTransformer3DModel'
        print(f"  Updated _class_name: {old_class} → LongCatTransformer3DModel")
    else:
        config['_class_name'] = 'LongCatTransformer3DModel'
        print(f"  Added _class_name: LongCatTransformer3DModel")
    
    # Fix num_heads -> num_attention_heads for FastVideo compatibility
    if 'num_heads' in config and 'num_attention_heads' not in config:
        config['num_attention_heads'] = config.pop('num_heads')
        print(f"  Updated num_heads → num_attention_heads")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("  ✓ Transformer config updated")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LongCat weights to FastVideo native format"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to original LongCat weights (LongCat-Video/weights/LongCat-Video/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output directory for native weights",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after conversion",
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    
    # Check source directory
    if not source_dir.exists():
        print(f"❌ Error: Source directory not found: {source_dir}")
        return 1
    
    # Check for dit/transformer directory (original uses 'dit', we output to 'transformer')
    transformer_source = source_dir / "dit"
    if not transformer_source.exists():
        print(f"❌ Error: DiT directory not found in source")
        return 1
    
    print("=" * 60)
    print("LongCat → FastVideo Native Conversion")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Step 1: Convert transformer weights
    print("[Step 1/4] Converting transformer weights...")
    
    # Load source weights
    shard_files = sorted(glob.glob(str(transformer_source / "*.safetensors")))
    if not shard_files:
        print(f"❌ Error: No safetensors files found in {transformer_source}")
        return 1
    
    print(f"  Found {len(shard_files)} shard(s)")
    source_weights = {}
    for shard_file in shard_files:
        print(f"  Loading {Path(shard_file).name}...")
        source_weights.update(load_file(shard_file))
    
    print(f"  Loaded {len(source_weights)} parameters")
    
    # Convert
    converted_weights = convert_transformer_weights(source_weights)
    print(f"  Converted to {len(converted_weights)} parameters")
    
    # Validate if requested
    if args.validate:
        if not validate_conversion(source_weights, converted_weights):
            print("\n❌ Validation failed!")
            return 1
        print("\n✓ Validation passed!")
    
    # Save
    transformer_output = output_dir / "transformer"
    transformer_output.mkdir(parents=True, exist_ok=True)
    output_file = transformer_output / "model.safetensors"
    print(f"\n  Saving to {output_file}...")
    save_file(converted_weights, str(output_file))
    size_gb = output_file.stat().st_size / (1024**3)
    print(f"  ✓ Saved ({size_gb:.2f} GB)")
    print()
    
    # Step 2: Copy other components
    print("[Step 2/4] Copying other components...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    components = ["vae", "text_encoder", "tokenizer", "scheduler"]
    for component in components:
        copy_component(source_dir, output_dir, component)
    
    # Copy LoRA if exists
    lora_source = source_dir / "lora"
    if lora_source.exists():
        lora_dest = output_dir / "lora"
        if lora_dest.exists():
            shutil.rmtree(lora_dest)
        shutil.copytree(lora_source, lora_dest)
        print(f"  ✓ lora copied")
    
    print()
    
    # Step 3: Update transformer config
    print("[Step 3/4] Updating transformer config...")
    
    # Copy config.json from source if exists
    source_config = transformer_source / "config.json"
    output_config = transformer_output / "config.json"
    if source_config.exists():
        shutil.copy(source_config, output_config)
        print(f"  Copied config.json")
    
    update_transformer_config(transformer_output)
    print()
    
    # Step 4: Create model_index.json
    print("[Step 4/4] Creating model_index.json...")
    model_index_path = output_dir / "model_index.json"
    with open(model_index_path, 'w') as f:
        json.dump(create_model_index(), f, indent=2)
    print(f"  ✓ Created {model_index_path}")
    print()
    
    print("=" * 60)
    print("✓ Conversion Complete!")
    print("=" * 60)
    print(f"Native weights ready at: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Test loading the model:")
    print("     from fastvideo import VideoGenerator")
    print(f"     generator = VideoGenerator.from_pretrained('{output_dir}')")
    print()
    print("  2. Generate a test video:")
    print("     video = generator.generate_video(")
    print("         prompt='A cat playing piano',")
    print("         num_inference_steps=50")
    print("     )")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
