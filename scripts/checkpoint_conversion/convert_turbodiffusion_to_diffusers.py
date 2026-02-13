#!/usr/bin/env python3
"""
Convert TurboDiffusion .pth checkpoint to Diffusers safetensors format.

This script:
1. Loads the TurboDiffusion .pth checkpoint
2. Applies weight key renaming to match FastVideo/Diffusers format
3. Reshapes patch_embedding from [D, C*P] to [D, C, P_t, P_h, P_w]
4. Saves as sharded safetensors files compatible with diffusers

Usage:
    python convert_turbodiffusion_to_diffusers.py \
        --input_path /path/to/TurboWan2.1-T2V-1.3B-480P.pth \
        --output_dir /path/to/output/transformer \
        --reference_repo Wan-AI/Wan2.1-T2V-1.3B-Diffusers
"""

import argparse
import os
import re
import json
import torch
import shutil
import glob
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

# Weight mapping from TurboDiffusion -> Diffusers/FastVideo format
TURBODIFFUSION_WEIGHT_MAPPING = {
    # Self attention
    r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.to_q.\2",
    r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.to_k.\2",
    r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.to_v.\2",
    r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.to_out.\2",
    r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$": r"blocks.\1.norm_q.\2",
    r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$": r"blocks.\1.norm_k.\2",
    # Cross attention
    r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
    r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
    r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
    r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": r"blocks.\1.attn2.to_out.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$": r"blocks.\1.attn2.norm_q.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$": r"blocks.\1.attn2.norm_k.\2",
    # Norms and FFN
    r"^blocks\.(\d+)\.norm1\.(.*)$": r"blocks.\1.norm1.\2",
    r"^blocks\.(\d+)\.norm3\.(.*)$": r"blocks.\1.self_attn_residual_norm.norm.\2",
    r"^blocks\.(\d+)\.norm2\.(.*)$": r"blocks.\1.norm3.\2",
    r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
    r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
    r"^blocks\.(\d+)\.modulation$": r"blocks.\1.scale_shift_table",
    # Embeddings - DON'T add .proj here! WanVideoArchConfig's param_names_mapping will add it
    # patch_embedding.weight stays as patch_embedding.weight (HF format needs this)
    r"^text_embedding\.0\.(.*)$": r"condition_embedder.text_embedder.fc_in.\1",
    r"^text_embedding\.2\.(.*)$": r"condition_embedder.text_embedder.fc_out.\1",
    r"^time_embedding\.0\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_in.\1",
    r"^time_embedding\.2\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_out.\1",
    r"^time_projection\.1\.(.*)$": r"condition_embedder.time_modulation.linear.\1",
    # Head
    r"^head\.head\.(.*)$": r"proj_out.\1",
    r"^head\.norm\.(.*)$": r"norm_out.\1",
    r"^head\.modulation$": r"scale_shift_table",
    # SLA proj_l weights - include them! They're the distilled attention weights
    r"^blocks\.(\d+)\.self_attn\.attn_op\.local_attn\.proj_l\.(.*)$": r"blocks.\1.attn1.attn_impl.proj_l.\2",
}

# No keys to skip - we want all weights including proj_l
SKIP_PATTERNS = []


def should_skip_key(key: str) -> bool:
    """Check if a key should be skipped (SLA-specific weights)."""
    for pattern in SKIP_PATTERNS:
        if re.match(pattern, key):
            return True
    return False


def convert_key(turbo_key: str) -> str:
    """Convert TurboDiffusion key to Diffusers format."""
    for pattern, replacement in TURBODIFFUSION_WEIGHT_MAPPING.items():
        if re.match(pattern, turbo_key):
            return re.sub(pattern, replacement, turbo_key)
    return turbo_key  # Return unchanged if no pattern matches


def reshape_patch_embedding(tensor: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Reshape patch_embedding from [D, C*P_t*P_h*P_w] to [D, C, P_t, P_h, P_w]."""
    if len(tensor.shape) == 2 and len(target_shape) == 5:
        return tensor.view(target_shape)
    return tensor


def get_reference_shapes(reference_repo: str) -> dict:
    """Download reference model and get expected shapes for patch_embedding."""
    print(f"Downloading reference model shapes from {reference_repo}...")
    
    # Download just the transformer config and a weight file to get shapes
    local_dir = snapshot_download(
        repo_id=reference_repo,
        allow_patterns=["transformer/config.json", "transformer/diffusion_pytorch_model*.safetensors"],
        local_dir_use_symlinks=False
    )
    
    # Load the first safetensors file to get shapes    
    weight_files = glob.glob(os.path.join(local_dir, "transformer", "*.safetensors"))
    shapes = {}
    
    for wf in weight_files:
        with safe_open(wf, framework="pt") as f:
            for key in f.keys():
                shapes[key] = f.get_tensor(key).shape
    
    return shapes


def main():
    parser = argparse.ArgumentParser(description="Convert TurboDiffusion checkpoint to Diffusers format")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to TurboDiffusion .pth checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for converted safetensors")
    parser.add_argument("--reference_repo", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                        help="Reference HF repo to get expected tensor shapes")
    parser.add_argument("--skip_sla_weights", action="store_true", default=False,
                        help="Skip SLA-specific weights (proj_l) that aren't in base model")
    
    args = parser.parse_args()
    
    # Load TurboDiffusion checkpoint
    print(f"Loading TurboDiffusion checkpoint from {args.input_path}...")
    turbo_state_dict = torch.load(args.input_path, map_location="cpu", weights_only=True)
    print(f"Loaded {len(turbo_state_dict)} keys")
    
    # Get reference shapes for reshaping
    ref_shapes = get_reference_shapes(args.reference_repo)
    print(f"Got {len(ref_shapes)} reference shapes")
    
    # Convert keys and reshape tensors
    converted_state_dict = {}
    skipped_keys = []
    
    for turbo_key, tensor in turbo_state_dict.items():
        # Skip SLA-specific weights if requested
        if args.skip_sla_weights and should_skip_key(turbo_key):
            skipped_keys.append(turbo_key)
            continue
        
        # Convert key name
        new_key = convert_key(turbo_key)
        
        # Reshape patch_embedding if needed
        if "patch_embedding" in new_key and new_key in ref_shapes:
            target_shape = ref_shapes[new_key]
            if tensor.shape != target_shape:
                print(f"Reshaping {new_key}: {tensor.shape} -> {target_shape}")
                tensor = reshape_patch_embedding(tensor, target_shape)
        
        # Verify shape matches reference if available
        if new_key in ref_shapes:
            if tensor.shape != ref_shapes[new_key]:
                print(f"WARNING: Shape mismatch for {new_key}: got {tensor.shape}, expected {ref_shapes[new_key]}")
        
        converted_state_dict[new_key] = tensor
    
    print(f"\nConversion summary:")
    print(f"  Converted: {len(converted_state_dict)} keys")
    print(f"  Skipped (SLA): {len(skipped_keys)} keys")
    
    if skipped_keys:
        print(f"\nSkipped SLA keys (first 5):")
        for k in skipped_keys[:5]:
            print(f"  - {k}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save as safetensors
    output_path = os.path.join(args.output_dir, "diffusion_pytorch_model.safetensors")
    print(f"\nSaving to {output_path}...")
    save_file(converted_state_dict, output_path)
    
    # Copy config.json from reference
    ref_local = snapshot_download(
        repo_id=args.reference_repo,
        allow_patterns=["transformer/config.json"],
        local_dir_use_symlinks=False
    )
    src_config = os.path.join(ref_local, "transformer", "config.json")
    dst_config = os.path.join(args.output_dir, "config.json")
    shutil.copy(src_config, dst_config)
    print(f"Copied config.json")
    
    print(f"\nDone! Converted weights saved to: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Use create_hf_repo.py to create a complete diffusers repo:")
    print(f"     python scripts/checkpoint_conversion/create_hf_repo.py \\")
    print(f"       --repo_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \\")
    print(f"       --local_dir /tmp/turbodiffusion-wan \\")
    print(f"       --checkpoint_dir {args.output_dir} \\")
    print(f"       --push_to_hub --upload_repo_id YOUR_USERNAME/TurboWan-Diffusers")


if __name__ == "__main__":
    main()
