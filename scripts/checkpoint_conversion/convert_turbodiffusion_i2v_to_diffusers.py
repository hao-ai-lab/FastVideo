#!/usr/bin/env python3
"""
Convert TurboDiffusion I2V .pth checkpoint to Diffusers safetensors format.

TurboDiffusion I2V uses two models: high-noise and low-noise.
This script converts both checkpoints to Diffusers format.

Usage:
    python convert_turbodiffusion_i2v_to_diffusers.py \
        --high_noise_path /path/to/TurboWan2.2-I2V-A14B-high-720P.pth \
        --low_noise_path /path/to/TurboWan2.2-I2V-A14B-low-720P.pth \
        --output_dir /path/to/output \
        --reference_repo Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
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
# Same as T2V but may need additional I2V-specific mappings
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
    # I2V-specific cross attention (add_k/v_proj for image context)
    r"^blocks\.(\d+)\.cross_attn\.add_k\.(.*)$": r"blocks.\1.attn2.add_k_proj.\2",
    r"^blocks\.(\d+)\.cross_attn\.add_v\.(.*)$": r"blocks.\1.attn2.add_v_proj.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_add_k\.(.*)$": r"blocks.\1.attn2.norm_added_k.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_add_q\.(.*)$": r"blocks.\1.attn2.norm_added_q.\2",
    # Norms and FFN
    r"^blocks\.(\d+)\.norm1\.(.*)$": r"blocks.\1.norm1.\2",
    r"^blocks\.(\d+)\.norm3\.(.*)$": r"blocks.\1.self_attn_residual_norm.norm.\2",
    r"^blocks\.(\d+)\.norm2\.(.*)$": r"blocks.\1.norm3.\2",
    r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
    r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
    r"^blocks\.(\d+)\.modulation$": r"blocks.\1.scale_shift_table",
    # Embeddings
    r"^text_embedding\.0\.(.*)$": r"condition_embedder.text_embedder.fc_in.\1",
    r"^text_embedding\.2\.(.*)$": r"condition_embedder.text_embedder.fc_out.\1",
    r"^time_embedding\.0\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_in.\1",
    r"^time_embedding\.2\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_out.\1",
    r"^time_projection\.1\.(.*)$": r"condition_embedder.time_modulation.linear.\1",
    # Head
    r"^head\.head\.(.*)$": r"proj_out.\1",
    r"^head\.norm\.(.*)$": r"norm_out.\1",
    r"^head\.modulation$": r"scale_shift_table",
    # SLA proj_l weights
    r"^blocks\.(\d+)\.self_attn\.attn_op\.local_attn\.proj_l\.(.*)$": r"blocks.\1.attn1.attn_impl.proj_l.\2",
}

SKIP_PATTERNS = []


def should_skip_key(key: str) -> bool:
    for pattern in SKIP_PATTERNS:
        if re.match(pattern, key):
            return True
    return False


def convert_key(turbo_key: str) -> str:
    for pattern, replacement in TURBODIFFUSION_WEIGHT_MAPPING.items():
        if re.match(pattern, turbo_key):
            return re.sub(pattern, replacement, turbo_key)
    return turbo_key


def reshape_patch_embedding(tensor: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    if len(tensor.shape) == 2 and len(target_shape) == 5:
        return tensor.view(target_shape)
    return tensor


def get_reference_shapes(reference_repo: str) -> dict:
    print(f"Downloading reference model shapes from {reference_repo}...")
    
    local_dir = snapshot_download(
        repo_id=reference_repo,
        allow_patterns=["transformer/config.json", "transformer/diffusion_pytorch_model*.safetensors"],
        local_dir_use_symlinks=False
    )
    
    weight_files = glob.glob(os.path.join(local_dir, "transformer", "*.safetensors"))
    shapes = {}
    
    for wf in weight_files:
        with safe_open(wf, framework="pt") as f:
            for key in f.keys():
                shapes[key] = f.get_tensor(key).shape
    
    return shapes, local_dir


def convert_checkpoint(input_path: str, output_dir: str, ref_shapes: dict, model_name: str) -> None:
    """Convert a single TurboDiffusion checkpoint to Diffusers format."""
    print(f"\n{'='*60}")
    print(f"Converting {model_name}: {input_path}")
    print(f"{'='*60}")
    
    turbo_state_dict = torch.load(input_path, map_location="cpu", weights_only=True)
    print(f"Loaded {len(turbo_state_dict)} keys")
    
    converted_state_dict = {}
    skipped_keys = []
    
    for turbo_key, tensor in turbo_state_dict.items():
        if should_skip_key(turbo_key):
            skipped_keys.append(turbo_key)
            continue
        
        new_key = convert_key(turbo_key)
        
        if "patch_embedding" in new_key and new_key in ref_shapes:
            target_shape = ref_shapes[new_key]
            if tensor.shape != target_shape:
                print(f"Reshaping {new_key}: {tensor.shape} -> {target_shape}")
                tensor = reshape_patch_embedding(tensor, target_shape)
        
        if new_key in ref_shapes:
            if tensor.shape != ref_shapes[new_key]:
                print(f"WARNING: Shape mismatch for {new_key}: got {tensor.shape}, expected {ref_shapes[new_key]}")
        
        converted_state_dict[new_key] = tensor
    
    print(f"Converted: {len(converted_state_dict)} keys, Skipped: {len(skipped_keys)} keys")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "diffusion_pytorch_model.safetensors")
    print(f"Saving to {output_path}...")
    save_file(converted_state_dict, output_path)
    
    return converted_state_dict


def main():
    parser = argparse.ArgumentParser(description="Convert TurboDiffusion I2V checkpoints to Diffusers format")
    parser.add_argument("--high_noise_path", type=str, required=True,
                        help="Path to high-noise TurboDiffusion .pth checkpoint")
    parser.add_argument("--low_noise_path", type=str, required=True,
                        help="Path to low-noise TurboDiffusion .pth checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for converted safetensors")
    parser.add_argument("--reference_repo", type=str, default="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                        help="Reference HF repo to get expected tensor shapes")
    
    args = parser.parse_args()
    
    # Get reference shapes
    ref_shapes, ref_local_dir = get_reference_shapes(args.reference_repo)
    print(f"Got {len(ref_shapes)} reference shapes")
    
    # Convert high-noise model
    high_noise_output = os.path.join(args.output_dir, "transformer_high")
    convert_checkpoint(args.high_noise_path, high_noise_output, ref_shapes, "high-noise")
    
    # Convert low-noise model
    low_noise_output = os.path.join(args.output_dir, "transformer_low")
    convert_checkpoint(args.low_noise_path, low_noise_output, ref_shapes, "low-noise")
    
    # Copy config.json to both
    src_config = os.path.join(ref_local_dir, "transformer", "config.json")
    shutil.copy(src_config, os.path.join(high_noise_output, "config.json"))
    shutil.copy(src_config, os.path.join(low_noise_output, "config.json"))
    print("Copied config.json to both transformer directories")
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"High-noise model: {high_noise_output}")
    print(f"Low-noise model: {low_noise_output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
