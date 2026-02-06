#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Convert official Hunyuan-GameCraft weights to FastVideo-compatible format.

This script converts the official PyTorch checkpoint format to a Diffusers-style
safetensors format that can be loaded by FastVideo.

Usage:
    python convert_gamecraft_to_fastvideo.py \
        --source /path/to/official_weights/gamecraft_model_states.pt \
        --output /path/to/converted_weights/HunyuanGameCraft \
        --load-key module
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import save_file


# Mapping from official GameCraft checkpoint keys to FastVideo keys
# Order matters - more specific patterns should come first
PARAM_NAME_MAPPING = [
    # ==================== SingleTokenRefiner / txt_in ====================
    # Note: Official has individual_token_refiner.blocks, FastVideo has refiner_blocks
    
    # MLP in refiner blocks: official uses mlp.fc1, mlp.fc2; FastVideo uses mlp.fc_in, mlp.fc_out
    (r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.mlp\.fc1\.(.*)$",
     r"txt_in.refiner_blocks.\1.mlp.fc_in.\2"),
    (r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.mlp\.fc2\.(.*)$",
     r"txt_in.refiner_blocks.\1.mlp.fc_out.\2"),
    # adaLN_modulation: official uses Sequential (0, 1), FastVideo uses direct linear
    (r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.adaLN_modulation\.1\.(.*)$",
     r"txt_in.refiner_blocks.\1.adaLN_modulation.linear.\2"),
    
    # Individual token refiner blocks (in txt_in) - general patterns after specific ones
    (r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.(.*)$",
     r"txt_in.refiner_blocks.\1.\2"),
    
    # t_embedder in txt_in: official uses mlp.0, mlp.2; FastVideo uses mlp.fc_in, mlp.fc_out
    (r"^txt_in\.t_embedder\.mlp\.0\.(.*)$", r"txt_in.t_embedder.mlp.fc_in.\1"),
    (r"^txt_in\.t_embedder\.mlp\.2\.(.*)$", r"txt_in.t_embedder.mlp.fc_out.\1"),
    
    # c_embedder in txt_in: official uses linear_1, linear_2; FastVideo uses fc_in, fc_out
    (r"^txt_in\.c_embedder\.linear_1\.(.*)$", r"txt_in.c_embedder.fc_in.\1"),
    (r"^txt_in\.c_embedder\.linear_2\.(.*)$", r"txt_in.c_embedder.fc_out.\1"),
    
    # input_embedder in txt_in (direct mapping)
    (r"^txt_in\.input_embedder\.(.*)$", r"txt_in.input_embedder.\1"),
    
    # ==================== Time embedder ====================
    # time_in: official uses mlp.0, mlp.2; FastVideo uses mlp.fc_in, mlp.fc_out
    (r"^time_in\.mlp\.0\.(.*)$", r"time_in.mlp.fc_in.\1"),
    (r"^time_in\.mlp\.2\.(.*)$", r"time_in.mlp.fc_out.\1"),
    
    # ==================== Guidance embedder ====================
    # guidance_in: same as time_in
    (r"^guidance_in\.mlp\.0\.(.*)$", r"guidance_in.mlp.fc_in.\1"),
    (r"^guidance_in\.mlp\.2\.(.*)$", r"guidance_in.mlp.fc_out.\1"),
    
    # ==================== Vector embedder ====================
    # vector_in: official uses in_layer, out_layer; FastVideo uses fc_in, fc_out
    (r"^vector_in\.in_layer\.(.*)$", r"vector_in.fc_in.\1"),
    (r"^vector_in\.out_layer\.(.*)$", r"vector_in.fc_out.\1"),
    
    # ==================== Double stream blocks ====================
    # img_mod: official uses ModulateDiT with linear; FastVideo uses ModulateProjection with linear
    (r"^double_blocks\.(\d+)\.img_mod\.linear\.(.*)$",
     r"double_blocks.\1.img_mod.linear.\2"),
    (r"^double_blocks\.(\d+)\.txt_mod\.linear\.(.*)$",
     r"double_blocks.\1.txt_mod.linear.\2"),
    
    # QKV and projections in double blocks
    (r"^double_blocks\.(\d+)\.img_attn_qkv\.(.*)$",
     r"double_blocks.\1.img_attn_qkv.\2"),
    (r"^double_blocks\.(\d+)\.img_attn_q_norm\.(.*)$",
     r"double_blocks.\1.img_attn_q_norm.\2"),
    (r"^double_blocks\.(\d+)\.img_attn_k_norm\.(.*)$",
     r"double_blocks.\1.img_attn_k_norm.\2"),
    (r"^double_blocks\.(\d+)\.img_attn_proj\.(.*)$",
     r"double_blocks.\1.img_attn_proj.\2"),
    
    (r"^double_blocks\.(\d+)\.txt_attn_qkv\.(.*)$",
     r"double_blocks.\1.txt_attn_qkv.\2"),
    (r"^double_blocks\.(\d+)\.txt_attn_q_norm\.(.*)$",
     r"double_blocks.\1.txt_attn_q_norm.\2"),
    (r"^double_blocks\.(\d+)\.txt_attn_k_norm\.(.*)$",
     r"double_blocks.\1.txt_attn_k_norm.\2"),
    (r"^double_blocks\.(\d+)\.txt_attn_proj\.(.*)$",
     r"double_blocks.\1.txt_attn_proj.\2"),
    
    # MLP in double blocks: official uses fc1, fc2; FastVideo uses fc_in, fc_out
    (r"^double_blocks\.(\d+)\.img_mlp\.fc1\.(.*)$",
     r"double_blocks.\1.img_mlp.fc_in.\2"),
    (r"^double_blocks\.(\d+)\.img_mlp\.fc2\.(.*)$",
     r"double_blocks.\1.img_mlp.fc_out.\2"),
    (r"^double_blocks\.(\d+)\.txt_mlp\.fc1\.(.*)$",
     r"double_blocks.\1.txt_mlp.fc_in.\2"),
    (r"^double_blocks\.(\d+)\.txt_mlp\.fc2\.(.*)$",
     r"double_blocks.\1.txt_mlp.fc_out.\2"),
    
    # Norms in double blocks (direct mapping)
    (r"^double_blocks\.(\d+)\.img_norm1\.(.*)$",
     r"double_blocks.\1.img_norm1.\2"),
    (r"^double_blocks\.(\d+)\.img_norm2\.(.*)$",
     r"double_blocks.\1.img_norm2.\2"),
    (r"^double_blocks\.(\d+)\.txt_norm1\.(.*)$",
     r"double_blocks.\1.txt_norm1.\2"),
    (r"^double_blocks\.(\d+)\.txt_norm2\.(.*)$",
     r"double_blocks.\1.txt_norm2.\2"),
    
    # ==================== Single stream blocks ====================
    # linear1, linear2 (direct mapping)
    (r"^single_blocks\.(\d+)\.linear1\.(.*)$",
     r"single_blocks.\1.linear1.\2"),
    (r"^single_blocks\.(\d+)\.linear2\.(.*)$",
     r"single_blocks.\1.linear2.\2"),
    
    # QK norms
    (r"^single_blocks\.(\d+)\.q_norm\.(.*)$",
     r"single_blocks.\1.q_norm.\2"),
    (r"^single_blocks\.(\d+)\.k_norm\.(.*)$",
     r"single_blocks.\1.k_norm.\2"),
    
    # pre_norm in single blocks
    (r"^single_blocks\.(\d+)\.pre_norm\.(.*)$",
     r"single_blocks.\1.pre_norm.\2"),
    
    # modulation in single blocks
    (r"^single_blocks\.(\d+)\.modulation\.linear\.(.*)$",
     r"single_blocks.\1.modulation.linear.\2"),
    
    # ==================== Final layer ====================
    # adaLN_modulation: official uses Sequential (0, 1), FastVideo uses direct linear
    (r"^final_layer\.adaLN_modulation\.1\.(.*)$",
     r"final_layer.adaLN_modulation.linear.\1"),
    (r"^final_layer\.linear\.(.*)$",
     r"final_layer.linear.\1"),
    (r"^final_layer\.norm_final\.(.*)$",
     r"final_layer.norm_final.\1"),
    
    # ==================== Image patch embedding ====================
    (r"^img_in\.proj\.(.*)$", r"img_in.proj.\1"),
    
    # ==================== CameraNet ====================
    # Direct mapping for CameraNet components
    (r"^camera_net\.(.*)$", r"camera_net.\1"),
]


def apply_mapping(key: str) -> str:
    """Apply the mapping rules to convert a checkpoint key."""
    for pattern, replacement in PARAM_NAME_MAPPING:
        if re.match(pattern, key):
            return re.sub(pattern, replacement, key)
    # If no mapping matches, return the key unchanged
    return key


def convert_state_dict(state_dict: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """Convert official GameCraft state_dict to FastVideo format."""
    converted = OrderedDict()
    unmapped_keys = []
    
    for key, value in state_dict.items():
        new_key = apply_mapping(key)
        converted[new_key] = value
        
        if new_key == key:
            # Key wasn't mapped - might be an issue
            unmapped_keys.append(key)
    
    if unmapped_keys:
        print(f"\nWarning: {len(unmapped_keys)} keys were not mapped (may be fine if they already match):")
        for k in unmapped_keys[:20]:  # Show first 20
            print(f"  - {k}")
        if len(unmapped_keys) > 20:
            print(f"  ... and {len(unmapped_keys) - 20} more")
    
    return converted


def load_official_checkpoint(path: Path, load_key: str = "module") -> dict[str, torch.Tensor]:
    """Load official GameCraft checkpoint."""
    print(f"Loading checkpoint from {path}")
    state_dict = torch.load(path, map_location="cpu")
    
    # Handle nested state_dict
    if load_key in state_dict:
        state_dict = state_dict[load_key]
        print(f"Extracted '{load_key}' key from checkpoint")
    elif load_key == ".":
        pass  # Use as-is
    else:
        available_keys = list(state_dict.keys())[:10]
        raise KeyError(f"Key '{load_key}' not found. Available keys (first 10): {available_keys}")
    
    print(f"Loaded {len(state_dict)} parameters")
    return state_dict


def create_transformer_config(hidden_size: int = 3072,
                              num_attention_heads: int = 24,
                              num_layers: int = 20,
                              num_single_layers: int = 40) -> dict:
    """Create a config.json for the transformer."""
    return {
        "_class_name": "HunyuanGameCraftTransformer3DModel",
        "_diffusers_version": "0.33.0.dev0",
        "attention_head_dim": hidden_size // num_attention_heads,
        "guidance_embeds": False,  # Official checkpoint doesn't have guidance_in
        "hidden_size": hidden_size,
        "in_channels": 33,  # 16 latent + 16 cond + 1 mask (multitask_mask_training)
        "mlp_ratio": 4.0,
        "num_attention_heads": num_attention_heads,
        "num_layers": num_layers,
        "num_refiner_layers": 2,
        "num_single_layers": num_single_layers,
        "out_channels": 16,
        "patch_size": 2,
        "patch_size_t": 1,
        "pooled_projection_dim": 768,
        "rope_axes_dim": [16, 56, 56],
        "rope_theta": 256,
        "text_embed_dim": 4096,
        "camera_in_channels": 6,
        "camera_downscale_coef": 8,
    }


def create_model_index() -> dict:
    """Create model_index.json for the Diffusers repo layout."""
    return {
        "_class_name": "HunyuanGameCraftPipeline",
        "_diffusers_version": "0.33.0.dev0",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "LlamaModel"],
        "text_encoder_2": ["transformers", "CLIPTextModel"],
        "tokenizer": ["transformers", "LlamaTokenizerFast"],
        "tokenizer_2": ["transformers", "CLIPTokenizer"],
        "transformer": ["diffusers", "HunyuanGameCraftTransformer3DModel"],
        "vae": ["diffusers", "AutoencoderKLHunyuanVideo"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert official Hunyuan-GameCraft weights to FastVideo format"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to official checkpoint file (*.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for converted weights"
    )
    parser.add_argument(
        "--load-key",
        type=str,
        default="module",
        help="Key to extract from checkpoint (module, ema, or . for root)"
    )
    parser.add_argument(
        "--transformer-only",
        action="store_true",
        help="Only save transformer weights (no full repo structure)"
    )
    parser.add_argument(
        "--print-keys",
        action="store_true",
        help="Print original and converted keys for debugging"
    )
    
    args = parser.parse_args()
    
    source_path = Path(args.source)
    output_dir = Path(args.output)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_path}")
    
    # Load and convert
    state_dict = load_official_checkpoint(source_path, args.load_key)
    
    if args.print_keys:
        print("\nOriginal keys (first 50):")
        for i, key in enumerate(list(state_dict.keys())[:50]):
            print(f"  {key}")
        if len(state_dict) > 50:
            print(f"  ... and {len(state_dict) - 50} more")
    
    converted = convert_state_dict(state_dict)
    
    if args.print_keys:
        print("\nConverted keys (first 50):")
        for i, key in enumerate(list(converted.keys())[:50]):
            print(f"  {key}")
    
    # Create output structure
    if args.transformer_only:
        # Save directly to output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_path = output_dir / "diffusion_pytorch_model.safetensors"
        save_file(converted, str(weights_path))
        print(f"\nSaved transformer weights to {weights_path}")
        
        config_path = output_dir / "config.json"
        with config_path.open("w") as f:
            json.dump(create_transformer_config(), f, indent=2)
        print(f"Saved config to {config_path}")
    else:
        # Create full Diffusers repo structure
        transformer_dir = output_dir / "transformer"
        transformer_dir.mkdir(parents=True, exist_ok=True)
        
        weights_path = transformer_dir / "diffusion_pytorch_model.safetensors"
        save_file(converted, str(weights_path))
        print(f"\nSaved transformer weights to {weights_path}")
        
        config_path = transformer_dir / "config.json"
        with config_path.open("w") as f:
            json.dump(create_transformer_config(), f, indent=2)
        print(f"Saved config to {config_path}")
        
        # Save model_index.json
        model_index_path = output_dir / "model_index.json"
        with model_index_path.open("w") as f:
            json.dump(create_model_index(), f, indent=2)
        print(f"Saved model_index.json to {model_index_path}")
    
    print("\nConversion complete!")
    print(f"Total parameters converted: {len(converted)}")


if __name__ == "__main__":
    main()
