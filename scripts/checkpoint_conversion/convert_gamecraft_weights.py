# SPDX-License-Identifier: Apache-2.0
"""
Convert Hunyuan-GameCraft-1.0 weights from DeepSpeed format to FastVideo format.

The official GameCraft checkpoint (mp_rank_00_model_states.pt) uses slightly different
naming conventions than FastVideo. This script converts the weights to be compatible
with FastVideo's HunyuanGameCraft model.

Usage:
    python scripts/checkpoint_conversion/convert_gamecraft_weights.py \
        --input Hunyuan-GameCraft-1.0/weights/gamecraft_models/mp_rank_00_model_states.pt \
        --output official_weights/hunyuan-gamecraft/transformer/diffusion_pytorch_model.safetensors
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import save_file


# Mapping from official GameCraft naming to FastVideo naming
# GameCraft weights are already close to FastVideo format, but need minor adjustments
PARAM_NAME_MAP: dict[str, str] = {
    # MLP naming: fc1 -> fc_in, fc2 -> fc_out
    r"^(.*)\.img_mlp\.fc1\.(.*)$": r"\1.img_mlp.fc_in.\2",
    r"^(.*)\.img_mlp\.fc2\.(.*)$": r"\1.img_mlp.fc_out.\2",
    r"^(.*)\.txt_mlp\.fc1\.(.*)$": r"\1.txt_mlp.fc_in.\2",
    r"^(.*)\.txt_mlp\.fc2\.(.*)$": r"\1.txt_mlp.fc_out.\2",
    
    # Single block MLP naming
    r"^single_blocks\.(\d+)\.mlp\.fc1\.(.*)$": r"single_blocks.\1.mlp.fc_in.\2",
    r"^single_blocks\.(\d+)\.mlp\.fc2\.(.*)$": r"single_blocks.\1.mlp.fc_out.\2",
    
    # Token refiner naming - rename individual_token_refiner.blocks to refiner_blocks
    # This is applied first, then subsequent mappings handle the sub-components
    r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.(.*)$": r"txt_in.refiner_blocks.\1.\2",
    
    # Vector in naming
    r"^vector_in\.in_layer\.(.*)$": r"vector_in.fc_in.\1",
    r"^vector_in\.out_layer\.(.*)$": r"vector_in.fc_out.\1",
    
    # Time embedder naming
    r"^time_in\.mlp\.0\.(.*)$": r"time_in.mlp.fc_in.\1",
    r"^time_in\.mlp\.2\.(.*)$": r"time_in.mlp.fc_out.\1",
    
    # Guidance embedder naming (if present)
    r"^guidance_in\.mlp\.0\.(.*)$": r"guidance_in.mlp.fc_in.\1",
    r"^guidance_in\.mlp\.2\.(.*)$": r"guidance_in.mlp.fc_out.\1",
    
    # Final layer adaLN modulation
    r"^final_layer\.adaLN_modulation\.1\.(.*)$": r"final_layer.adaLN_modulation.linear.\1",
    
    # Text in c_embedder (context embedder) naming
    r"^txt_in\.c_embedder\.linear_1\.(.*)$": r"txt_in.c_embedder.fc_in.\1",
    r"^txt_in\.c_embedder\.linear_2\.(.*)$": r"txt_in.c_embedder.fc_out.\1",
    
    # Text in t_embedder (time embedder) naming
    r"^txt_in\.t_embedder\.mlp\.0\.(.*)$": r"txt_in.t_embedder.mlp.fc_in.\1",
    r"^txt_in\.t_embedder\.mlp\.2\.(.*)$": r"txt_in.t_embedder.mlp.fc_out.\1",
    
    # Refiner block MLP naming - need to match the actual checkpoint names
    r"^txt_in\.refiner_blocks\.(\d+)\.mlp\.fc1\.(.*)$": r"txt_in.refiner_blocks.\1.mlp.fc_in.\2",
    r"^txt_in\.refiner_blocks\.(\d+)\.mlp\.fc2\.(.*)$": r"txt_in.refiner_blocks.\1.mlp.fc_out.\2",
    
    # Refiner block adaLN modulation naming - matches .adaLN_modulation.1.weight/bias
    r"^txt_in\.refiner_blocks\.(\d+)\.adaLN_modulation\.1\.(.*)$": r"txt_in.refiner_blocks.\1.adaLN_modulation.linear.\2",
    
    # Camera net - keep as-is (already correct)
    # r"^camera_net\.(.*)$": r"camera_net.\1",
}


def apply_mapping(key: str) -> str:
    """Apply name mapping to convert official key to FastVideo key.
    
    Mappings are applied repeatedly until no more changes occur.
    This allows for chained transformations.
    """
    prev_key = None
    while prev_key != key:
        prev_key = key
        for pattern, replacement in PARAM_NAME_MAP.items():
            if re.match(pattern, key):
                key = re.sub(pattern, replacement, key)
                break  # Apply only one mapping per iteration
    return key


def load_deepspeed_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """Load weights from a DeepSpeed checkpoint file."""
    checkpoint = torch.load(path, map_location="cpu")
    
    if "module" in checkpoint:
        state_dict = checkpoint["module"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected dict state_dict, got {type(state_dict)}")
    
    return state_dict


def convert_weights(
    input_path: Path,
    output_dir: Path,
    save_config: bool = True,
    verbose: bool = False,
) -> dict[str, int]:
    """
    Convert GameCraft weights to FastVideo format.
    
    Returns:
        dict with conversion statistics
    """
    print(f"Loading checkpoint from {input_path}")
    state_dict = load_deepspeed_checkpoint(input_path)
    
    print(f"Found {len(state_dict)} parameters")
    
    # Convert weights
    converted_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    renamed_count = 0
    unchanged_count = 0
    
    for key, value in state_dict.items():
        new_key = apply_mapping(key)
        if new_key != key:
            if verbose:
                print(f"  {key} -> {new_key}")
            renamed_count += 1
        else:
            unchanged_count += 1
        converted_state_dict[new_key] = value
    
    print(f"Renamed {renamed_count} parameters, {unchanged_count} unchanged")
    
    # Create output directory structure
    transformer_dir = output_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as safetensors
    output_path = transformer_dir / "diffusion_pytorch_model.safetensors"
    print(f"Saving converted weights to {output_path}")
    save_file(converted_state_dict, str(output_path))
    
    # Save config if requested
    if save_config:
        config = {
            "_class_name": "HunyuanGameCraftTransformer3DModel",
            "_fastvideo_version": "0.1.0",
            "in_channels": 33,  # 16 + 16 + 1 for concat mode
            "out_channels": 16,
            "hidden_size": 3072,
            "num_attention_heads": 24,
            "attention_head_dim": 128,
            "num_layers": 20,
            "num_single_layers": 40,
            "num_refiner_layers": 2,
            "patch_size": [1, 2, 2],
            "rope_axes_dim": [16, 56, 56],
            "text_embed_dim": 4096,
            "pooled_projection_dim": 768,
            "mlp_ratio": 4.0,
            "guidance_embeds": False,
            "camera_net": True,
        }
        config_path = transformer_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to {config_path}")
    
    # Verify camera_net weights
    camera_keys = [k for k in converted_state_dict.keys() if "camera_net" in k]
    print(f"\nCamera net weights ({len(camera_keys)}):")
    for k in camera_keys:
        print(f"  {k}: {converted_state_dict[k].shape}")
    
    # Verify img_in weights (should be 33 channels)
    if "img_in.proj.weight" in converted_state_dict:
        img_in_shape = converted_state_dict["img_in.proj.weight"].shape
        print(f"\nimg_in.proj.weight shape: {img_in_shape}")
        expected_in_channels = 33
        actual_in_channels = img_in_shape[1]
        if actual_in_channels != expected_in_channels:
            print(f"  WARNING: Expected {expected_in_channels} input channels, got {actual_in_channels}")
        else:
            print(f"  OK: 33 input channels (16 latent + 16 gt_latent + 1 mask)")
    
    return {
        "total": len(converted_state_dict),
        "renamed": renamed_count,
        "unchanged": unchanged_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert Hunyuan-GameCraft-1.0 weights to FastVideo format."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to official GameCraft checkpoint (mp_rank_00_model_states.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="official_weights/hunyuan-gamecraft",
        help="Output directory for converted weights"
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Don't save config.json"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print all renamed parameters"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    stats = convert_weights(
        input_path=input_path,
        output_dir=output_dir,
        save_config=not args.no_config,
        verbose=args.verbose,
    )
    
    print(f"\nConversion complete!")
    print(f"  Total parameters: {stats['total']}")
    print(f"  Renamed: {stats['renamed']}")
    print(f"  Unchanged: {stats['unchanged']}")


if __name__ == "__main__":
    main()
