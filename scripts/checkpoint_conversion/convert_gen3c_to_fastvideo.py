#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Convert GEN3C checkpoint (nvidia/GEN3C-Cosmos-7B) to FastVideo format.

This script:
1. Downloads the official GEN3C checkpoint from HuggingFace
2. Applies param_names_mapping to convert weights to FastVideo naming conventions
3. Saves the converted weights as safetensors in a diffusers-style layout

Usage:
    # Download and convert from HuggingFace
    python convert_gen3c_to_fastvideo.py --download nvidia/GEN3C-Cosmos-7B --output ./gen3c_fastvideo

    # Convert from local checkpoint
    python convert_gen3c_to_fastvideo.py --source ./model.pt --output ./gen3c_fastvideo

    # Analyze checkpoint structure only
    python convert_gen3c_to_fastvideo.py --source ./model.pt --analyze
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

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


# Parameter name mapping from official GEN3C checkpoint to FastVideo format
# Based on fastvideo/configs/models/dits/gen3c.py
PARAM_NAMES_MAPPING: dict[str, str] = {
    # Patch embedding: net.x_embedder.proj.1.weight -> patch_embed.proj.weight
    r"^net\.x_embedder\.proj\.1\.(.*)$": r"patch_embed.proj.\1",

    # Time embedding
    r"^net\.t_embedder\.0\.(.*)$": r"time_embed.time_proj.\1",
    r"^net\.t_embedder\.1\.linear_1\.(.*)$": r"time_embed.t_embedder.linear_1.\1",
    r"^net\.t_embedder\.1\.linear_2\.(.*)$": r"time_embed.t_embedder.linear_2.\1",
    
    # Augment sigma embedding (GEN3C-specific)
    r"^net\.augment_sigma_embedder\.0\.(.*)$": r"augment_sigma_embed.time_proj.\1",
    r"^net\.augment_sigma_embedder\.1\.linear_1\.(.*)$": r"augment_sigma_embed.t_embedder.linear_1.\1",
    r"^net\.augment_sigma_embedder\.1\.linear_2\.(.*)$": r"augment_sigma_embed.t_embedder.linear_2.\1",

    # Affine embedding norm
    r"^net\.affline_norm\.(.*)$": r"affine_norm.\1",

    # Extra positional embeddings (learnable per-axis)
    r"^net\.extra_pos_embedder\.pos_emb_t$": r"learnable_pos_embed.pos_emb_t",
    r"^net\.extra_pos_embedder\.pos_emb_h$": r"learnable_pos_embed.pos_emb_h",
    r"^net\.extra_pos_embedder\.pos_emb_w$": r"learnable_pos_embed.pos_emb_w",

    # Transformer blocks: net.blocks.blockN -> transformer_blocks.N
    # Self-attention (FA block, index 0)
    r"^net\.blocks\.block(\d+)\.blocks\.0\.block\.self_attn\.q_proj\.(.*)$": r"transformer_blocks.\1.attn1.to_q.\2",
    r"^net\.blocks\.block(\d+)\.blocks\.0\.block\.self_attn\.k_proj\.(.*)$": r"transformer_blocks.\1.attn1.to_k.\2",
    r"^net\.blocks\.block(\d+)\.blocks\.0\.block\.self_attn\.v_proj\.(.*)$": r"transformer_blocks.\1.attn1.to_v.\2",
    r"^net\.blocks\.block(\d+)\.blocks\.0\.block\.self_attn\.output_proj\.(.*)$": r"transformer_blocks.\1.attn1.to_out.\2",
    r"^net\.blocks\.block(\d+)\.blocks\.0\.block\.self_attn\.q_norm\.(.*)$": r"transformer_blocks.\1.attn1.norm_q.\2",
    r"^net\.blocks\.block(\d+)\.blocks\.0\.block\.self_attn\.k_norm\.(.*)$": r"transformer_blocks.\1.attn1.norm_k.\2",
    r"^net\.blocks\.block(\d+)\.blocks\.0\.adaLN_modulation\.(.*)$": r"transformer_blocks.\1.adaln_modulation_self_attn.\2",

    # Cross-attention (CA block, index 1)
    r"^net\.blocks\.block(\d+)\.blocks\.1\.block\.cross_attn\.q_proj\.(.*)$": r"transformer_blocks.\1.attn2.to_q.\2",
    r"^net\.blocks\.block(\d+)\.blocks\.1\.block\.cross_attn\.k_proj\.(.*)$": r"transformer_blocks.\1.attn2.to_k.\2",
    r"^net\.blocks\.block(\d+)\.blocks\.1\.block\.cross_attn\.v_proj\.(.*)$": r"transformer_blocks.\1.attn2.to_v.\2",
    r"^net\.blocks\.block(\d+)\.blocks\.1\.block\.cross_attn\.output_proj\.(.*)$": r"transformer_blocks.\1.attn2.to_out.\2",
    r"^net\.blocks\.block(\d+)\.blocks\.1\.block\.cross_attn\.q_norm\.(.*)$": r"transformer_blocks.\1.attn2.norm_q.\2",
    r"^net\.blocks\.block(\d+)\.blocks\.1\.block\.cross_attn\.k_norm\.(.*)$": r"transformer_blocks.\1.attn2.norm_k.\2",
    r"^net\.blocks\.block(\d+)\.blocks\.1\.adaLN_modulation\.(.*)$": r"transformer_blocks.\1.adaln_modulation_cross_attn.\2",

    # MLP (FF block, index 2)
    r"^net\.blocks\.block(\d+)\.blocks\.2\.block\.mlp\.layer1\.(.*)$": r"transformer_blocks.\1.mlp.fc_in.\2",
    r"^net\.blocks\.block(\d+)\.blocks\.2\.block\.mlp\.layer2\.(.*)$": r"transformer_blocks.\1.mlp.fc_out.\2",
    r"^net\.blocks\.block(\d+)\.blocks\.2\.adaLN_modulation\.(.*)$": r"transformer_blocks.\1.adaln_modulation_mlp.\2",

    # Final layer
    r"^net\.final_layer\.linear\.(.*)$": r"final_layer.proj_out.\1",
    r"^net\.final_layer\.adaLN_modulation\.(.*)$": r"final_layer.adaln_modulation.\1",
}

# Keys to skip (dynamically computed or training metadata)
SKIP_PATTERNS = [
    "net.pos_embedder.",  # RoPE computed dynamically
    "net.accum_",  # Training accumulation metadata
]


def apply_mapping(key: str) -> str | None:
    """Apply parameter name mapping to convert from official to FastVideo format."""
    # Check if key should be skipped
    for pattern in SKIP_PATTERNS:
        if key.startswith(pattern):
            return None
    
    # Apply mapping patterns
    for pattern, replacement in PARAM_NAMES_MAPPING.items():
        if re.match(pattern, key):
            return re.sub(pattern, replacement, key)
    
    # If no mapping found, return original key (will be reported)
    return key


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """Load checkpoint from .pt file."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "ema"):
            if key in checkpoint:
                return checkpoint[key]
        return checkpoint
    
    return checkpoint


def analyze_checkpoint(state_dict: dict[str, torch.Tensor]) -> None:
    """Print checkpoint structure analysis."""
    print("\n" + "=" * 80)
    print("CHECKPOINT ANALYSIS")
    print("=" * 80)
    
    # Count parameters
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"\nTotal parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"Total keys: {len(state_dict)}")
    
    # Analyze key prefixes
    prefixes: dict[str, int] = {}
    for key in state_dict:
        parts = key.split(".")
        prefix = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    
    print("\nKey prefixes:")
    for prefix, count in sorted(prefixes.items()):
        print(f"  {prefix}: {count}")
    
    # Print first 100 keys with shapes
    print("\nFirst 100 keys with shapes:")
    for i, (key, value) in enumerate(state_dict.items()):
        if i >= 100:
            print(f"  ... and {len(state_dict) - 100} more keys")
            break
        print(f"  {key}: {list(value.shape)}")
    
    # Identify GEN3C-specific layers
    print("\nGEN3C-specific layers:")
    gen3c_patterns = [
        "augment_sigma",
        "x_embedder",  # Has more input channels than standard Cosmos
        "extra_pos_embedder",
    ]
    for key in state_dict:
        for pattern in gen3c_patterns:
            if pattern in key:
                print(f"  {key}: {list(state_dict[key].shape)}")
                break
    
    print("=" * 80 + "\n")


def convert_weights(
    state_dict: dict[str, torch.Tensor],
    verbose: bool = False,
) -> tuple[OrderedDict[str, torch.Tensor], list[str], list[str]]:
    """Convert weights from official format to FastVideo format.
    
    Returns:
        converted: Converted state dict
        unmapped: List of keys that weren't mapped (kept as-is)
        skipped: List of keys that were skipped
    """
    converted = OrderedDict()
    unmapped = []
    skipped = []
    
    for key, value in state_dict.items():
        new_key = apply_mapping(key)
        
        if new_key is None:
            skipped.append(key)
            if verbose:
                print(f"  Skipped: {key}")
        elif new_key == key:
            # No mapping found, but not in skip list
            unmapped.append(key)
            converted[key] = value
            if verbose:
                print(f"  Unmapped: {key}")
        else:
            converted[new_key] = value
            if verbose:
                print(f"  {key} -> {new_key}")
    
    return converted, unmapped, skipped


def write_component(
    output_dir: Path,
    name: str,
    weights: OrderedDict[str, torch.Tensor],
    config: dict | None = None,
) -> None:
    """Write component weights and config to output directory."""
    component_dir = output_dir / name
    component_dir.mkdir(parents=True, exist_ok=True)
    
    # Save weights
    output_file = component_dir / "model.safetensors"
    save_file(weights, str(output_file))
    print(f"Saved {name} weights to {output_file}")
    print(f"  {len(weights)} tensors, {sum(t.numel() for t in weights.values()):,} parameters")
    
    # Save config
    if config is not None:
        config_path = component_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        print(f"Saved {name} config to {config_path}")


def build_transformer_config() -> dict:
    """Build transformer config for Gen3CTransformer3DModel."""
    return {
        "_class_name": "Gen3CTransformer3DModel",
        "in_channels": 16,
        "out_channels": 16,
        "num_attention_heads": 16,
        "attention_head_dim": 128,
        "num_layers": 28,
        "mlp_ratio": 4.0,
        "text_embed_dim": 1024,
        "adaln_lora_dim": 256,
        "use_adaln_lora": True,
        "add_augment_sigma_embedding": True,
        "frame_buffer_max": 2,
        "max_size": [128, 240, 240],
        "patch_size": [1, 2, 2],
        "rope_scale": [2.0, 1.0, 1.0],
        "extra_pos_embed_type": "learnable",
        "concat_padding_mask": True,
        "affine_emb_norm": True,
        "qk_norm": "rms_norm",
        "eps": 1e-6,
    }


def build_model_index() -> dict:
    """Build model_index.json for the converted model."""
    return {
        "_class_name": "Gen3CPipeline",
        "_diffusers_version": "0.33.0.dev0",
        "transformer": ["fastvideo", "Gen3CTransformer3DModel"],
        "vae": ["fastvideo", "AutoencoderKLWan"],
        "text_encoder": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "T5Tokenizer"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }


def download_checkpoint(
    repo_id: str,
    filename: str = "model.pt",
    token: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Download checkpoint from HuggingFace Hub."""
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is required for --download. Install with: pip install huggingface_hub")
    
    print(f"Downloading {filename} from {repo_id}...")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    print(f"Downloaded to {path}")
    return Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GEN3C checkpoint to FastVideo format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download and convert from HuggingFace
    python convert_gen3c_to_fastvideo.py --download nvidia/GEN3C-Cosmos-7B --output ./gen3c_fastvideo

    # Convert from local checkpoint
    python convert_gen3c_to_fastvideo.py --source ./model.pt --output ./gen3c_fastvideo

    # Analyze checkpoint structure only
    python convert_gen3c_to_fastvideo.py --source ./model.pt --analyze
        """,
    )
    
    parser.add_argument(
        "--source",
        type=str,
        help="Path to input .pt checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for converted weights",
    )
    parser.add_argument(
        "--download",
        type=str,
        help="HuggingFace repo ID to download checkpoint from (e.g., nvidia/GEN3C-Cosmos-7B)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="model.pt",
        help="Filename to download from HuggingFace (default: model.pt)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Only analyze checkpoint structure, don't convert",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed conversion info",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it exists",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.download and args.source:
        raise ValueError("Use either --download or --source, not both")
    if not args.download and not args.source:
        raise ValueError("Either --download or --source is required")
    if not args.analyze and not args.output:
        raise ValueError("--output is required when not using --analyze")
    
    # Get checkpoint path
    if args.download:
        checkpoint_path = download_checkpoint(
            repo_id=args.download,
            filename=args.filename,
            token=args.token,
        )
    else:
        checkpoint_path = Path(args.source)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = load_checkpoint(checkpoint_path)
    
    # Analyze if requested
    if args.analyze:
        analyze_checkpoint(state_dict)
        return
    
    # Analyze before conversion
    analyze_checkpoint(state_dict)
    
    # Convert weights
    print("\nConverting weights...")
    converted, unmapped, skipped = convert_weights(state_dict, verbose=args.verbose)
    
    print(f"\nConversion summary:")
    print(f"  Converted: {len(converted)} tensors")
    print(f"  Skipped: {len(skipped)} tensors (dynamic/metadata)")
    print(f"  Unmapped: {len(unmapped)} tensors (kept original names)")
    
    if unmapped:
        print("\nWarning: The following keys were not mapped:")
        for key in unmapped[:20]:
            print(f"  {key}")
        if len(unmapped) > 20:
            print(f"  ... and {len(unmapped) - 20} more")
    
    # Write output
    output_dir = Path(args.output)
    if output_dir.exists() and not args.force:
        raise FileExistsError(f"Output directory exists: {output_dir}. Use --force to overwrite.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write transformer weights
    transformer_config = build_transformer_config()
    write_component(output_dir, "transformer", converted, transformer_config)
    
    # Write model_index.json
    model_index = build_model_index()
    model_index_path = output_dir / "model_index.json"
    with model_index_path.open("w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2)
        f.write("\n")
    print(f"\nSaved model_index.json to {model_index_path}")
    
    print(f"\nConversion complete! Output saved to {output_dir}")
    print("\nTo use with FastVideo:")
    print(f"  from fastvideo.models.dits.gen3c import Gen3CTransformer3DModel")
    print(f"  model = Gen3CTransformer3DModel.from_pretrained('{output_dir}/transformer')")


if __name__ == "__main__":
    main()
