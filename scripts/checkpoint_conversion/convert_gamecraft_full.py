# SPDX-License-Identifier: Apache-2.0
"""
Convert the full Hunyuan-GameCraft-1.0 model to FastVideo diffusers format.

This script converts:
1. Transformer (DiT) weights from DeepSpeed format
2. VAE weights
3. Text encoders (LLaVA-LLaMA-3-8B and CLIP)
4. Creates model_index.json and scheduler config

Usage:
    python scripts/checkpoint_conversion/convert_gamecraft_full.py \
        --input-dir Hunyuan-GameCraft-1.0/weights \
        --output-dir official_weights/hunyuan-gamecraft-diffusers

    # Or specify individual paths:
    python scripts/checkpoint_conversion/convert_gamecraft_full.py \
        --transformer Hunyuan-GameCraft-1.0/weights/gamecraft_models/mp_rank_00_model_states.pt \
        --vae Hunyuan-GameCraft-1.0/weights/stdmodels/vae_3d/hyvae/checkpoint-step-270000.ckpt \
        --text-encoder Hunyuan-GameCraft-1.0/weights/stdmodels/llava-llama-3-8b-v1_1-transformers \
        --text-encoder-2 Hunyuan-GameCraft-1.0/weights/stdmodels/openai_clip-vit-large-patch14 \
        --output-dir official_weights/hunyuan-gamecraft-diffusers
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

# Import conversion functions
from convert_gamecraft_weights import convert_weights as convert_transformer
from convert_gamecraft_vae import convert_gamecraft_vae


def create_model_index(output_dir: Path) -> None:
    """Create model_index.json for the pipeline."""
    model_index = {
        "_class_name": "HunyuanGameCraftPipeline",
        "_diffusers_version": "0.30.0",
        "scheduler": [
            "diffusers",
            "FlowMatchEulerDiscreteScheduler"
        ],
        "text_encoder": [
            "transformers",
            "LlamaModel"
        ],
        "text_encoder_2": [
            "transformers",
            "CLIPTextModel"
        ],
        "tokenizer": [
            "transformers",
            "AutoTokenizer"
        ],
        "tokenizer_2": [
            "transformers",
            "CLIPTokenizer"
        ],
        "transformer": [
            "fastvideo",
            "HunyuanGameCraftTransformer3DModel"
        ],
        "vae": [
            "fastvideo",
            "AutoencoderKLCausal3D"
        ]
    }
    
    with open(output_dir / "model_index.json", "w") as f:
        json.dump(model_index, f, indent=2)
    print(f"Created model_index.json")


def create_scheduler_config(output_dir: Path) -> None:
    """Create scheduler config for FlowMatchEulerDiscreteScheduler."""
    scheduler_dir = output_dir / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    
    scheduler_config = {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "_diffusers_version": "0.30.0",
        "base_image_seq_len": 256,
        "base_shift": 0.5,
        "invert_sigmas": False,
        "max_shift": 1.15,
        "num_train_timesteps": 1000,
        "shift": 7.0,
        "use_dynamic_shifting": True
    }
    
    with open(scheduler_dir / "scheduler_config.json", "w") as f:
        json.dump(scheduler_config, f, indent=2)
    print(f"Created scheduler config")


def copy_text_encoder(src_dir: Path, dst_dir: Path, name: str) -> None:
    """Copy text encoder files to output directory."""
    if not src_dir.exists():
        print(f"Warning: {name} not found at {src_dir}")
        return
    
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all files
    for src_file in src_dir.iterdir():
        if src_file.is_file():
            shutil.copy2(src_file, dst_dir / src_file.name)
    
    print(f"Copied {name} to {dst_dir}")


def copy_tokenizer(src_dir: Path, dst_dir: Path, name: str) -> None:
    """Copy tokenizer files to output directory."""
    if not src_dir.exists():
        print(f"Warning: {name} not found at {src_dir}")
        return
    
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy tokenizer-related files
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
    ]
    
    for filename in tokenizer_files:
        src_file = src_dir / filename
        if src_file.exists():
            shutil.copy2(src_file, dst_dir / filename)
    
    print(f"Copied {name} to {dst_dir}")


def convert_full_model(
    transformer_path: Path | None = None,
    vae_path: Path | None = None,
    text_encoder_path: Path | None = None,
    text_encoder_2_path: Path | None = None,
    input_dir: Path | None = None,
    output_dir: Path = Path("official_weights/hunyuan-gamecraft-diffusers"),
    verbose: bool = False,
) -> None:
    """Convert all components of GameCraft to diffusers format."""
    
    # Resolve paths from input_dir if individual paths not provided
    if input_dir is not None:
        if transformer_path is None:
            transformer_path = input_dir / "gamecraft_models" / "mp_rank_00_model_states.pt"
        if vae_path is None:
            vae_path = input_dir / "stdmodels" / "vae_3d" / "hyvae" / "checkpoint-step-270000.ckpt"
            if not vae_path.exists():
                vae_path = input_dir / "stdmodels" / "vae_3d" / "hyvae" / "pytorch_model.pt"
        if text_encoder_path is None:
            text_encoder_path = input_dir / "stdmodels" / "llava-llama-3-8b-v1_1-transformers"
        if text_encoder_2_path is None:
            text_encoder_2_path = input_dir / "stdmodels" / "openai_clip-vit-large-patch14"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting GameCraft to diffusers format at {output_dir}")
    print("=" * 60)
    
    # 1. Convert transformer
    if transformer_path and transformer_path.exists():
        print(f"\n[1/5] Converting transformer from {transformer_path}")
        convert_transformer(
            input_path=transformer_path,
            output_dir=output_dir,
            save_config=True,
            verbose=verbose,
        )
    else:
        print(f"\n[1/5] Skipping transformer (not found: {transformer_path})")
    
    # 2. Convert VAE
    if vae_path and vae_path.exists():
        print(f"\n[2/5] Converting VAE from {vae_path}")
        convert_gamecraft_vae(
            input_path=vae_path,
            output_dir=output_dir / "vae",
            copy_config=True,
        )
    else:
        print(f"\n[2/5] Skipping VAE (not found: {vae_path})")
    
    # 3. Copy text encoder (LLaMA)
    if text_encoder_path and text_encoder_path.exists():
        print(f"\n[3/5] Copying text encoder from {text_encoder_path}")
        copy_text_encoder(text_encoder_path, output_dir / "text_encoder", "text_encoder")
        copy_tokenizer(text_encoder_path, output_dir / "tokenizer", "tokenizer")
    else:
        print(f"\n[3/5] Skipping text_encoder (not found: {text_encoder_path})")
    
    # 4. Copy text encoder 2 (CLIP)
    if text_encoder_2_path and text_encoder_2_path.exists():
        print(f"\n[4/5] Copying text encoder 2 from {text_encoder_2_path}")
        copy_text_encoder(text_encoder_2_path, output_dir / "text_encoder_2", "text_encoder_2")
        copy_tokenizer(text_encoder_2_path, output_dir / "tokenizer_2", "tokenizer_2")
    else:
        print(f"\n[4/5] Skipping text_encoder_2 (not found: {text_encoder_2_path})")
    
    # 5. Create model_index.json and scheduler
    print(f"\n[5/5] Creating model_index.json and scheduler config")
    create_model_index(output_dir)
    create_scheduler_config(output_dir)
    
    print("\n" + "=" * 60)
    print(f"Conversion complete! Output: {output_dir}")
    print("\nDirectory structure:")
    for item in sorted(output_dir.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(output_dir)
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {rel_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert full Hunyuan-GameCraft-1.0 model to FastVideo diffusers format."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Path to Hunyuan-GameCraft-1.0/weights directory"
    )
    parser.add_argument(
        "--transformer",
        type=str,
        default=None,
        help="Path to transformer checkpoint (mp_rank_00_model_states.pt)"
    )
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="Path to VAE checkpoint"
    )
    parser.add_argument(
        "--text-encoder",
        type=str,
        default=None,
        help="Path to text encoder (LLaVA-LLaMA) directory"
    )
    parser.add_argument(
        "--text-encoder-2",
        type=str,
        default=None,
        help="Path to text encoder 2 (CLIP) directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="official_weights/hunyuan-gamecraft-diffusers",
        help="Output directory for converted model"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed conversion info"
    )
    
    args = parser.parse_args()
    
    # Require either input-dir or individual paths
    if args.input_dir is None and args.transformer is None:
        parser.error("Either --input-dir or individual component paths must be provided")
    
    convert_full_model(
        transformer_path=Path(args.transformer) if args.transformer else None,
        vae_path=Path(args.vae) if args.vae else None,
        text_encoder_path=Path(args.text_encoder) if args.text_encoder else None,
        text_encoder_2_path=Path(args.text_encoder_2) if args.text_encoder_2 else None,
        input_dir=Path(args.input_dir) if args.input_dir else None,
        output_dir=Path(args.output_dir),
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
