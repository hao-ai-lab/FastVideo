# SPDX-License-Identifier: Apache-2.0
"""
Convert GameCraft VAE weights to FastVideo safetensors format.

The official GameCraft VAE (checkpoint-step-270000.ckpt or pytorch_model.pt) uses
a "vae." prefix. This script extracts and converts for FastVideo GameCraftVAE.

Usage:
    python scripts/checkpoint_conversion/convert_gamecraft_vae.py \
        --input Hunyuan-GameCraft-1.0/weights/stdmodels/vae_3d/hyvae/checkpoint-step-270000.ckpt \
        --output official_weights/hunyuan-gamecraft/vae
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert_gamecraft_vae(
    input_path: Path,
    output_dir: Path,
    copy_config: bool = True,
) -> dict:
    """Convert official GameCraft VAE checkpoint to FastVideo format."""
    ckpt = torch.load(input_path, map_location="cpu", weights_only=True)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Extract VAE weights and strip "vae." prefix
    vae_sd = {k.replace("vae.", ""): v for k, v in state_dict.items() if k.startswith("vae.")}
    print(f"Extracted {len(vae_sd)} VAE parameters")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(vae_sd, output_dir / "diffusion_pytorch_model.safetensors")
    print(f"Saved to {output_dir / 'diffusion_pytorch_model.safetensors'}")

    if copy_config:
        config_src = input_path.parent / "config.json"
        if config_src.exists():
            config_dst = output_dir / "config.json"
            config = json.loads(config_src.read_text())
            config["_class_name"] = "AutoencoderKLCausal3D"
            with open(config_dst, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Saved config to {config_dst}")
        else:
            print(f"Warning: config.json not found at {config_src}")

    return {"total": len(vae_sd)}


def main():
    parser = argparse.ArgumentParser(
        description="Convert GameCraft VAE weights to FastVideo safetensors format."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to official VAE checkpoint (.ckpt or .pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="official_weights/hunyuan-gamecraft/vae",
        help="Output directory for converted weights",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Don't copy config.json",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    stats = convert_gamecraft_vae(
        input_path=input_path,
        output_dir=Path(args.output),
        copy_config=not args.no_config,
    )
    print(f"\nConversion complete! Total parameters: {stats['total']}")


if __name__ == "__main__":
    main()
