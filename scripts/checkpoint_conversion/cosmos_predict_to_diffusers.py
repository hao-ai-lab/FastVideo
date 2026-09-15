# SPDX-License-Identifier: Apache-2.0
"""
Convert official Cosmos Predict (Cosmos 2.5) checkpoints to FastVideo / Diffusers format.

This script processes:
1. Transformer (DiT) weights and configuration
2. VAE (AutoencoderKLCosmos) weights and configuration
3. Text Encoder (Qwen2.5-VL) & Tokenizer assets
4. Scheduler config (EDM / FlowMatch)
5. Root model_index.json

Usage:
    python scripts/checkpoint_conversion/cosmos_predict_to_diffusers.py \
        --src nvidia/Cosmos-1.0-Prompt2World-7B-Video \
        --dst converted_weights/cosmos_predict
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import torch
from safetensors.torch import save_file, load_file
from huggingface_hub import snapshot_download


def create_model_index(output_dir: Path, pipeline_name: str = "CosmosPredictPipeline") -> None:
    """Create root model_index.json for the pipeline."""
    model_index = {
        "_class_name": pipeline_name,
        "_diffusers_version": "0.32.0",
        "scheduler": ["diffusers", "EDMEulerScheduler"],
        "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
        "tokenizer": ["transformers", "AutoTokenizer"],
        "transformer": ["diffusers", "CosmosTransformer3DModel"],
        "vae": ["diffusers", "AutoencoderKLCosmos"]
    }

    with open(output_dir / "model_index.json", "w") as f:
        json.dump(model_index, f, indent=2)
    print("Created model_index.json")


def create_scheduler_config(output_dir: Path) -> None:
    """Create scheduler_config.json for EDMEulerScheduler."""
    scheduler_dir = output_dir / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)

    scheduler_config = {
        "_class_name": "EDMEulerScheduler",
        "_diffusers_version": "0.32.0",
        "num_train_timesteps": 1000,
        "sigma_data": 0.5,
        "sigma_max": 80.0,
        "sigma_min": 0.002,
        "sigma_schedule": "exponential",
        "prediction_type": "v_prediction"
    }

    with open(scheduler_dir / "scheduler_config.json", "w") as f:
        json.dump(scheduler_config, f, indent=2)
    print("Created scheduler/scheduler_config.json")


def convert_transformer_weights(
    state_dict: Dict[str, torch.Tensor],
    use_condition_mask: bool = False,
) -> Dict[str, torch.Tensor]:
    """Convert official transformer state dict keys to FastVideo / Diffusers format."""
    new_state_dict: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        new_key = key
        # Prefix cleanups if extracted from a compound checkpoint
        if new_key.startswith("model.diffusion_model."):
            new_key = new_key[len("model.diffusion_model."):]
        elif new_key.startswith("net."):
            new_key = new_key[len("net."):]

        new_state_dict[new_key] = tensor

    return new_state_dict


def convert_cosmos_predict_checkpoint(
    src_path: str,
    dst_path: str,
    model_family: str = "cosmos_predict",
    hf_token: str | None = None,
) -> None:
    """Main entrypoint to convert Cosmos Predict checkpoint."""
    output_dir = Path(dst_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    src_is_local = os.path.exists(src_path)
    if not src_is_local:
        print(f"Downloading checkpoint from Hugging Face: {src_path}")
        src_dir = Path(snapshot_download(repo_id=src_path, token=hf_token))
    else:
        src_dir = Path(src_path)

    print(f"Converting from source: {src_dir} to {output_dir}")

    # 1. Model Index & Scheduler
    create_model_index(output_dir)
    create_scheduler_config(output_dir)

    # 2. Transformer
    src_transformer_dir = src_dir / "transformer"
    dst_transformer_dir = output_dir / "transformer"
    dst_transformer_dir.mkdir(parents=True, exist_ok=True)

    if (src_transformer_dir / "config.json").exists():
        shutil.copy2(src_transformer_dir / "config.json", dst_transformer_dir / "config.json")
    
    # Process transformer weights
    transformer_weights: Dict[str, torch.Tensor] = {}
    if src_transformer_dir.exists():
        for file in src_transformer_dir.glob("*.safetensors"):
            transformer_weights.update(load_file(str(file)))
        if not transformer_weights:
            for file in src_transformer_dir.glob("*.bin"):
                transformer_weights.update(torch.load(str(file), map_location="cpu"))
    
    if transformer_weights:
        converted_transformer = convert_transformer_weights(transformer_weights)
        save_file(converted_transformer, str(dst_transformer_dir / "diffusion_pytorch_model.safetensors"))
        print(f"Saved {len(converted_transformer)} transformer weights")

    # 3. VAE
    src_vae_dir = src_dir / "vae"
    dst_vae_dir = output_dir / "vae"
    dst_vae_dir.mkdir(parents=True, exist_ok=True)

    if (src_vae_dir / "config.json").exists():
        shutil.copy2(src_vae_dir / "config.json", dst_vae_dir / "config.json")

    vae_weights: Dict[str, torch.Tensor] = {}
    if src_vae_dir.exists():
        for file in src_vae_dir.glob("*.safetensors"):
            vae_weights.update(load_file(str(file)))
        if not vae_weights:
            for file in src_vae_dir.glob("*.bin"):
                vae_weights.update(torch.load(str(file), map_location="cpu"))

    if vae_weights:
        save_file(vae_weights, str(dst_vae_dir / "diffusion_pytorch_model.safetensors"))
        print(f"Saved {len(vae_weights)} VAE weights")

    # 4. Text Encoder & Tokenizer
    for sub in ["text_encoder", "tokenizer"]:
        src_sub = src_dir / sub
        dst_sub = output_dir / sub
        if src_sub.exists():
            dst_sub.mkdir(parents=True, exist_ok=True)
            for f in src_sub.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst_sub / f.name)
            print(f"Copied {sub} assets to {dst_sub}")

    print(f"Conversion complete! Converted model layout ready at {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Cosmos Predict checkpoints to FastVideo format.")
    parser.add_argument("--src", type=str, required=True, help="HuggingFace repo ID or local checkpoint path.")
    parser.add_argument("--dst", type=str, default="converted_weights/cosmos_predict", help="Output directory.")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace authentication token if needed.")
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    convert_cosmos_predict_checkpoint(
        src_path=args.src,
        dst_path=args.dst,
        hf_token=token,
    )


if __name__ == "__main__":
    main()
