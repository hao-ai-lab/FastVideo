# SPDX-License-Identifier: Apache-2.0
"""
Extract LlamaModel text encoder from LLaVA weights for GameCraft/HunyuanVideo.

LLaVA stores the text encoder as `language_model.model.XXX`, but FastVideo/HunyuanVideo
expects just `XXX` (e.g., `layers.0.self_attn.q_proj.weight` instead of 
`language_model.model.layers.0.self_attn.q_proj.weight`).

This script extracts just the LlamaModel weights and renames them.

Usage:
    python scripts/checkpoint_conversion/extract_llava_text_encoder.py \
        --input Hunyuan-GameCraft-1.0/weights/stdmodels/llava-llama-3-8b-v1_1-transformers \
        --output official_weights/hunyuan-gamecraft-diffusers/text_encoder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import OrderedDict

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def extract_text_encoder(
    input_dir: Path,
    output_dir: Path,
) -> dict:
    """Extract LlamaModel from LLaVA weights."""
    
    # Find all safetensor shards
    shard_files = sorted(input_dir.glob("model-*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensor files found in {input_dir}")
    
    print(f"Found {len(shard_files)} weight shards")
    
    # Collect all weights, filtering for language_model.model prefix
    extracted_weights: OrderedDict[str, torch.Tensor] = OrderedDict()
    
    for shard_file in shard_files:
        print(f"Processing {shard_file.name}...")
        with safe_open(str(shard_file), framework="pt") as f:
            for key in f.keys():
                # Extract only language_model.model weights
                if key.startswith("language_model.model."):
                    # Strip the prefix
                    new_key = key.replace("language_model.model.", "")
                    extracted_weights[new_key] = f.get_tensor(key)
    
    print(f"Extracted {len(extracted_weights)} weights")
    
    # Save to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as single file (it should be <10GB which is fine for safetensors)
    output_file = output_dir / "model.safetensors"
    print(f"Saving to {output_file}...")
    save_file(extracted_weights, str(output_file))
    
    # Create config.json for LlamaModel
    config = {
        "_name_or_path": "llava-llama-3-8b-v1_1-text_encoder",
        "architectures": ["LlamaModel"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "max_position_embeddings": 8192,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "rope_theta": 500000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.48.0",
        "use_cache": True,
        "vocab_size": 128320
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config.json")
    
    # Copy tokenizer files if they exist
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json", 
        "special_tokens_map.json",
    ]
    for fname in tokenizer_files:
        src = input_dir / fname
        if src.exists():
            import shutil
            shutil.copy2(src, output_dir / fname)
            print(f"Copied {fname}")
    
    return {"total": len(extracted_weights)}


def extract_clip_text_encoder(
    input_dir: Path,
    output_dir: Path,
) -> dict:
    """Extract CLIPTextModel from full CLIP model weights."""
    
    # Find weight file
    weight_files = list(input_dir.glob("model.safetensors")) + \
                   list(input_dir.glob("pytorch_model.bin"))
    
    if not weight_files:
        raise FileNotFoundError(f"No weight files found in {input_dir}")
    
    weight_file = weight_files[0]
    print(f"Loading {weight_file}...")
    
    if weight_file.suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(str(weight_file))
    else:
        state_dict = torch.load(weight_file, map_location="cpu")
    
    # Extract only text_model weights
    extracted_weights: OrderedDict[str, torch.Tensor] = OrderedDict()
    
    for key, value in state_dict.items():
        # CLIP full model has text_model.XXX, we want just XXX
        if key.startswith("text_model."):
            new_key = key.replace("text_model.", "")
            extracted_weights[new_key] = value
    
    if not extracted_weights:
        # Maybe it's already just the text model
        extracted_weights = OrderedDict(state_dict)
    
    print(f"Extracted {len(extracted_weights)} weights")
    
    # Save to output directory  
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "model.safetensors"
    print(f"Saving to {output_file}...")
    save_file(extracted_weights, str(output_file))
    
    # Create config.json for CLIPTextModel
    config = {
        "_name_or_path": "openai/clip-vit-large-patch14",
        "architectures": ["CLIPTextModel"],
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "dropout": 0.0,
        "eos_token_id": 2,
        "hidden_act": "quick_gelu",
        "hidden_size": 768,
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 77,
        "model_type": "clip_text_model",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 1,
        "projection_dim": 768,
        "torch_dtype": "float16",
        "transformers_version": "4.48.0",
        "vocab_size": 49408
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config.json")
    
    # Copy tokenizer files if they exist
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    for fname in tokenizer_files:
        src = input_dir / fname
        if src.exists():
            import shutil
            shutil.copy2(src, output_dir / fname)
            print(f"Copied {fname}")
    
    return {"total": len(extracted_weights)}


def main():
    parser = argparse.ArgumentParser(
        description="Extract text encoder from LLaVA or CLIP model weights."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing model weights"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for extracted weights"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["llama", "clip"],
        default="llama",
        help="Type of text encoder to extract"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if args.type == "llama":
        stats = extract_text_encoder(input_dir, output_dir)
    else:
        stats = extract_clip_text_encoder(input_dir, output_dir)
    
    print(f"\nExtraction complete! Total weights: {stats['total']}")


if __name__ == "__main__":
    main()
