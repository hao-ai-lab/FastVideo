#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert DreamX-World-5B-Cam raw transformer weights to FastVideo-loadable format.

The GD-ML/DreamX-World-5B-Cam repository stores the transformer as raw
DreamX/Wan official shards. FastVideo's TransformerLoader expects a Diffusers-like
transformer folder with a config.json and safetensors whose keys can be mapped by
WanVideoConfig.param_names_mapping. This script performs the raw official ->
Diffusers-like key rename and writes the DreamX 5B-Cam transformer config.

Example:
    python scripts/checkpoint_conversion/dreamx_world_to_diffusers.py \
        --source official_weights/dreamx_world \
        --output converted_weights/dreamx_world
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import OrderedDict
from pathlib import Path

import torch
from huggingface_hub import save_torch_state_dict
from safetensors import safe_open
from safetensors.torch import load_file


OFFICIAL_TO_DIFFUSERS_MAPPING: dict[str, str] = {
    r"^text_embedding\.0\.(.*)$": r"condition_embedder.text_embedder.linear_1.\1",
    r"^text_embedding\.2\.(.*)$": r"condition_embedder.text_embedder.linear_2.\1",
    r"^time_embedding\.0\.(.*)$": r"condition_embedder.time_embedder.linear_1.\1",
    r"^time_embedding\.2\.(.*)$": r"condition_embedder.time_embedder.linear_2.\1",
    r"^time_projection\.1\.(.*)$": r"condition_embedder.time_proj.\1",
    r"^img_emb\.proj\.0\.(.*)$": r"condition_embedder.image_embedder.norm1.\1",
    r"^img_emb\.proj\.1\.(.*)$": r"condition_embedder.image_embedder.ff.net.0.proj.\1",
    r"^img_emb\.proj\.3\.(.*)$": r"condition_embedder.image_embedder.ff.net.2.\1",
    r"^img_emb\.proj\.4\.(.*)$": r"condition_embedder.image_embedder.norm2.\1",
    r"^head\.modulation": r"scale_shift_table",
    r"^head\.head\.(.*)$": r"proj_out.\1",
    r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.attn1.to_q.\2",
    r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.attn1.to_k.\2",
    r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.attn1.to_v.\2",
    r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.attn1.to_out.0.\2",
    r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$": r"blocks.\1.attn1.norm_q.\2",
    r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$": r"blocks.\1.attn1.norm_k.\2",
    r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
    r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
    r"^blocks\.(\d+)\.cross_attn\.k_img\.(.*)$": r"blocks.\1.attn2.add_k_proj.\2",
    r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
    r"^blocks\.(\d+)\.cross_attn\.v_img\.(.*)$": r"blocks.\1.attn2.add_v_proj.\2",
    r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": r"blocks.\1.attn2.to_out.0.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$": r"blocks.\1.attn2.norm_q.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$": r"blocks.\1.attn2.norm_k.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_k_img\.(.*)$": r"blocks.\1.attn2.norm_added_k.\2",
    r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.net.0.proj.\2",
    r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.net.2.\2",
    r"^blocks\.(\d+)\.modulation": r"blocks.\1.scale_shift_table",
    r"^blocks\.(\d+)\.norm3\.(.*)$": r"blocks.\1.norm2.\2",
}

TRANSFORMER_CONFIG: dict[str, object] = {
    "_class_name": "DreamXWorldTransformer3DModel",
    "patch_size": [1, 2, 2],
    "text_len": 512,
    "num_attention_heads": 24,
    "attention_head_dim": 128,
    "in_channels": 48,
    "out_channels": 48,
    "text_dim": 4096,
    "freq_dim": 256,
    "ffn_dim": 14336,
    "num_layers": 30,
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "eps": 1e-6,
    "image_dim": None,
    "added_kv_proj_dim": None,
    "rope_max_seq_len": 1024,
    "add_control_adapter": True,
    "cam_method": "prope",
    "attn_compress": 1,
    "cam_self_attn_layers": None,
}

MODEL_INDEX: dict[str, object] = {
    "_class_name": "DreamXWorldPipeline",
    "_diffusers_version": "0.31.0",
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "AutoTokenizer"],
    "transformer": ["diffusers", "DreamXWorldTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
}

REUSED_COMPONENTS = ("scheduler", "text_encoder", "tokenizer", "vae")


def map_transformer_key(key: str) -> str:
    for pattern, replacement in OFFICIAL_TO_DIFFUSERS_MAPPING.items():
        if re.match(pattern, key):
            return re.sub(pattern, replacement, key)
    return key


def _safetensor_files(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix != ".safetensors":
            raise ValueError(f"Only .safetensors files are supported, got {source}")
        return [source]

    index_path = source / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        return sorted({source / shard for shard in index["weight_map"].values()})

    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files found under {source}")
    return files


def convert_transformer(source: Path, output: Path, max_shard_size: str) -> None:
    transformer_dir = output / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    (transformer_dir / "config.json").write_text(json.dumps(TRANSFORMER_CONFIG, indent=2) + "\n")

    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    for shard in _safetensor_files(source):
        print(f"loading {shard}")
        for key, tensor in load_file(shard, device="cpu").items():
            new_key = map_transformer_key(key)
            if new_key in converted:
                raise ValueError(f"Duplicate converted key: {new_key}")
            converted[new_key] = tensor

    print(f"saving {len(converted)} tensors to {transformer_dir}")
    save_torch_state_dict(converted, transformer_dir, max_shard_size=max_shard_size)


def _copy_or_link_component(component: str, component_source: Path, output: Path, symlink: bool) -> None:
    src = component_source / component
    dst = output / component
    if not src.exists():
        raise FileNotFoundError(f"Missing reused component source: {src}")
    if dst.exists() or dst.is_symlink():
        print(f"keeping existing {dst}")
        return
    if symlink:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
        print(f"linked {dst} -> {src}")
    elif src.is_dir():
        shutil.copytree(src, dst)
        print(f"copied {src} -> {dst}")
    else:
        shutil.copy2(src, dst)
        print(f"copied {src} -> {dst}")


def write_model_index(output: Path, component_source: Path | None, symlink_components: bool) -> None:
    if component_source is not None:
        for component in REUSED_COMPONENTS:
            _copy_or_link_component(component, component_source, output, symlink_components)
    missing = [component for component in REUSED_COMPONENTS if not (output / component).exists()]
    if missing:
        print("skipping model_index.json; missing reused components: " + ", ".join(missing))
        print("pass --component-source <Wan2.2 Diffusers root> to copy or link reused components")
        return
    (output / "model_index.json").write_text(json.dumps(MODEL_INDEX, indent=2) + "\n")
    print(f"wrote {output / 'model_index.json'}")


def analyze(source: Path) -> None:
    total = 0
    unchanged = 0
    examples: list[tuple[str, str]] = []
    for shard in _safetensor_files(source):
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            for key in tensors:
                total += 1
                new_key = map_transformer_key(key)
                unchanged += int(new_key == key)
                if len(examples) < 20 and new_key != key:
                    examples.append((key, new_key))
    print(f"total_keys={total} unchanged_keys={unchanged}")
    for old, new in examples:
        print(f"{old} -> {new}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="DreamX raw transformer directory or safetensors file")
    parser.add_argument("--output", type=Path, required=True, help="Output model root; transformer/ is created inside it")
    parser.add_argument("--max-shard-size", default="10GB")
    parser.add_argument(
        "--component-source",
        type=Path,
        help="Optional Wan2.2 Diffusers root whose scheduler/text_encoder/tokenizer/vae components are reused.",
    )
    parser.add_argument(
        "--symlink-components",
        action="store_true",
        help="Symlink reused components from --component-source instead of copying them.",
    )
    parser.add_argument("--analyze", action="store_true", help="Only print key mapping summary")
    args = parser.parse_args()

    if args.analyze:
        analyze(args.source)
    else:
        convert_transformer(args.source, args.output, args.max_shard_size)
        write_model_index(args.output, args.component_source, args.symlink_components)


if __name__ == "__main__":
    main()
