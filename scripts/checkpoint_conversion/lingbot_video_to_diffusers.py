#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert official LingBot-Video components into a native FastVideo layout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import fields
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from fastvideo.configs.models.dits.lingbot_video import LingBotVideoConfig
from fastvideo.configs.models.encoders.lingbot_video import LingBotVideoQwen3VLTextConfig


LANGUAGE_PREFIX = "model.language_model."
PASSTHROUGH_COMPONENTS = ("transformer", "vae", "scheduler")
DENSE_PIPELINE_CLASS = "LingBotVideoDensePipeline"
MOE_PIPELINE_CLASS = "LingBotVideoMoePipeline"


def _copy_tree_with_weight_links(source: Path, destination: Path) -> None:
    """Copy metadata and hard-link immutable weight files on the same filesystem."""
    destination.mkdir(parents=True, exist_ok=False)
    for source_path in sorted(source.rglob("*")):
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
        elif source_path.suffix == ".safetensors":
            os.link(source_path, destination_path)
        else:
            shutil.copy2(source_path, destination_path)


def _fuse_language_shard(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Filter Qwen3-VL language tensors and fuse native QKV and gate/up weights."""
    converted = {
        name[len(LANGUAGE_PREFIX) :]: tensor for name, tensor in state.items() if name.startswith(LANGUAGE_PREFIX)
    }
    layer_ids = {int(name.split(".")[1]) for name in converted if name.startswith("layers.")}
    for layer_id in sorted(layer_ids):
        attention = f"layers.{layer_id}.self_attn"
        qkv_names = [f"{attention}.{kind}_proj.weight" for kind in ("q", "k", "v")]
        present_qkv = [name in converted for name in qkv_names]
        if any(present_qkv):
            if not all(present_qkv):
                raise KeyError(f"QKV tensors cross shard boundary for layer {layer_id}")
            converted[f"{attention}.qkv_proj.weight"] = torch.cat([converted.pop(name) for name in qkv_names], dim=0)
        mlp = f"layers.{layer_id}.mlp"
        gate_up_names = [f"{mlp}.{kind}_proj.weight" for kind in ("gate", "up")]
        present_gate_up = [name in converted for name in gate_up_names]
        if any(present_gate_up):
            if not all(present_gate_up):
                raise KeyError(f"gate/up tensors cross shard boundary for layer {layer_id}")
            converted[f"{mlp}.gate_up_proj.weight"] = torch.cat([converted.pop(name) for name in gate_up_names], dim=0)
    return converted


def _convert_text_encoder(source: Path, destination: Path) -> dict[str, str]:
    """Convert each source shard independently and emit a valid safetensors index."""
    destination.mkdir(parents=True, exist_ok=False)
    source_index = json.loads((source / "model.safetensors.index.json").read_text())
    source_shards = sorted(set(source_index["weight_map"].values()))
    weight_map: dict[str, str] = {}
    total_size = 0
    for shard_name in source_shards:
        source_state = load_file(source / shard_name, device="cpu")
        converted = _fuse_language_shard(source_state)
        if not converted:
            continue
        output_name = shard_name
        save_file(converted, destination / output_name, metadata={"format": "pt"})
        for name, tensor in converted.items():
            if name in weight_map:
                raise ValueError(f"duplicate converted text-encoder tensor: {name}")
            weight_map[name] = output_name
            total_size += tensor.numel() * tensor.element_size()
    index = {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))}
    (destination / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return weight_map


def _write_text_encoder_config(source: Path, destination: Path) -> dict:
    """Flatten the compound Qwen3-VL config into FastVideo's text-model bucket."""
    official = json.loads((source / "config.json").read_text())
    text = dict(official["text_config"])
    valid = {item.name for item in fields(LingBotVideoQwen3VLTextConfig().arch_config)}
    config = {key: value for key, value in text.items() if key in valid}
    config.update(
        {
            "_class_name": "LingBotVideoQwen3VLTextModel",
            "architectures": ["LingBotVideoQwen3VLTextModel"],
            "pad_token_id": 151643,
            "text_len": 37698,
            "output_hidden_states": True,
            "require_processor": True,
            "rope_scaling": None,
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
        }
    )
    LingBotVideoQwen3VLTextConfig().update_model_arch(config)
    (destination / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    for filename in ("generation_config.json",):
        candidate = source / filename
        if candidate.is_file():
            shutil.copy2(candidate, destination / filename)
    return config


def _validate_transformer_config(path: Path) -> None:
    """Exercise the same architecture update used by the production transformer loader."""
    config = json.loads(path.read_text())
    if config.get("_class_name") != "LingBotVideoTransformer3DModel":
        raise ValueError("converted transformer config has the wrong _class_name")
    LingBotVideoConfig().update_model_arch(config)


def _write_model_index(destination: Path, *, has_refiner: bool) -> None:
    """Declare the native T2V components, including an optional refiner DiT."""
    model_index = {
        "_class_name": MOE_PIPELINE_CLASS if has_refiner else DENSE_PIPELINE_CLASS,
        "_diffusers_version": "0.39.0",
        "workload_type": "video-generation",
        "transformer": ["diffusers", "LingBotVideoTransformer3DModel"],
        "vae": ["diffusers", "AutoencoderKLWan"],
        "text_encoder": ["transformers", "LingBotVideoQwen3VLTextModel"],
        "tokenizer": ["transformers", "Qwen3VLProcessor"],
        "scheduler": ["diffusers", "FlowUniPCMultistepScheduler"],
    }
    if has_refiner:
        model_index["transformer_2"] = ["diffusers", "LingBotVideoTransformer3DModel"]
    (destination / "model_index.json").write_text(json.dumps(model_index, indent=2, sort_keys=True) + "\n")


def convert(source: Path, destination: Path) -> None:
    """Create a Dense or MoE/refiner T2V layout without altering source assets."""
    required = (*PASSTHROUGH_COMPONENTS, "text_encoder", "processor")
    missing = [name for name in required if not (source / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing official component directories: {missing}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty destination: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for component in PASSTHROUGH_COMPONENTS:
        _copy_tree_with_weight_links(source / component, destination / component)
    _validate_transformer_config(destination / "transformer" / "config.json")
    refiner_source = source / "refiner"
    has_refiner = refiner_source.is_dir()
    if has_refiner:
        _copy_tree_with_weight_links(refiner_source, destination / "transformer_2")
        _validate_transformer_config(destination / "transformer_2" / "config.json")
    text_destination = destination / "text_encoder"
    weight_map = _convert_text_encoder(source / "text_encoder", text_destination)
    text_config = _write_text_encoder_config(source / "text_encoder", text_destination)
    _copy_tree_with_weight_links(source / "processor", destination / "tokenizer")
    _write_model_index(destination, has_refiner=has_refiner)
    print(f"converted text tensors: {len(weight_map)}")
    print(f"text architecture: {text_config['architectures'][0]}")
    print(f"output: {destination}")


def parse_args() -> argparse.Namespace:
    """Parse explicit local source and destination paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the deterministic local conversion."""
    args = parse_args()
    convert(args.src.resolve(), args.dst.resolve())


if __name__ == "__main__":
    main()
