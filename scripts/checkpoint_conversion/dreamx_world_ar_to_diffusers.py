#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert DreamX-World-5B autoregressive weights to FastVideo layout.

The HF repository stores one raw official ``model.safetensors`` whose keys match
FastVideo's native ``DreamXWorldARTransformer3DModel``. The converter writes a
Diffusers-like root with ``transformer/config.json`` and reusable Wan2.2
components. Use ``--symlink-transformer`` locally to avoid duplicating the 21GB
AR tensor file.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

TRANSFORMER_CONFIG: dict[str, object] = {
    "_class_name": "DreamXWorldARTransformer3DModel",
    "model_type": "ti2v",
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
    "local_attn_size": 12,
    "sink_size": 3,
    "cross_attn_norm": True,
    "qk_norm": True,
    "eps": 1e-6,
    "add_control_adapter": True,
    "cam_method": "prope",
    "attn_compress": 4,
    "cam_self_attn_layers": list(range(30)),
    "num_frames_per_block": 3,
}

MODEL_INDEX: dict[str, object] = {
    "_class_name": "DreamXWorldARPipeline",
    "_diffusers_version": "0.31.0",
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "AutoTokenizer"],
    "transformer": ["diffusers", "DreamXWorldARTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
}

REUSED_COMPONENTS = ("scheduler", "text_encoder", "tokenizer", "vae")


def _source_safetensors(source: Path) -> Path:
    if source.is_file():
        return source
    path = source / "model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"Missing AR model.safetensors under {source}")
    return path


def convert_transformer(source: Path, output: Path, symlink_transformer: bool) -> None:
    src = _source_safetensors(source)
    transformer_dir = output / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    (transformer_dir / "config.json").write_text(json.dumps(TRANSFORMER_CONFIG, indent=2) + "\n")
    dst = transformer_dir / "model.safetensors"
    if dst.exists() or dst.is_symlink():
        return
    if symlink_transformer:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def _copy_or_link_component(component: str, component_source: Path, output: Path, symlink: bool) -> None:
    src = component_source / component
    dst = output / component
    if not src.exists():
        raise FileNotFoundError(f"Missing reused component source: {src}")
    if dst.exists() or dst.is_symlink():
        return
    if symlink:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    elif src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def write_model_index(output: Path, component_source: Path | None, symlink_components: bool) -> None:
    if component_source is not None:
        for component in REUSED_COMPONENTS:
            _copy_or_link_component(component, component_source, output, symlink_components)
    missing = [component for component in REUSED_COMPONENTS if not (output / component).exists()]
    if missing:
        print("skipping model_index.json; missing reused components: " + ", ".join(missing))
        return
    (output / "model_index.json").write_text(json.dumps(MODEL_INDEX, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--component-source", type=Path)
    parser.add_argument("--symlink-components", action="store_true")
    parser.add_argument("--symlink-transformer", action="store_true")
    args = parser.parse_args()

    convert_transformer(args.source, args.output, args.symlink_transformer)
    write_model_index(args.output, args.component_source, args.symlink_components)


if __name__ == "__main__":
    main()
