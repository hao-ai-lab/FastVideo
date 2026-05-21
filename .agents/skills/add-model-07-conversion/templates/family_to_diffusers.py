#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert <model_family> official weights to a FastVideo Diffusers-style tree.

This template supports both separate component sources and a monolithic pipeline
checkpoint that must be split by component prefix. Replace every TODO before
using it for a real port.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover - optional local conversion dependency
    snapshot_download = None


# TODO: fill with authoritative component prefixes for monolithic checkpoints.
# Example: {"model.model.": "transformer", "pretransform.model.": "vae"}
COMPONENT_PREFIXES: dict[str, str] = {}

# TODO: fill with component-specific source paths for separate-component repos.
# Example: {"transformer": "transformer/model.safetensors", "vae": "vae/"}
SEPARATE_COMPONENT_PATHS: dict[str, str] = {}

# TODO: copy passthrough dirs that are already loadable by FastVideo/Diffusers.
PASSTHROUGH_SUBFOLDERS: tuple[str, ...] = ("tokenizer", "scheduler")

# TODO: add regex renames derived from Phase 4 key/shape dumps.
PARAM_NAME_MAP: dict[str, str] = {}

# TODO: include training-only or dynamically-computed keys that must not load.
SKIP_PATTERNS: tuple[str, ...] = ()


def _hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_API_KEY")
    )


def resolve_src(src: str, revision: str | None) -> Path:
    if os.path.exists(src):
        return Path(src)
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is required when --src is a repo id")
    return Path(snapshot_download(repo_id=src, revision=revision, token=_hf_token()))


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    if path.is_dir():
        weights: dict[str, torch.Tensor] = {}
        for shard in sorted(path.glob("*.safetensors")):
            weights.update(load_file(str(shard)))
        if weights:
            return weights
        raise FileNotFoundError(f"No safetensors found in {path}")

    if path.suffix == ".safetensors":
        return load_file(str(path))

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "module", "ema"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def should_skip_key(key: str) -> bool:
    return any(re.search(pattern, key) for pattern in SKIP_PATTERNS)


def apply_mapping(key: str) -> str | None:
    if should_skip_key(key):
        return None
    for pattern, replacement in PARAM_NAME_MAP.items():
        if re.match(pattern, key):
            return re.sub(pattern, replacement, key)
    return key


def split_monolithic(
    state: dict[str, torch.Tensor],
) -> dict[str, OrderedDict[str, torch.Tensor]]:
    components: dict[str, OrderedDict[str, torch.Tensor]] = {
        name: OrderedDict() for name in set(COMPONENT_PREFIXES.values())
    }
    intentionally_skipped: list[str] = []
    unowned: list[str] = []
    for key, value in state.items():
        if should_skip_key(key):
            intentionally_skipped.append(key)
            continue
        for prefix, component in COMPONENT_PREFIXES.items():
            if key.startswith(prefix):
                mapped = apply_mapping(key[len(prefix):])
                if mapped is not None:
                    components[component][mapped] = value
                break
        else:
            unowned.append(key)
    if unowned:
        sample = ", ".join(unowned[:10])
        raise ValueError(
            f"Unowned monolithic keys: {len(unowned)}. "
            f"Add COMPONENT_PREFIXES or SKIP_PATTERNS entries. Sample: {sample}"
        )
    if intentionally_skipped:
        print(f"Intentionally skipped {len(intentionally_skipped)} keys")
    return {name: weights for name, weights in components.items() if weights}


def load_separate_components(src_dir: Path) -> dict[str, OrderedDict[str, torch.Tensor]]:
    components: dict[str, OrderedDict[str, torch.Tensor]] = {}
    for component, rel_path in SEPARATE_COMPONENT_PATHS.items():
        state = load_checkpoint(src_dir / rel_path)
        converted: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key, value in state.items():
            mapped = apply_mapping(key)
            if mapped is not None:
                converted[mapped] = value
        components[component] = converted
    return components


def build_component_configs(_src_dir: Path) -> dict[str, dict[str, Any]]:
    # TODO: emit config content accepted by FastVideo loaders. Most components use
    # config.json; schedulers use scheduler_config.json.
    return {
        "transformer": {"_class_name": "<FastVideoTransformerClass>"},
        "vae": {"_class_name": "<FastVideoVAEClass>"},
    }


def config_filename(component: str) -> str:
    if component == "scheduler":
        return "scheduler_config.json"
    return "config.json"


def source_label(src: str) -> str:
    if os.path.exists(src):
        return Path(src).name
    return src


def build_model_index(
    src: str,
    revision: str | None,
    available_components: set[str],
) -> dict[str, Any]:
    # TODO: match the target pipeline and every required component.
    index: dict[str, Any] = {
        "_class_name": "<FastVideoPipelineClass>",
        "_diffusers_version": "0.30.0",
        "_fastvideo_converted_from": source_label(src),
        # Existing transformer/VAE loaders expect "diffusers" even when
        # _class_name names a FastVideo-native class registered in FastVideo.
        "transformer": ["diffusers", "<FastVideoTransformerClass>"],
        "vae": ["diffusers", "<FastVideoVAEClass>"],
    }
    if revision:
        index["_fastvideo_converted_revision"] = revision
    return {
        key: value
        for key, value in index.items()
        if key.startswith("_") or key in available_components
    }


def validate_component_configs(configs: dict[str, dict[str, Any]]) -> None:
    # TODO: instantiate each FastVideo config and call update_model_arch(...) or
    # update_model_config(...) with this JSON so unknown emitted keys fail here.
    placeholder_configs = [
        name for name, config in configs.items() if "<" in json.dumps(config)
    ]
    if placeholder_configs:
        raise ValueError(f"Replace config placeholders for: {placeholder_configs}")


def verify_conversion(
    dst_dir: Path,
    components: dict[str, OrderedDict[str, torch.Tensor]],
) -> None:
    del dst_dir, components
    # TODO: load each emitted stateful component through its production loader and
    # assert strict load, or document exact allowed missing/unexpected keys.
    raise NotImplementedError(
        "Implement production config validation and strict-load checks"
    )


def write_component(
    dst_dir: Path,
    name: str,
    state: dict[str, torch.Tensor],
    config: dict[str, Any] | None,
) -> None:
    component_dir = dst_dir / name
    if component_dir.exists() and any(component_dir.iterdir()):
        shutil.rmtree(component_dir)
    component_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        dict(state), str(component_dir / "diffusion_pytorch_model.safetensors")
    )
    if config is not None:
        config_path = component_dir / config_filename(name)
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
    print(f"Wrote {name}: {len(state)} tensors")


def copy_passthrough(src_dir: Path, dst_dir: Path) -> list[str]:
    copied: list[str] = []
    for subfolder in PASSTHROUGH_SUBFOLDERS:
        src = src_dir / subfolder
        if not src.is_dir():
            continue
        dst = dst_dir / subfolder
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        copied.append(subfolder)
        print(f"Copied {subfolder}/")
    return copied


def default_monolithic_checkpoint(src_path: Path) -> Path:
    if src_path.is_file():
        return src_path
    return src_path / "model.safetensors"


def convert(
    src: str,
    dst: str,
    layout: str,
    revision: str | None,
) -> None:
    src_path = resolve_src(src, revision)
    dst_dir = Path(dst)
    dst_dir.mkdir(parents=True, exist_ok=True)
    model_index_path = dst_dir / "model_index.json"

    if layout in {"monolithic", "raw_official"}:
        # TODO: replace model.safetensors with the official monolithic file name.
        components = split_monolithic(
            load_checkpoint(default_monolithic_checkpoint(src_path))
        )
    elif layout in {"separate_components", "mixed"}:
        if not src_path.is_dir():
            raise ValueError(f"{layout} layout requires a source directory: {src_path}")
        components = load_separate_components(src_path)
    else:
        raise ValueError(f"Unsupported template layout: {layout}")

    copied = (
        copy_passthrough(src_path, dst_dir) if src_path.is_dir() else []
    )
    configs = build_component_configs(src_path if src_path.is_dir() else src_path.parent)
    validate_component_configs(configs)
    for name, state in components.items():
        write_component(dst_dir, name, state, configs.get(name))

    available = set(components) | set(copied)
    with model_index_path.open("w", encoding="utf-8") as f:
        json.dump(build_model_index(src, revision, available), f, indent=2)
        f.write("\n")
    print(f"Wrote {dst_dir / 'model_index.json'}")
    verify_conversion(dst_dir, components)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src", required=True, help="HF repo id, local dir, or checkpoint path"
    )
    parser.add_argument("--revision", help="HF branch, tag, or commit for repo sources")
    parser.add_argument(
        "--dst",
        required=True,
        help="Output converted_weights/<model_family> directory",
    )
    parser.add_argument(
        "--layout",
        choices=("raw_official", "monolithic", "separate_components", "mixed"),
        required=True,
        help="Official source layout",
    )
    args = parser.parse_args()
    convert(args.src, args.dst, args.layout, args.revision)


if __name__ == "__main__":
    main()
