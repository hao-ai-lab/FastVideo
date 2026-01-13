# SPDX-License-Identifier: Apache-2.0
"""
Convert LTX-2 weights to FastVideo naming conventions and split by component.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover - optional dependency
    snapshot_download = None


PARAM_NAME_MAP: dict[str, str] = {
    r"^model\.diffusion_model\.(.*)$": r"\1",
}

COMPONENT_PREFIXES: dict[str, tuple[str, ...]] = {
    "transformer": ("model.diffusion_model.",),
    "vae": ("vae.",),
    "audio_vae": ("audio_vae.",),
    "vocoder": ("vocoder.",),
    "text_embedding_projection": ("text_embedding_projection.", "model.text_embedding_projection."),
}


def _find_shards(model_path: Path) -> list[Path]:
    if model_path.is_file():
        return [model_path]

    index_files = list(model_path.glob("*.safetensors.index.json"))
    if index_files:
        with index_files[0].open("r", encoding="utf-8") as f:
            index = json.load(f)
        return sorted({model_path / shard for shard in index["weight_map"].values()})
    return sorted(Path(p) for p in glob.glob(str(model_path / "*.safetensors")))


def _apply_mapping(key: str) -> str:
    for pattern, replacement in PARAM_NAME_MAP.items():
        if re.match(pattern, key):
            return re.sub(pattern, replacement, key)
    return key


def _load_weights(shards: list[Path]) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    for shard in shards:
        weights.update(load_file(str(shard)))
    return weights


def _read_metadata_config(path: Path) -> dict:
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata()
    if not metadata or "config" not in metadata:
        return {}
    return json.loads(metadata["config"])


def _filter_transformer_config(config: dict) -> dict:
    transformer = config.get("transformer", {})
    allowed = {
        "num_attention_heads",
        "attention_head_dim",
        "num_layers",
        "cross_attention_dim",
        "caption_channels",
        "norm_eps",
        "attention_type",
        "positional_embedding_theta",
        "positional_embedding_max_pos",
        "timestep_scale_multiplier",
        "use_middle_indices_grid",
        "rope_type",
        "frequencies_precision",
        "in_channels",
        "out_channels",
        "audio_num_attention_heads",
        "audio_attention_head_dim",
        "audio_in_channels",
        "audio_out_channels",
        "audio_cross_attention_dim",
        "audio_positional_embedding_max_pos",
        "av_ca_timestep_scale_multiplier",
    }
    filtered = {k: v for k, v in transformer.items() if k in allowed}
    if "frequencies_precision" in filtered:
        filtered["double_precision_rope"] = filtered["frequencies_precision"] == "float64"
        del filtered["frequencies_precision"]
    return filtered


def _build_text_embedding_projection_config(
    gemma_model_path: str = "",
) -> dict:
    return {
        "architectures": ["LTX2GemmaTextEncoderModel"],
        "hidden_size": 3840,
        "num_hidden_layers": 48,
        "num_attention_heads": 30,
        "text_len": 1024,
        "pad_token_id": 0,
        "eos_token_id": 2,
        "gemma_model_path": gemma_model_path,
        "gemma_dtype": "bfloat16",
        "padding_side": "left",
        "feature_extractor_in_features": 3840 * 49,
        "feature_extractor_out_features": 3840,
        "connector_num_attention_heads": 30,
        "connector_attention_head_dim": 128,
        "connector_num_layers": 2,
        "connector_positional_embedding_theta": 10000.0,
        "connector_positional_embedding_max_pos": [4096],
        "connector_rope_type": "split",
        "connector_double_precision_rope": True,
        "connector_num_learnable_registers": 128,
    }


def _wrap_component_config(
    component_name: str,
    component_config: dict | None,
    class_name: str | None = None,
) -> dict | None:
    if component_config is None:
        return None
    wrapped = {component_name: component_config}
    if class_name is not None:
        wrapped["_class_name"] = class_name
    return wrapped


def _split_component_weights(weights: dict[str, torch.Tensor]) -> dict[str, OrderedDict]:
    components: dict[str, OrderedDict] = {name: OrderedDict() for name in COMPONENT_PREFIXES}
    for key, value in weights.items():
        if key.startswith("model.diffusion_model.audio_embeddings_connector."):
            new_key = key.replace("model.diffusion_model.audio_embeddings_connector.", "audio_embeddings_connector.")
            components["text_embedding_projection"][new_key] = value
            continue
        if key.startswith("model.diffusion_model.video_embeddings_connector."):
            new_key = key.replace("model.diffusion_model.video_embeddings_connector.", "embeddings_connector.")
            components["text_embedding_projection"][new_key] = value
            continue

        matched = False
        for component, prefixes in COMPONENT_PREFIXES.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    components[component][new_key] = value
                    matched = True
                    break
            if matched:
                break
    return {name: weights for name, weights in components.items() if weights}


def _write_component(
    output_dir: Path,
    name: str,
    weights: OrderedDict,
    config: dict | None,
    dir_name: str | None = None,
) -> None:
    component_dir = output_dir / (dir_name or name)
    component_dir.mkdir(parents=True, exist_ok=True)
    output_file = component_dir / "model.safetensors"
    save_file(weights, str(output_file))
    print(f"Saved {name} weights to {output_file}")

    if config is not None:
        config_path = component_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        print(f"Saved {name} config to {config_path}")


def _build_model_index(
    transformer_class_name: str,
    vae_class_name: str,
    pipeline_class_name: str,
    diffusers_version: str,
) -> dict:
    return {
        "_class_name": pipeline_class_name,
        "_diffusers_version": diffusers_version,
        "transformer": ["diffusers", transformer_class_name],
        "vae": ["diffusers", vae_class_name],
        "text_encoder": ["transformers", "LTX2GemmaTextEncoderModel"],
        "tokenizer": ["transformers", "AutoTokenizer"],
        "audio_vae": ["diffusers", "LTX2AudioDecoder"],
        "vocoder": ["diffusers", "LTX2Vocoder"],
    }


def _write_model_index(output_dir: Path, model_index: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_index_path = output_dir / "model_index.json"
    with model_index_path.open("w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2)
        f.write("\n")
    print(f"Saved model_index.json to {model_index_path}")


def convert_components(
    source_path: Path,
    output_dir: Path,
    metadata_config: dict,
    transformer_class_name: str,
    components_to_write: set[str] | None = None,
    emit_diffusers_repo: bool = True,
    pipeline_class_name: str = "LTX2Pipeline",
    diffusers_version: str = "0.33.0.dev0",
    gemma_model_path: str = "",
) -> None:
    shards = _find_shards(source_path)
    if not shards:
        raise FileNotFoundError(f"No safetensors found in {source_path}")

    weights = _load_weights(shards)
    split_weights = _split_component_weights(weights)
    if components_to_write is not None:
        split_weights = {name: weights for name, weights in split_weights.items() if name in components_to_write}

    transformer_weights = split_weights.get("transformer", OrderedDict())
    converted_transformer = OrderedDict()
    for key, value in transformer_weights.items():
        new_key = _apply_mapping(f"model.diffusion_model.{key}")
        converted_transformer[new_key] = value
    split_weights["transformer"] = converted_transformer

    transformer_config = _filter_transformer_config(metadata_config)
    if transformer_config:
        transformer_config["_class_name"] = transformer_class_name

    component_configs: dict[str, dict | None] = {
        "transformer": transformer_config or None,
        "vae": _wrap_component_config(
            "vae",
            metadata_config.get("vae"),
            class_name="CausalVideoAutoencoder",
        ),
        "audio_vae": _wrap_component_config(
            "audio_vae",
            metadata_config.get("audio_vae"),
            class_name="LTX2AudioDecoder",
        ),
        "vocoder": _wrap_component_config(
            "vocoder",
            metadata_config.get("vocoder"),
            class_name="LTX2Vocoder",
        ),
        "text_embedding_projection": _build_text_embedding_projection_config(
            gemma_model_path=gemma_model_path
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, component_weights in split_weights.items():
        _write_component(output_dir, name, component_weights, component_configs.get(name))
        if emit_diffusers_repo and name == "text_embedding_projection":
            _write_component(
                output_dir,
                name,
                component_weights,
                component_configs.get(name),
                dir_name="text_encoder",
            )
    if emit_diffusers_repo:
        required_for_index = {
            "transformer",
            "vae",
            "audio_vae",
            "vocoder",
            "text_embedding_projection",
        }
        if components_to_write is not None and not required_for_index.issubset(components_to_write):
            print("Skipping model_index.json; not all diffusers components were written.")
            return
        if not required_for_index.issubset(split_weights.keys()):
            print("Skipping model_index.json; missing diffusers components in weights.")
            return
        vae_class_name = (component_configs.get("vae") or {}).get(
            "_class_name", "CausalVideoAutoencoder"
        )
        model_index = _build_model_index(
            transformer_class_name=transformer_class_name,
            vae_class_name=vae_class_name,
            pipeline_class_name=pipeline_class_name,
            diffusers_version=diffusers_version,
        )
        _write_model_index(output_dir, model_index)


def update_transformer_config(config_path: Path, class_name: str) -> None:
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    config["_class_name"] = class_name
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print(f"Updated _class_name in {config_path} -> {class_name}")


def maybe_download(repo_id: str, target_dir: Path, token: str | None, allow_patterns: str | None) -> Path:
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is required for --download")
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=allow_patterns,
    )
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LTX-2 weights to FastVideo format")
    parser.add_argument("--source", type=str, help="Path to transformer weights directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for converted weights")
    parser.add_argument("--download", type=str, help="HF repo id to download before conversion")
    parser.add_argument("--allow-patterns", type=str, help="Limit HF download to matching files")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"), help="HF token (or set HF_TOKEN)")
    parser.add_argument("--update-config", action="store_true", help="Update source config.json _class_name")
    parser.add_argument("--class-name", type=str, default="LTX2Transformer3DModel")
    parser.add_argument(
        "--diffusers-repo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit a diffusers-style repo layout with model_index.json.",
    )
    parser.add_argument(
        "--pipeline-class-name",
        type=str,
        default="LTX2Pipeline",
        help="Pipeline class name for model_index.json.",
    )
    parser.add_argument(
        "--diffusers-version",
        type=str,
        default="0.33.0.dev0",
        help="Diffusers version for model_index.json.",
    )
    parser.add_argument(
        "--transformer-only",
        action="store_true",
        help="Only convert transformer weights (no component split).",
    )
    parser.add_argument(
        "--components",
        type=str,
        default="",
        help=(
            "Comma-separated component list to write "
            "(transformer,vae,audio_vae,vocoder,text_embedding_projection)."
        ),
    )
    parser.add_argument(
        "--gemma-path",
        type=str,
        default="",
        help="Optional local Gemma model path to copy into the output repo.",
    )

    args = parser.parse_args()

    if args.download:
        if args.source:
            raise ValueError("Use either --download or --source, not both.")
        source_dir = maybe_download(args.download, Path(args.output) / "download", args.token, args.allow_patterns)
    else:
        if not args.source:
            raise ValueError("--source is required when not using --download")
        source_dir = Path(args.source)

    output_dir = Path(args.output)
    shards = _find_shards(source_dir)
    if not shards:
        raise FileNotFoundError(f"No safetensors found in {source_dir}")
    metadata_path = shards[0]
    metadata_config = _read_metadata_config(metadata_path)
    components_to_write: set[str] | None = None
    if args.transformer_only:
        components_to_write = {"transformer"}
    elif args.components:
        components_to_write = {
            component.strip()
            for component in args.components.split(",")
            if component.strip()
        }

    gemma_model_path = ""
    if args.gemma_path:
        gemma_src = Path(args.gemma_path)
        if not gemma_src.is_dir():
            raise ValueError(f"--gemma-path must be a directory: {gemma_src}")
        gemma_dest = output_dir / "text_encoder" / "gemma"
        if gemma_dest.exists():
            shutil.rmtree(gemma_dest)
        gemma_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(gemma_src, gemma_dest)
        gemma_model_path = "gemma"

    convert_components(
        source_dir,
        output_dir,
        metadata_config,
        args.class_name,
        components_to_write=components_to_write,
        emit_diffusers_repo=args.diffusers_repo,
        pipeline_class_name=args.pipeline_class_name,
        diffusers_version=args.diffusers_version,
        gemma_model_path=gemma_model_path,
    )

    if args.update_config:
        if source_dir.is_dir():
            update_transformer_config(source_dir / "config.json", args.class_name)


if __name__ == "__main__":
    main()
