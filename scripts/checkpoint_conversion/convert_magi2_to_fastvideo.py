#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create a FastVideo MAGI-2 repository from the official checkpoint layout.

The published ``sand-ai/MAGI-2-preview`` snapshot already stores tensors in the
names and dtypes that the FastVideo-native loaders consume. This converter keeps
the tensor files byte-for-byte and changes only the component directory layout.
Hard links avoid a second 286 GiB allocation while producing regular files that
Hugging Face upload tools can publish as a self-contained repository.

Example:
    python scripts/checkpoint_conversion/convert_magi2_to_fastvideo.py \
        --source official_weights/magi2 \
        --output converted_weights/magi2
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


SOURCE_REPOSITORY = "sand-ai/MAGI-2-preview"
SOURCE_REVISION = "2dea51b64db47ee5b4402d36fd90829a0c58913b"

COMPONENT_DIRECTORY_MAPPING: dict[str, str] = {
    "preview": "transformer",
    "refiner": "transformer_2",
    "text_encoder": "text_encoder",
    "vae": "image_encoder",
    "turbo_vae": "vae",
    "stable-audio-open-1.0": "audio_vae",
}

REQUIRED_COMPONENT_FILES: dict[str, tuple[str, ...]] = {
    "preview": ("model.safetensors.index.json",),
    "refiner": ("model.safetensors.index.json",),
    "text_encoder": ("config.json", "model.safetensors.index.json", "tokenizer.json"),
    "vae": ("Wan2.2_VAE.pth",),
    "turbo_vae": ("TurboV3-Wan22-TinyShallow_7_7.json", "checkpoint.ckpt"),
    "stable-audio-open-1.0": ("model_config.json", "model.safetensors"),
}

# The distilled decoder checkpoint contains these feature-matching heads for
# training. MAGI-2 inference strictly loads the 88 ``decoder.*`` tensors.
TURBO_VAE_SKIPPED_KEYS: tuple[str, ...] = (
    "aligned_feature_projection_heads.0.0.conv.weight",
    "aligned_feature_projection_heads.0.0.conv.bias",
    "aligned_feature_projection_heads.0.1.conv.weight",
    "aligned_feature_projection_heads.0.1.conv.bias",
    "aligned_feature_projection_heads.1.0.conv.weight",
    "aligned_feature_projection_heads.1.0.conv.bias",
    "aligned_feature_projection_heads.1.1.conv.weight",
    "aligned_feature_projection_heads.1.1.conv.bias",
)

MODEL_INDEX: dict[str, object] = {
    "_class_name": "Magi2Pipeline",
    "_diffusers_version": "0.37.0",
    "transformer": ["fastvideo.models.dits.magi2", "Magi2PreviewDiT"],
    "transformer_2": ["fastvideo.models.dits.magi2_refiner", "Magi2RefinerDiT"],
    "text_encoder": ["fastvideo.models.encoders.qwen3_5", "Magi2Qwen35TextEncoder"],
    "image_encoder": ["fastvideo.models.vaes.magi2_wan_loader", "Magi2WanImageEncoder"],
    "vae": ["fastvideo.models.vaes.magi2_turbo_vae", "Magi2TurboVAEModel"],
    "audio_vae": ["fastvideo.models.vaes.magi2_audio_vae", "Magi2AudioVAE"],
    "scheduler": [
        "fastvideo.models.schedulers.scheduling_flow_unipc_multistep",
        "FlowUniPCMultistepScheduler",
    ],
}

SCHEDULER_CONFIG: dict[str, object] = {
    "_class_name": "FlowUniPCMultistepScheduler",
    "_diffusers_version": "0.37.0",
    "disable_corrector": [],
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "predict_x0": True,
    "prediction_type": "flow_prediction",
    "shift": 1.0,
    "solver_order": 2,
    "solver_type": "bh2",
}

TRANSFORMER_CONFIGS: dict[str, dict[str, object]] = {
    "transformer": {
        "_class_name": "Magi2PreviewDiT",
        "_diffusers_version": "0.37.0",
        "source_subfolder": "preview",
    },
    "transformer_2": {
        "_class_name": "Magi2RefinerDiT",
        "_diffusers_version": "0.37.0",
        "source_subfolder": "refiner",
    },
}


def _validate_source(source: Path) -> None:
    """Require component metadata and every shard referenced by an index."""
    missing_files = [
        str(source / component / relative_path)
        for component, relative_paths in REQUIRED_COMPONENT_FILES.items()
        for relative_path in relative_paths
        if not (source / component / relative_path).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "The MAGI-2 source snapshot is incomplete; missing: "
            + ", ".join(missing_files)
        )

    missing_shards: list[str] = []
    for component, relative_paths in REQUIRED_COMPONENT_FILES.items():
        for relative_path in relative_paths:
            if not relative_path.endswith(".index.json"):
                continue
            index_path = source / component / relative_path
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index_payload.get("weight_map")
            if not isinstance(weight_map, dict):
                raise ValueError(f"Checkpoint index has no weight_map object: {index_path}")
            if not weight_map or not all(isinstance(shard_name, str) for shard_name in weight_map.values()):
                raise ValueError(f"Checkpoint index has an invalid weight_map object: {index_path}")
            shard_names = set(weight_map.values())
            missing_shards.extend(
                str(index_path.parent / shard_name)
                for shard_name in sorted(shard_names)
                if not (index_path.parent / shard_name).is_file()
            )
    if missing_shards:
        raise FileNotFoundError(
            "The MAGI-2 source snapshot is missing indexed checkpoint shards: "
            + ", ".join(missing_shards)
        )


def _hardlink_component(source: Path, destination: Path) -> None:
    """Replicate one component tree with regular-file hard links."""
    destination.mkdir(parents=True)
    for source_path in sorted(source.rglob("*")):
        relative_path = source_path.relative_to(source)
        destination_path = destination / relative_path
        if source_path.is_dir():
            destination_path.mkdir()
            continue
        if not source_path.is_file():
            raise ValueError(f"Unsupported checkpoint entry: {source_path}")
        os.link(source_path, destination_path)


def _write_json(path: Path, payload: object) -> None:
    """Write stable, reviewable JSON with a trailing newline."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _conversion_manifest() -> dict[str, object]:
    """Describe directory and tensor-name transformations for every component."""
    return {
        "format_version": 1,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
        },
        "component_directory_mapping": COMPONENT_DIRECTORY_MAPPING,
        "tensor_mapping": {
            "transformer": "identity; expert_bias_ema replaces expert_bias during inference loading",
            "transformer_2": "identity",
            "text_encoder": "identity; Transformers Qwen3.5 loads the language-model tensors",
            "image_encoder": "identity; ignore decoder.* and conv2.* tensors",
            "vae": "strip module.; load decoder.* tensors",
            "audio_vae": "strip pretransform.model.; map the published sequential decoder to OobleckDecoder",
        },
        "skipped_checkpoint_keys": {
            "vae/checkpoint.ckpt": list(TURBO_VAE_SKIPPED_KEYS),
        },
    }


def convert_checkpoint_layout(source: Path, output: Path) -> None:
    """Hard-link official components and add FastVideo repository metadata."""
    source = source.resolve()
    output = output.resolve()
    _validate_source(source)
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    output.mkdir(parents=True)

    for source_name, destination_name in COMPONENT_DIRECTORY_MAPPING.items():
        _hardlink_component(source / source_name, output / destination_name)

    scheduler_dir = output / "scheduler"
    scheduler_dir.mkdir()
    _write_json(scheduler_dir / "scheduler_config.json", SCHEDULER_CONFIG)
    for component_name, config in TRANSFORMER_CONFIGS.items():
        _write_json(output / component_name / "config.json", config)
    _write_json(output / "model_index.json", MODEL_INDEX)
    _write_json(output / "magi2_conversion_manifest.json", _conversion_manifest())


def main() -> None:
    """Parse source and destination paths and create the converted repository."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Local snapshot of sand-ai/MAGI-2-preview.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination FastVideo model repository; the path must not exist.",
    )
    arguments = parser.parse_args()
    convert_checkpoint_layout(arguments.source, arguments.output)


if __name__ == "__main__":
    main()
