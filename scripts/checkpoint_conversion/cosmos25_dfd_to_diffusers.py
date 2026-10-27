#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Package the Cosmos Predict2.5 DFD I2V student for FastVideo.

The public checkpoint is a PyTorch distributed-checkpoint directory produced
from the upstream student network. This converter materializes that state dict,
normalizes its network prefix to the existing Cosmos25 mapping contract, drops
training-only tensors, and combines it with the reusable Cosmos25 components
from an existing FastVideo-loadable package.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REQUIRED_BASE_PATHS = (
    "model_index.json",
    "transformer/config.json",
    "vae",
    "text_encoder",
    "tokenizer",
)
TRANSFORMER_FILENAME = "diffusion_pytorch_model.safetensors"
SKIP_KEY_FRAGMENTS = (
    ".accum_",
    ".logvar_linear.",
    ".pos_embedder.",
    "._extra_state",
)


class ConversionError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _flatten_tensors(value: object, prefix: str = "") -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        return {}
    tensors: dict[str, torch.Tensor] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        name = f"{prefix}.{key}" if prefix else key
        if torch.is_tensor(item):
            tensors[name] = item
        elif isinstance(item, Mapping):
            tensors.update(_flatten_tensors(item, name))
    return tensors


def load_dfd_dcp_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"DFD distributed checkpoint directory not found: {checkpoint_dir}")
    if not (checkpoint_dir / ".metadata").is_file():
        raise ConversionError(f"DFD checkpoint has no .metadata file: {checkpoint_dir}")

    try:
        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    except ImportError as exc:
        raise ConversionError("This conversion requires torch.distributed.checkpoint.format_utils") from exc

    with tempfile.TemporaryDirectory(prefix="cosmos25-dfd-") as temp_dir:
        materialized_path = Path(temp_dir) / "student.pt"
        dcp_to_torch_save(checkpoint_dir, materialized_path)
        checkpoint = torch.load(materialized_path, map_location="cpu", weights_only=True)

    tensors = _flatten_tensors(checkpoint)
    if not tensors:
        raise ConversionError(f"DFD checkpoint contains no tensors: {checkpoint_dir}")
    return tensors


def normalize_dfd_student_state_dict(state_dict: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Normalize DCP keys to the native `net.*` Cosmos25 conversion surface."""
    student: dict[str, torch.Tensor] = {}
    for source_key, tensor in state_dict.items():
        if not isinstance(source_key, str) or not torch.is_tensor(tensor):
            continue

        key = source_key.replace("._checkpoint_wrapped_module", "")
        wrapper_removed = True
        while wrapper_removed:
            wrapper_removed = False
            for wrapper_prefix in ("model.", "module."):
                if key.startswith(wrapper_prefix):
                    key = key[len(wrapper_prefix) :]
                    wrapper_removed = True

        if key.startswith("net.transformer."):
            key = "net." + key[len("net.transformer.") :]
        elif key.startswith("transformer."):
            key = "net." + key[len("transformer.") :]
        elif not key.startswith("net."):
            key = "net." + key

        if any(fragment in key for fragment in SKIP_KEY_FRAGMENTS):
            continue
        student[key] = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()

    if not student:
        raise ConversionError("No DFD student tensors remained after key normalization")
    return student


def _validate_base_model(base_model: Path) -> None:
    missing = [relative for relative in REQUIRED_BASE_PATHS if not (base_model / relative).exists()]
    if missing:
        raise ConversionError(f"Base Cosmos25 package is missing required paths: {missing}")


def _prepare_output(dst: Path, overwrite: bool) -> None:
    if dst.exists() and any(dst.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {dst}. Pass --overwrite to replace it.")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)


def _remove_base_transformer_weights(transformer_dir: Path) -> None:
    for pattern in ("*.safetensors", "*.safetensors.index.json", "*.bin", "*.pt"):
        for path in transformer_dir.glob(pattern):
            path.unlink()


def _write_dfd_metadata(dst: Path) -> None:
    model_index_path = dst / "model_index.json"
    model_index = _read_json(model_index_path)
    model_index["is_distilled"] = True
    model_index["distillation_method"] = "dfd"
    model_index["scheduler"] = ["diffusers", "Cosmos25DFDScheduler"]
    _write_json(model_index_path, model_index)

    transformer_config_path = dst / "transformer" / "config.json"
    transformer_config = _read_json(transformer_config_path)
    transformer_config["rope_enable_fps_modulation"] = True
    transformer_config["use_crossattn_projection"] = True
    _write_json(transformer_config_path, transformer_config)

    scheduler_dir = dst / "scheduler"
    if scheduler_dir.exists():
        shutil.rmtree(scheduler_dir)
    _write_json(
        scheduler_dir / "scheduler_config.json",
        {
            "_class_name": "Cosmos25DFDScheduler",
            "_diffusers_version": "0.37.0.dev0",
            "num_train_timesteps": 1000,
            "sigma_data": 1.0,
        },
    )


def _verify_output(dst: Path, expected_keys: set[str]) -> None:
    model_index = _read_json(dst / "model_index.json")
    if model_index.get("scheduler") != ["diffusers", "Cosmos25DFDScheduler"]:
        raise ConversionError("model_index.json does not select Cosmos25DFDScheduler")
    if model_index.get("distillation_method") != "dfd":
        raise ConversionError("model_index.json does not identify the DFD checkpoint")

    transformer_config = _read_json(dst / "transformer/config.json")
    if transformer_config.get("rope_enable_fps_modulation") is not True:
        raise ConversionError("DFD transformer config must enable FPS-modulated RoPE")
    if transformer_config.get("use_crossattn_projection") is not True:
        raise ConversionError("DFD transformer config must enable the Reason1 projection")

    scheduler_config = _read_json(dst / "scheduler/scheduler_config.json")
    if scheduler_config.get("_class_name") != "Cosmos25DFDScheduler":
        raise ConversionError("scheduler_config.json has the wrong _class_name")

    weights_path = dst / "transformer" / TRANSFORMER_FILENAME
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        actual_keys = set(handle.keys())
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ConversionError(
            f"Converted transformer key mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )


def convert_checkpoint(
    src_dcp: Path,
    base_model: Path,
    dst: Path,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    src_dcp = src_dcp.expanduser().resolve()
    base_model = base_model.expanduser().resolve()
    dst = dst.expanduser().resolve()
    _validate_base_model(base_model)
    _prepare_output(dst, overwrite)

    raw_state = load_dfd_dcp_state_dict(src_dcp)
    student = normalize_dfd_student_state_dict(raw_state)

    shutil.copytree(base_model, dst, dirs_exist_ok=True, symlinks=False)
    transformer_dir = dst / "transformer"
    _remove_base_transformer_weights(transformer_dir)
    save_file(
        student,
        str(transformer_dir / TRANSFORMER_FILENAME),
        metadata={"format": "pt", "model_type": "cosmos25_dfd_student"},
    )
    _write_dfd_metadata(dst)
    _verify_output(dst, set(student))

    return {
        "student_tensors": len(student),
        "student_parameters": sum(tensor.numel() for tensor in student.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dcp", required=True, type=Path)
    parser.add_argument(
        "--base-model",
        required=True,
        type=Path,
        help="FastVideo-loadable Cosmos25 package supplying configs, Reason1, tokenizer, and VAE",
    )
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = convert_checkpoint(
        args.src_dcp,
        args.base_model,
        args.dst,
        overwrite=args.overwrite,
    )
    print(f"Converted {report['student_tensors']} student tensors ({report['student_parameters']:,} parameters)")
    print(f"Output: {args.dst.expanduser().resolve()}")


if __name__ == "__main__":
    main()
