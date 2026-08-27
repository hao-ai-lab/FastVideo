#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Package NVIDIA's Cosmos Predict2.5 2B distilled student for FastVideo.

The released checkpoint is a native PyTorch state dict. FastVideo's existing
Cosmos25 parameter mapping consumes its ``net.*`` names directly, so this
converter deliberately does not rename tensors. It isolates the student from
teacher/critic/training state and combines it with the non-transformer assets
from an existing FastVideo-loadable Cosmos Predict2.5 package.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file

STATE_DICT_KEYS = ("state_dict", "model", "ema", "ema_model", "module")
STUDENT_SKIP_PREFIXES = ("net.accum_",)
REQUIRED_BASE_PATHS = (
    "model_index.json",
    "transformer/config.json",
    "vae",
    "text_encoder",
    "tokenizer",
)
TRANSFORMER_FILENAME = "diffusion_pytorch_model.safetensors"


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


def _has_student_tensors(value: object) -> bool:
    return isinstance(value, Mapping) and any(
        isinstance(key, str) and key.startswith("net.") and torch.is_tensor(tensor) for key, tensor in value.items()
    )


def extract_student_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """Find the official state dict and retain only the distilled student."""
    state_dict: Mapping[Any, Any] | None = (
        cast(Mapping[Any, Any], checkpoint) if _has_student_tensors(checkpoint) else None
    )
    if state_dict is None and isinstance(checkpoint, Mapping):
        for key in STATE_DICT_KEYS:
            candidate = checkpoint.get(key)
            if _has_student_tensors(candidate):
                state_dict = cast(Mapping[Any, Any], candidate)
                break

    if state_dict is None:
        raise ConversionError("Could not find a tensor state dict containing native 'net.*' student keys")

    student = {
        key: tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        for key, tensor in state_dict.items()
        if (
            isinstance(key, str)
            and key.startswith("net.")
            and not key.startswith(STUDENT_SKIP_PREFIXES)
            and torch.is_tensor(tensor)
        )
    }
    if not student:
        raise ConversionError("The resolved checkpoint contains no 'net.*' tensors")
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


def _write_distilled_metadata(dst: Path) -> None:
    model_index_path = dst / "model_index.json"
    model_index = _read_json(model_index_path)
    model_index["is_distilled"] = True
    model_index["scheduler"] = ["diffusers", "Cosmos25DistilledScheduler"]
    _write_json(model_index_path, model_index)

    scheduler_dir = dst / "scheduler"
    if scheduler_dir.exists():
        shutil.rmtree(scheduler_dir)
    _write_json(
        scheduler_dir / "scheduler_config.json",
        {
            "_class_name": "Cosmos25DistilledScheduler",
            "_diffusers_version": "0.37.0.dev0",
            "num_train_timesteps": 1000,
            "sigma_data": 1.0,
        },
    )


def _verify_output(dst: Path, expected_keys: set[str]) -> None:
    model_index = _read_json(dst / "model_index.json")
    if model_index.get("scheduler") != ["diffusers", "Cosmos25DistilledScheduler"]:
        raise ConversionError("model_index.json does not select Cosmos25DistilledScheduler")

    scheduler_config = _read_json(dst / "scheduler/scheduler_config.json")
    if scheduler_config.get("_class_name") != "Cosmos25DistilledScheduler":
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
    src_checkpoint: Path,
    base_model: Path,
    dst: Path,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    src_checkpoint = src_checkpoint.expanduser().resolve()
    base_model = base_model.expanduser().resolve()
    dst = dst.expanduser().resolve()
    if not src_checkpoint.is_file():
        raise FileNotFoundError(f"Distilled checkpoint not found: {src_checkpoint}")
    _validate_base_model(base_model)
    _prepare_output(dst, overwrite)

    checkpoint = torch.load(src_checkpoint, map_location="cpu", weights_only=True)
    student = extract_student_state_dict(checkpoint)

    shutil.copytree(base_model, dst, dirs_exist_ok=True, symlinks=False)
    transformer_dir = dst / "transformer"
    _remove_base_transformer_weights(transformer_dir)
    save_file(
        student,
        str(transformer_dir / TRANSFORMER_FILENAME),
        metadata={"format": "pt", "model_type": "cosmos25_distilled_student"},
    )
    _write_distilled_metadata(dst)
    _verify_output(dst, set(student))

    return {
        "student_tensors": len(student),
        "student_parameters": sum(tensor.numel() for tensor in student.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--base-model",
        required=True,
        type=Path,
        help="Existing FastVideo-loadable Cosmos25 package supplying configs, encoder, tokenizer, and VAE",
    )
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = convert_checkpoint(
        args.src_checkpoint,
        args.base_model,
        args.dst,
        overwrite=args.overwrite,
    )
    print(f"Converted {report['student_tensors']} student tensors ({report['student_parameters']:,} parameters)")
    print(f"Output: {args.dst.expanduser().resolve()}")


if __name__ == "__main__":
    main()
