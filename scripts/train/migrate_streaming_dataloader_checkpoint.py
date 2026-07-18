# SPDX-License-Identifier: Apache-2.0
"""Create audited rank-local dataloader sidecars for a legacy checkpoint.

The legacy checkpoint is never modified.  This utility only supports the
deterministic, single-process-per-rank OpenVid streaming layout used by the
GB200 run and deliberately refuses ambiguous epoch-boundary migrations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from fastvideo.dataset.parquet_dataset_streaming_style import (
    reconstruct_streaming_dataset_state,
)
from fastvideo.train.utils.checkpoint import (
    _DATALOADER_STATE_VERSION,
    _atomic_torch_save,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_checkpoint_config(
    checkpoint_config: dict[str, Any],
    run_config: dict[str, Any],
) -> None:
    critical_paths = (
        ("training", "distributed", "num_gpus"),
        ("training", "distributed", "sp_size"),
        ("training", "data", "data_path"),
        ("training", "data", "dataloader_type"),
        ("training", "data", "streaming_manifest_path"),
        ("training", "data", "streaming_read_batch_size"),
        ("training", "data", "streaming_shuffle_row_groups"),
        ("training", "data", "dataloader_num_workers"),
        ("training", "data", "train_batch_size"),
        ("training", "data", "seed"),
        ("training", "loop", "gradient_accumulation_steps"),
    )
    mismatches = []
    for path in critical_paths:
        checkpoint_value = _nested(checkpoint_config, *path)
        run_value = _nested(run_config, *path)
        if checkpoint_value != run_value:
            mismatches.append((".".join(path), checkpoint_value, run_value))
    if mismatches:
        formatted = "; ".join(
            f"{path}: checkpoint={old!r}, config={new!r}"
            for path, old, new in mismatches
        )
        raise ValueError(f"Run config does not match checkpoint metadata: {formatted}")


def migrate(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir == checkpoint_dir or checkpoint_dir in output_dir.parents:
        raise ValueError("Migration output must be outside the legacy checkpoint")
    if not (checkpoint_dir / "dcp").is_dir():
        raise FileNotFoundError(f"Missing DCP directory: {checkpoint_dir / 'dcp'}")

    metadata_path = checkpoint_dir / "metadata.json"
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    step = int(metadata["step"])
    if step != int(args.expected_step):
        raise ValueError(f"Checkpoint step mismatch: expected {args.expected_step}, got {step}")

    with config_path.open(encoding="utf-8") as handle:
        run_config = yaml.safe_load(handle)
    if not isinstance(run_config, dict):
        raise ValueError("Run config must be a YAML mapping")
    checkpoint_config = metadata.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("Legacy checkpoint metadata has no embedded run config")
    _validate_checkpoint_config(checkpoint_config, run_config)

    distributed = _nested(run_config, "training", "distributed")
    data = _nested(run_config, "training", "data")
    loop = _nested(run_config, "training", "loop")
    if str(data["dataloader_type"]).strip().lower() != "streaming":
        raise ValueError("This migration only supports the streaming dataloader")
    if int(data["dataloader_num_workers"]) != 0:
        raise ValueError("Legacy migration currently requires dataloader_num_workers=0")

    world_size = int(distributed["num_gpus"])
    if world_size != int(args.expected_world_size):
        raise ValueError(
            f"World-size mismatch: expected {args.expected_world_size}, config has {world_size}"
        )
    sp_world_size = int(distributed["sp_size"])
    batch_size = int(data["train_batch_size"])
    grad_accum = int(loop["gradient_accumulation_steps"])
    consumed_batches = int(args.consumed_batches_per_rank)
    expected_batches = step * grad_accum
    if consumed_batches != expected_batches:
        raise ValueError(
            f"Consumed batch count mismatch: explicitly supplied {consumed_batches}, "
            f"but step*gradient_accumulation_steps is {expected_batches}"
        )
    yielded_samples = consumed_batches * batch_size

    manifest_path = Path(data["streaming_manifest_path"]).expanduser().resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    dataset_root = os.path.realpath(os.path.expanduser(str(data["data_path"])))
    if manifest.get("dataset_root") != dataset_root:
        raise ValueError("Streaming manifest belongs to a different dataset root")

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("dataloader_state_rank*.pt"))
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Migration output already contains {len(existing)} sidecars; use --overwrite"
        )

    sidecars = []
    for rank in range(world_size):
        dataset_state = reconstruct_streaming_dataset_state(
            manifest,
            global_rank=rank,
            world_size=world_size,
            sp_world_size=sp_world_size,
            num_workers=0,
            batch_size=batch_size,
            read_batch_size=int(data["streaming_read_batch_size"]),
            seed=int(data["seed"]),
            shuffle_row_groups=bool(data["streaming_shuffle_row_groups"]),
            yielded_samples=yielded_samples,
        )
        payload = {
            "version": _DATALOADER_STATE_VERSION,
            "rank": rank,
            "world_size": world_size,
            "step": step,
            "state_kind": "dataset",
            "state": dataset_state,
        }
        path = output_dir / f"dataloader_state_rank{rank}.pt"
        _atomic_torch_save(payload, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if loaded != payload:
            raise RuntimeError(f"Sidecar round-trip mismatch: {path}")
        sidecars.append(
            {
                "rank": rank,
                "path": path.name,
                "sha256": _sha256(path),
                "row_group_position": dataset_state["row_group_position"],
                "row_offset": dataset_state["row_offset"],
                "yielded_samples": dataset_state["yielded_samples"],
            }
        )

    receipt = {
        "schema": "fastvideo-legacy-streaming-dataloader-migration/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(checkpoint_dir),
        "source_checkpoint_metadata_sha256": _sha256(metadata_path),
        "source_config": str(config_path),
        "source_config_sha256": _sha256(config_path),
        "manifest": str(manifest_path),
        "manifest_fingerprint": manifest["fingerprint"],
        "step": step,
        "world_size": world_size,
        "sp_world_size": sp_world_size,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "consumed_batches_per_rank": consumed_batches,
        "yielded_samples_per_rank": yielded_samples,
        "sidecars": sidecars,
    }
    _write_json_atomic(output_dir / "migration_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-step", required=True, type=int)
    parser.add_argument("--expected-world-size", required=True, type=int)
    parser.add_argument("--consumed-batches-per-rank", required=True, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    receipt = migrate(args)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
