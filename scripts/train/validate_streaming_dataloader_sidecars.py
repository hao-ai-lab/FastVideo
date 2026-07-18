# SPDX-License-Identifier: Apache-2.0
"""Validate migrated T2V streaming sidecars under the real process topology."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import yaml

from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v
from fastvideo.dataset.parquet_dataset_streaming_style import (
    build_parquet_streaming_style_dataloader,
)
from fastvideo.distributed import (
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.train.utils.checkpoint import CheckpointConfig, CheckpointManager


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _next_manifest_id(dataset: Any, state: dict[str, Any]) -> str:
    shard = dataset._worker_shard(0)
    position = int(state["row_group_position"])
    offset = int(state["row_offset"])
    item = shard[position]
    if offset == int(item["rows"]):
        position += 1
        offset = 0
        item = shard[position]
    parquet_file = pq.ParquetFile(os.path.join(dataset.path, item["file"]))
    ids = parquet_file.read_row_group(int(item["row_group"]), columns=["id"]).column("id")
    return str(ids[offset].as_py())


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate(args: argparse.Namespace) -> None:
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        config_path = Path(args.config).expanduser().resolve()
        checkpoint_dir = Path(args.checkpoint).expanduser().resolve()
        sidecar_dir = Path(args.sidecar_dir).expanduser().resolve()
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        training = config["training"]
        distributed = training["distributed"]
        data = training["data"]
        if world_size != int(distributed["num_gpus"]):
            raise ValueError(
                f"Runtime world_size={world_size} does not match config num_gpus={distributed['num_gpus']}"
            )
        if int(distributed["sp_size"]) != 1:
            raise ValueError("This validator currently requires sp_size=1")

        dataset, dataloader = build_parquet_streaming_style_dataloader(
            path=str(data["data_path"]),
            batch_size=int(data["train_batch_size"]),
            num_data_workers=int(data["dataloader_num_workers"]),
            parquet_schema=pyarrow_schema_t2v,
            manifest_path=str(data["streaming_manifest_path"]),
            cfg_rate=float(data["training_cfg_rate"]),
            text_padding_length=512,
            seed=int(data["seed"]),
            read_batch_size=int(data["streaming_read_batch_size"]),
            shuffle_row_groups=bool(data["streaming_shuffle_row_groups"]),
        )
        sidecar_path = sidecar_dir / f"dataloader_state_rank{rank}.pt"
        payload = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        expected_id = _next_manifest_id(dataset, payload["state"])

        os.environ["FASTVIDEO_DATALOADER_STATE_DIR"] = str(sidecar_dir)
        manager = CheckpointManager(
            method=object(),
            dataloader=dataloader,
            output_dir=str(checkpoint_dir.parent),
            config=CheckpointConfig(save_steps=0, keep_last=0),
        )
        manager._load_dataloader_snapshot(checkpoint_dir, int(args.expected_step))
        actual_id = str(next(iter(dataloader))["info_list"][0]["id"])
        if actual_id != expected_id:
            raise AssertionError(
                f"Rank {rank} resumed at {actual_id!r}; expected {expected_id!r}"
            )

        result = {
            "rank": rank,
            "sidecar": sidecar_path.name,
            "sidecar_sha256": _sha256(sidecar_path),
            "next_id": actual_id,
            "row_group_position": payload["state"]["row_group_position"],
            "row_offset": payload["state"]["row_offset"],
        }
        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, result)
        if rank == 0:
            receipt = {
                "schema": "fastvideo-streaming-dataloader-sidecar-validation/v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint": str(checkpoint_dir),
                "checkpoint_metadata_sha256": _sha256(checkpoint_dir / "metadata.json"),
                "config": str(config_path),
                "config_sha256": _sha256(config_path),
                "sidecar_dir": str(sidecar_dir),
                "step": int(args.expected_step),
                "world_size": world_size,
                "results": gathered,
            }
            _write_json_atomic(Path(args.output).expanduser().resolve(), receipt)
            print(json.dumps(receipt, indent=2, sort_keys=True))
        dist.barrier()
    finally:
        cleanup_dist_env_and_memory()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sidecar-dir", required=True)
    parser.add_argument("--expected-step", required=True, type=int)
    parser.add_argument("--output", required=True)
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
