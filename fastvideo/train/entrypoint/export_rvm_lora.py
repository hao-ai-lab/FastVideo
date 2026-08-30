# SPDX-License-Identifier: Apache-2.0
"""Export a sharded FastH3 RVM checkpoint as an inference LoRA safetensors file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file

from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
from fastvideo.layers.lora.linear import BaseLayerWithLoRA
from fastvideo.train.utils.builder import build_from_config
from fastvideo.train.utils.config import load_run_config
from fastvideo.training.checkpointing_utils import ModelWrapper


def materialize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(tensor, "full_tensor"):
        tensor = tensor.full_tensor()
    elif hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    return tensor.detach().float().cpu().contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--sp-size", type=int, default=None)
    parser.add_argument("--hsdp-replicate-dim", type=int, default=None)
    parser.add_argument("--hsdp-shard-dim", type=int, default=None)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", args.num_gpus))
    if world_size != args.num_gpus:
        raise ValueError(f"torchrun WORLD_SIZE={world_size} does not match --num-gpus={args.num_gpus}")
    sp_size = args.sp_size or args.num_gpus
    replicate = args.hsdp_replicate_dim or max(1, args.num_gpus // sp_size)
    shard = args.hsdp_shard_dim or sp_size

    cfg = load_run_config(
        args.config,
        overrides=[
            "--training.distributed.num_gpus",
            str(args.num_gpus),
            "--training.distributed.sp_size",
            str(sp_size),
            "--training.distributed.tp_size",
            "1",
            "--training.distributed.hsdp_replicate_dim",
            str(replicate),
            "--training.distributed.hsdp_shard_dim",
            str(shard),
        ],
    )
    maybe_init_distributed_environment_and_model_parallel(1, sp_size)
    _tc, method, _dataloader, _start = build_from_config(cfg)

    checkpoint = args.checkpoint.resolve()
    dcp_dir = checkpoint / "dcp" if checkpoint.name != "dcp" else checkpoint
    if not (dcp_dir / ".metadata").is_file():
        raise FileNotFoundError(f"No complete DCP checkpoint found at {dcp_dir}")
    dcp.load(
        {"roles.student.transformer": ModelWrapper(method.student.transformer)},
        checkpoint_id=str(dcp_dir),
    )

    tensors: dict[str, torch.Tensor] = {}
    layer_count = 0
    for name, module in method.student.transformer.named_modules():
        if not isinstance(module, BaseLayerWithLoRA):
            continue
        if module.lora_A is None or module.lora_B is None:
            continue
        tensors[f"{name}.lora_A"] = materialize_tensor(module.lora_A)
        tensors[f"{name}.lora_B"] = materialize_tensor(module.lora_B)
        tensors[f"{name}.lora_alpha"] = torch.tensor(
            int(module.lora_alpha or module.lora_rank or 1), dtype=torch.int64
        )
        layer_count += 1
    if not tensors:
        raise RuntimeError("The checkpoint contained no trainable LoRA tensors")

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            tensors,
            str(args.output),
            metadata={
                "format": "fastvideo-rvm-lora",
                "base_model": str(cfg.training.model_path),
                "source_checkpoint": str(checkpoint),
            },
        )
        manifest: dict[str, Any] = {
            "adapter": str(args.output),
            "source_checkpoint": str(checkpoint),
            "base_model": str(cfg.training.model_path),
            "lora_layers": layer_count,
            "tensor_keys": len(tensors),
        }
        manifest_path = args.output.with_suffix(args.output.suffix + ".json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2))
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
