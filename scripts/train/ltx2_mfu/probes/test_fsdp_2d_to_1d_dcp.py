#!/usr/bin/env python3
"""Smoke a FastVideo-style FSDP2 2-D-to-1-D DCP resume."""

from __future__ import annotations

import argparse
import gc
import os
import shutil
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from fastvideo.training.checkpointing_utils import ModelWrapper, OptimizerWrapper


def _local_clone(value: Any) -> Any:
    if isinstance(value, DTensor):
        value = value.to_local()
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _local_clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_local_clone(item) for item in value)
    return value


def _assert_equal(expected: Any, actual: Any, path: str = "state") -> None:
    if isinstance(expected, torch.Tensor):
        if isinstance(actual, DTensor):
            actual = actual.to_local()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=lambda msg: f"{path}: {msg}")
        return
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), path
        for key in expected:
            _assert_equal(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual), path
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _assert_equal(left, right, f"{path}[{index}]")
        return
    assert actual == expected, path


def _build(mesh, seed: int):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 8),
    ).cuda()
    fully_shard(model, mesh=mesh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    return model, optimizer


def _step(model, optimizer) -> torch.Tensor:
    inputs = torch.arange(128, device="cuda", dtype=torch.float32).reshape(8, 16) / 128
    targets = torch.arange(64, device="cuda", dtype=torch.float32).reshape(8, 8) / 64
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4

    if rank == 0:
        shutil.rmtree(args.checkpoint, ignore_errors=True)
    dist.barrier()

    old_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(1, world_size),
        mesh_dim_names=("replicate", "shard"),
    )
    old_model, old_optimizer = _build(old_mesh, seed=11)
    first_loss = _step(old_model, old_optimizer)
    old_model_wrapper = ModelWrapper(old_model)
    old_optimizer_wrapper = OptimizerWrapper(old_model, old_optimizer)
    expected_model = _local_clone(old_model_wrapper.state_dict())
    expected_optimizer = _local_clone(old_optimizer_wrapper.state_dict())
    dcp.save(
        {"model": old_model_wrapper, "optimizer": old_optimizer_wrapper},
        checkpoint_id=args.checkpoint,
    )

    del old_model_wrapper, old_optimizer_wrapper, old_optimizer, old_model
    gc.collect()
    torch.cuda.empty_cache()

    new_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size, ),
        mesh_dim_names=("shard", ),
    )
    new_model, new_optimizer = _build(new_mesh, seed=99)
    _step(new_model, new_optimizer)
    new_model_wrapper = ModelWrapper(new_model)
    new_optimizer_wrapper = OptimizerWrapper(new_model, new_optimizer)
    dcp.load(
        {"model": new_model_wrapper, "optimizer": new_optimizer_wrapper},
        checkpoint_id=args.checkpoint,
    )

    _assert_equal(expected_model, new_model_wrapper.state_dict(), "model")
    _assert_equal(expected_optimizer, new_optimizer_wrapper.state_dict(), "optimizer")
    resumed_loss = _step(new_model, new_optimizer)
    assert torch.isfinite(resumed_loss)

    if rank == 0:
        print({
            "world_size": world_size,
            "first_loss": float(first_loss),
            "resumed_loss": float(resumed_loss),
            "old_mesh": [1, world_size],
            "new_mesh": [world_size],
            "model_optimizer_state_exact": True,
        })
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
