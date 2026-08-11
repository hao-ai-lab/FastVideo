# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the sharded base-weight cache.

Runs on a single-process gloo group with a (1, 1) CPU device mesh: DTensor
round-trip through write/load, the FQN reconciliation matrix (allowed
zero-init params, disallowed extras, shape mismatches), and the
never-fail-the-run contract.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from fastvideo.models.loader.shard_cache import (
    ShardCacheContext,
    try_load_from_shard_cache,
    write_shard_cache,
)


@pytest.fixture(scope="module")
def cpu_mesh():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29581")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    return init_device_mesh("cpu", (1, 1), mesh_dim_names=("replicate", "shard"))


def _make_model(cpu_mesh, *, extra_param: str | None = None, weight_rows: int = 8) -> nn.Module:
    model = nn.Module()
    placements = (Replicate(), Shard(0))
    weight = distribute_tensor(torch.randn(weight_rows, 4), cpu_mesh, placements)
    bias = distribute_tensor(torch.randn(weight_rows), cpu_mesh, placements)
    model.register_parameter("weight", nn.Parameter(weight))
    model.register_parameter("bias", nn.Parameter(bias))
    model.register_buffer("scale", torch.full((1, ), 2.0))
    if extra_param is not None:
        extra = distribute_tensor(torch.randn(4, 4), cpu_mesh, placements)
        model.register_parameter(extra_param.replace(".", "_"), nn.Parameter(extra))
        # register under the dotted name via a child module for realism
    model.reverse_param_names_mapping = {"weight": ("hf.weight", None, None)}
    return model


def _ctx(tmp_path) -> ShardCacheContext:
    return ShardCacheContext(entry_dir=tmp_path / "entry", key="testkey", shard_index=0, num_shards=1, is_writer=True)


def test_round_trip_restores_tensors_and_reverse_mapping(cpu_mesh, tmp_path):
    src = _make_model(cpu_mesh)
    ctx = _ctx(tmp_path)
    write_shard_cache(src, ctx)

    dst = _make_model(cpu_mesh)
    with torch.no_grad():
        dst.weight.mul_(0)
        dst.bias.mul_(0)
    dst.reverse_param_names_mapping = {}
    assert try_load_from_shard_cache(dst, ctx, torch.device("cpu"))
    assert torch.equal(dst.weight.to_local(), src.weight.to_local())
    assert torch.equal(dst.bias.to_local(), src.bias.to_local())
    assert torch.equal(dst.scale, src.scale)
    assert dst.reverse_param_names_mapping == {"weight": ("hf.weight", None, None)}


def test_allowed_new_param_zero_inits_on_hit(cpu_mesh, tmp_path):
    src = _make_model(cpu_mesh)
    ctx = _ctx(tmp_path)
    write_shard_cache(src, ctx)

    dst = _make_model(cpu_mesh)
    gate = distribute_tensor(torch.randn(4, 4), cpu_mesh, (Replicate(), Shard(0)))
    dst.register_parameter("to_gate_compress", nn.Parameter(gate))
    assert try_load_from_shard_cache(dst, ctx, torch.device("cpu"))
    assert torch.equal(dst.to_gate_compress.to_local(), torch.zeros(4, 4))


def test_disallowed_missing_param_misses_without_mutation(cpu_mesh, tmp_path):
    src = _make_model(cpu_mesh)
    ctx = _ctx(tmp_path)
    write_shard_cache(src, ctx)

    dst = _make_model(cpu_mesh)
    mystery = distribute_tensor(torch.randn(4, 4), cpu_mesh, (Replicate(), Shard(0)))
    dst.register_parameter("mystery", nn.Parameter(mystery))
    before = dst.weight.to_local().clone()
    assert not try_load_from_shard_cache(dst, ctx, torch.device("cpu"))
    assert torch.equal(dst.weight.to_local(), before)


def test_shape_mismatch_misses(cpu_mesh, tmp_path):
    src = _make_model(cpu_mesh)
    ctx = _ctx(tmp_path)
    write_shard_cache(src, ctx)

    dst = _make_model(cpu_mesh, weight_rows=9)
    assert not try_load_from_shard_cache(dst, ctx, torch.device("cpu"))


def test_missing_entry_misses_cleanly(cpu_mesh, tmp_path):
    dst = _make_model(cpu_mesh)
    ctx = ShardCacheContext(entry_dir=tmp_path / "absent", key="k2", shard_index=0, num_shards=1, is_writer=True)
    assert not try_load_from_shard_cache(dst, ctx, torch.device("cpu"))
