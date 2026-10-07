# SPDX-License-Identifier: Apache-2.0
"""The rank that reads a checkpoint shard must be the rank that broadcasts it.

safetensors_weights_iterator broadcasts every tensor from node-group rank 0.
The local rank is the CUDA device ordinal, and a node can have no rank on
device 0 (for example, Ray workers on GPUs 2 and 3 of a 4-GPU node). The
reader must therefore be node-group rank 0, not the rank with local rank 0.
"""
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import fastvideo.models.loader.weight_utils as weight_utils

WEIGHT = torch.arange(6, dtype=torch.float32)
RECEIVED = torch.full((6, ), -1.0)


@pytest.fixture
def shard(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"w": WEIGHT}, str(path))
    return str(path)


def _load(monkeypatch, shard, rank_in_group, local_rank):
    group = SimpleNamespace(rank_in_group=rank_in_group, local_rank=local_rank, world_size=2, device_group=object())
    monkeypatch.setattr(weight_utils, "_get_initialized_node_group", lambda: group)
    monkeypatch.setattr(weight_utils.parallel_state, "get_local_torch_device", lambda: torch.device("cpu"))
    sent = []

    def fake_broadcast(tensor, src, group=None, async_op=False):
        if rank_in_group == 0:
            sent.append(tensor.clone())
        else:
            tensor.copy_(RECEIVED)

    monkeypatch.setattr(weight_utils.dist, "broadcast", fake_broadcast)
    monkeypatch.setattr(weight_utils.dist, "get_global_rank", lambda group, rank: 0)
    return dict(weight_utils.safetensors_weights_iterator([shard])), sent


def test_group_rank_zero_reads_even_when_its_device_is_not_zero(monkeypatch, shard):
    loaded, sent = _load(monkeypatch, shard, rank_in_group=0, local_rank=2)
    assert torch.equal(loaded["w"], WEIGHT)
    assert len(sent) == 1 and torch.equal(sent[0], WEIGHT)


def test_other_ranks_receive_even_when_their_device_is_zero(monkeypatch, shard):
    loaded, sent = _load(monkeypatch, shard, rank_in_group=1, local_rank=0)
    assert torch.equal(loaded["w"], RECEIVED)
    assert sent == []
