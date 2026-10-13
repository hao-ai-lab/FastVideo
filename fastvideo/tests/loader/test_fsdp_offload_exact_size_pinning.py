# SPDX-License-Identifier: Apache-2.0
"""FSDP CPU offload must pin shards at their exact size (one GPU required).

``CPUOffloadPolicy(pin_memory=True)`` pins each shard through PyTorch's caching
host allocator, which rounds every allocation up to a power of two. For the
non-power-of-two weights of real encoders and DiTs that costs up to ~2x the
model size in page-locked host memory. ``shard_model`` instead registers the
shards in exact-size ``PinnedTensorArena`` allocations, for both loader orders:
weights loaded before sharding (text encoders) and sharding on meta followed by
``load_model_from_full_model_state_dict`` (DiTs).
"""
import copy

import pytest
import torch
from torch import nn
from torch.distributed import init_device_mesh
from torch.distributed.tensor import DTensor

from fastvideo.models.loader.fsdp_load import load_model_from_full_model_state_dict, shard_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA host registration requires one GPU")

_SHAPES = ((96, 160), (160, 96), (96, 40))


class _Block(nn.Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.proj(x))


class _Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block(i, o) for i, o in _SHAPES[:-1])
        self.head = nn.Linear(*_SHAPES[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def _shard_conditions():
    return [lambda name, module: isinstance(module, _Block)]


def _offload_mesh():
    return init_device_mesh("cuda", mesh_shape=(1, 1), mesh_dim_names=("offload", "replicate"))["offload"]


def _fsdp_params(model: nn.Module):
    for module in model.modules():
        if hasattr(module, "_get_fsdp_state"):
            for group in module._get_fsdp_state()._fsdp_param_groups:
                yield from group.fsdp_params


def _assert_exact_size_pinned(model: nn.Module) -> None:
    arenas = {}
    logical_bytes = 0
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), name
        local = param.to_local()
        assert local.device.type == "cpu", name
        assert local.is_pinned(), f"{name} is not pinned"
    for fsdp_param in _fsdp_params(model):
        data = fsdp_param._sharded_param_data
        assert data.is_pinned()
        assert fsdp_param.sharded_param._local_tensor.data_ptr() == data.data_ptr()
        arena = getattr(data, "_pinned_arena", None)
        assert arena is not None, "shard was pinned by the power-of-two caching host allocator"
        assert arena.buffer is not None, "This GPU must exercise registration, not the fallback"
        arenas[id(arena)] = arena
        logical_bytes += data.numel() * data.element_size()
    registered = sum(arena.buffer.numel() for arena in arenas.values())
    assert registered <= logical_bytes + len(arenas) * 4096 + 256 * len(list(_fsdp_params(model)))


@torch.no_grad()
def test_weights_loaded_before_sharding_are_pinned_at_exact_size(distributed_setup):
    torch.manual_seed(0)
    model = _Model().to(torch.bfloat16)
    reference = copy.deepcopy(model).cuda()
    x = torch.randn(4, _SHAPES[0][0], device="cuda", dtype=torch.bfloat16)
    expected = reference(x)

    shard_model(model, cpu_offload=True, mesh=_offload_mesh(), fsdp_shard_conditions=_shard_conditions(),
                pin_cpu_memory=True)

    _assert_exact_size_pinned(model)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    # FSDP's lazy init re-validates sharded params on the first forward; the arena must survive it.
    _assert_exact_size_pinned(model)


@torch.no_grad()
def test_weights_loaded_after_sharding_on_meta_are_pinned_at_exact_size(distributed_setup):
    torch.manual_seed(0)
    source = _Model().to(torch.bfloat16)
    x = torch.randn(4, _SHAPES[0][0], device="cuda", dtype=torch.bfloat16)
    expected = copy.deepcopy(source).cuda()(x)

    with torch.device("meta"):
        model = _Model().to(torch.bfloat16)
    shard_model(model, cpu_offload=True, mesh=_offload_mesh(), fsdp_shard_conditions=_shard_conditions(),
                pin_cpu_memory=True)
    load_model_from_full_model_state_dict(
        model,
        iter(list(source.state_dict().items())),
        torch.device("cuda"),
        torch.bfloat16,
        strict=True,
        cpu_offload=True,
        param_names_mapping=lambda name: (name, None, None),
        training_mode=False,
    )

    _assert_exact_size_pinned(model)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    # FSDP's lazy init re-validates sharded params on the first forward; the arena must survive it.
    _assert_exact_size_pinned(model)


@torch.no_grad()
def test_pin_cpu_memory_false_keeps_pageable_shards(distributed_setup):
    model = _Model().to(torch.bfloat16)
    shard_model(model, cpu_offload=True, mesh=_offload_mesh(), fsdp_shard_conditions=_shard_conditions(),
                pin_cpu_memory=False)
    for name, param in model.named_parameters():
        assert param.to_local().device.type == "cpu", name
        assert not param.to_local().is_pinned(), name
