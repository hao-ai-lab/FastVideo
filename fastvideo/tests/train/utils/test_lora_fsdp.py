# SPDX-License-Identifier: Apache-2.0
"""Distributed ownership contracts for training-side LoRA adapters."""

from __future__ import annotations

import os
from pathlib import Path
import socket

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed._composable.fsdp import FSDPModule, fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3Config
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.lora.linear import BaseLayerWithLoRA
from fastvideo.models.loader.fsdp_load import load_model_from_full_model_state_dict
from fastvideo.models.loader.utils import get_param_names_mapping
from fastvideo.train.utils.lora import enable_lora_training, finalize_lora_training
from fastvideo.training.checkpointing_utils import ModelWrapper


class _ArchConfig:
    exclude_lora_layers: list[str] = []


class _Config:
    arch_config = _ArchConfig()


class _TinyBlock(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.to_q = ReplicatedLinear(4, 4, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.to_q(value)[0]


class _TinyTransformer(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.block = _TinyBlock()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


def _prepare_lora(model: _TinyTransformer) -> None:
    converted = enable_lora_training(
        model,
        lora_rank=2,
        lora_alpha=2,
        lora_target_modules=["to_q"],
        prepare_for_fsdp=True,
        initialization_seed=42,
    )
    assert converted == 1


def test_meta_checkpoint_load_aliases_base_and_initializes_adapters() -> None:
    with torch.device("meta"):
        model = _TinyTransformer()
    _prepare_lora(model)
    base_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    incompatible = load_model_from_full_model_state_dict(
        model,
        iter([("block.to_q.weight", base_weight)]),
        torch.device("cpu"),
        torch.float32,
        strict=True,
        param_names_mapping=lambda name: (name, None, None),
    )

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    wrapper = model.block.to_q
    assert isinstance(wrapper, BaseLayerWithLoRA)
    torch.testing.assert_close(wrapper.base_layer.weight, base_weight)
    assert wrapper.cpu_weight is None
    assert wrapper.lora_A is not None and bool(torch.count_nonzero(wrapper.lora_A))
    assert wrapper.lora_B is not None and not bool(torch.count_nonzero(wrapper.lora_B))
    assert model.reverse_param_names_mapping["block.to_q.base_layer.weight"][0] == "block.to_q.weight"


def test_h3_hf_mapping_is_applied_before_lora_checkpoint_alias() -> None:
    class _Attention(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.to_out = ReplicatedLinear(4, 4, bias=False)

    class _MappedBlock(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.attn = _Attention()

    class _MappedTransformer(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.config = _Config()
            self.block = _MappedBlock()

    with torch.device("meta"):
        model = _MappedTransformer()
    assert enable_lora_training(
        model,
        lora_rank=2,
        lora_target_modules=["attn.to_out"],
        prepare_for_fsdp=True,
    ) == 1
    base_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    mapping = get_param_names_mapping(MiniMaxH3Config().arch_config.param_names_mapping)

    incompatible = load_model_from_full_model_state_dict(
        model,
        iter([("block.attn.to_out.0.weight", base_weight)]),
        torch.device("cpu"),
        torch.float32,
        strict=True,
        param_names_mapping=mapping,
    )

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    torch.testing.assert_close(model.block.attn.to_out.base_layer.weight, base_weight)
    assert model.reverse_param_names_mapping["block.attn.to_out.base_layer.weight"][0] == (
        "block.attn.to_out.0.weight")


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _materialize_adapter_initialization(model: _TinyTransformer) -> None:
    initializer = model._fastvideo_missing_parameter_initializer
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            initialized = initializer(name, parameter.shape, parameter.dtype)
            if initialized is not None:
                parameter.copy_(initialized)


def _assert_matches_rank_zero(value: torch.Tensor) -> None:
    reference = value.detach().clone()
    dist.broadcast(reference, src=0)
    torch.testing.assert_close(value, reference, rtol=0.0, atol=0.0)


def _fsdp_worker(rank: int, port: int, checkpoint_path: str) -> None:
    os.environ.update({
        "RANK": str(rank),
        "WORLD_SIZE": "2",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "TORCHDYNAMO_DISABLE": "1",
    })
    dist.init_process_group("gloo")
    try:
        torch.manual_seed(7)
        model = _TinyTransformer()
        _prepare_lora(model)
        _materialize_adapter_initialization(model)

        torch.manual_seed(7)
        reference = _TinyTransformer()
        _prepare_lora(reference)
        _materialize_adapter_initialization(reference)
        finalize_lora_training(reference)

        mesh = init_device_mesh("cpu", (2, ))
        fully_shard(model.block, mesh=mesh)
        fully_shard(model, mesh=mesh)
        assert finalize_lora_training(model) == 1

        wrapper = model.block.to_q
        assert isinstance(model.block, FSDPModule)
        assert isinstance(wrapper, BaseLayerWithLoRA)
        assert isinstance(wrapper.lora_A, DTensor)
        assert isinstance(wrapper.lora_B, DTensor)
        assert wrapper.lora_A.placements == (Shard(0), )
        assert wrapper.lora_B.placements == (Shard(0), )
        _assert_matches_rank_zero(wrapper.lora_A.full_tensor())
        _assert_matches_rank_zero(wrapper.lora_B.full_tensor())

        optimizer = torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.1)
        reference_optimizer = torch.optim.SGD(
            [parameter for parameter in reference.parameters() if parameter.requires_grad],
            lr=0.1,
        )
        for _ in range(2):
            local_input = torch.ones(2, 4) * (rank + 1)
            model(local_input).sum().backward()

            # FSDP averages the two rank-local losses.  Compare both adapter
            # gradients to that exact unsharded objective, not just each other.
            ((reference(torch.ones(2, 4)).sum() + reference(torch.ones(2, 4) * 2).sum()) / 2).backward()
            for name in ("lora_A", "lora_B"):
                distributed_parameter = getattr(wrapper, name)
                reference_parameter = getattr(reference.block.to_q, name)
                full_gradient = distributed_parameter.grad.full_tensor()
                _assert_matches_rank_zero(full_gradient)
                torch.testing.assert_close(full_gradient, reference_parameter.grad, rtol=1e-6, atol=1e-6)

            optimizer.step()
            reference_optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)

        torch.testing.assert_close(wrapper.lora_A.full_tensor(), reference.block.to_q.lora_A, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(wrapper.lora_B.full_tensor(), reference.block.to_q.lora_B, rtol=1e-6, atol=1e-6)

        expected_a = wrapper.lora_A.full_tensor().detach().clone()
        expected_b = wrapper.lora_B.full_tensor().detach().clone()
        checkpoint_state = {"transformer": ModelWrapper(model)}
        dcp.save(checkpoint_state, checkpoint_id=checkpoint_path)
        with torch.no_grad():
            wrapper.lora_A.zero_()
            wrapper.lora_B.zero_()
        dcp.load(checkpoint_state, checkpoint_id=checkpoint_path)
        torch.testing.assert_close(wrapper.lora_A.full_tensor(), expected_a, rtol=0.0, atol=0.0)
        torch.testing.assert_close(wrapper.lora_B.full_tensor(), expected_b, rtol=0.0, atol=0.0)
    finally:
        dist.destroy_process_group()


def test_two_rank_fsdp_owns_synchronizes_and_resumes_lora_gradients(tmp_path: Path) -> None:
    """Exercise real two-rank FSDP/DTensor ownership and DCP resume."""
    mp.start_processes(
        _fsdp_worker,
        args=(_free_port(), str(tmp_path / "dcp")),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
