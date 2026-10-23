# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for the modular trainer's DDP strategy."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from fastvideo.train.utils.distributed_strategy import (
    build_replicated_model_from_scratch,
    normalize_distributed_strategy,
    unwrap_ddp_module,
    wrap_module_ddp,
)
from fastvideo.training.checkpointing_utils import ModelWrapper


class _TinyModel(torch.nn.Module):

    def __init__(self, width: int = 2) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(width, width)
        self.public_value = 17

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)

    def helper(self) -> int:
        return self.public_value


@pytest.fixture
def local_process_group(tmp_path: Path):
    owns_group = not dist.is_initialized()
    if owns_group:
        rendezvous = tmp_path / "ddp_rendezvous"
        dist.init_process_group(
            "gloo",
            init_method=f"file://{rendezvous}",
            rank=0,
            world_size=1,
        )
    try:
        yield
    finally:
        if owns_group:
            dist.destroy_process_group()


def test_strategy_normalization() -> None:
    assert normalize_distributed_strategy(None) == "fsdp"
    assert normalize_distributed_strategy(" DDP ") == "ddp"
    with pytest.raises(ValueError, match="strategy must be one of"):
        normalize_distributed_strategy("deepspeed")


def test_replicated_initialization_is_seeded() -> None:
    first = build_replicated_model_from_scratch(
        _TinyModel,
        {"width": 3},
        device=torch.device("cpu"),
        default_dtype=torch.float32,
        seed=123,
    )
    second = build_replicated_model_from_scratch(
        _TinyModel,
        {"width": 3},
        device=torch.device("cpu"),
        default_dtype=torch.float32,
        seed=123,
    )
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(left, right)


def test_ddp_adapter_forward_delegation_and_checkpoint(
    local_process_group,
) -> None:
    del local_process_group
    wrapped = wrap_module_ddp(
        _TinyModel(),
        device=torch.device("cpu"),
        broadcast_buffers=False,
    )
    assert wrapped.helper() == 17
    assert unwrap_ddp_module(wrapped) is wrapped.module

    output = wrapped(torch.ones(2, 2)).sum()
    output.backward()
    assert all(parameter.grad is not None for parameter in wrapped.parameters())

    # torch.distributed.checkpoint canonicalizes DDP's ``module.`` prefix.
    # ModelWrapper must apply the same canonicalization when filtering out
    # frozen parameters, otherwise it silently emits an empty checkpoint.
    state = ModelWrapper(wrapped).state_dict()
    assert set(state) == {"linear.weight", "linear.bias"}
