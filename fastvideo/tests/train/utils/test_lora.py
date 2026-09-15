# SPDX-License-Identifier: Apache-2.0
"""Tests for modular-trainer LoRA synchronization."""

from __future__ import annotations

import torch

from fastvideo.layers.lora.linear import BaseLayerWithLoRA
from fastvideo.train.utils import lora as lora_utils


def test_synchronize_lora_gradients_sums_sp_and_averages_dp(monkeypatch, ) -> None:
    model = torch.nn.Sequential(
        BaseLayerWithLoRA(
            torch.nn.Linear(3, 2, bias=False),
            lora_rank=1,
            lora_alpha=1,
            training_mode=True,
        ), )
    layer = model[0]
    assert layer.lora_A is not None
    assert layer.lora_B is not None
    layer.lora_A.grad = torch.ones_like(layer.lora_A)
    layer.lora_B.grad = torch.full_like(layer.lora_B, 2.0)

    monkeypatch.setattr(lora_utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(lora_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(lora_utils.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(lora_utils, "get_dp_world_size", lambda: 2)
    all_reduce_calls: list[int] = []

    def _all_reduce(tensor: torch.Tensor, *, op: object) -> None:
        assert op == lora_utils.dist.ReduceOp.SUM
        all_reduce_calls.append(tensor.numel())
        tensor.mul_(4.0)

    monkeypatch.setattr(lora_utils.dist, "all_reduce", _all_reduce)

    lora_utils.synchronize_lora_gradients(model)

    assert all_reduce_calls == [
        layer.lora_A.numel() + layer.lora_B.numel()
    ]
    torch.testing.assert_close(
        layer.lora_A.grad,
        torch.full_like(layer.lora_A, 2.0),
    )
    torch.testing.assert_close(
        layer.lora_B.grad,
        torch.full_like(layer.lora_B, 4.0),
    )
