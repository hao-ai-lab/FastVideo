# SPDX-License-Identifier: Apache-2.0
"""CPU contract: merging an adapter stored wider than the base rounds once, after a wide add."""

from __future__ import annotations

import pytest
import torch

from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.lora import linear as lora_linear
from fastvideo.layers.lora.linear import get_lora_layer

IN, OUT, RANK = 32, 24, 16


@pytest.fixture(autouse=True)
def _cpu_merge(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    monkeypatch.setattr(lora_linear, "get_local_torch_device", lambda: torch.device("cpu"))


def _merged(adapter_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    layer = ReplicatedLinear(IN, OUT, bias=False, params_dtype=torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(OUT, IN))
    base = layer.weight.detach().clone()
    wrapped = get_lora_layer(layer)
    a, b = torch.randn(RANK, IN, dtype=adapter_dtype), torch.randn(OUT, RANK, dtype=adapter_dtype) * 0.1
    wrapped.set_lora_weights(a, b, lora_alpha=RANK, strength=1.0)
    wrapped.merge_lora_weights()
    return wrapped.base_layer.weight.detach(), base, a, b


def test_fp32_adapter_is_added_in_fp32_and_rounded_once() -> None:
    merged, base, a, b = _merged(torch.float32)
    expected = (base.float() + b @ a).to(torch.bfloat16)
    assert merged.dtype == torch.bfloat16
    assert torch.equal(merged, expected)


def test_bf16_adapter_keeps_the_bf16_merge() -> None:
    merged, base, a, b = _merged(torch.bfloat16)
    assert torch.equal(merged, base + b @ a)
