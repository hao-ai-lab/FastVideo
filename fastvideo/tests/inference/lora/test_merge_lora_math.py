# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

import fastvideo.layers.lora.linear as lora_linear
from fastvideo.layers.lora.linear import BaseLayerWithLoRA


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    monkeypatch.setattr(lora_linear, "get_local_torch_device", lambda: torch.device("cpu"))


def _make_layer(in_dim=4, out_dim=3):
    base = nn.Linear(in_dim, out_dim, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.zeros(out_dim, in_dim))
    return BaseLayerWithLoRA(base)


def _ab():
    A = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    B = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    return A, B


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0, 2.0])
def test_merge_applies_strength(strength):
    A, B = _ab()
    layer = _make_layer()
    layer.set_lora_weights(A.clone(), B.clone(), lora_alpha=2, strength=strength, accumulate=False)
    expected = strength * (B @ A)
    torch.testing.assert_close(layer.base_layer.weight.data.cpu(), expected)


def test_merge_applies_alpha_scale():
    A, B = _ab()
    layer = _make_layer()
    layer.set_lora_weights(A.clone(), B.clone(), lora_alpha=4, strength=1.0, accumulate=False)
    expected = (4.0 / 2.0) * (B @ A)
    torch.testing.assert_close(layer.base_layer.weight.data.cpu(), expected)


def test_stacked_accumulate_sums_deltas():
    A1, B1 = _ab()
    A2 = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    B2 = torch.tensor([[0.0], [0.0], [3.0]])
    layer = _make_layer()
    layer.set_lora_weights(A1.clone(), B1.clone(), lora_alpha=2, strength=1.0, accumulate=False)
    layer.set_lora_weights(A2.clone(), B2.clone(), lora_alpha=1, strength=1.0, accumulate=True)
    expected = (B1 @ A1) + (B2 @ A2)
    torch.testing.assert_close(layer.base_layer.weight.data.cpu(), expected)
