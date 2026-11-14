# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

from fastvideo import envs
from fastvideo.layers.linear import ReplicatedLinear
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


def test_stacked_accumulate_preserves_forward_across_unmerge_remerge():
    # Issue #2096: unmerging must preserve every active adapter's contribution.
    with (torch.no_grad(), torch._dynamo.config.patch(disable=True), envs.FASTVIDEO_LORA_SIDE_BRANCH.override(False)):
        dtype = torch.bfloat16
        base = ReplicatedLinear(1, 1, bias=False, params_dtype=dtype)
        base.weight.fill_(10)
        layer = BaseLayerWithLoRA(base)
        x = torch.tensor([[2]], dtype=dtype)
        A = torch.ones(1, 1, dtype=dtype)

        layer.set_lora_weights(A, torch.tensor([[2]], dtype=dtype))
        layer.set_lora_weights(A, torch.tensor([[5]], dtype=dtype), accumulate=True)
        outputs = [layer(x)[0].item()]

        layer.unmerge_lora_weights()
        outputs.append(layer(x)[0].item())

        layer.merge_lora_weights()
        outputs.append(layer(x)[0].item())

        # (base 10 + adapter X 2 + adapter Y 5) * input 2 = 34 in every state.
        assert outputs == [34.0, 34.0, 34.0], f"Merged, unmerged, remerged outputs: {outputs}"


@pytest.fixture
def scalar_stack():
    with (torch.no_grad(), torch._dynamo.config.patch(disable=True), envs.FASTVIDEO_LORA_SIDE_BRANCH.override(False)):
        dtype = torch.bfloat16
        base = ReplicatedLinear(1, 1, bias=False, params_dtype=dtype)
        base.weight.fill_(10)
        layer = BaseLayerWithLoRA(base)
        A = torch.ones(1, 1, dtype=dtype)
        layer.set_lora_weights(A, torch.tensor([[2]], dtype=dtype), lora_alpha=2, strength=0.5)
        layer.set_lora_weights(A, torch.tensor([[5]], dtype=dtype), accumulate=True)
        yield layer, torch.tensor([[2]], dtype=dtype)


@pytest.mark.parametrize("unmerge_before_update", [False, True])
@pytest.mark.parametrize("accumulate", [False, True])
def test_update_stacked_adapters_preserves_rank_scale_and_replacement(scalar_stack, unmerge_before_update, accumulate):
    layer, x = scalar_stack
    if unmerge_before_update:
        layer.unmerge_lora_weights()

    # A different rank and negative scale: B @ A = 2, alpha / rank * strength = -1.
    layer.set_lora_weights(torch.ones(2, 1, dtype=x.dtype),
                           torch.ones(1, 2, dtype=x.dtype),
                           lora_alpha=4,
                           strength=-0.5,
                           accumulate=accumulate)
    expected = 30.0 if accumulate else 16.0
    for _ in range(2):
        assert layer(x)[0].item() == expected
        layer.unmerge_lora_weights()
        assert layer.base_layer.weight.item() == 10.0
        assert layer(x)[0].item() == expected
        layer.merge_lora_weights()
        layer.merge_lora_weights()
        assert layer(x)[0].item() == expected
    assert layer.cpu_weight.item() == 10.0


@pytest.mark.parametrize("unmerge_before_update", [False, True])
def test_direct_accumulating_merge_preserves_stack(scalar_stack, unmerge_before_update):
    layer, x = scalar_stack
    if unmerge_before_update:
        layer.unmerge_lora_weights()

    layer.merge_lora_weights(accumulate=True)
    assert layer(x)[0].item() == 44.0
    layer.unmerge_lora_weights()
    assert layer(x)[0].item() == 44.0
    layer.merge_lora_weights()
    assert layer(x)[0].item() == 44.0


def test_unmerged_stack_supports_fullgraph_capture(scalar_stack):
    layer, x = scalar_stack
    layer.unmerge_lora_weights()
    with torch._dynamo.config.patch(disable=False):
        compiled = torch.compile(layer, backend="eager", fullgraph=True)
        assert compiled(x)[0].item() == 34.0


@pytest.mark.parametrize("change, expected", [("strength", 29.0), ("factors", 36.0)])
def test_unmerged_stack_uses_live_current_adapter(scalar_stack, change, expected):
    layer, x = scalar_stack
    layer.unmerge_lora_weights()
    if change == "strength":
        layer.lora_strength = 0.5
    else:
        layer.lora_B.fill_(6)
    assert layer(x)[0].item() == expected
