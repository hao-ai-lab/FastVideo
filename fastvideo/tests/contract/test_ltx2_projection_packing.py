# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from torch import nn

from fastvideo.models.dits.ltx2 import (
    _init_attention_projections,
    _project_attention_inputs,
)


@pytest.mark.parametrize("context_dim", [None, 7], ids=["self_qkv", "cross_kv"])
def test_packed_attention_projections_match_split_forward_and_backward(
        context_dim: int | None) -> None:
    query_dim, inner_dim = 5, 3
    split, packed = nn.Module(), nn.Module()
    for module, pack in ((split, False), (packed, True)):
        _init_attention_projections(
            module,
            query_dim=query_dim,
            context_dim=context_dim,
            inner_dim=inner_dim,
            quant_config=None,
            prefix="attn",
            pack_attention_projections=pack,
        )
        module.double()

    torch.manual_seed(0)
    with torch.no_grad():
        for parameter in split.parameters():
            parameter.copy_(torch.randn_like(parameter))
        if context_dim is None:
            packed.to_qkv.weight.copy_(torch.cat(
                [split.to_q.weight, split.to_k.weight, split.to_v.weight]))
            packed.to_qkv.bias.copy_(torch.cat(
                [split.to_q.bias, split.to_k.bias, split.to_v.bias]))
        else:
            packed.to_q.load_state_dict(split.to_q.state_dict())
            packed.to_kv.weight.copy_(torch.cat(
                [split.to_k.weight, split.to_v.weight]))
            packed.to_kv.bias.copy_(torch.cat(
                [split.to_k.bias, split.to_v.bias]))
    for module in (split, packed):
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    split_x = torch.randn(2, 4, query_dim, dtype=torch.float64,
                          requires_grad=True)
    packed_x = split_x.detach().clone().requires_grad_(True)
    if context_dim is None:
        split_context, packed_context = split_x, packed_x
    else:
        split_context = torch.randn(2,
                                    3,
                                    context_dim,
                                    dtype=torch.float64,
                                    requires_grad=True)
        packed_context = split_context.detach().clone().requires_grad_(True)

    split_outputs = _project_attention_inputs(split, split_x, split_context)
    packed_outputs = _project_attention_inputs(packed, packed_x,
                                               packed_context)
    for actual, expected in zip(packed_outputs, split_outputs):
        torch.testing.assert_close(actual, expected)

    coefficients = (0.5, 1.5, -2.0)
    sum(coefficient * output.square().sum()
        for coefficient, output in zip(coefficients,
                                       split_outputs)).backward()
    sum(coefficient * output.square().sum()
        for coefficient, output in zip(coefficients,
                                       packed_outputs)).backward()
    torch.testing.assert_close(packed_x.grad, split_x.grad)
    if context_dim is not None:
        torch.testing.assert_close(packed_context.grad, split_context.grad)

    split_weight_grads = [
        split.to_q.weight.grad,
        split.to_k.weight.grad,
        split.to_v.weight.grad,
    ]
    split_bias_grads = [
        split.to_q.bias.grad,
        split.to_k.bias.grad,
        split.to_v.bias.grad,
    ]
    assert set(split.state_dict()) == {
        "to_q.weight",
        "to_q.bias",
        "to_k.weight",
        "to_k.bias",
        "to_v.weight",
        "to_v.bias",
    }
    if context_dim is None:
        packed_weight_grads = packed.to_qkv.weight.grad.chunk(3)
        packed_bias_grads = packed.to_qkv.bias.grad.chunk(3)
        assert set(packed.state_dict()) == {"to_qkv.weight", "to_qkv.bias"}
    else:
        torch.testing.assert_close(packed.to_q.weight.grad,
                                   split.to_q.weight.grad)
        torch.testing.assert_close(packed.to_q.bias.grad, split.to_q.bias.grad)
        packed_weight_grads = (packed.to_q.weight.grad,
                               *packed.to_kv.weight.grad.chunk(2))
        packed_bias_grads = (packed.to_q.bias.grad,
                             *packed.to_kv.bias.grad.chunk(2))
        assert set(packed.state_dict()) == {
            "to_q.weight",
            "to_q.bias",
            "to_kv.weight",
            "to_kv.bias",
        }
    for actual, expected in zip(packed_weight_grads, split_weight_grads):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(packed_bias_grads, split_bias_grads):
        torch.testing.assert_close(actual, expected)
    assert sum(parameter.numel() for parameter in packed.parameters()) == sum(
        parameter.numel() for parameter in split.parameters())


@pytest.mark.parametrize("context_dim", [None, 7])
def test_packed_attention_projections_reject_linear_quantization(
        context_dim: int | None) -> None:
    module = nn.Module()
    with pytest.raises(ValueError, match="do not yet support linear quantization"):
        _init_attention_projections(
            module,
            query_dim=5,
            context_dim=context_dim,
            inner_dim=3,
            quant_config=object(),  # type: ignore[arg-type]
            prefix="attn",
            pack_attention_projections=True,
        )
    assert not tuple(module.parameters())
