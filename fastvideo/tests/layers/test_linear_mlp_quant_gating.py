# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for PR #1390's S2-1 plumbing finding: the dormant FP4 shape-tracking path must stay
gated off unless explicitly enabled, and the MLP quant_config=None default path must keep using ReplicatedLinear's
unquantized fallback.
"""
from __future__ import annotations

import torch

from fastvideo.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from fastvideo.layers.mlp import MLP


def test_replicated_linear_shape_tracking_default_off() -> None:
    ReplicatedLinear.reset_shape_tracking()
    assert ReplicatedLinear.enable_shape_tracking is False

    linear = ReplicatedLinear(input_size=8, output_size=4)
    linear(torch.randn(2, 8))

    assert len(ReplicatedLinear._shape_to_layer_types) == 0


def test_replicated_linear_shape_tracking_enabled_records_unique_shapes() -> None:
    ReplicatedLinear.reset_shape_tracking()
    ReplicatedLinear.enable_shape_tracking = True
    try:
        linear = ReplicatedLinear(input_size=8, output_size=4)
        linear(torch.randn(2, 8))
        linear(torch.randn(3, 8))

        assert len(ReplicatedLinear._shape_to_layer_types) == 2
        for layer_types in ReplicatedLinear._shape_to_layer_types.values():
            assert "ReplicatedLinear" in layer_types

        ReplicatedLinear.reset_shape_tracking()
        assert len(ReplicatedLinear._shape_to_layer_types) == 0
    finally:
        ReplicatedLinear.enable_shape_tracking = False


def test_mlp_quant_config_none_uses_unquantized_path() -> None:
    mlp = MLP(input_dim=8, mlp_hidden_dim=16)

    assert isinstance(mlp.fc_in.quant_method, UnquantizedLinearMethod)
    assert isinstance(mlp.fc_out.quant_method, UnquantizedLinearMethod)

    output = mlp.forward(torch.randn(2, 8))
    assert output.shape == (2, 8)
