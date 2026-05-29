# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for PR #1390's S2-1 plumbing finding: the dormant FP4 shape-tracking path must stay
gated off unless explicitly enabled, and the MLP quant_config=None default path must keep using ReplicatedLinear's
unquantized fallback.
"""
from __future__ import annotations

from typing import cast

import torch

from fastvideo.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from fastvideo.layers.mlp import MLP

ShapeTrackingKey = tuple[object, object]


def _tracked_unique_shapes() -> set[ShapeTrackingKey]:
    return cast(set[ShapeTrackingKey], getattr(ReplicatedLinear, "_unique_shapes"))


def _tracked_layer_types_by_shape() -> dict[ShapeTrackingKey, list[str]]:
    return cast(dict[ShapeTrackingKey, list[str]], getattr(ReplicatedLinear, "_shape_to_layer_types"))


def test_replicated_linear_shape_tracking_default_off() -> None:
    ReplicatedLinear.reset_shape_tracking()
    assert ReplicatedLinear.enable_shape_tracking is False

    linear = ReplicatedLinear(input_size=8, output_size=4)
    linear(torch.randn(2, 8))

    assert len(_tracked_unique_shapes()) == 0
    assert len(_tracked_layer_types_by_shape()) == 0


def test_replicated_linear_shape_tracking_enabled_records_unique_shapes() -> None:
    ReplicatedLinear.reset_shape_tracking()
    ReplicatedLinear.enable_shape_tracking = True
    try:
        linear = ReplicatedLinear(input_size=8, output_size=4)
        linear(torch.randn(2, 8))
        linear(torch.randn(3, 8))

        assert len(_tracked_unique_shapes()) == 2
        assert len(_tracked_layer_types_by_shape()) == 2
        for layer_types in _tracked_layer_types_by_shape().values():
            assert "ReplicatedLinear" in layer_types

        ReplicatedLinear.reset_shape_tracking()
        assert len(_tracked_unique_shapes()) == 0
        assert len(_tracked_layer_types_by_shape()) == 0
    finally:
        ReplicatedLinear.enable_shape_tracking = False


def test_mlp_quant_config_none_uses_unquantized_path() -> None:
    mlp = MLP(input_dim=8, mlp_hidden_dim=16)

    assert isinstance(mlp.fc_in.quant_method, UnquantizedLinearMethod)
    assert isinstance(mlp.fc_out.quant_method, UnquantizedLinearMethod)

    output = mlp.forward(torch.randn(2, 8))
    assert output.shape == (2, 8)
