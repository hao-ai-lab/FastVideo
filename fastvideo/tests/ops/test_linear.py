# SPDX-License-Identifier: Apache-2.0

import torch

from fastvideo.layers.linear import ReplicatedLinear


def test_replicated_linear_shape_tracking_disabled_by_default():
    ReplicatedLinear.reset_shape_tracking()
    ReplicatedLinear.enable_shape_tracking = False

    layer = ReplicatedLinear(4, 6, bias=False)
    x = torch.randn(2, 4)

    layer(x)

    assert ReplicatedLinear.get_shape_mapping() == {}


def test_replicated_linear_shape_tracking_records_shapes_when_enabled():
    ReplicatedLinear.reset_shape_tracking()
    ReplicatedLinear.enable_shape_tracking = True

    try:
        layer = ReplicatedLinear(4, 6, bias=False, prefix="test.linear")
        x = torch.randn(2, 4)

        output, _ = layer(x)
        output_again, _ = layer(x)

        shape_mapping = ReplicatedLinear.get_shape_mapping()
        shape_key = (x.shape, output.shape)

        assert output.shape == output_again.shape
        assert shape_key in shape_mapping
        assert shape_mapping[shape_key] == ["ReplicatedLinear"]
    finally:
        ReplicatedLinear.enable_shape_tracking = False
        ReplicatedLinear.reset_shape_tracking()
