import unittest

import torch
import torch.nn as nn
from torch.testing import assert_close

from fastvideo.layers.quantization.absmax_fp8 import (
    AbsMaxFP8LinearMethod,
    AbsMaxFP8MergedParameter,
    AbsMaxFP8Parameter,
)


class TestAbsMaxFP8LinearMethod(unittest.TestCase):
    def test_convert_scale_none(self):
        method = AbsMaxFP8LinearMethod()
        scale = method._convert_scale(None)
        self.assertIsInstance(scale, AbsMaxFP8Parameter)
        self.assertEqual(scale.dtype, torch.float32)
        assert_close(scale, torch.tensor([1.0], dtype=torch.float32))

    def test_convert_scale_scalar(self):
        method = AbsMaxFP8LinearMethod()
        scale = method._convert_scale(2.5)
        self.assertIsInstance(scale, AbsMaxFP8Parameter)
        self.assertEqual(scale.dtype, torch.float32)
        assert_close(scale, torch.tensor([2.5], dtype=torch.float32))

    def test_convert_scale_rejects_non_float32(self):
        method = AbsMaxFP8LinearMethod()
        scale = torch.tensor([1.0], dtype=torch.float16)
        with self.assertRaisesRegex(NotImplementedError, "float32"):
            method._convert_scale(scale)

    def test_create_weights_rejects_invalid_dtype(self):
        method = AbsMaxFP8LinearMethod()
        layer = nn.Module()
        with self.assertRaisesRegex(AssertionError, "only supports"):
            method.create_weights(
                layer=layer,
                input_size_per_partition=2,
                output_partition_sizes=[3],
                input_size=2,
                output_size=3,
                params_dtype=torch.float32,
            )

    def test_absmax_fp8_parameter_weight_loader(self):
        param = AbsMaxFP8Parameter(torch.zeros(1), requires_grad=False)
        param.weight_loader(param, torch.tensor(3.0))
        assert_close(param, torch.tensor([3.0]))

    def test_absmax_fp8_merged_parameter_weight_loader(self):
        method = AbsMaxFP8LinearMethod()
        output_partition_sizes = [2, 3, 4]
        param = method._merged_placeholder(output_partition_sizes)
        self.assertIsInstance(param, AbsMaxFP8MergedParameter)
        param.weight_loader(param, torch.tensor(7.0), share_id="k")
        expected = torch.ones(sum(output_partition_sizes), dtype=torch.float32)
        expected[2:5] = 7.0
        assert_close(param, expected)

    def test_absmax_fp8_merged_parameter_rejects_invalid_share_id(self):
        method = AbsMaxFP8LinearMethod()
        param = method._merged_placeholder([2, 2, 2])
        with self.assertRaisesRegex(ValueError, "requires share_id"):
            param.weight_loader(param, torch.tensor(1.0), share_id="bad")

    def test_apply_matches_linear(self):
        method = AbsMaxFP8LinearMethod()
        layer = nn.Module()
        method.create_weights(
            layer=layer,
            input_size_per_partition=3,
            output_partition_sizes=[2],
            input_size=3,
            output_size=2,
            params_dtype=torch.float16,
        )
        weight_fp16 = torch.tensor(
            [[1.0, -2.0, 3.0], [4.0, 0.5, -1.5]], dtype=torch.float16
        )
        layer.weight.data = weight_fp16.to(dtype=torch.float8_e4m3fn)
        layer.scale_weight.data = torch.tensor([2.0, 3.0], dtype=torch.float32)
        layer.scale_input.data = torch.tensor([4.0], dtype=torch.float32)
        x = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.float16)
        expected = torch.nn.functional.linear(
            x * layer.scale_input.data.to(dtype=torch.float16),
            weight_fp16
            * layer.scale_weight.data.to(dtype=torch.float16).unsqueeze(1),
        ).to(dtype=torch.float16)
        output = method.apply(layer, x, bias=None)
        assert_close(output, expected)