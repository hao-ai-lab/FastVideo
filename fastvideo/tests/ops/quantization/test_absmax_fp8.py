import unittest

import torch
import torch.nn as nn
from torch.testing import assert_close

from fastvideo.layers.quantization.absmax_fp8 import (
    AbsMaxFP8Config,
    AbsMaxFP8LinearMethod,
    AbsMaxFP8MergedParameter,
    AbsMaxFP8Parameter,
    _quantize_input_dynamic,
    _quantize_input_static,
    _supports_fp8_compute,
    _FP8_DTYPE,
    _FP8_MAX,
)
_HAVE_FP8_GPU = torch.cuda.is_available() and (
    torch.cuda.get_device_capability()[0] > 8
    or (
        torch.cuda.get_device_capability()[0] == 8
        and torch.cuda.get_device_capability()[1] >= 9
    )
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
                params_dtype=torch.int8,
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

    def test_apply_dequant_matches_linear(self):
        """Dequant fallback path produces the same result as manual dequant."""
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
        layer.weight_scale.data = torch.tensor([2.0, 3.0], dtype=torch.float32)
        layer.input_scale.data = torch.tensor([4.0], dtype=torch.float32)
        x = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.float16)
        expected = torch.nn.functional.linear(
            x * layer.input_scale.data.to(dtype=torch.float16),
            weight_fp16
            * layer.weight_scale.data.to(dtype=torch.float16).unsqueeze(1),
        ).to(dtype=torch.float16)
        output = method._apply_dequant(layer, x, bias=None)
        assert_close(output, expected)


class TestFP8Helpers(unittest.TestCase):
    def test_quantize_input_dynamic_roundtrip(self):
        x = torch.randn(4, 8, dtype=torch.float32)
        x_fp8, scale = _quantize_input_dynamic(x)
        self.assertEqual(x_fp8.dtype, _FP8_DTYPE)
        self.assertEqual(scale.dtype, torch.float32)
        # Reconstructed values should be close to originals
        x_recon = x_fp8.float() * scale
        assert_close(x_recon, x, atol=scale.item() * 2, rtol=0.15)

    def test_quantize_input_static(self):
        x = torch.randn(4, 8, dtype=torch.float32)
        scale = torch.tensor([0.5], dtype=torch.float32)
        x_fp8 = _quantize_input_static(x, scale)
        self.assertEqual(x_fp8.dtype, _FP8_DTYPE)

    def test_quantize_input_dynamic_zero_tensor(self):
        x = torch.zeros(2, 4, dtype=torch.float32)
        x_fp8, scale = _quantize_input_dynamic(x)
        self.assertEqual(x_fp8.dtype, _FP8_DTYPE)
        self.assertTrue(scale.item() > 0)


class TestAbsMaxFP8Config(unittest.TestCase):
    def test_from_config_accepts_exact_absmax_fp8_name(self):
        config = AbsMaxFP8Config.from_config({"quant_method": "AbsMaxFP8"})
        self.assertIsInstance(config, AbsMaxFP8Config)

    def test_from_config_rejects_other_quant_method(self):
        with self.assertRaisesRegex(ValueError, "incompatible quant_method"):
            AbsMaxFP8Config.from_config({"quant_method": "fp8"})


@unittest.skipUnless(_HAVE_FP8_GPU, "Requires sm89+ GPU with FP8 tensor cores")
class TestFP8Compute(unittest.TestCase):
    """Tests that run the true FP8 torch._scaled_mm path on compatible GPUs."""

    def test_fp8_per_tensor_scale(self):
        """Per-tensor scale path produces reasonable output shape and values."""
        device = "cuda"
        method = AbsMaxFP8LinearMethod()
        layer = nn.Module()
        method.create_weights(
            layer=layer,
            input_size_per_partition=128,
            output_partition_sizes=[64],
            input_size=128,
            output_size=64,
            params_dtype=torch.bfloat16,
        )
        # Simulate loaded FP8 weights
        w_bf16 = torch.randn(64, 128, dtype=torch.bfloat16, device=device)
        layer.weight.data = w_bf16.to(torch.float8_e4m3fn)
        layer.weight_scale.data = torch.tensor(
            [1.0], dtype=torch.float32, device=device
        )
        layer.input_scale.data = torch.tensor(
            [1.0], dtype=torch.float32, device=device
        )
        layer.to(device)

        x = torch.randn(4, 128, dtype=torch.bfloat16, device=device)
        out = method.apply(layer, x, bias=None)

        self.assertEqual(out.shape, (4, 64))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertFalse(out.isnan().any())

    def test_fp8_per_row_scale(self):
        """Per-row (channelwise) scale path for merged QKV-like layers."""
        device = "cuda"
        method = AbsMaxFP8LinearMethod()
        layer = nn.Module()
        method.create_weights(
            layer=layer,
            input_size_per_partition=128,
            output_partition_sizes=[64],
            input_size=128,
            output_size=64,
            params_dtype=torch.bfloat16,
        )
        w_bf16 = torch.randn(64, 128, dtype=torch.bfloat16, device=device)
        layer.weight.data = w_bf16.to(torch.float8_e4m3fn)
        # Per-row scale: one value per output row — replace data in-place
        layer.weight_scale = nn.Parameter(
            torch.rand(64, dtype=torch.float32, device=device) + 0.5,
            requires_grad=False,
        )
        layer.input_scale.data = torch.tensor(
            [1.0], dtype=torch.float32, device=device
        )
        layer.to(device)

        x = torch.randn(4, 128, dtype=torch.bfloat16, device=device)
        out = method.apply(layer, x, bias=None)

        self.assertEqual(out.shape, (4, 64))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertFalse(out.isnan().any())

    def test_fp8_3d_input(self):
        """Verify that batched 3D input [B, S, D] works correctly."""
        device = "cuda"
        method = AbsMaxFP8LinearMethod()
        layer = nn.Module()
        method.create_weights(
            layer=layer,
            input_size_per_partition=128,
            output_partition_sizes=[64],
            input_size=128,
            output_size=64,
            params_dtype=torch.bfloat16,
        )
        w_bf16 = torch.randn(64, 128, dtype=torch.bfloat16, device=device)
        layer.weight.data = w_bf16.to(torch.float8_e4m3fn)
        layer.weight_scale.data = torch.tensor(
            [1.0], dtype=torch.float32, device=device
        )
        layer.input_scale.data = torch.tensor(
            [1.0], dtype=torch.float32, device=device
        )
        layer.to(device)

        x = torch.randn(2, 8, 128, dtype=torch.bfloat16, device=device)
        out = method.apply(layer, x, bias=None)

        self.assertEqual(out.shape, (2, 8, 64))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_fp8_with_bias(self):
        """Verify bias is correctly added in the FP8 path."""
        device = "cuda"
        method = AbsMaxFP8LinearMethod()
        layer = nn.Module()
        method.create_weights(
            layer=layer,
            input_size_per_partition=128,
            output_partition_sizes=[64],
            input_size=128,
            output_size=64,
            params_dtype=torch.bfloat16,
        )
        w_bf16 = torch.randn(64, 128, dtype=torch.bfloat16, device=device)
        layer.weight.data = w_bf16.to(torch.float8_e4m3fn)
        layer.weight_scale.data = torch.tensor(
            [1.0], dtype=torch.float32, device=device
        )
        layer.input_scale.data = torch.tensor(
            [1.0], dtype=torch.float32, device=device
        )
        layer.to(device)

        bias = torch.randn(64, dtype=torch.bfloat16, device=device)
        x = torch.randn(4, 128, dtype=torch.bfloat16, device=device)

        out_no_bias = method.apply(layer, x, bias=None)
        # Weight is already transposed from the first apply() call above
        out_with_bias = method.apply(layer, x, bias=bias)

        diff = out_with_bias - out_no_bias
        assert_close(diff, bias.unsqueeze(0).expand_as(diff), atol=0.2, rtol=0.2)

    def test_fp8_matches_dequant_approximately(self):
        """FP8 path output should approximate the dequant path."""
        device = "cuda"
        method = AbsMaxFP8LinearMethod()

        # Build a dequant-only layer for reference
        layer_ref = nn.Module()
        method.create_weights(
            layer=layer_ref,
            input_size_per_partition=128,
            output_partition_sizes=[64],
            input_size=128,
            output_size=64,
            params_dtype=torch.bfloat16,
        )
        w_fp8 = torch.randn(64, 128, dtype=torch.bfloat16, device=device).to(
            torch.float8_e4m3fn
        )
        layer_ref.weight.data = w_fp8.clone()
        layer_ref.weight_scale.data = torch.tensor(
            [1.0], dtype=torch.float32, device=device
        )
        layer_ref.input_scale.data = torch.tensor(
            [1.0], dtype=torch.float32, device=device
        )
        layer_ref.to(device)

        # Build a second identical layer for the FP8 compute path
        layer_fp8 = nn.Module()
        method.create_weights(
            layer=layer_fp8,
            input_size_per_partition=128,
            output_partition_sizes=[64],
            input_size=128,
            output_size=64,
            params_dtype=torch.bfloat16,
        )
        layer_fp8.weight.data = w_fp8.clone()
        layer_fp8.weight_scale.data = torch.tensor(
            [1.0], dtype=torch.float32, device=device
        )
        layer_fp8.input_scale.data = torch.tensor(
            [1.0], dtype=torch.float32, device=device
        )
        layer_fp8.to(device)

        x = torch.randn(4, 128, dtype=torch.bfloat16, device=device)

        # Dequant reference (uses original [out, in] weight layout)
        dequant_out = method._apply_dequant(layer_ref, x, bias=None)

        # FP8 compute (apply() triggers lazy transpose on layer_fp8)
        fp8_out = method.apply(layer_fp8, x, bias=None)

        # Should be approximately equal (FP8 quantization introduces some error)
        assert_close(fp8_out, dequant_out, atol=1.0, rtol=0.2)