import unittest

import torch
import torch.nn as nn
from torch.testing import assert_close

from fastvideo.layers.quantization.utils.quant_utils import (
    FP8_DTYPE,
    GroupShape,
    QuantKey,
    ScaleDesc,
    is_layer_skipped,
    kDynamicTensorScale,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
    kStaticTensorScale,
)
from fastvideo.layers.quantization.fp8_utils import (
    FP8_MAX,
    is_fp8_dtype,
    quantize_input_dynamic,
    quantize_input_static,
    supports_fp8_compute,
)
from fastvideo.layers.quantization.input_quant_fp8 import QuantFP8
from fastvideo.layers.quantization.kernels.scaled_mm import (
    choose_fp8_kernel,
    init_fp8_kernel,
)
from fastvideo.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from fastvideo.layers.quantization.kernels.scaled_mm.pytorch import (
    ChannelWiseTorchFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
)
from fastvideo.layers.quantization.fp8 import (
    ACTIVATION_GRANULARITIES,
    ACTIVATION_SCHEMES,
    Fp8Config,
    Fp8LinearMethod,
    Fp8OnlineLinearMethod,
)
from fastvideo.layers.quantization import get_quantization_config

_HAVE_CUDA = torch.cuda.is_available()
_HAVE_FP8_GPU = _HAVE_CUDA and (
    torch.cuda.get_device_capability()[0] > 8
    or (
        torch.cuda.get_device_capability()[0] == 8
        and torch.cuda.get_device_capability()[1] >= 9
    )
)

SEED = 0


def _ref_dynamic_per_tensor_fp8_quant(x: torch.Tensor):
    fp8_max = torch.finfo(FP8_DTYPE).max
    min_scale = 1.0 / (fp8_max * 512.0)
    x_max = x.abs().amax().float()
    scale = (x_max / fp8_max).clamp(min=min_scale)
    out = (x.float() / scale).clamp(-fp8_max, fp8_max).to(FP8_DTYPE)
    return out, scale


def _ref_dynamic_per_token_fp8_quant(x: torch.Tensor):
    fp8_max = torch.finfo(FP8_DTYPE).max
    min_scale = 1.0 / (fp8_max * 512.0)
    x_max = x.abs().max(dim=-1)[0].unsqueeze(-1).float()
    scale = (x_max / fp8_max).clamp(min=min_scale)
    out = (x.float() / scale).clamp(-fp8_max, fp8_max).to(FP8_DTYPE)
    return out, scale


def _baseline_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=None):
    output = torch.mm(
        scale_a * a.float(),
        scale_b * b.float(),
    ).to(out_dtype)
    if bias is not None:
        output = output + bias
    return output


class TestGroupShape(unittest.TestCase):
    def test_per_tensor(self):
        gs = GroupShape.PER_TENSOR
        self.assertTrue(gs.is_per_tensor())
        self.assertFalse(gs.is_per_token())
        self.assertFalse(gs.is_per_channel())

    def test_per_token(self):
        gs = GroupShape.PER_TOKEN
        self.assertFalse(gs.is_per_tensor())
        self.assertTrue(gs.is_per_token())
        self.assertFalse(gs.is_per_channel())

    def test_per_channel(self):
        gs = GroupShape.PER_CHANNEL
        self.assertFalse(gs.is_per_tensor())
        self.assertFalse(gs.is_per_token())
        self.assertTrue(gs.is_per_channel())

    def test_custom_group(self):
        gs = GroupShape(1, 128)
        self.assertTrue(gs.is_per_group())
        self.assertFalse(gs.is_per_tensor())


class TestScaleDesc(unittest.TestCase):
    def test_static_per_tensor(self):
        sd = kStaticTensorScale
        self.assertTrue(sd.static)
        self.assertTrue(sd.group_shape.is_per_tensor())
        self.assertEqual(sd.dtype, torch.float32)

    def test_dynamic_per_tensor(self):
        sd = kDynamicTensorScale
        self.assertFalse(sd.static)
        self.assertTrue(sd.group_shape.is_per_tensor())

    def test_str(self):
        s = str(kStaticTensorScale)
        self.assertIn("static", s)
        self.assertIn("per_tensor", s)


class TestQuantKey(unittest.TestCase):
    def test_fp8_static_tensor_sym(self):
        qk = kFp8StaticTensorSym
        self.assertEqual(qk.dtype, FP8_DTYPE)
        self.assertTrue(qk.symmetric)
        self.assertTrue(qk.scale.static)

    def test_fp8_dynamic_tensor_sym(self):
        qk = kFp8DynamicTensorSym
        self.assertEqual(qk.dtype, FP8_DTYPE)
        self.assertTrue(qk.symmetric)
        self.assertFalse(qk.scale.static)

    def test_str(self):
        s = str(kFp8StaticTensorSym)
        self.assertIn("QuantKey", s)
        self.assertIn("symmetric", s)


class TestIsLayerSkipped(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(
            is_layer_skipped("model.layers.0.mlp", ["model.layers.0.mlp"])
        )

    def test_no_match(self):
        self.assertFalse(
            is_layer_skipped("model.layers.0.mlp", ["model.layers.1.mlp"])
        )

    def test_proj_name_match(self):
        self.assertTrue(
            is_layer_skipped("model.layers.0.mlp.gate_proj", ["gate_proj"])
        )

    def test_empty_ignored(self):
        self.assertFalse(is_layer_skipped("anything", []))


class TestFp8Utils(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(SEED)

    def test_is_fp8_dtype(self):
        self.assertTrue(is_fp8_dtype(torch.float8_e4m3fn))
        self.assertTrue(is_fp8_dtype(torch.float8_e5m2))
        self.assertFalse(is_fp8_dtype(torch.bfloat16))
        self.assertFalse(is_fp8_dtype(torch.float32))

    def test_supports_fp8_compute_returns_bool(self):
        result = supports_fp8_compute()
        self.assertIsInstance(result, bool)

    def test_quantize_input_dynamic_roundtrip(self):
        x = torch.randn(4, 8, dtype=torch.float32)
        x_fp8, scale = quantize_input_dynamic(x)
        self.assertEqual(x_fp8.dtype, FP8_DTYPE)
        self.assertEqual(scale.dtype, torch.float32)
        x_recon = x_fp8.float() * scale
        assert_close(x_recon, x, atol=scale.item() * 2, rtol=0.15)

    def test_quantize_input_dynamic_zero_tensor(self):
        x = torch.zeros(2, 4, dtype=torch.float32)
        x_fp8, scale = quantize_input_dynamic(x)
        self.assertEqual(x_fp8.dtype, FP8_DTYPE)
        self.assertGreater(scale.item(), 0)

    def test_quantize_input_static(self):
        x = torch.randn(4, 8, dtype=torch.float32)
        scale = torch.tensor(0.5, dtype=torch.float32)
        x_fp8 = quantize_input_static(x, scale)
        self.assertEqual(x_fp8.dtype, FP8_DTYPE)

    def test_quantize_input_dynamic_vs_reference(self):
        for shape in [(1, 17), (4, 1024), (7, 256)]:
            with self.subTest(shape=shape):
                x = torch.randn(*shape, dtype=torch.float32)
                out, scale = quantize_input_dynamic(x)
                ref_out, ref_scale = _ref_dynamic_per_tensor_fp8_quant(x)
                assert_close(out.float(), ref_out.float())
                assert_close(scale, ref_scale)


class TestQuantFP8(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(SEED)

    def test_dynamic_per_tensor(self):
        quant = QuantFP8(static=False, group_shape=GroupShape.PER_TENSOR)
        x = torch.randn(4, 16, dtype=torch.float32)
        x_fp8, scale = quant(x)
        self.assertEqual(x_fp8.dtype, FP8_DTYPE)
        self.assertEqual(x_fp8.shape, (4, 16))
        self.assertEqual(scale.shape, (1,))

    def test_dynamic_per_token(self):
        quant = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)
        x = torch.randn(4, 16, dtype=torch.float32)
        x_fp8, scale = quant(x)
        self.assertEqual(x_fp8.dtype, FP8_DTYPE)
        self.assertEqual(scale.shape, (4, 1))

    def test_static_per_tensor(self):
        quant = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)
        x = torch.randn(4, 16, dtype=torch.float32)
        scale = torch.tensor([[0.01]], dtype=torch.float32)
        x_fp8, scale_out = quant(x, scale=scale)
        self.assertEqual(x_fp8.dtype, FP8_DTYPE)
        self.assertIs(scale_out, scale)

    def test_static_requires_scale(self):
        quant = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)
        x = torch.randn(4, 16, dtype=torch.float32)
        with self.assertRaises(AssertionError):
            quant(x)

    def test_dynamic_rejects_scale(self):
        quant = QuantFP8(static=False, group_shape=GroupShape.PER_TENSOR)
        x = torch.randn(4, 16, dtype=torch.float32)
        scale = torch.tensor([[0.01]], dtype=torch.float32)
        with self.assertRaises(AssertionError):
            quant(x, scale=scale)

    def test_num_token_padding(self):
        quant = QuantFP8(
            static=False, group_shape=GroupShape.PER_TENSOR, num_token_padding=8
        )
        x = torch.randn(3, 16, dtype=torch.float32)
        x_fp8, _ = quant(x)
        self.assertEqual(x_fp8.shape[0], 8)
        self.assertEqual(x_fp8.shape[1], 16)

    def test_dynamic_per_tensor_vs_reference(self):
        for shape in [(1, 17), (4, 1024), (7, 5137)]:
            with self.subTest(shape=shape):
                x = torch.randn(*shape, dtype=torch.float32)
                quant = QuantFP8(static=False, group_shape=GroupShape.PER_TENSOR)
                out, scale = quant(x)
                ref_out, ref_scale = _ref_dynamic_per_tensor_fp8_quant(x)
                assert_close(out.float(), ref_out.float())
                assert_close(scale.squeeze(), ref_scale)

    def test_dynamic_per_token_vs_reference(self):
        for shape in [(1, 17), (4, 1024), (7, 5137)]:
            with self.subTest(shape=shape):
                x = torch.randn(*shape, dtype=torch.float32)
                quant = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)
                out, scale = quant(x)
                ref_out, ref_scale = _ref_dynamic_per_token_fp8_quant(x)
                assert_close(out.float(), ref_out.float())
                assert_close(scale, ref_scale)


class TestKernelDispatch(unittest.TestCase):
    def _make_config(self, act_per_tensor=True, weight_per_tensor=True):
        act_key = kFp8DynamicTensorSym
        if not act_per_tensor:
            act_key = QuantKey(
                FP8_DTYPE,
                ScaleDesc(torch.float32, False, GroupShape.PER_TOKEN),
                symmetric=True,
            )
        weight_key = kFp8StaticTensorSym
        if not weight_per_tensor:
            weight_key = QuantKey(
                FP8_DTYPE,
                ScaleDesc(torch.float32, True, GroupShape.PER_CHANNEL),
                symmetric=True,
            )
        return FP8ScaledMMLinearLayerConfig(
            weight_quant_key=weight_key,
            activation_quant_key=act_key,
            out_dtype=torch.bfloat16,
        )

    def test_per_tensor_selects_per_tensor_kernel(self):
        config = self._make_config(act_per_tensor=True, weight_per_tensor=True)
        kernel_cls = choose_fp8_kernel(config, compute_capability=89)
        self.assertIs(kernel_cls, PerTensorTorchFP8ScaledMMLinearKernel)

    def test_channelwise_selects_channelwise_kernel(self):
        config = self._make_config(
            act_per_tensor=True, weight_per_tensor=False
        )
        kernel_cls = choose_fp8_kernel(config, compute_capability=89)
        self.assertIs(kernel_cls, ChannelWiseTorchFP8ScaledMMLinearKernel)

    def test_no_kernel_below_sm89(self):
        config = self._make_config()
        with self.assertRaises(ValueError):
            choose_fp8_kernel(config, compute_capability=80)

    def test_per_tensor_kernel_is_supported(self):
        ok, _ = PerTensorTorchFP8ScaledMMLinearKernel.is_supported(
            compute_capability=89
        )
        self.assertTrue(ok)

    def test_per_tensor_kernel_not_supported_below_89(self):
        ok, reason = PerTensorTorchFP8ScaledMMLinearKernel.is_supported(
            compute_capability=80
        )
        self.assertFalse(ok)
        self.assertIn("89", reason)

    def test_per_token_activation_selects_channelwise_kernel(self):
        config = self._make_config(act_per_tensor=False, weight_per_tensor=True)
        kernel_cls = choose_fp8_kernel(config, compute_capability=89)
        self.assertIs(kernel_cls, ChannelWiseTorchFP8ScaledMMLinearKernel)

    def test_init_fp8_kernel_per_token(self):
        kernel = init_fp8_kernel(
            activation_quant_key=kFp8DynamicTokenSym,
            weight_quant_key=kFp8StaticTensorSym,
            out_dtype=torch.bfloat16,
        )
        self.assertIsInstance(kernel, ChannelWiseTorchFP8ScaledMMLinearKernel)
        self.assertTrue(kernel.quant_fp8.use_per_token_if_dynamic)


class TestFp8Config(unittest.TestCase):
    def test_default_construction(self):
        config = Fp8Config()
        self.assertFalse(config.is_checkpoint_fp8_serialized)
        self.assertEqual(config.activation_scheme, "dynamic")
        self.assertEqual(config.ignored_layers, [])

    def test_construction_with_params(self):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="static",
            ignored_layers=["lm_head"],
        )
        self.assertTrue(config.is_checkpoint_fp8_serialized)
        self.assertEqual(config.activation_scheme, "static")
        self.assertEqual(config.ignored_layers, ["lm_head"])

    def test_rejects_invalid_activation_scheme(self):
        with self.assertRaises(ValueError):
            Fp8Config(activation_scheme="invalid")

    def test_get_name(self):
        config = Fp8Config()
        self.assertEqual(config.get_name(), "fp8")

    def test_get_supported_act_dtypes(self):
        config = Fp8Config()
        dtypes = config.get_supported_act_dtypes()
        self.assertIn(torch.bfloat16, dtypes)
        self.assertIn(torch.float16, dtypes)

    def test_get_min_capability(self):
        self.assertEqual(Fp8Config.get_min_capability(), 75)

    def test_get_quant_method_returns_none_for_non_linear(self):
        config = Fp8Config(is_checkpoint_fp8_serialized=True)
        result = config.get_quant_method(nn.Module(), prefix="test")
        self.assertIsNone(result)

    def test_default_activation_granularity(self):
        config = Fp8Config()
        self.assertEqual(config.activation_granularity, "per_tensor")

    def test_per_token_activation_granularity(self):
        config = Fp8Config(activation_granularity="per_token")
        self.assertEqual(config.activation_granularity, "per_token")

    def test_rejects_invalid_activation_granularity(self):
        with self.assertRaises(ValueError):
            Fp8Config(activation_granularity="per_channel")

    def test_from_config(self):
        config = Fp8Config.from_config({
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
        })
        self.assertIsInstance(config, Fp8Config)
        self.assertTrue(config.is_checkpoint_fp8_serialized)
        self.assertEqual(config.activation_scheme, "dynamic")


class TestRegistry(unittest.TestCase):
    def test_get_fp8_config(self):
        config_cls = get_quantization_config("fp8")
        self.assertIs(config_cls, Fp8Config)

    def test_get_absmax_config(self):
        from fastvideo.layers.quantization.absmax_fp8 import AbsMaxFP8Config
        config_cls = get_quantization_config("AbsMaxFP8")
        self.assertIs(config_cls, AbsMaxFP8Config)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            get_quantization_config("nonexistent")


class TestFp8LinearMethodKernelSelection(unittest.TestCase):

    @unittest.skipUnless(_HAVE_FP8_GPU, "Requires sm89+ GPU")
    def test_per_tensor_granularity_selects_per_tensor_kernel(self):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            activation_granularity="per_tensor",
        )
        method = Fp8LinearMethod(config)
        self.assertIsNotNone(method.fp8_linear)
        self.assertIsInstance(
            method.fp8_linear, PerTensorTorchFP8ScaledMMLinearKernel
        )

    @unittest.skipUnless(_HAVE_FP8_GPU, "Requires sm89+ GPU")
    def test_per_token_granularity_selects_channelwise_kernel(self):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            activation_granularity="per_token",
        )
        method = Fp8LinearMethod(config)
        self.assertIsNotNone(method.fp8_linear)
        self.assertIsInstance(
            method.fp8_linear, ChannelWiseTorchFP8ScaledMMLinearKernel
        )
        self.assertTrue(method.fp8_linear.quant_fp8.use_per_token_if_dynamic)

    @unittest.skipUnless(_HAVE_FP8_GPU, "Requires sm89+ GPU")
    def test_static_scheme_ignores_granularity(self):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="static",
            activation_granularity="per_token",
        )
        method = Fp8LinearMethod(config)
        self.assertIsNotNone(method.fp8_linear)
        self.assertIsInstance(
            method.fp8_linear, PerTensorTorchFP8ScaledMMLinearKernel
        )

    @unittest.skipUnless(_HAVE_FP8_GPU, "Requires sm89+ GPU")
    def test_online_method_inherits_granularity(self):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=False,
            activation_scheme="dynamic",
            activation_granularity="per_token",
        )
        method = Fp8OnlineLinearMethod(config)
        self.assertIsNotNone(method.fp8_linear)
        self.assertIsInstance(
            method.fp8_linear, ChannelWiseTorchFP8ScaledMMLinearKernel
        )


class TestFp8LinearMethodDequant(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(SEED)

    def _make_layer_and_method(self, out_features=4, in_features=8):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True, activation_scheme="dynamic"
        )
        method = Fp8LinearMethod(config)
        layer = nn.Module()
        method.create_weights(
            layer=layer,
            input_size_per_partition=in_features,
            output_partition_sizes=[out_features],
            input_size=in_features,
            output_size=out_features,
            params_dtype=torch.bfloat16,
        )
        return layer, method

    def test_create_weights_registers_parameters(self):
        layer, method = self._make_layer_and_method()
        self.assertTrue(hasattr(layer, "weight"))
        self.assertTrue(hasattr(layer, "weight_scale"))
        self.assertEqual(layer.weight.dtype, FP8_DTYPE)
        self.assertEqual(layer.weight_scale.dtype, torch.float32)

    def test_create_weights_no_input_scale_for_dynamic(self):
        layer, method = self._make_layer_and_method()
        self.assertFalse(hasattr(layer, "input_scale"))

    def test_create_weights_has_input_scale_for_static(self):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True, activation_scheme="static"
        )
        method = Fp8LinearMethod(config)
        layer = nn.Module()
        method.create_weights(
            layer=layer,
            input_size_per_partition=8,
            output_partition_sizes=[4],
            input_size=8,
            output_size=4,
            params_dtype=torch.bfloat16,
        )
        self.assertTrue(hasattr(layer, "input_scale"))
        self.assertEqual(layer.input_scale.dtype, torch.float32)

    def test_dequant_path_produces_correct_shape(self):
        layer, method = self._make_layer_and_method(out_features=4, in_features=8)
        w_bf16 = torch.randn(4, 8, dtype=torch.bfloat16)
        layer.weight.data = w_bf16.to(FP8_DTYPE)
        layer.weight_scale.data = torch.tensor([1.0], dtype=torch.float32)
        method.process_weights_after_loading(layer)

        x = torch.randn(2, 8, dtype=torch.bfloat16)
        out = method._apply_dequant(layer, x, bias=None)
        self.assertEqual(out.shape, (2, 4))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_dequant_path_with_bias(self):
        layer, method = self._make_layer_and_method(out_features=4, in_features=8)
        w_bf16 = torch.randn(4, 8, dtype=torch.bfloat16)
        layer.weight.data = w_bf16.to(FP8_DTYPE)
        layer.weight_scale.data = torch.tensor([1.0], dtype=torch.float32)
        method.process_weights_after_loading(layer)

        x = torch.randn(2, 8, dtype=torch.bfloat16)
        bias = torch.randn(4, dtype=torch.bfloat16)

        out_no_bias = method._apply_dequant(layer, x, bias=None)
        out_with_bias = method._apply_dequant(layer, x, bias=bias)

        diff = out_with_bias - out_no_bias
        assert_close(diff, bias.unsqueeze(0).expand_as(diff), atol=0.05, rtol=0.05)


class TestFp8OnlineLinearMethod(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(SEED)

    def test_create_weights_in_original_dtype(self):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=False, activation_scheme="dynamic"
        )
        method = Fp8OnlineLinearMethod(config)
        layer = nn.Module()
        method.create_weights(
            layer=layer,
            input_size_per_partition=8,
            output_partition_sizes=[4],
            input_size=8,
            output_size=4,
            params_dtype=torch.bfloat16,
        )
        self.assertEqual(layer.weight.dtype, torch.bfloat16)

    def test_process_weights_quantizes_to_fp8(self):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=False, activation_scheme="dynamic"
        )
        method = Fp8OnlineLinearMethod(config)
        layer = nn.Module()
        method.create_weights(
            layer=layer,
            input_size_per_partition=8,
            output_partition_sizes=[4],
            input_size=8,
            output_size=4,
            params_dtype=torch.bfloat16,
        )
        layer.weight.data = torch.randn(4, 8, dtype=torch.bfloat16)
        method.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.dtype, FP8_DTYPE)
        self.assertTrue(hasattr(layer, "weight_scale"))
        self.assertIsNone(layer.input_scale)


@unittest.skipUnless(_HAVE_FP8_GPU, "Requires sm89+ GPU with FP8 tensor cores")
class TestFp8ComputePath(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    def _make_fp8_layer(self, out_features=64, in_features=128):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True, activation_scheme="dynamic"
        )
        method = Fp8LinearMethod(config)
        self.assertIsNotNone(method.fp8_linear)

        layer = nn.Module()
        method.create_weights(
            layer=layer,
            input_size_per_partition=in_features,
            output_partition_sizes=[out_features],
            input_size=in_features,
            output_size=out_features,
            params_dtype=torch.bfloat16,
        )

        w_bf16 = torch.randn(
            out_features, in_features, dtype=torch.bfloat16, device="cuda"
        )
        layer.weight.data = w_bf16.to(FP8_DTYPE)
        layer.weight_scale.data = torch.tensor(
            [1.0], dtype=torch.float32, device="cuda"
        )
        layer.to("cuda")

        method.process_weights_after_loading(layer)
        return layer, method

    def test_fp8_kernel_output_shape(self):
        layer, method = self._make_fp8_layer()
        x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
        out = method.apply(layer, x, bias=None)
        self.assertEqual(out.shape, (4, 64))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertFalse(out.isnan().any())

    def test_fp8_kernel_3d_input(self):
        layer, method = self._make_fp8_layer()
        x = torch.randn(2, 8, 128, dtype=torch.bfloat16, device="cuda")
        out = method.apply(layer, x, bias=None)
        self.assertEqual(out.shape, (2, 8, 64))

    def test_fp8_kernel_with_bias(self):
        layer, method = self._make_fp8_layer()
        x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
        bias = torch.randn(64, dtype=torch.bfloat16, device="cuda")

        out_no_bias = method.apply(layer, x, bias=None)
        out_with_bias = method.apply(layer, x, bias=bias)

        diff = out_with_bias - out_no_bias
        expected = bias.unsqueeze(0).expand_as(diff)
        rel_diff = (
            (diff.float() - expected.float()).abs().mean()
            / expected.float().abs().mean()
        )
        self.assertLess(rel_diff.item(), 0.03)

    def test_fp8_vs_dequant_approximate(self):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True, activation_scheme="dynamic"
        )
        method = Fp8LinearMethod(config)

        w_fp8 = torch.randn(
            64, 128, dtype=torch.bfloat16, device="cuda"
        ).to(FP8_DTYPE)

        layer_dq = nn.Module()
        layer_dq.weight = nn.Parameter(w_fp8.clone(), requires_grad=False)
        layer_dq.weight_scale = nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32, device="cuda"),
            requires_grad=False,
        )
        layer_dq.input_scale = None
        layer_dq.orig_dtype = torch.bfloat16
        layer_dq._fp8_weight_transposed = False

        layer_fp8, method_fp8 = self._make_fp8_layer()
        layer_fp8.weight.data = w_fp8.clone().t()
        layer_fp8.weight_scale.data = torch.tensor(
            1.0, dtype=torch.float32, device="cuda"
        )

        x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")

        dequant_out = method._apply_dequant(layer_dq, x, bias=None)
        fp8_out = method_fp8.apply(layer_fp8, x, bias=None)

        rel_diff = (
            (fp8_out.float() - dequant_out.float()).abs().mean()
            / dequant_out.float().abs().mean()
        )
        self.assertLess(rel_diff.item(), 0.05)

    def test_online_quantization_end_to_end(self):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=False, activation_scheme="dynamic"
        )
        method = Fp8OnlineLinearMethod(config)
        layer = nn.Module()
        method.create_weights(
            layer=layer,
            input_size_per_partition=128,
            output_partition_sizes=[64],
            input_size=128,
            output_size=64,
            params_dtype=torch.bfloat16,
        )
        layer.weight.data = torch.randn(
            64, 128, dtype=torch.bfloat16, device="cuda"
        )
        layer.to("cuda")
        method.process_weights_after_loading(layer)

        x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
        out = method.apply(layer, x, bias=None)
        self.assertEqual(out.shape, (4, 64))
        self.assertFalse(out.isnan().any())

    def test_fp8_gemm_vs_baseline(self):
        for M, K, N in [(1, 128, 64), (32, 128, 64), (33, 256, 128)]:
            with self.subTest(M=M, K=K, N=N):
                device = "cuda"
                a = torch.randn(M, K, device=device).to(FP8_DTYPE)
                b = torch.randn(N, K, device=device).to(FP8_DTYPE).t()

                scale_a = torch.tensor(1.0, device=device, dtype=torch.float32)
                scale_b = torch.tensor(1.0, device=device, dtype=torch.float32)

                output = torch._scaled_mm(
                    a, b,
                    scale_a=scale_a,
                    scale_b=scale_b,
                    out_dtype=torch.bfloat16,
                )
                if isinstance(output, tuple):
                    output = output[0]

                baseline = _baseline_scaled_mm(
                    a, b, scale_a, scale_b, torch.bfloat16
                )
                assert_close(output, baseline, rtol=1e-1, atol=1e-1)

    def test_fp8_kernel_with_non_unit_scales(self):
        device = "cuda"
        M, K, N = 16, 128, 64

        w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
        w_absmax = w_bf16.abs().amax().float()
        w_scale = (w_absmax / FP8_MAX).clamp(min=1.0 / (FP8_MAX * 512.0))
        w_fp8 = (
            (w_bf16.float() / w_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        )

        config = Fp8Config(
            is_checkpoint_fp8_serialized=True, activation_scheme="dynamic"
        )
        method = Fp8LinearMethod(config)
        layer = nn.Module()
        method.create_weights(
            layer=layer,
            input_size_per_partition=K,
            output_partition_sizes=[N],
            input_size=K,
            output_size=N,
            params_dtype=torch.bfloat16,
        )
        layer.weight.data = w_fp8
        layer.weight_scale.data = w_scale.view(1).to(device)
        layer.to(device)
        method.process_weights_after_loading(layer)

        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        fp8_out = method.apply(layer, x, bias=None)

        w_dequant = w_fp8.float() * w_scale.to(device)
        ref_out = (x.float() @ w_dequant.t().float()).to(torch.bfloat16)

        rel_diff = (
            (fp8_out.float() - ref_out.float()).abs().mean()
            / ref_out.float().abs().mean()
        )
        self.assertLess(rel_diff.item(), 0.05)

    def test_online_quant_vs_bf16_reference(self):
        device = "cuda"
        M, K, N = 16, 128, 64

        w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)

        config = Fp8Config(
            is_checkpoint_fp8_serialized=False, activation_scheme="dynamic"
        )
        method = Fp8OnlineLinearMethod(config)
        layer = nn.Module()
        method.create_weights(
            layer=layer,
            input_size_per_partition=K,
            output_partition_sizes=[N],
            input_size=K,
            output_size=N,
            params_dtype=torch.bfloat16,
        )
        layer.weight.data = w_bf16.clone()
        layer.to(device)
        method.process_weights_after_loading(layer)

        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        fp8_out = method.apply(layer, x, bias=None)

        ref_out = (x @ w_bf16.t()).to(torch.bfloat16)

        rel_diff = (
            (fp8_out.float() - ref_out.float()).abs().mean()
            / ref_out.float().abs().mean()
        )
        self.assertLess(rel_diff.item(), 0.1)

    def test_init_fp8_kernel_factory(self):
        kernel = init_fp8_kernel(
            activation_quant_key=kFp8DynamicTensorSym,
            weight_quant_key=kFp8StaticTensorSym,
            out_dtype=torch.bfloat16,
        )
        self.assertIsNotNone(kernel)
        self.assertIsNotNone(kernel.quant_fp8)

    def test_per_token_offline_end_to_end(self):
        device = "cuda"
        M, K, N = 16, 128, 64

        w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
        w_absmax = w_bf16.abs().amax().float()
        w_scale = (w_absmax / FP8_MAX).clamp(min=1.0 / (FP8_MAX * 512.0))
        w_fp8 = (
            (w_bf16.float() / w_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        )

        config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            activation_granularity="per_token",
        )
        method = Fp8LinearMethod(config)
        layer = nn.Module()
        method.create_weights(
            layer=layer,
            input_size_per_partition=K,
            output_partition_sizes=[N],
            input_size=K,
            output_size=N,
            params_dtype=torch.bfloat16,
        )
        layer.weight.data = w_fp8
        layer.weight_scale.data = w_scale.view(1).to(device)
        layer.to(device)
        method.process_weights_after_loading(layer)

        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        fp8_out = method.apply(layer, x, bias=None)

        self.assertEqual(fp8_out.shape, (M, N))
        self.assertEqual(fp8_out.dtype, torch.bfloat16)
        self.assertFalse(fp8_out.isnan().any())

        w_dequant = w_fp8.float() * w_scale.to(device)
        ref_out = (x.float() @ w_dequant.t().float()).to(torch.bfloat16)

        rel_diff = (
            (fp8_out.float() - ref_out.float()).abs().mean()
            / ref_out.float().abs().mean()
        )
        self.assertLess(rel_diff.item(), 0.1)

    def test_per_token_online_end_to_end(self):
        device = "cuda"
        M, K, N = 16, 128, 64

        w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)

        config = Fp8Config(
            is_checkpoint_fp8_serialized=False,
            activation_scheme="dynamic",
            activation_granularity="per_token",
        )
        method = Fp8OnlineLinearMethod(config)
        layer = nn.Module()
        method.create_weights(
            layer=layer,
            input_size_per_partition=K,
            output_partition_sizes=[N],
            input_size=K,
            output_size=N,
            params_dtype=torch.bfloat16,
        )
        layer.weight.data = w_bf16.clone()
        layer.to(device)
        method.process_weights_after_loading(layer)

        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        fp8_out = method.apply(layer, x, bias=None)

        self.assertEqual(fp8_out.shape, (M, N))
        self.assertFalse(fp8_out.isnan().any())

        ref_out = (x @ w_bf16.t()).to(torch.bfloat16)
        rel_diff = (
            (fp8_out.float() - ref_out.float()).abs().mean()
            / ref_out.float().abs().mean()
        )
        self.assertLess(rel_diff.item(), 0.1)


class TestFp8ConfigIgnoredLayers(unittest.TestCase):

    def test_default_ignored_layers_empty(self):
        config = Fp8Config()
        self.assertEqual(config.ignored_layers, [])

    def test_ignored_layers_set(self):
        config = Fp8Config(ignored_layers=["proj_out", "norm"])
        self.assertEqual(config.ignored_layers, ["proj_out", "norm"])

    def test_ignored_layers_none_becomes_empty(self):
        config = Fp8Config(ignored_layers=None)
        self.assertEqual(config.ignored_layers, [])

    def test_default_min_layer_size_zero(self):
        config = Fp8Config()
        self.assertEqual(config.min_layer_size, 0)

    def test_min_layer_size_set(self):
        config = Fp8Config(min_layer_size=4096)
        self.assertEqual(config.min_layer_size, 4096)


class TestBridgeIgnoredLayers(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(SEED)

    def _make_model_with_named_layers(self):
        model = nn.Module()
        model.proj_qkv = nn.Linear(64, 192, bias=False)
        model.proj_out = nn.Linear(192, 64, bias=True)
        model.norm = nn.Linear(64, 64, bias=False)
        model.mlp_up = nn.Linear(64, 256, bias=False)
        model.tiny = nn.Linear(4, 4, bias=False)
        return model

    def test_online_ignored_layers_by_name(self):
        from fastvideo.layers.quantization.dit_fp8_bridge import (
            prepare_model_for_online_fp8,
        )

        config = Fp8Config(
            is_checkpoint_fp8_serialized=False,
            activation_scheme="dynamic",
            ignored_layers=["proj_out", "norm"],
        )
        model = self._make_model_with_named_layers()
        prepare_model_for_online_fp8(
            model, output_dtype=torch.bfloat16, quant_config=config
        )

        injected_names = [
            name for name, m in model.named_modules()
            if isinstance(m, nn.Linear) and hasattr(m, "_fp8_output_dtype")
        ]
        self.assertIn("proj_qkv", injected_names)
        self.assertIn("mlp_up", injected_names)
        self.assertIn("tiny", injected_names)
        self.assertNotIn("proj_out", injected_names)
        self.assertNotIn("norm", injected_names)

    def test_online_min_layer_size_skips_small(self):
        from fastvideo.layers.quantization.dit_fp8_bridge import (
            prepare_model_for_online_fp8,
        )

        config = Fp8Config(
            is_checkpoint_fp8_serialized=False,
            activation_scheme="dynamic",
            min_layer_size=100,
        )
        model = self._make_model_with_named_layers()
        prepare_model_for_online_fp8(
            model, output_dtype=torch.bfloat16, quant_config=config
        )

        self.assertFalse(hasattr(model.tiny, "_fp8_output_dtype"))
        self.assertTrue(hasattr(model.proj_qkv, "_fp8_output_dtype"))

    def test_online_both_filters_combine(self):
        from fastvideo.layers.quantization.dit_fp8_bridge import (
            prepare_model_for_online_fp8,
        )

        config = Fp8Config(
            is_checkpoint_fp8_serialized=False,
            activation_scheme="dynamic",
            ignored_layers=["norm"],
            min_layer_size=100,
        )
        model = self._make_model_with_named_layers()
        prepare_model_for_online_fp8(
            model, output_dtype=torch.bfloat16, quant_config=config
        )

        self.assertFalse(hasattr(model.norm, "_fp8_output_dtype"))
        self.assertFalse(hasattr(model.tiny, "_fp8_output_dtype"))
        self.assertTrue(hasattr(model.proj_qkv, "_fp8_output_dtype"))
        self.assertTrue(hasattr(model.proj_out, "_fp8_output_dtype"))
        self.assertTrue(hasattr(model.mlp_up, "_fp8_output_dtype"))

    def test_offline_ignored_layers_by_name(self):
        from fastvideo.layers.quantization.dit_fp8_bridge import (
            prepare_model_for_fp8,
        )

        config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            ignored_layers=["proj_out"],
        )
        model = self._make_model_with_named_layers()
        fp8_prefixes = {"proj_qkv", "proj_out", "mlp_up"}
        prepare_model_for_fp8(
            model,
            fp8_module_prefixes=fp8_prefixes,
            output_dtype=torch.bfloat16,
            quant_config=config,
        )

        self.assertFalse(hasattr(model.proj_out, "_fp8_output_dtype"))
        self.assertTrue(hasattr(model.proj_qkv, "_fp8_output_dtype"))
        self.assertTrue(hasattr(model.mlp_up, "_fp8_output_dtype"))

    def test_offline_min_layer_size_skips_small(self):
        from fastvideo.layers.quantization.dit_fp8_bridge import (
            prepare_model_for_fp8,
        )

        config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            min_layer_size=5000,
        )
        model = self._make_model_with_named_layers()
        fp8_prefixes = {"proj_qkv", "norm", "tiny"}
        prepare_model_for_fp8(
            model,
            fp8_module_prefixes=fp8_prefixes,
            output_dtype=torch.bfloat16,
            quant_config=config,
        )

        self.assertFalse(hasattr(model.tiny, "_fp8_output_dtype"))
        self.assertFalse(hasattr(model.norm, "_fp8_output_dtype"))
        self.assertTrue(hasattr(model.proj_qkv, "_fp8_output_dtype"))

    def test_empty_ignored_layers_skips_nothing(self):
        from fastvideo.layers.quantization.dit_fp8_bridge import (
            prepare_model_for_online_fp8,
        )

        config = Fp8Config(
            is_checkpoint_fp8_serialized=False,
            activation_scheme="dynamic",
            ignored_layers=[],
            min_layer_size=0,
        )
        model = self._make_model_with_named_layers()
        prepare_model_for_online_fp8(
            model, output_dtype=torch.bfloat16, quant_config=config
        )

        linear_count = sum(
            1 for m in model.modules()
            if isinstance(m, nn.Linear) and hasattr(m, "_fp8_output_dtype")
        )
        self.assertEqual(linear_count, 5)

    def test_get_quant_method_with_ignored_layers(self):
        from fastvideo.layers.linear import LinearBase, UnquantizedLinearMethod

        config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            ignored_layers=["lm_head"],
        )
        layer = LinearBase(16, 32)
        result = config.get_quant_method(layer, prefix="lm_head")
        self.assertIsInstance(result, UnquantizedLinearMethod)

        result_normal = config.get_quant_method(layer, prefix="model.layers.0.qkv")
        self.assertIsInstance(result_normal, Fp8LinearMethod)


class TestOnlineFp8Bridge(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(SEED)

    def _make_simple_model(self):
        model = nn.Sequential(
            nn.Linear(16, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 8, bias=True),
        )
        return model

    def test_prepare_online_sets_orig_dtype(self):
        from fastvideo.layers.quantization.dit_fp8_bridge import (
            prepare_model_for_online_fp8,
        )

        model = self._make_simple_model()
        prepare_model_for_online_fp8(model, output_dtype=torch.bfloat16)

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self.assertEqual(module.orig_dtype, torch.bfloat16)
                self.assertEqual(module._fp8_output_dtype, torch.bfloat16)

    def test_prepare_online_does_not_change_weight_dtype(self):
        from fastvideo.layers.quantization.dit_fp8_bridge import (
            prepare_model_for_online_fp8,
        )

        model = self._make_simple_model()
        for module in model.modules():
            if isinstance(module, nn.Linear):
                original_dtype = module.weight.dtype

        prepare_model_for_online_fp8(model, output_dtype=torch.bfloat16)

        for module in model.modules():
            if isinstance(module, nn.Linear):
                self.assertEqual(module.weight.dtype, original_dtype)

    def test_prepare_online_does_not_register_scale_params(self):
        from fastvideo.layers.quantization.dit_fp8_bridge import (
            prepare_model_for_online_fp8,
        )

        model = self._make_simple_model()
        prepare_model_for_online_fp8(model, output_dtype=torch.bfloat16)

        for module in model.modules():
            if isinstance(module, nn.Linear):
                self.assertFalse(
                    hasattr(module, "weight_scale")
                    and isinstance(module.weight_scale, nn.Parameter)
                )

    def test_prepare_online_injects_forward_hook(self):
        from fastvideo.layers.quantization.dit_fp8_bridge import (
            prepare_model_for_online_fp8,
        )

        model = self._make_simple_model()
        original_forwards = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                original_forwards[name] = module.forward

        prepare_model_for_online_fp8(model, output_dtype=torch.bfloat16)

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self.assertIsNot(module.forward, original_forwards[name])

    def test_prepare_online_lazy_quantization_on_forward(self):
        from fastvideo.layers.quantization.dit_fp8_bridge import (
            prepare_model_for_online_fp8,
        )

        model = nn.Linear(16, 8, bias=False)
        model.weight.data = torch.randn(8, 16, dtype=torch.float32)
        prepare_model_for_online_fp8(model, output_dtype=torch.bfloat16)

        self.assertFalse(getattr(model, "_fp8_weights_processed", False))

        x = torch.randn(2, 16, dtype=torch.bfloat16)
        out = model(x)

        self.assertTrue(model._fp8_weights_processed)
        self.assertEqual(model.weight.dtype, FP8_DTYPE)
        self.assertTrue(hasattr(model, "weight_scale"))
        self.assertIsNone(model.input_scale)
        self.assertEqual(out.shape, (2, 8))

    def test_prepare_online_with_explicit_quant_config(self):
        from fastvideo.layers.quantization.dit_fp8_bridge import (
            prepare_model_for_online_fp8,
        )

        config = Fp8Config(
            is_checkpoint_fp8_serialized=False,
            activation_scheme="dynamic",
        )
        model = nn.Linear(16, 8, bias=False)
        model.weight.data = torch.randn(8, 16, dtype=torch.float32)
        prepare_model_for_online_fp8(
            model, output_dtype=torch.bfloat16, quant_config=config
        )

        x = torch.randn(2, 16, dtype=torch.bfloat16)
        out = model(x)

        self.assertEqual(out.shape, (2, 8))
        self.assertEqual(model.weight.dtype, FP8_DTYPE)

    def test_prepare_offline_vs_online_count(self):
        from fastvideo.layers.quantization.dit_fp8_bridge import (
            prepare_model_for_fp8,
            prepare_model_for_online_fp8,
        )

        model_offline = self._make_simple_model()
        prepare_model_for_fp8(
            model_offline,
            fp8_module_prefixes={"0"},
            output_dtype=torch.bfloat16,
        )
        offline_count = sum(
            1 for m in model_offline.modules()
            if isinstance(m, nn.Linear) and hasattr(m, "_fp8_output_dtype")
        )

        model_online = self._make_simple_model()
        prepare_model_for_online_fp8(
            model_online, output_dtype=torch.bfloat16
        )
        online_count = sum(
            1 for m in model_online.modules()
            if isinstance(m, nn.Linear) and hasattr(m, "_fp8_output_dtype")
        )

        self.assertEqual(offline_count, 1)
        self.assertEqual(online_count, 2)


if __name__ == "__main__":
    unittest.main()
