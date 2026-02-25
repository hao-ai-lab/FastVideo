# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/layers/quantization/fp8.py
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from fastvideo.layers.quantization import QuantizationMethods
from fastvideo.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from fastvideo.layers.quantization.fp8_utils import supports_fp8_compute
from fastvideo.layers.quantization.kernels.scaled_mm import init_fp8_kernel
from fastvideo.layers.quantization.utils.quant_utils import (
    FP8_DTYPE,
    is_layer_skipped,
    kFp8DynamicTensorSym,
    kFp8StaticTensorSym,
)
from fastvideo.logger import init_logger
from fastvideo.models.parameter import ModelWeightParameter, PerTensorScaleParameter
from fastvideo.models.utils import set_weight_attrs

logger = init_logger(__name__)

ACTIVATION_SCHEMES = ["static", "dynamic"]

FP8_MAX = torch.finfo(FP8_DTYPE).max


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(
                f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []

    def get_name(self) -> QuantizationMethods:
        return "fp8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Fp8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
        )

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            if self.is_checkpoint_fp8_serialized:
                return Fp8LinearMethod(self)
            return Fp8OnlineLinearMethod(self)
        return None


class Fp8LinearMethod(LinearMethodBase):
    """
    Linear method for FP8.

    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Limitations:
    1. Only supports float8_e4m3fn due to torch._scaled_mm constraints.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config

        if quant_config.activation_scheme == "static":
            activation_quant_key = kFp8StaticTensorSym
        else:
            activation_quant_key = kFp8DynamicTensorSym

        if supports_fp8_compute():
            self.fp8_linear = init_fp8_kernel(
                activation_quant_key=activation_quant_key,
                weight_quant_key=kFp8StaticTensorSym,
                out_dtype=None,
            )
        else:
            self.fp8_linear = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.orig_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=FP8_DTYPE,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = PerTensorScaleParameter(
            data=torch.ones(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        if self.quant_config.activation_scheme == "static":
            input_scale = PerTensorScaleParameter(
                data=torch.ones(len(output_partition_sizes),
                                dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("input_scale", input_scale)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        weight = layer.weight.data
        weight_scale = layer.weight_scale.data
        input_scale = getattr(layer, "input_scale", None)
        if input_scale is not None:
            input_scale = input_scale.data

        logical_widths = getattr(layer, "logical_widths", [weight.shape[0]])

        if weight_scale.numel() > 1:
            max_w_scale = weight_scale.max()
            start = 0
            for i, width in enumerate(logical_widths):
                shard = weight[start:start + width, :]
                if weight_scale[i] != max_w_scale:
                    dequant = shard.float() * weight_scale[i]
                    requant = ((dequant / max_w_scale).clamp(
                        -FP8_MAX, FP8_MAX).to(FP8_DTYPE))
                    weight[start:start + width, :] = requant
                start += width
            weight_scale = max_w_scale

            if input_scale is not None:
                input_scale = input_scale.max()
        else:
            weight_scale = weight_scale.squeeze()

        transposed = self.fp8_linear is not None
        if transposed:
            weight = weight.t()

        layer._fp8_weight_transposed = transposed
        layer.weight = nn.Parameter(weight, requires_grad=False)
        set_weight_attrs(layer.weight, {"output_dtype": layer.orig_dtype})
        layer.weight_scale = nn.Parameter(weight_scale, requires_grad=False)

        if input_scale is not None:
            layer.input_scale = nn.Parameter(input_scale, requires_grad=False)
        else:
            layer.input_scale = None

        if self.fp8_linear is not None:
            self.fp8_linear.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.fp8_linear is not None:
            return self.fp8_linear.apply_weights(layer, x, bias)
        return self._apply_dequant(layer, x, bias)

    def _apply_dequant(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        input_scale = layer.input_scale
        out_dtype = getattr(layer, "orig_dtype", torch.bfloat16)

        if weight_scale.dim() == 0:
            weight_scale = weight_scale.unsqueeze(0)

        w_dequant = (weight.to(out_dtype) *
                     weight_scale.to(out_dtype).unsqueeze(1))

        if getattr(layer, "_fp8_weight_transposed", False):
            w_dequant = w_dequant.t()

        if input_scale is not None:
            if input_scale.dim() == 0:
                input_scale = input_scale.unsqueeze(0)
            x = x.to(out_dtype) * input_scale.to(out_dtype)
        else:
            x = x.to(out_dtype)

        return F.linear(x, w_dequant, bias).to(out_dtype)


class Fp8OnlineLinearMethod(Fp8LinearMethod):
    """Quantizes bf16/fp16 weights to FP8 at load time (no pre-quantized checkpoint needed)."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.orig_dtype = params_dtype

        weight = nn.Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        from fastvideo.layers.quantization.fp8_utils import FP8_MIN_SCALE

        weight = layer.weight.data
        w_absmax = weight.abs().amax().float()
        w_scale = (w_absmax / FP8_MAX).clamp(min=FP8_MIN_SCALE)
        w_fp8 = ((weight.float() / w_scale).clamp(-FP8_MAX,
                                                  FP8_MAX).to(FP8_DTYPE))

        transposed = self.fp8_linear is not None
        if transposed:
            w_fp8 = w_fp8.t()

        layer._fp8_weight_transposed = transposed
        layer.weight = nn.Parameter(w_fp8, requires_grad=False)
        set_weight_attrs(layer.weight, {"output_dtype": layer.orig_dtype})
        layer.weight_scale = nn.Parameter(w_scale, requires_grad=False)
        layer.input_scale = None

        if self.fp8_linear is not None:
            self.fp8_linear.process_weights_after_loading(layer)
