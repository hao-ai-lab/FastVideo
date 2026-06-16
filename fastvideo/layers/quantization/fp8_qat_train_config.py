# SPDX-License-Identifier: Apache-2.0
"""FP8 (e4m3) quantization-aware *training* linear method (straight-through estimator).

Mirror of ``nvfp4_qat_train_config.py`` but for FP8. The weight stays a trainable
bf16/fp32 master that is fake-quantized to FP8 on every forward, with a
full-precision backward (STE), so the model learns to absorb FP8 linear error.

The STE lives in ``fastvideo.layers.fp8linear._LinearFWD8BWD16Fn`` (FP8 forward
via ``torch._scaled_mm`` on sm89+, with a bf16 fake-quant fallback on older GPUs;
full-precision backward). This method bridges it into the standard
``quant_config`` path, so it activates via ``transformer_quant="fp8_qat_train"``
on the same Wan-2.1 layers as the FP4 path (to_q/k/v/out + ffn). No conversion is
needed: the weight is kept in full precision and quantized on the fly each step.

Unlike the FP4 path this needs no flashinfer and runs on any sm89+ GPU (and even
older ones via the bf16 fallback), not just Blackwell.
"""
import logging
from typing import Any

import torch
from torch.nn.parameter import Parameter

from fastvideo.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from fastvideo.models.utils import set_weight_attrs

logger = logging.getLogger(__name__)


class FP8QATTrainQuantizeMethod(QuantizeMethodBase):

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int, output_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        # Trainable master weight, fake-quantized to FP8 on each forward.
        weight = Parameter(torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=params_dtype,
        ),
                           requires_grad=True)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        # FP8 forward + full-precision backward (STE).
        from fastvideo.layers.fp8linear import _LinearFWD8BWD16Fn
        return _LinearFWD8BWD16Fn.apply(x, layer.weight, bias, "tensor")


class FP8QATTrainConfig(QuantizationConfig):

    def __init__(self) -> None:
        super().__init__()

    def get_name(self):
        return "fp8_qat_train"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls):
        return 89

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FP8QATTrainConfig":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from fastvideo.layers.linear import LinearBase
        fp8_layers = ["ffn.fc_in", "ffn.fc_out", "to_q", "to_k", "to_v", "to_out"]
        if isinstance(layer, LinearBase) and any(layer_name in prefix for layer_name in fp8_layers):
            return FP8QATTrainQuantizeMethod()
        return None
