# SPDX-License-Identifier: Apache-2.0
"""NVFP4 quantization-aware *training* linear method (straight-through estimator).

The inference ``nvfp4_qat`` config quantizes each weight to FP4 once at load time
(``convert_model_to_fp4``) and has no gradient path — it is inference only. For
QAT *finetuning* the weight must stay a trainable bf16/fp32 master that is
fake-quantized to FP4 on every forward, with a full-precision backward (a
straight-through estimator), so the model learns to absorb FP4 linear error.

That STE already exists in ``fastvideo.layers.fp4linear._LinearFWD4BWD16Fn``
(FP4 forward, full-precision backward) but is otherwise unwired. This method
bridges it into the standard ``quant_config`` path, so it activates via
``transformer_quant="nvfp4_qat_train"`` on the same Wan-2.1 layers as nvfp4_qat
(to_q/k/v/out + ffn). No ``convert_model_to_fp4`` is needed: the weight is kept
in full precision and quantized on the fly each step.
"""
import logging
from typing import Any

import torch
from torch.nn.parameter import Parameter

from fastvideo.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from fastvideo.models.utils import set_weight_attrs

logger = logging.getLogger(__name__)


class NVFP4QATTrainQuantizeMethod(QuantizeMethodBase):

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int, output_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        # Trainable master weight, fake-quantized to FP4 on each forward.
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
        # FP4 forward + full-precision backward (STE).
        from fastvideo.layers.fp4linear import _LinearFWD4BWD16Fn
        return _LinearFWD4BWD16Fn.apply(x, layer.weight, bias, "cutlass", 16, True)


class NVFP4QATTrainConfig(QuantizationConfig):

    def __init__(self) -> None:
        super().__init__()

    def get_name(self):
        return "nvfp4_qat_train"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls):
        return 100

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NVFP4QATTrainConfig":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from fastvideo.layers.linear import LinearBase
        fp4_layers = ["ffn.fc_in", "ffn.fc_out", "to_q", "to_k", "to_v", "to_out"]
        if isinstance(layer, LinearBase) and any(layer_name in prefix for layer_name in fp4_layers):
            return NVFP4QATTrainQuantizeMethod()
        return None
