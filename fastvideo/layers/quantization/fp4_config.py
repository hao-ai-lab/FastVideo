# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any

import torch
from torch.nn.parameter import Parameter

from fastvideo.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from fastvideo.models.utils import set_weight_attrs

try:
    import flashinfer
except ImportError:
    flashinfer = None

logger = logging.getLogger(__name__)


def _require_flashinfer() -> Any:
    if flashinfer is None:
        raise ImportError("flashinfer is required for FP4 quantization. "
                          "Please install flashinfer to use the fp4 quantization backend.")
    return flashinfer


class FP4QuantizeMethod(QuantizeMethodBase):

    def __init__(self) -> None:
        super().__init__()
        self.weight_fp4 = None
        self.weight_scale = None

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int],
                       input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs):
        """Create weights for a linear layer. Note the corrected signature to match LinearMethodBase."""
        weight = Parameter(torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=params_dtype,
        ),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    @torch.compile
    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """Apply FP4 quantized computation."""
        flashinfer_mod = _require_flashinfer()
        out_dim = layer.weight.shape[0]
        original_shape = x.shape
        assert x.dtype == torch.bfloat16 or x.dtype == torch.float16, f"only allow bf16/fp16 inputs to fp4 linear, got {x.dtype}"

        x = x.view(-1, x.shape[-1])

        x_global_sf = (448 * 6) / x.float().abs().nan_to_num().max()
        x_fp4, x_scale = flashinfer_mod.nvfp4_quantize(
            x,
            x_global_sf,
            sfLayout=flashinfer_mod.SfLayout.layout_128x4,
            do_shuffle=False,
        )
        weight_fp4 = layer._fp4_weight
        weight_scale = layer._fp4_weight_scale
        weight_global_sf = layer._weight_global_sf

        out = flashinfer_mod.mm_fp4(
            x_fp4,
            weight_fp4.T,
            x_scale,
            weight_scale.T,
            1.0 / (x_global_sf * weight_global_sf),
            torch.bfloat16,
            None,
            backend="cutlass",
        )

        if bias is not None:
            if bias.device != out.device or bias.dtype != out.dtype:
                bias = bias.to(device=out.device, dtype=out.dtype)
            out = out + bias

        if len(original_shape) == 3:
            out = out.view(original_shape[0], original_shape[1], out_dim)

        return out


class FP4Config(QuantizationConfig):

    def __init__(self) -> None:
        super().__init__()

    def get_name(self):
        return "fp4"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls):
        return 100

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FP4Config":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from fastvideo.layers.linear import LinearBase
        fp4_layers = ["ffn.fc_in", "ffn.fc_out", "to_q", "to_k", "to_v", "to_out"]
        if isinstance(layer, LinearBase) and any(layer_name in prefix for layer_name in fp4_layers):
            return FP4QuantizeMethod()
        return None


@torch.compile
def convert_model_to_fp4(model: torch.nn.Module):
    flashinfer_mod = _require_flashinfer()
    from torch.distributed.tensor import DTensor  # type: ignore
    for mod in model.modules():
        qm = getattr(mod, "quant_method", None)
        if isinstance(qm, FP4QuantizeMethod):
            weight = getattr(mod, "weight", None)
            if weight is None:
                continue
            weight_local = weight.to_local() if isinstance(weight, DTensor) else weight  # type: ignore[arg-type]
            weight_global_sf = (448 * 6) / weight_local.float().abs().nan_to_num().max()
            fp4_w, fp4_s = flashinfer_mod.nvfp4_quantize(
                weight_local,
                weight_global_sf,
                sfLayout=flashinfer_mod.SfLayout.layout_128x4,
                do_shuffle=False,
            )
            mod.register_buffer("_fp4_weight", fp4_w, persistent=False)
            mod.register_buffer("_fp4_weight_scale", fp4_s, persistent=False)
            mod.register_buffer("_weight_global_sf",
                                torch.tensor(weight_global_sf, dtype=torch.bfloat16),
                                persistent=False)
