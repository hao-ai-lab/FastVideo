# SPDX-License-Identifier: Apache-2.0
"""Generic FP8 quantization backed by ``torch._scaled_mm``.

Matches linear layers by suffix (``to_q/k/v/to_out``, ``ffn.fc_in/fc_out``).
Supports per-tensor (default, fast) and per-channel (higher accuracy) granularity.
Falls back to bf16 dequant on GPUs older than sm89.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from v2._vendor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from v2._vendor.models.utils import set_weight_attrs

logger = logging.getLogger(__name__)

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)

_FP8_SUFFIXES = (
    "ffn.fc_in",
    "ffn.fc_out",
    "to_q",
    "to_k",
    "to_v",
    "to_out",
)


def _supports_fp8_compute() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] > 8 or (cap[0] == 8 and cap[1] >= 9)


def _quantize_tensorwise(x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_absmax = x_2d.abs().amax().float()
    x_scale = (x_absmax / FP8_MAX).clamp(min=FP8_MIN_SCALE)
    x_fp8 = (x_2d / x_scale.to(x_2d.dtype)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, x_scale.view(1)


def _quantize_rowwise(x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_absmax = x_2d.abs().amax(dim=-1, keepdim=True).float()
    x_scale = (x_absmax / FP8_MAX).clamp(min=FP8_MIN_SCALE)
    x_fp8 = (x_2d / x_scale.to(x_2d.dtype)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, x_scale


class FP8QuantizeMethod(QuantizeMethodBase):
    """FP8 linear method.

    ``granularity='tensor'`` uses per-tensor weight and dynamic activation
    scales. ``granularity='channel'`` uses per-output-channel weight and
    per-token activation scales.
    """

    def __init__(self, granularity: str = "tensor"):
        super().__init__()
        self.granularity = granularity

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_dim = layer._fp8_weight.shape[0]
        original_shape = x.shape

        if not _supports_fp8_compute():
            return self._apply_dequant(layer, x, bias)

        if self.granularity == "channel":
            x_fp8, x_scale = _quantize_rowwise(x.reshape(-1, x.shape[-1]))
        else:
            x_fp8, x_scale = _quantize_tensorwise(x.reshape(-1, x.shape[-1]))

        w_fp8 = layer._fp8_weight
        w_scale = layer._fp8_weight_scale
        scale_a = x_scale.to(device=x_fp8.device, dtype=torch.float32).contiguous()
        scale_b = w_scale.view(1, -1) if self.granularity == "channel" else w_scale
        scale_b = scale_b.to(device=w_fp8.device, dtype=torch.float32).contiguous()

        out = torch._scaled_mm(
            x_fp8,
            w_fp8.t(),
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.bfloat16,
        )
        if isinstance(out, tuple):
            out = out[0]
        if bias is not None:
            out = out + bias
        return out.view(*original_shape[:-1], out_dim)

    def _apply_dequant(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_dim = layer._fp8_weight.shape[0]
        original_shape = x.shape
        w_fp8 = layer._fp8_weight
        w_scale = layer._fp8_weight_scale.to(x.dtype)
        weight = w_fp8.to(x.dtype) * w_scale.unsqueeze(1)
        out = F.linear(x, weight, bias)
        return out.view(*original_shape[:-1], out_dim)


class FP8Config(QuantizationConfig):
    """FP8 e4m3 quantization via suffix matching on standard linear layer names."""

    def __init__(self, granularity: str = "tensor"):
        super().__init__()
        if granularity not in ("tensor", "channel"):
            raise ValueError(f"granularity must be 'tensor' or 'channel', got {granularity!r}")
        self.granularity = granularity

    def get_name(self) -> str:
        return "FP8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FP8Config:
        return cls(granularity=config.get("granularity", "tensor"))

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from v2._vendor.layers.linear import LinearBase

        if isinstance(layer, LinearBase) and any(s in prefix for s in _FP8_SUFFIXES):
            return FP8QuantizeMethod(granularity=self.granularity)
        return None


def convert_model_to_fp8(model: torch.nn.Module) -> None:
    """Quantize all FP8-tagged linear layers in-place after weights are loaded."""
    import gc
    from torch.distributed.tensor import DTensor

    with torch.no_grad():
        for mod in model.modules():
            qm = getattr(mod, "quant_method", None)
            if not isinstance(qm, FP8QuantizeMethod):
                continue
            weight = getattr(mod, "weight", None)
            if weight is None:
                continue
            weight_local = weight.to_local() if isinstance(weight, DTensor) else weight
            if getattr(qm, "granularity", "tensor") == "channel":
                w_absmax = weight_local.detach().abs().amax(dim=1).nan_to_num().float()
                w_scale = (w_absmax / FP8_MAX).clamp(min=FP8_MIN_SCALE)
                w_fp8 = (weight_local / w_scale.to(weight_local.dtype).unsqueeze(1)).clamp(-FP8_MAX,
                                                                                           FP8_MAX).to(FP8_DTYPE)
            else:
                w_absmax = weight_local.detach().abs().amax().nan_to_num().to(torch.float32)
                w_scale = (w_absmax / FP8_MAX).clamp(min=FP8_MIN_SCALE).view(1)
                w_fp8 = (weight_local / w_scale.to(weight_local.dtype)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
            mod.register_buffer("_fp8_weight", w_fp8.contiguous(), persistent=False)
            mod.register_buffer("_fp8_weight_scale", w_scale.to(torch.float32), persistent=False)
            removed_weight = mod._parameters.pop("weight", None)
            if removed_weight is not None:
                removed_weight.grad = None
            del removed_weight, weight, weight_local, w_absmax, w_scale, w_fp8

    gc.collect()
    torch.cuda.empty_cache()


__all__ = [
    "FP8Config",
    "FP8QuantizeMethod",
    "convert_model_to_fp8",
]
