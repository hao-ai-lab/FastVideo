# SPDX-License-Identifier: Apache-2.0
"""MXFP8 quantization for MiniMax-H3 transformer-block feed-forward layers."""

from __future__ import annotations

import re
from typing import Any

import torch
from torch.nn.parameter import Parameter

from fastvideo.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from fastvideo.logger import init_logger
from fastvideo.models.utils import set_weight_attrs

logger = init_logger(__name__)

_MINIMAX_H3_FF_PREFIX = re.compile(r"(?:^|\.)transformer_blocks\.\d+\.ff\.(?:fc_in|fc_out)$")
_MINIMAX_H3_ATTN_PREFIX = re.compile(r"(?:^|\.)transformer_blocks\.\d+\.attn\.(?:to_q|to_k|to_v|to_out)$")


class MXFP8QuantizeMethod(QuantizeMethodBase):
    """Dynamically quantize activations against prequantized MXFP8 weights."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        """Create the BF16 checkpoint weight and non-persistent MXFP8 buffers."""
        del input_size, output_size
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
        layer.register_buffer("_mxfp8_weight", None, persistent=False)
        layer.register_buffer("_mxfp8_weight_scale", None, persistent=False)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Prequantize one adapter-merged BF16 linear weight."""
        from fastvideo.layers.mxfp8linear import quantize_mxfp8_weight_blockwise

        quantized_weight, blocked_scales = quantize_mxfp8_weight_blockwise(layer.weight.detach())
        layer._mxfp8_weight = quantized_weight
        layer._mxfp8_weight_scale = blocked_scales

    def apply(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        bias: torch.Tensor | None = None,
        pre_quantized: tuple[torch.Tensor, torch.Tensor, None] | None = None,
    ) -> torch.Tensor:
        """Quantize one BF16 activation and apply the prequantized linear.

        ``pre_quantized`` lets several linears that share the same input (Q/K/V) reuse one
        MXFP8 activation quantization instead of quantizing it once per projection.
        """
        from fastvideo.layers.mxfp8linear import quantize_mxfp8_blockwise

        original_shape = hidden_states.shape
        if pre_quantized is not None:
            activation_values, activation_scales, _ = pre_quantized
            if activation_values.dim() > 2:
                activation_values = activation_values.reshape(-1, activation_values.shape[-1])
            if activation_scales.dim() > 2:
                activation_scales = activation_scales.reshape(-1, activation_scales.shape[-1])
        else:
            hidden_states_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
            activation_values, activation_scales = quantize_mxfp8_blockwise(hidden_states_2d)
        output_2d = self.apply_quantized(layer, activation_values, activation_scales, bias)
        return output_2d.reshape(*original_shape[:-1], layer.output_size)

    def quantize_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Pre-quantize an activation once for reuse across to_q/to_k/to_v."""
        from fastvideo.layers.mxfp8linear import quantize_mxfp8_blockwise

        x_2d = x.reshape(-1, x.shape[-1])
        activation_values, activation_scales = quantize_mxfp8_blockwise(x_2d)
        return activation_values, activation_scales, None

    def wants_prequantized_input(self) -> bool:
        return True

    def apply_quantized(
        self,
        layer: torch.nn.Module,
        activation_values: torch.Tensor,
        activation_scales: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply one linear to an activation that is already in MXFP8."""
        from fastvideo.layers.mxfp8linear import mxfp8_scaled_mm

        if layer._mxfp8_weight is None or layer._mxfp8_weight_scale is None:
            raise RuntimeError(f"MXFP8 weight buffers are not initialized for {layer.prefix}.")
        return mxfp8_scaled_mm(
            activation_values,
            activation_scales,
            layer._mxfp8_weight,
            layer._mxfp8_weight_scale,
            bias,
        )


class MXFP8Config(QuantizationConfig):
    """Select MXFP8 for the main MiniMax-H3 transformer-block FFN linears."""

    def get_name(self) -> str:
        return "MXFP8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MXFP8Config:
        del config
        return cls()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        from fastvideo.layers.linear import LinearBase

        if not isinstance(layer, LinearBase):
            return None
        if _MINIMAX_H3_FF_PREFIX.search(prefix):
            return MXFP8QuantizeMethod()
        if _MINIMAX_H3_ATTN_PREFIX.search(prefix):
            # Opt-in extension to the attention projections (NVlabs' full "six big
            # Linears"). Kept separate from the FFN set so the quality ladder can
            # measure FFN-only and FFN+attention independently.
            import fastvideo.envs as envs
            if envs.FASTVIDEO_H3_MXFP8_ATTN.get():
                return MXFP8QuantizeMethod()
        return None


def convert_model_to_mxfp8(model: torch.nn.Module) -> int:
    """Prequantize every MXFP8-tagged weight and return the converted count."""
    import fastvideo.envs as envs

    release_bf16 = envs.FASTVIDEO_H3_MXFP8_RELEASE_BF16.get()
    converted_count = 0
    with torch.no_grad():
        for module in model.modules():
            quant_method = getattr(module, "quant_method", None)
            if not isinstance(quant_method, MXFP8QuantizeMethod):
                continue
            if converted_count == 0 and (not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10):
                raise RuntimeError("MXFP8 inference requires an NVIDIA Blackwell GPU with compute capability 10.0+.")
            quant_method.process_weights_after_loading(module)
            converted_count += 1
            if release_bf16 and module.weight is not None and module.weight.numel():
                # Exact-tier memory win: the quantized buffers are the only tensors used by
                # ``MXFP8QuantizeMethod.apply`` from here on, so hand the BF16 storage back.
                # Keep a zero-element Parameter (not None) so ``.weight.dtype`` reads stay valid.
                module.weight.data = torch.empty(0, dtype=module.weight.dtype, device=module.weight.device)

    if converted_count:
        logger.info("Prequantized %d MiniMax-H3 feed-forward linear weights to MXFP8", converted_count)
        torch.cuda.empty_cache()
    return converted_count


__all__ = [
    "MXFP8Config",
    "MXFP8QuantizeMethod",
    "convert_model_to_mxfp8",
]
