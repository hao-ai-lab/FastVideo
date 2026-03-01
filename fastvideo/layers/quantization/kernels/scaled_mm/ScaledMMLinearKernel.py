# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/layers/quantization/kernels/scaled_mm/ScaledMMLinearKernel.py
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from fastvideo.layers.quantization.input_quant_fp8 import QuantFP8
from fastvideo.layers.quantization.utils.quant_utils import FP8_DTYPE, QuantKey


@dataclass
class FP8ScaledMMLinearLayerConfig:
    weight_quant_key: QuantKey
    activation_quant_key: QuantKey
    out_dtype: torch.dtype | None


class FP8ScaledMMLinearKernel(ABC):

    @classmethod
    @abstractmethod
    def is_supported(
            cls,
            compute_capability: int | None = None) -> tuple[bool, str | None]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(
            cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        raise NotImplementedError

    def __init__(self, c: FP8ScaledMMLinearLayerConfig,
                 layer_param_names: Sequence[str]) -> None:
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c
        self.layer_param_names = layer_param_names

        act_scale = c.activation_quant_key.scale
        self.quant_fp8 = QuantFP8(
            static=act_scale.static,
            group_shape=act_scale.group_shape,
        )

    def process_weights_after_loading(  # noqa: B027
            self, layer: torch.nn.Module) -> None:
        pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w, w_s, x_s, x_s_ub = self._get_layer_params(layer)

        x_2d = x.view(-1, x.shape[-1])
        output_shape = [*x.shape[:-1], w.shape[1]]
        out_dtype = x.dtype if self.config.out_dtype is None else self.config.out_dtype

        x_2d_q = x_2d
        if x.dtype != FP8_DTYPE:
            x_2d_q, x_s = self.quant_fp8(x_2d, x_s, x_s_ub)

        return self.apply_scaled_mm(
            A=x_2d_q,
            B=w,
            out_dtype=out_dtype,
            As=x_s,
            Bs=w_s,
            bias=bias,
            output_shape=output_shape,
        )

    def _get_layer_params(
        self, layer: torch.nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor
               | None]:
        w, w_s, x_s, x_s_ub = self.layer_param_names
        return (
            getattr(layer, w),
            getattr(layer, w_s),
            getattr(layer, x_s, None),
            getattr(layer, x_s_ub, None),
        )

    @abstractmethod
    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        raise NotImplementedError
