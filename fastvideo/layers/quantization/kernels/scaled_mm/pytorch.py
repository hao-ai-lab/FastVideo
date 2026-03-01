# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/layers/quantization/kernels/scaled_mm/pytorch.py
import math

import torch

from fastvideo.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)


class PerTensorTorchFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):

    @classmethod
    def is_supported(
            cls,
            compute_capability: int | None = None) -> tuple[bool, str | None]:
        if not torch.cuda.is_available():
            return False, "requires CUDA."
        if compute_capability is not None and compute_capability < 89:
            return False, "requires compute capability 89 and above."
        return True, None

    @classmethod
    def can_implement(
            cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        per_tensor_activation = (
            c.activation_quant_key.scale.group_shape.is_per_tensor())
        per_tensor_weight = (
            c.weight_quant_key.scale.group_shape.is_per_tensor())
        if not (per_tensor_activation and per_tensor_weight):
            return False, "requires per tensor activation and weight scales."
        return True, None

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
        output = torch._scaled_mm(A,
                                  B,
                                  out_dtype=out_dtype,
                                  scale_a=As,
                                  scale_b=Bs,
                                  bias=bias)
        if type(output) is tuple and len(output) == 2:
            output = output[0]
        n_rows = math.prod(output_shape[:-1])
        return torch.narrow(output, 0, 0, n_rows).view(*output_shape)


class ChannelWiseTorchFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):

    @classmethod
    def is_supported(
            cls,
            compute_capability: int | None = None) -> tuple[bool, str | None]:
        if not torch.cuda.is_available():
            return False, "requires CUDA."
        if compute_capability is not None and compute_capability < 89:
            return False, "requires compute capability 89 and above."
        return True, None

    @classmethod
    def can_implement(
            cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        per_tensor_activation = (
            c.activation_quant_key.scale.group_shape.is_per_tensor())
        per_tensor_weight = (
            c.weight_quant_key.scale.group_shape.is_per_tensor())
        if per_tensor_activation and per_tensor_weight:
            return False, "cannot be used with per tensor activation and weight scales."
        return True, None

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
        dummy = torch.ones(1, dtype=torch.float32, device=A.device)
        output = torch._scaled_mm(
            A,
            B,
            scale_a=dummy,
            scale_b=dummy,
            out_dtype=torch.float32,
        )
        if type(output) is tuple and len(output) == 2:
            output = output[0]

        n_rows = math.prod(output_shape[:-1])
        output = torch.narrow(output, 0, 0, n_rows)

        x_scale = torch.narrow(As, 0, 0, n_rows) if As.numel() > 1 else As

        output = output * x_scale * Bs.t()
        if bias is not None:
            output = output + bias
        return output.to(out_dtype).view(*output_shape)
