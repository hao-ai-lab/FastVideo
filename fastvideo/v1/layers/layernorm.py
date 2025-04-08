# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/layers/layernorm.py
"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from fastvideo.v1.layers.custom_op import CustomOp


@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (None if var_hidden_size == hidden_size
                                       else var_hidden_size)
        self.has_weight = has_weight

        self.weight = torch.ones(hidden_size)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        hidden_size = x.shape[-1]
        if hidden_size != self.hidden_size:
            raise ValueError("Expected hidden_size to be "
                             f"{self.hidden_size}, but found: {hidden_size}")

        if self.variance_size_override is None:
            x_var = x
        else:
            if hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {hidden_size}")

            x_var = x[:, :, :self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype)
        if self.has_weight:
            x = x * self.weight
        if residual is None:
            return x
        else:
            return x, residual

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)

        from vllm import _custom_ops as ops

        if residual is not None:
            ops.fused_add_rms_norm(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            return x, residual
        out = torch.empty_like(x)
        ops.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        return out

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


class ScaleResidual(nn.Module):
    """
    Applies gated residual connection.
    """

    def __init__(self, prefix: str = ""):
        super().__init__()

    def forward(self, residual: torch.Tensor, x: torch.Tensor,
                gate: torch.Tensor) -> torch.Tensor:
        """Apply gated residual connection."""
        return residual + x * gate


class ScaleResidualLayerNormScaleShift(nn.Module):
    """
    Fused operation that combines:
    1. Gated residual connection
    2. LayerNorm
    3. Scale and shift operations
    
    This reduces memory bandwidth by combining memory-bound operations.
    """

    def __init__(
        self,
        hidden_size: int,
        norm_type: str = "rms",
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        dtype: torch.dtype = torch.float32,
        prefix: str = "",
    ):
        super().__init__()
        if norm_type == "rms":
            self.norm = RMSNorm(hidden_size,
                                has_weight=elementwise_affine,
                                eps=eps,
                                dtype=dtype)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(hidden_size,
                                     elementwise_affine=elementwise_affine,
                                     eps=eps,
                                     dtype=dtype)
        else:
            raise NotImplementedError(f"Norm type {norm_type} not implemented")

    def forward(self, residual: torch.Tensor, x: torch.Tensor,
                gate: torch.Tensor, shift: torch.Tensor,
                scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply gated residual connection, followed by layernorm and 
        scale/shift in a single fused operation.
        
        Returns:
            Tuple containing:
            - normalized and modulated output
            - residual value (value after residual connection 
              but before normalization)
        """
        # Apply residual connection with gating
        residual_output = residual + x * gate
        # Apply normalization
        normalized = self.norm(residual_output)
        # Apply scale and shift
        modulated = normalized * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return modulated, residual_output


class LayerNormScaleShift(nn.Module):
    """
    Fused operation that combines LayerNorm with scale and shift operations.
    This reduces memory bandwidth by combining memory-bound operations.
    """

    def __init__(
        self,
        hidden_size: int,
        norm_type: str = "rms",
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        dtype: torch.dtype = torch.float32,
        prefix: str = "",
    ):
        super().__init__()
        if norm_type == "rms":
            self.norm = RMSNorm(hidden_size,
                                has_weight=elementwise_affine,
                                eps=eps)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(hidden_size,
                                     elementwise_affine=elementwise_affine,
                                     eps=eps,
                                     dtype=dtype)
        else:
            raise NotImplementedError(f"Norm type {norm_type} not implemented")

    def forward(self, x: torch.Tensor, shift: torch.Tensor,
                scale: torch.Tensor) -> torch.Tensor:
        """Apply ln followed by scale and shift in a single fused operation."""
        normalized = self.norm(x)
        return normalized * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
