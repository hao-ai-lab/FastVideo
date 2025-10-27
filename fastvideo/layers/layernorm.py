# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/layers/layernorm.py
"""Custom normalization layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.layers.custom_op import CustomOp


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
        var_hidden_size: int | None = None,
        has_weight: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (None if var_hidden_size == hidden_size
                                       else var_hidden_size)
        self.has_weight = has_weight

        from fastvideo.platforms import current_platform

        self.weight = torch.ones(hidden_size) if current_platform.is_cuda_alike(
        ) else torch.ones(hidden_size, dtype=dtype)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)

    # if we do fully_shard(model.layer_norm), and we call layer_form.forward_native(input) instead of layer_norm(input),
    # we need to call model.layer_norm.register_fsdp_forward_method(model, "forward_native") to make sure fsdp2 hooks are triggered
    # for mixed precision and cpu offloading

    # the even better way might be fully_shard(model.layer_norm, mp_policy=, cpu_offloading=), and call model.layer_norm(input). everything should work out of the box
    # because fsdp2 hooks will be triggered with model.layer_norm.__call__
    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        # x.shape: [batch_size, seq_len, inner_dim]
        if gate.dim() == 4:
            # gate.shape: [batch_size, num_frames, 1, inner_dim]
            num_frames = gate.shape[1]
            frame_seqlen = x.shape[1] // num_frames
            return residual + (x.unflatten(
                dim=1, sizes=(num_frames, frame_seqlen)) * gate).flatten(1, 2)
        else:
            # gate.shape: [batch_size, 1, inner_dim]
            return residual + x * gate


# adapted from Diffusers: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/normalization.py
# NOTE(will): Needed to match behavior of diffusers and wan2.1 even while using
# FSDP's MixedPrecisionPolicy
class FP32LayerNorm(nn.LayerNorm):

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


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
        compute_dtype: torch.dtype | None = None,
        prefix: str = "",
    ):
        super().__init__()
        if norm_type == "rms":
            self.norm = RMSNorm(hidden_size,
                                has_weight=elementwise_affine,
                                eps=eps,
                                dtype=dtype)
        elif norm_type == "layer":
            if compute_dtype == torch.float32:
                self.norm = FP32LayerNorm(hidden_size,
                                          elementwise_affine=elementwise_affine,
                                          eps=eps)
            else:
                self.norm = nn.LayerNorm(hidden_size,
                                         elementwise_affine=elementwise_affine,
                                         eps=eps,
                                         dtype=dtype)
        else:
            raise NotImplementedError(f"Norm type {norm_type} not implemented")

    def forward(self, residual: torch.Tensor, x: torch.Tensor,
                gate: torch.Tensor | int, shift: torch.Tensor,
                scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply gated residual connection, followed by layernorm and 
        scale/shift in a single fused operation.
        
        Returns:
            Tuple containing:
            - normalized and modulated output
            - residual value (value after residual connection 
              but before normalization)
        """
        # x.shape: [batch_size, seq_len, inner_dim]
        # Apply residual connection with gating
        if isinstance(gate, int):
            # used by cross-attention, should be 1
            assert gate == 1
            residual_output = residual + x
        elif isinstance(gate, torch.Tensor):
            if gate.dim() == 4:
                # gate.shape: [batch_size, num_frames, 1, inner_dim]
                num_frames = gate.shape[1]
                frame_seqlen = x.shape[1] // num_frames
                residual_output = residual + (
                    x.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) *
                    gate).flatten(1, 2)
            else:
                # used by bidirectional self attention
                # gate.shape: [batch_size, 1, inner_dim]
                residual_output = residual + x * gate
        else:
            raise ValueError(f"Gate type {type(gate)} not supported")
        # residual_output.shape: [batch_size, seq_len, inner_dim]

        # Apply normalization
        normalized = self.norm(residual_output)
        # Apply scale and shift
        if isinstance(scale, torch.Tensor) and scale.dim() == 4:
            # scale.shape: [batch_size, num_frames, 1, inner_dim]
            # shift.shape: [batch_size, num_frames, 1, inner_dim]
            num_frames = scale.shape[1]
            frame_seqlen = normalized.shape[1] // num_frames
            modulated = (
                normalized.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) *
                (1.0 + scale) + shift).flatten(1, 2)
        else:
            modulated = normalized * (1.0 + scale) + shift
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
        compute_dtype: torch.dtype | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.compute_dtype = compute_dtype
        if norm_type == "rms":
            self.norm = RMSNorm(hidden_size,
                                has_weight=elementwise_affine,
                                eps=eps)
        elif norm_type == "layer":
            if self.compute_dtype == torch.float32:
                self.norm = FP32LayerNorm(hidden_size,
                                          elementwise_affine=elementwise_affine,
                                          eps=eps)
            else:
                self.norm = nn.LayerNorm(hidden_size,
                                         elementwise_affine=elementwise_affine,
                                         eps=eps,
                                         dtype=dtype)
        else:
            raise NotImplementedError(f"Norm type {norm_type} not implemented")

    def forward(self, x: torch.Tensor, shift: torch.Tensor,
                scale: torch.Tensor) -> torch.Tensor:
        """Apply ln followed by scale and shift in a single fused operation."""
        # x.shape: [batch_size, seq_len, inner_dim]
        normalized = self.norm(x)
        if self.compute_dtype == torch.float32:
            normalized = normalized.float()

        if scale.dim() == 4:
            # scale.shape: [batch_size, num_frames, 1, inner_dim]
            num_frames = scale.shape[1]
            frame_seqlen = normalized.shape[1] // num_frames
            output = (
                normalized.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) *
                (1.0 + scale) + shift).flatten(1, 2)
        else:
            # scale.shape: [batch_size, 1, inner_dim]
            # shift.shape: [batch_size, 1, inner_dim]
            output = normalized * (1.0 + scale) + shift

        if self.compute_dtype == torch.float32:
            output = output.to(x.dtype)

        return output
