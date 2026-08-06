# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MAGI-2 preview model (magi2_preview) - all layers and model definition."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import partial
from typing import Callable, List, Literal, Optional, Tuple, TypeAlias, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange, repeat
from flash_attn_interface import flash_attn_3_cuda
from magi_attn_extensions.fa3_interface_with_sink import (
    FA3VarlenFuncWithSink,
    fa3_func_with_sink,
    fa3_varlen_func_with_sink,
)
from magi_compiler import magi_compile, magi_register_custom_op
from fastvideo.configs.models.dits.magi2 import (
    Magi2MoEConfig,
    Magi2PreviewArchConfig,
    Magi2PreviewVideoConfig,
)
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.magi2_runtime import psm
from fastvideo.models.dits.magi2_runtime.all_to_all_primitive import (
    batch_scatter_head_gather_seqlen,
    scatter_seqlen_gather_head,
)
from fastvideo.models.dits.magi2_runtime.context_parallel import ulysses_scheduler
from fastvideo.models.dits.magi2_runtime.flash_mh_moe import (
    compute_topk_probs_and_indices as _mh_moe_compute_topk,
    flash_mh_moe_fwd as _mh_moe_fwd,
    flash_mh_moe_global_sort as _mh_moe_global_sort,
)
from torch.library import triton_op, wrap_triton


logger = logging.getLogger(__name__)

ModelConfig = Magi2PreviewArchConfig
MoEConfig = Magi2MoEConfig


def env_is_true(name: str) -> bool:
    """Interpret one MAGI-2 environment variable as a Boolean flag."""
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def _resolve_params_dtype(value: torch.dtype | str) -> torch.dtype:
    """Convert a serialized MAGI-2 parameter dtype into a PyTorch dtype."""
    if isinstance(value, torch.dtype):
        return value
    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "torch.bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "torch.float16": torch.float16,
        "float32": torch.float32,
        "torch.float32": torch.float32,
    }
    try:
        return dtype_by_name[value]
    except KeyError as error:
        raise ValueError(f"Unsupported MAGI-2 parameter dtype: {value!r}") from error

# ============================================================
# enum
# ============================================================


class MLPActivationType(Enum):
    """Enumeration of supported activation functions for MLP"""

    SWIGLU7 = "swiglu7"


class Modality(IntEnum):
    VIDEO = 0
    AUDIO = 1
    TEXT = 2
    # Special marker modality used ONLY at input to identify time tokens.
    # The model will remap TIME -> TEXT internally before passing to ModalityDispatcher,
    # so we do NOT need to increase num_modality or change parameter shapes.
    TIME = 3


# ============================================================
# activation
# ============================================================


@torch.compile
def swiglu7(
    x, alpha: float = 1.702, limit: float = 7.0, out_dtype: Optional[torch.dtype] = None
):
    """Apply the bounded SwiGLU7 activation in the requested output dtype."""
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer (from GPT-OSS)
    return (out_glu * (x_linear + 1)).to(out_dtype)


def create_activation_func(activation_type: MLPActivationType) -> Callable:
    if activation_type is MLPActivationType.SWIGLU7:
        return swiglu7
    raise ValueError(f"Unknown activation type: {activation_type}")


# ============================================================
# _common
# ============================================================




def _maybe_gather(
    input: torch.Tensor, gather_ids: Optional[torch.Tensor]
) -> torch.Tensor:
    if gather_ids is None:
        return input
    gather_ids = gather_ids.to(input.device)
    if gather_ids.dim() == 1:
        return input.index_select(0, gather_ids)
    return torch.gather(input, 0, gather_ids)


def _expand_grouped_bias(bias: torch.Tensor, m_splits: list[int]) -> torch.Tensor:
    chunks = []
    for expert_idx, split in enumerate(m_splits):
        if split > 0:
            chunks.append(bias[expert_idx].expand(split, -1))
    return torch.cat(chunks, dim=0) if chunks else bias.new_empty((0, bias.size(-1)))


def _m_splits_from(
    cu_seqlens: Optional[torch.Tensor], m_splits: Optional[list[int] | torch.Tensor]
) -> list[int]:
    if isinstance(m_splits, torch.Tensor):
        return [int(v) for v in m_splits.detach().cpu().tolist()]
    if m_splits is not None:
        return [int(v) for v in m_splits]
    if cu_seqlens is None:
        raise ValueError("m_splits or cu_seqlens is required for grouped linear")
    return [int(v) for v in torch.diff(cu_seqlens).detach().cpu().tolist()]


def _torch_grouped_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    m_splits: list[int],
    gather_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply expert-specific linear projections to contiguous token groups."""
    ordered_input = _maybe_gather(input, gather_ids)
    outputs = []
    start = 0
    for expert_idx, split in enumerate(m_splits):
        end = start + split
        if end > start:
            outputs.append(
                F.linear(
                    ordered_input[start:end],
                    weight[expert_idx],
                    None if bias is None else bias[expert_idx],
                )
            )
        start = end
    return (
        torch.cat(outputs, dim=0)
        if outputs
        else ordered_input.new_empty((0, weight.size(1)))
    )



class GroupedLinearBase(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int = 1,
        num_patterns: int = 1,
        bias: bool = False,
        device=None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Allocate expert-stacked weights for grouped linear projections."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.num_patterns = num_patterns
        self.weight = nn.Parameter(
            torch.empty(
                num_experts * out_features, in_features, device=device, dtype=dtype
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(num_experts * out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        input: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        m_splits: Optional[list[int]] = None,
        gather_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project tokens with one expert or expert-specific grouped weights."""
        weight = self.weight.view(self.num_experts, self.out_features, self.in_features)
        bias = (
            self.bias.view(self.num_experts, self.out_features)
            if self.bias is not None
            else None
        )
        if self.num_experts == 1:
            return F.linear(input, weight[0], None if bias is None else bias[0])
        m_splits = _m_splits_from(cu_seqlens, m_splits)
        return _torch_grouped_linear(input, weight, bias, m_splits, gather_ids)


@triton.jit
def _multi_head_rms_norm_fwd_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    weight_bias,
    eps,
    x_stride_t,
    x_stride_h,
    x_stride_d,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    weight_stride_h,
    weight_stride_d,
    T,
    D: tl.constexpr,
    acc_dtype: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Normalize each token head with root mean square statistics in Triton."""
    block_idx_t = tl.program_id(0)
    h = tl.program_id(1)
    t_start = block_idx_t * BLOCK_SIZE_T
    d_block_offs = tl.arange(0, BLOCK_SIZE_D)

    for t_off in tl.static_range(0, BLOCK_SIZE_T):
        t = t_start + t_off

        if t < T:
            x_start_ptr = x_ptr + t * x_stride_t + h * x_stride_h
            out_start_ptr = out_ptr + t * out_stride_t + h * out_stride_h
            weight_start_ptr = weight_ptr + h * weight_stride_h

            rms_sum = tl.zeros([], dtype=acc_dtype)
            for d_block_start in tl.static_range(0, D, BLOCK_SIZE_D):
                d_ptrs = d_block_start + d_block_offs
                d_mask = d_ptrs < D
                x = tl.load(
                    x_start_ptr + d_ptrs * x_stride_d, mask=d_mask, other=0.0
                ).to(acc_dtype)
                rms_sum += tl.sum(x * x, axis=0)

            rms_norm_inv = tl.rsqrt(rms_sum / D + eps)

            for d_block_start in tl.static_range(0, D, BLOCK_SIZE_D):
                d_ptrs = d_block_start + d_block_offs
                d_mask = d_ptrs < D
                x = tl.load(
                    x_start_ptr + d_ptrs * x_stride_d, mask=d_mask, other=0.0
                ).to(acc_dtype)
                weight = tl.load(
                    weight_start_ptr + d_ptrs * weight_stride_d, mask=d_mask, other=0.0
                ).to(acc_dtype)
                out = x * rms_norm_inv * (weight + weight_bias)
                tl.store(out_start_ptr + d_ptrs * out_stride_d, out, mask=d_mask)


@triton_op("magi2::multi_head_rms_norm_fwd", mutates_args={})
def _multi_head_rms_norm_fwd_op(
    t: torch.Tensor,
    w: torch.Tensor,
    weight_bias: float,
    eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Launch multi-head root mean square normalization with Triton."""
    out = torch.empty(t.shape, dtype=out_dtype, device=t.device)
    T, H, D = t.shape
    block_size_d = min(8192, triton.next_power_of_2(D))
    block_size_t = 1
    grid = (triton.cdiv(T, block_size_t), H)

    wrap_triton(_multi_head_rms_norm_fwd_kernel)[grid](
        t,
        out,
        w,
        weight_bias,
        eps,
        t.stride(0),
        t.stride(1),
        t.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        w.stride(0),
        w.stride(1),
        T,
        D,
        acc_dtype=tl.float32,
        BLOCK_SIZE_T=block_size_t,
        BLOCK_SIZE_D=block_size_d,
    )
    return out


@triton.jit
def _grouped_multi_head_rms_norm_fwd_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    cu_group_sizes_ptr,
    weight_bias,
    eps,
    x_stride_t,
    x_stride_b,
    x_stride_h,
    x_stride_d,
    out_stride_t,
    out_stride_b,
    out_stride_h,
    out_stride_d,
    weight_stride_g,
    weight_stride_h,
    weight_stride_d,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    acc_dtype: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Normalize modality-grouped token heads with Triton."""
    block_idx_t = tl.program_id(0)
    h = tl.program_id(1)

    g = -1
    t_start = -1

    accum_blocks = 0
    for curr_g in range(G):
        g_start = tl.load(cu_group_sizes_ptr + curr_g)
        g_end = tl.load(cu_group_sizes_ptr + curr_g + 1)
        g_size = g_end - g_start
        g_blocks = tl.cdiv(g_size, BLOCK_SIZE_T)

        is_in_group = (block_idx_t >= accum_blocks) & (
            block_idx_t < accum_blocks + g_blocks
        )
        if is_in_group:
            g = curr_g
            rel_block_idx = block_idx_t - accum_blocks
            t_start = g_start + rel_block_idx * BLOCK_SIZE_T

        accum_blocks += g_blocks

    if g == -1:
        return

    t_end = tl.load(cu_group_sizes_ptr + g + 1)
    d_block_offs = tl.arange(0, BLOCK_SIZE_D)

    for t_off in tl.static_range(0, BLOCK_SIZE_T):
        t = t_start + t_off

        if (t < t_end) & (t < T):
            for b in tl.static_range(0, B):
                x_start_ptr = x_ptr + t * x_stride_t + b * x_stride_b + h * x_stride_h
                out_start_ptr = (
                    out_ptr + t * out_stride_t + b * out_stride_b + h * out_stride_h
                )
                weight_start_ptr = (
                    weight_ptr + g * weight_stride_g + h * weight_stride_h
                )

                rms_sum = tl.zeros([], dtype=acc_dtype)
                for d_block_start in tl.static_range(0, D, BLOCK_SIZE_D):
                    d_ptrs = d_block_start + d_block_offs
                    d_mask = d_ptrs < D
                    x = tl.load(
                        x_start_ptr + d_ptrs * x_stride_d, mask=d_mask, other=0.0
                    ).to(acc_dtype)
                    rms_sum += tl.sum(x * x, axis=0)

                rms_norm_inv = tl.rsqrt(rms_sum / D + eps)

                for d_block_start in tl.static_range(0, D, BLOCK_SIZE_D):
                    d_ptrs = d_block_start + d_block_offs
                    d_mask = d_ptrs < D
                    x = tl.load(
                        x_start_ptr + d_ptrs * x_stride_d, mask=d_mask, other=0.0
                    ).to(acc_dtype)
                    weight = tl.load(
                        weight_start_ptr + d_ptrs * weight_stride_d,
                        mask=d_mask,
                        other=0.0,
                    ).to(acc_dtype)
                    out = x * rms_norm_inv * (weight + weight_bias)
                    tl.store(out_start_ptr + d_ptrs * out_stride_d, out, mask=d_mask)


@triton_op("magi2::grouped_multi_head_rms_norm_fwd", mutates_args={})
def _grouped_multi_head_rms_norm_fwd_op(
    t: torch.Tensor,
    w: torch.Tensor,
    cu_group_sizes: torch.Tensor,
    weight_bias: float,
    eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Launch grouped multi-head root mean square normalization with Triton."""
    out = torch.empty(t.shape, dtype=out_dtype, device=t.device)
    T, B, H, D = t.shape
    G = w.shape[0]
    block_size_d = min(8192, triton.next_power_of_2(D))
    block_size_t = 1
    grid = (triton.cdiv(T, block_size_t), H, 1)

    wrap_triton(_grouped_multi_head_rms_norm_fwd_kernel)[grid](
        t,
        out,
        w,
        cu_group_sizes,
        weight_bias,
        eps,
        t.stride(0),
        t.stride(1),
        t.stride(2),
        t.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        T,
        B,
        H,
        D,
        G,
        acc_dtype=tl.float32,
        BLOCK_SIZE_T=block_size_t,
        BLOCK_SIZE_D=block_size_d,
    )
    return out


def _shape_as_grouped_heads(
    x: torch.Tensor, num_heads: int, head_dim: int
) -> tuple[int, int]:
    """Validate a head tensor and derive its sequence and batch dimensions."""
    seqlen = x.shape[0]
    if x.shape[-1] == num_heads * head_dim:
        hidden = num_heads * head_dim
        return seqlen, x.numel() // (seqlen * hidden)
    if x.dim() >= 3 and x.shape[-2] == num_heads and x.shape[-1] == head_dim:
        hidden = num_heads * head_dim
        return seqlen, x.numel() // (seqlen * hidden)
    raise AssertionError(
        "x must have shape [t, *batch_dims, num_heads * head_dim] "
        "or [t, *batch_dims, num_heads, head_dim]."
    )


def _multi_head_rms_norm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float,
    weight_bias: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply Triton root mean square normalization to multi-head tokens."""
    t = x.view(-1, num_heads, head_dim)
    w = weight.view(num_heads, head_dim)
    return _multi_head_rms_norm_fwd_op(t, w, weight_bias, eps, out_dtype).view_as(x)


def _grouped_multi_head_rms_norm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    num_groups: int,
    num_heads: int,
    head_dim: int,
    cu_group_sizes: torch.Tensor,
    eps: float,
    weight_bias: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply Triton root mean square normalization to grouped token heads."""
    seqlen, batch = _shape_as_grouped_heads(x, num_heads, head_dim)
    t = x.view(seqlen, batch, num_heads, head_dim)
    w = weight.view(num_groups, num_heads, head_dim)
    return _grouped_multi_head_rms_norm_fwd_op(
        t, w, cu_group_sizes.contiguous(), weight_bias, eps, out_dtype
    ).view_as(x)




class MultiModalityRMSNorm(nn.Module):
    __constants__ = ["dim", "eps", "num_modality"]
    dim: int
    eps: float
    num_modality: int

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        device: torch.device | None = None,
        num_modality: int = 1,
        num_patterns: int = 1,
        out_dtype: torch.dtype | None = None,
        backend: Literal["torch", "triton"] = "triton",
    ):
        """Configure modality-specific normalization weights and execution backend."""
        super().__init__()
        if backend not in {"torch", "triton"}:
            raise ValueError(f"Unsupported RMSNorm backend: {backend}")
        self.dim = dim
        self.eps = eps
        self.num_modality = num_modality
        self.num_patterns = num_patterns
        self.out_dtype = out_dtype
        self.backend = backend

        self.compute_dtype = torch.float32
        self.weight_bias = 1.0

        self.weight = torch.nn.Parameter(
            torch.zeros(
                num_patterns * dim * num_modality, device=device, dtype=torch.float32
            )
        )

    def forward(
        self, x: torch.Tensor, modality_dispatcher: Optional[ModalityDispatcher] = None
    ) -> torch.Tensor:
        """Normalize tokens with single- or multi-modality weights."""
        target_dtype = self.out_dtype

        if self.num_modality > 1:
            assert modality_dispatcher is not None, (
                "modality_dispatcher is required for multi-modality RMSNorm"
            )
            return self.forward_multi_experts(x, modality_dispatcher, target_dtype)
        else:
            return self.forward_single_expert(x, target_dtype)

    def forward_multi_experts(
        self,
        x: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
        target_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Normalize modality-grouped tokens with Triton or PyTorch."""
        out_dtype = target_dtype if target_dtype is not None else x.dtype
        if self.backend == "triton" and x.is_cuda and not torch.is_grad_enabled():
            return _grouped_multi_head_rms_norm_triton(
                x=x,
                weight=self.weight,
                num_groups=self.num_modality,
                num_heads=self.num_patterns,
                head_dim=self.dim,
                cu_group_sizes=modality_dispatcher.cu_group_sizes.to(
                    device=x.device, dtype=torch.int32
                ),
                eps=self.eps,
                weight_bias=self.weight_bias,
                out_dtype=out_dtype,
            )
        return self._forward_multi_experts_torch_impl(
            x, modality_dispatcher, target_dtype
        )

    @torch.compile(dynamic=True, fullgraph=False)
    def _forward_multi_experts_torch_impl(
        self,
        x: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
        target_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        """Apply modality-specific root mean square normalization in PyTorch."""
        t, original_dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)

        weight_chunked = self.weight.chunk(self.num_modality, dim=-1)
        t_list = modality_dispatcher.dispatch(t)

        for i in range(self.num_modality):
            t_list[i] = t_list[i] * (
                weight_chunked[i].view(self.num_patterns, self.dim) + 1
            )

        t = modality_dispatcher.undispatch(*t_list)

        out_dtype = target_dtype if target_dtype is not None else original_dtype
        return t.to(out_dtype)

    def forward_single_expert(
        self, x: torch.Tensor, target_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Normalize tokens with one root mean square normalization weight set."""
        out_dtype = target_dtype if target_dtype is not None else x.dtype
        if (
            self.backend == "triton"
            and self.num_patterns > 1
            and x.is_cuda
            and not torch.is_grad_enabled()
        ):
            return _multi_head_rms_norm_triton(
                x=x,
                weight=self.weight,
                num_heads=self.num_patterns,
                head_dim=self.dim,
                eps=self.eps,
                weight_bias=self.weight_bias,
                out_dtype=out_dtype,
            )
        return self._forward_single_expert_torch_impl(x, target_dtype)

    @torch.compile(dynamic=True, fullgraph=False)
    def _forward_single_expert_torch_impl(
        self, x: torch.Tensor, target_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        t, original_dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)

        out_dtype = target_dtype if target_dtype is not None else original_dtype

        return (t * (self.weight.view(self.num_patterns, self.dim) + 1)).to(out_dtype)


class CustomGroupedLinear(GroupedLinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        num_patterns: int,
        num_layers_for_initialization: int,
        bias: bool = False,
        device=None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize grouped projection weights for expert token partitions."""
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            num_experts=num_experts,
            num_patterns=num_patterns,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        input: torch.Tensor,
        modality_dispatcher: Optional[Union[ModalityDispatcher, list[int]]] = None,
        gather_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Derive token-group boundaries and apply the grouped projection."""
        if isinstance(modality_dispatcher, ModalityDispatcher):
            m_splits = modality_dispatcher.group_size_cpu
            cu_seqlens = modality_dispatcher.cu_group_sizes
        elif isinstance(modality_dispatcher, torch.Tensor):
            m_splits = modality_dispatcher.tolist()
            cu_seqlens = None
        elif isinstance(modality_dispatcher, list):
            m_splits = modality_dispatcher
            cu_seqlens = None
        else:  # fall back to base linear which does not need split infos
            m_splits = None
            cu_seqlens = None

        return super().forward(
            input=input, cu_seqlens=cu_seqlens, m_splits=m_splits, gather_ids=gather_ids
        )


def create_linear(
    in_features: int,
    out_features: int,
    num_layers: int = 1,
    num_experts: int = 1,
    num_patterns: int = 1,
    bias: bool = True,
    device=None,
    dtype: torch.dtype | None = None,
) -> CustomGroupedLinear:
    """Create the grouped linear layer used by MAGI-2 transformer blocks."""
    return CustomGroupedLinear(
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        num_patterns=num_patterns,
        num_layers_for_initialization=num_layers,
        bias=bias,
        device=device,
        dtype=dtype,
    )


# ============================================================
# element_wise_rope
# ============================================================

Coords: TypeAlias = tuple[int, int, int]


class ElementWiseFourierEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        max_res: int = 224,
        temperature: float = 10000.0,
        in_pixels: bool = True,
        linear_bands: bool = False,
        learnable: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """Configure frequency bands for coordinate-wise Fourier embeddings.

        Args:
            dim: output feature dimension, total number of channels, must be divisible by 6
            max_res: maximum pixel-frequency resolution, used to build the pixel-domain frequency bands
            temperature: temperature parameter for the inverse-frequency mode
            in_pixels: True -> use pixel frequency bands, False -> use inverse frequency bands
            linear_bands: whether the pixel frequency bands are linearly spaced
            learnable: whether to make the frequency bands a trainable parameter
        """
        super().__init__()
        self.dim = dim
        self.in_pixels = in_pixels
        self.learnable = learnable
        self.temperature = temperature
        self.max_res = max_res
        self.linear_bands = linear_bands
        self.device = device
        self.dtype = dtype
        # make the frequency bands a trainable parameter or register them as a buffer
        bands = self.get_default_bands()
        if self.learnable:
            self.bands = nn.Parameter(bands)
        else:
            self.register_buffer("bands", bands)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode token coordinates as element-wise Fourier features.

        Args:
            coords: [L,9], column order is (time, row, col, T, H, W, ref_T, ref_H, ref_W)
        Returns:
            emb: [L, dim] element-wise Fourier embedding
        """
        # use slicing instead of unbind + stack to reduce intermediate tensors
        coords_xyz = coords[:, :3]  # [L,3] -> (t, h, w)
        sizes = coords[:, 3:6]  # [L,3] -> (T, H, W)
        refs = coords[:, 6:9]  # [L,3] -> (ref_T, ref_H, ref_W)

        # compute the scaling factor
        scales = (refs - 1) / (sizes - 1)  # [L,3]

        # NOTE: for points where both ref and size are 1 the scale is fixed to 1; anything else errors out
        scales[(refs == 1) & (sizes == 1)] = 1
        assert not scales.isnan().any(), "scales has nan"
        assert not scales.isinf().any(), "scales has inf"

        # center alignment, only for the h, w dims, not for the t dim
        centers = (sizes - 1) / 2  # [L,3]
        centers[:, 0] = 0  # do not center the time dimension
        coords_xyz = coords_xyz - centers  # [L,3]

        # project onto the frequency bands in one shot: [L,3,B]
        proj = coords_xyz.unsqueeze(-1) * scales.unsqueeze(-1) * self.bands

        # compute sin & cos and concatenate
        sin_proj = proj.sin()  # [L,3,B]
        cos_proj = proj.cos()

        return torch.cat((sin_proj, cos_proj), dim=1).flatten(1)

    def reset_parameters(self):
        bands = self.get_default_bands()
        self.bands.copy_(bands)

    def get_default_bands(self):
        if self.in_pixels:
            raise NotImplementedError("in_pixels are not implemented yet")
        else:
            bands = freq_bands(
                self.dim // 8, temperature=self.temperature, step=1, device=self.device
            ).to(self.dtype)
        return bands


def get_coords(
    shape: Coords,
    ref_feat_shape: Coords,
    offset_thw: Coords = (0, 0, 0),
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    time_positions: Optional[torch.Tensor] = None,
):
    """
    Build the feature-grid coordinates together with their original and reference size metadata.
    Args:
        feat_shape: [T, H, W] original feature-map size
        ref_feat_shape: [T_ref, H_ref, W_ref] reference feature-map size
        device: device the coordinate tensors live on
        time_positions: optional 1D tensor (length == ori_t) used in place of the default arange(ori_t)
                        as the time-dimension coordinates. Together with ref_t == ori_t this bypasses the linear rescale.
    Returns:
        coords: tensor of shape (T*H*W, 9) containing (t, h, w, T, H, W, ref_T, ref_H, ref_W)
    """
    ori_t, ori_h, ori_w = shape
    ref_t, ref_h, ref_w = ref_feat_shape

    offset_t, offset_h, offset_w = offset_thw
    # build the index ranges
    if time_positions is None:
        time_rng = torch.arange(ori_t, device=device, dtype=dtype) + offset_t
    else:
        time_rng = time_positions.to(device=device, dtype=dtype) + offset_t
    height_rng = torch.arange(ori_h, device=device, dtype=dtype) + offset_h
    width_rng = torch.arange(ori_w, device=device, dtype=dtype) + offset_w

    # meshgrid to build the 3D grid (T, H, W)
    time_grid, height_grid, width_grid = torch.meshgrid(
        time_rng, height_rng, width_rng, indexing="ij"
    )

    # stack and flatten
    coords_grid = torch.stack([time_grid, height_grid, width_grid], dim=-1)
    coords_flat = coords_grid.reshape(-1, 3)

    # build and broadcast the size metadata
    meta = torch.tensor(
        [ori_t, ori_h, ori_w, ref_t, ref_h, ref_w], device=device, dtype=dtype
    )
    meta_expanded = meta.expand(coords_flat.size(0), -1)

    # concatenate and return
    return torch.cat([coords_flat, meta_expanded], dim=-1)


# ============================================================
# nd_rotary_pos_embedding
# ============================================================


def ndgrid(*tensors) -> Tuple[torch.Tensor, ...]:
    """generate N-D grid in dimension order.

    The ndgrid function is like meshgrid except that the order of the first two input arguments are switched.

    That is, the statement
    [X1,X2,X3] = ndgrid(x1,x2,x3)

    produces the same result as

    [X2,X1,X3] = meshgrid(x2,x1,x3)

    This naming is based on MATLAB, the purpose is to avoid confusion due to torch's change to make
    torch.meshgrid behaviour move from matching ndgrid ('ij') indexing to numpy meshgrid defaults of ('xy').

    """
    try:
        return torch.meshgrid(*tensors, indexing="ij")
    except TypeError:
        # old PyTorch < 1.10 will follow this path as it does not have indexing arg,
        # the old behaviour of meshgrid was 'ij'
        return torch.meshgrid(*tensors)


def meshgrid(*tensors) -> Tuple[torch.Tensor, ...]:
    """generate N-D grid in spatial dim order.

    The meshgrid function is similar to ndgrid except that the order of the
    first two input and output arguments is switched.

    That is, the statement

    [X,Y,Z] = meshgrid(x,y,z)
    produces the same result as

    [Y,X,Z] = ndgrid(y,x,z)
    Because of this, meshgrid is better suited to problems in two- or three-dimensional Cartesian space,
    while ndgrid is better suited to multidimensional problems that aren't spatially based.
    """

    # NOTE: this will throw in PyTorch < 1.10 as meshgrid did not support indexing arg or have
    # capability of generating grid in xy order before then.
    return torch.meshgrid(*tensors, indexing="xy")


def pixel_freq_bands(
    num_bands: int,
    max_freq: float = 224.0,
    linear_bands: bool = True,
    device: Optional[torch.device] = None,
):
    """Create pixel-domain frequency bands scaled by pi."""
    if linear_bands:
        bands = torch.linspace(
            1.0, max_freq / 2, num_bands, dtype=torch.float32, device=device
        )
    else:
        bands = 2 ** torch.linspace(
            0, math.log(max_freq, 2) - 1, num_bands, dtype=torch.float32, device=device
        )
    return bands * torch.pi


def freq_bands(
    num_bands: int,
    temperature: float = 10000.0,
    step: int = 2,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create inverse-frequency bands for sinusoidal embeddings."""
    exp = (
        torch.arange(0, num_bands, step, dtype=torch.int64, device=device).to(
            torch.float32
        )
        / num_bands
    )
    bands = 1.0 / (temperature**exp)
    return bands


def build_sincos2d_pos_embed(
    feat_shape: List[int],
    dim: int = 64,
    temperature: float = 10000.0,
    reverse_coord: bool = False,
    interleave_sin_cos: bool = False,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build a two-dimensional sinusoidal embedding for a feature grid.

    Args:
        feat_shape:
        dim:
        temperature:
        reverse_coord: stack grid order W, H instead of H, W
        interleave_sin_cos: sin, cos, sin, cos stack instead of sin, sin, cos, cos
        dtype:
        device:

    Returns:

    """
    assert dim % 4 == 0, (
        "Embed dimension must be divisible by 4 for sin-cos 2D position embedding"
    )
    pos_dim = dim // 4
    bands = freq_bands(pos_dim, temperature=temperature, step=1, device=device)

    if reverse_coord:
        feat_shape = feat_shape[::-1]  # stack W, H instead of H, W
    grid = (
        torch.stack(
            ndgrid(
                [
                    torch.arange(s, device=device, dtype=torch.int64).to(torch.float32)
                    for s in feat_shape
                ]
            )
        )
        .flatten(1)
        .transpose(0, 1)
    )
    pos2 = grid.unsqueeze(-1) * bands.unsqueeze(0)
    # FIXME add support for unflattened spatial dim?

    stack_dim = (
        2 if interleave_sin_cos else 1
    )  # stack sin, cos, sin, cos  instead of sin sin cos cos
    pos_emb = torch.stack([torch.sin(pos2), torch.cos(pos2)], dim=stack_dim).flatten(1)
    return pos_emb.to(dtype=dtype)


def build_fourier_pos_embed(
    feat_shape: List[int],
    bands: Optional[torch.Tensor] = None,
    num_bands: int = 64,
    max_res: int = 224,
    temperature: float = 10000.0,
    linear_bands: bool = False,
    include_grid: bool = False,
    in_pixels: bool = True,
    ref_feat_shape: Optional[List[float]] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """Build sine and cosine Fourier features for a feature grid.

    Args:
        feat_shape: Feature shape for embedding.
        bands: Pre-calculated frequency bands.
        num_bands: Number of frequency bands (determines output dim).
        max_res: Maximum resolution for pixel based freq.
        temperature: Temperature for non-pixel freq.
        linear_bands: Linear band spacing for pixel based freq.
        include_grid: Include the spatial grid in output.
        in_pixels: Output in pixel freq.
        ref_feat_shape: Reference feature shape for resize / fine-tune.
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    if bands is None:
        if in_pixels:
            bands = pixel_freq_bands(
                num_bands, float(max_res), linear_bands=linear_bands, device=device
            )
        else:
            bands = freq_bands(
                num_bands, temperature=temperature, step=1, device=device
            )
    else:
        if device is None:
            device = bands.device
        if dtype is None:
            dtype = bands.dtype

    if in_pixels:
        t = [
            torch.linspace(-1.0, 1.0, steps=s, device=device, dtype=torch.float32)
            for s in feat_shape
        ]
    else:
        t = [
            torch.arange(s, device=device, dtype=torch.int64).to(torch.float32)
            for s in feat_shape
        ]
        # align spatial center (H/2,W/2) to (0,0)
        t[1] = t[1] - (feat_shape[1] - 1) / 2
        t[2] = t[2] - (feat_shape[2] - 1) / 2
    if ref_feat_shape is not None:
        # eva's scheme for resizing rope embeddings (ref shape = pretrain)
        # aligning to the endpoint e.g [0,1,2] -> [0, 0.4, 0.8, 1.2, 1.6, 2]
        t_rescaled = []
        for x, f, r in zip(t, feat_shape, ref_feat_shape):
            if f == 1:
                assert r == 1, "ref_feat_shape must be 1 when feat_shape is 1"
                t_rescaled.append(x)
            else:
                t_rescaled.append(x / (f - 1) * (r - 1))
    else:
        t_rescaled = t

    grid = torch.stack(ndgrid(t_rescaled), dim=-1)
    grid = grid.unsqueeze(-1)
    pos = grid * bands

    pos_sin, pos_cos = pos.sin().to(dtype=dtype), pos.cos().to(dtype)
    out = [grid, pos_sin, pos_cos] if include_grid else [pos_sin, pos_cos]
    return out


class FourierEmbed(nn.Module):
    def __init__(
        self,
        max_res: int = 224,
        num_bands: int = 64,
        concat_grid=True,
        keep_spatial=False,
    ):
        """Configure Fourier bands and the positional-feature output layout."""
        super().__init__()
        self.max_res = max_res
        self.num_bands = num_bands
        self.concat_grid = concat_grid
        self.keep_spatial = keep_spatial
        self.register_buffer(
            "bands", pixel_freq_bands(max_res, num_bands), persistent=False
        )

    def forward(self, x):
        """Append Fourier positional features to an image feature tensor."""
        B, C = x.shape[:2]
        feat_shape = x.shape[2:]
        emb = build_fourier_pos_embed(
            feat_shape,
            self.bands,
            include_grid=self.concat_grid,
            dtype=x.dtype,
            device=x.device,
        )
        emb = torch.cat(emb, dim=-1)
        emb = emb.transpose(-1, -2).flatten(len(feat_shape))
        batch_expand = (B,) + (-1,) * (x.ndim - 1)

        # FIXME support nD
        if self.keep_spatial:
            x = torch.cat(
                [x, emb.unsqueeze(0).expand(batch_expand).permute(0, 3, 1, 2)], dim=1
            )
        else:
            x = torch.cat(
                [x.permute(0, 2, 3, 1), emb.unsqueeze(0).expand(batch_expand)], dim=-1
            )
            x = x.reshape(B, feat_shape.numel(), -1)

        return x


def rot(x):
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)


def apply_rot_embed(x: torch.Tensor, sin_emb, cos_emb):
    if sin_emb.ndim == 3:
        return x * cos_emb.unsqueeze(1).expand_as(x) + rot(x) * sin_emb.unsqueeze(
            1
        ).expand_as(x)
    return x * cos_emb + rot(x) * sin_emb


def apply_rot_embed_list(x: List[torch.Tensor], sin_emb, cos_emb):
    if isinstance(x, torch.Tensor):
        x = [x]
    return [t * cos_emb + rot(t) * sin_emb for t in x]


def apply_rot_embed_cat(x: torch.Tensor, emb):
    sin_emb, cos_emb = emb.tensor_split(2, -1)
    if sin_emb.ndim == 3:
        return x * cos_emb.unsqueeze(1).expand_as(x) + rot(x) * sin_emb.unsqueeze(
            1
        ).expand_as(x)
    return x * cos_emb.unsqueeze(1).unsqueeze(2) + rot(x) * sin_emb.unsqueeze(
        1
    ).unsqueeze(2)


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(
            torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2
        )


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """Apply rotary embeddings to the rotary channels of attention heads.

    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(
        cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    sin = repeat(
        sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


def apply_keep_indices_nlc(x, pos_embed, keep_indices):
    pos_embed = pos_embed.unsqueeze(0).expand(x.shape[0], -1, -1)
    pos_embed = pos_embed.gather(
        1, keep_indices.unsqueeze(-1).expand(-1, -1, pos_embed.shape[-1])
    )
    return pos_embed


def build_rotary_pos_embed(
    feat_shape: List[int],
    bands: Optional[torch.Tensor] = None,
    dim: int = 64,
    max_res: int = 224,
    temperature: float = 10000.0,
    linear_bands: bool = False,
    in_pixels: bool = True,
    ref_feat_shape: Optional[List[float]] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
):
    """Build split sine and cosine rotary embeddings for a feature grid.

    Args:
        feat_shape: Spatial shape of the target tensor for embedding.
        bands: Optional pre-generated frequency bands
        dim: Output dimension of embedding tensor.
        max_res: Maximum resolution for pixel mode.
        temperature: Temperature (inv freq) for non-pixel mode
        linear_bands: Linearly (instead of log) spaced bands for pixel mode
        in_pixels: Pixel vs language (inv freq) mode.
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape,
        bands=bands,
        num_bands=dim // 8,
        max_res=max_res,
        temperature=temperature,
        linear_bands=linear_bands,
        in_pixels=in_pixels,
        ref_feat_shape=ref_feat_shape,
        device=device,
        dtype=dtype,
    )
    num_spatial_dim = 1
    # this would be much nicer as a .numel() call to torch.Size(), but torchscript sucks
    for x in feat_shape:
        num_spatial_dim *= x

    # the original timm implementation performs a repeat_interleave here
    # but in flash attn the repeat_interleave is done on the fly
    # this aligns with the flash attn implementation and does not run repeat_interleave here
    # sin_emb = sin_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    # cos_emb = cos_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    sin_emb = sin_emb.reshape(num_spatial_dim, -1)
    cos_emb = cos_emb.reshape(num_spatial_dim, -1)
    return sin_emb, cos_emb


class RotaryEmbedding(nn.Module):
    """Rotary position embedding

    NOTE: This is my initial attempt at impl rotary embedding for spatial use, it has not
    been well tested, and will likely change. It will be moved to its own file.

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(
        self,
        dim,
        max_res=224,
        temperature=10000,
        in_pixels=True,
        linear_bands: bool = False,
        feat_shape: Optional[List[int]] = None,
        ref_feat_shape: Optional[List[float]] = None,
    ):
        """Configure cached or shape-dependent sine and cosine rotary embeddings."""
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape

        if feat_shape is None:
            # only cache bands
            if in_pixels:
                bands = pixel_freq_bands(
                    dim // 8, float(max_res), linear_bands=linear_bands
                )
            else:
                bands = freq_bands(dim // 8, temperature=temperature, step=1)
                logger.debug("bands: %s", bands)
            self.register_buffer("bands", bands, persistent=False)
            self.pos_embed_sin = None
            self.pos_embed_cos = None
        else:
            # cache full sin/cos embeddings if shape provided up front
            emb_sin, emb_cos = build_rotary_pos_embed(
                feat_shape=feat_shape,
                dim=dim,
                max_res=max_res,
                linear_bands=linear_bands,
                in_pixels=in_pixels,
                ref_feat_shape=self.ref_feat_shape,
                temperature=self.temperature,
            )
            self.bands = None
            self.register_buffer("pos_embed_sin", emb_sin, persistent=False)
            self.register_buffer("pos_embed_cos", emb_cos, persistent=False)

    def get_embed(self, shape: Optional[List[int]] = None):
        """Return cached rotary embeddings or build them for a target shape."""
        if self.bands is not None:
            # rebuild embeddings every call, use if target shape changes
            assert shape is not None  # type: ignore[unreachable]
            return build_rotary_pos_embed(
                shape,
                self.bands,
                in_pixels=self.in_pixels,
                temperature=self.temperature,
            )
        else:
            return self.pos_embed_sin, self.pos_embed_cos

    def forward(self, x):
        # assuming channel-first tensor where spatial dim are >= 2
        sin_emb, cos_emb = self.get_embed(x.shape[2:])
        return apply_rot_embed(x, sin_emb, cos_emb)


class RotaryEmbeddingCat(nn.Module):
    """Rotary position embedding w/ concatenatd sin & cos

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(
        self,
        dim,
        max_res=224,
        temperature=10000,
        in_pixels=True,
        linear_bands: bool = False,
        feat_shape: Optional[List[int]] = None,
        ref_feat_shape: Optional[List[float]] = None,
    ):
        """Configure concatenated sine and cosine rotary embedding state."""
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.linear_bands = linear_bands
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape

        if feat_shape is None:
            self.bands = None
            self.pos_embed = None
        else:
            # cache full sin/cos embeddings if shape provided up front
            embeds = build_rotary_pos_embed(
                feat_shape=feat_shape,
                dim=dim,
                max_res=max_res,
                linear_bands=linear_bands,
                in_pixels=in_pixels,
                ref_feat_shape=self.ref_feat_shape,
                temperature=self.temperature,
            )
            self.bands = None
            self.register_buffer("pos_embed", torch.cat(embeds, -1), persistent=False)

    def get_embed(
        self,
        shape: Optional[List[int]] = None,
        ref_feat_shape: Optional[List[float]] = None,
    ):
        """Return concatenated rotary embeddings for a target feature shape."""
        if shape is not None:
            # rebuild bands and embeddings every call, use if target shape changes
            embeds = build_rotary_pos_embed(
                feat_shape=shape,
                dim=self.dim,
                max_res=self.max_res,
                linear_bands=self.linear_bands,
                in_pixels=self.in_pixels,
                ref_feat_shape=ref_feat_shape
                if ref_feat_shape
                else self.ref_feat_shape,
                temperature=self.temperature,
                device=torch.cuda.current_device(),
            )
            return torch.cat(embeds, -1)
        elif self.pos_embed is not None:
            return self.pos_embed  # type: ignore[unreachable]
        else:
            assert False, (
                "get_embed() requires pre-computed pos_embed or valid shape w/ pre-computed bands"
            )

    def forward(self, x):
        # assuming channel-first tensor where spatial dim are >= 2
        pos_embed = self.get_embed(x.shape[2:])
        return apply_rot_embed_cat(x, pos_embed)


class LearnableRotaryEmbeddingCat(nn.Module):
    """Rotary position embedding w/ concatenatd sin & cos

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(
        self,
        dim,
        max_res=224,
        temperature=10000,
        in_pixels=True,
        linear_bands: bool = False,
        feat_shape: Optional[List[int]] = None,
        ref_feat_shape: Optional[List[int]] = None,
    ):
        """Initialize trainable frequency bands for rotary embeddings."""
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.linear_bands = linear_bands
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape
        self.bands = nn.Parameter(self.get_default_bands())
        setattr(self.bands, "sequence_parallel", True)

    def get_default_bands(self):
        """Create the initial trainable rotary frequency bands."""
        if self.in_pixels:
            bands = pixel_freq_bands(
                self.dim // 8,
                float(self.max_res),
                linear_bands=self.linear_bands,
                device=torch.cuda.current_device(),
            )
        else:
            bands = freq_bands(
                self.dim // 8,
                temperature=self.temperature,
                step=1,
                device=torch.cuda.current_device(),
            )
        return bands

    def get_embed(
        self, shape: Optional[List[int]], ref_feat_shape: Optional[List[float]] = None
    ):
        """Build concatenated rotary embeddings from learned frequency bands."""
        # rebuild bands and embeddings every call, use if target shape changes
        embeds = build_rotary_pos_embed(
            feat_shape=shape,  # type: ignore[arg-type]
            bands=self.bands,  # use learned bands
            dim=self.dim,
            max_res=self.max_res,
            linear_bands=self.linear_bands,
            in_pixels=self.in_pixels,
            ref_feat_shape=ref_feat_shape if ref_feat_shape else self.ref_feat_shape,  # type: ignore[arg-type]
            temperature=self.temperature,
            device=torch.cuda.current_device(),
        )
        return torch.cat(embeds, -1)


# ============================================================
# modality_dispatcher
# ============================================================


def seqlens2cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.pad(torch.cumsum(seqlens, dim=0), (1, 0))


class ModalityDispatcher:
    """A torch.compile-friendly ModalityDispatcher.

    Design choices:
    - dispatch() has no @torch._dynamo.disable(), so it can be traced by Dynamo
    - _permute()/_inv_permute() have no side effects
    """

    num_modalities: int

    permuted_modality_mapping: torch.Tensor
    modality_mapping: torch.Tensor
    permute_mapping: torch.Tensor
    inv_permute_mapping: torch.Tensor

    group_size: torch.Tensor
    group_size_cpu: list[int]
    group_size_cpu_tensor: torch.Tensor
    cu_group_sizes: torch.Tensor
    expert_ranges: torch.Tensor

    @torch._dynamo.disable
    def __init__(self, modality_mapping: torch.Tensor, num_modalities: int):
        """Precompute token permutations and group boundaries by modality."""
        self.modality_mapping = modality_mapping
        self.num_modalities = num_modalities

        self.permute_mapping = torch.argsort(modality_mapping)
        self.inv_permute_mapping = torch.argsort(self.permute_mapping)
        self.permuted_modality_mapping = modality_mapping.index_select(
            0, self.permute_mapping
        )

        self.group_size = torch.bincount(
            self.permuted_modality_mapping, minlength=num_modalities
        ).to(torch.int32)

        # FIXME: these two causes a lot of trouble with torch compile
        # need to avoid using them as soon as possible, and eventually remove them
        self.group_size_cpu_tensor = self.group_size.to("cpu")
        self.group_size_cpu = self.group_size_cpu_tensor.tolist()

        self.cu_group_sizes = seqlens2cu_seqlens(self.group_size)

        end_indices = torch.cumsum(self.group_size, dim=0)
        start_indices = torch.cat(
            [
                torch.zeros(1, dtype=end_indices.dtype, device=end_indices.device),
                end_indices[:-1],
            ],
            dim=0,
        )
        self.expert_ranges = (
            torch.stack([start_indices, end_indices], dim=1).to(torch.int32).cpu()
        )

    def dispatch(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(x, self.group_size_cpu, dim=0))

    def undispatch(self, *processed_groups: torch.Tensor) -> torch.Tensor:
        return torch.cat(processed_groups, dim=0)

    def _permute(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(0, self.permute_mapping)

    def _inv_permute(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(0, self.inv_permute_mapping)


# ============================================================
# mhc
# ============================================================

MHCTensorTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _use_triton_backend(x: torch.Tensor, backend: str) -> bool:
    return backend == "triton" and x.is_cuda and not torch.is_grad_enabled()


@triton.jit
def _sigmoid_affine_fwd_kernel(
    x_ptr,
    alpha_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    matmul_scale,
    sigmoid_scale,
    n_cols: tl.constexpr,
    block_size: tl.constexpr,
):
    """Apply an elementwise sigmoid affine transform with Triton."""
    block_start = tl.program_id(0) * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    alpha = tl.load(alpha_ptr)
    bias = tl.load(bias_ptr + offsets % n_cols, mask=mask)
    out = sigmoid_scale * tl.sigmoid(alpha * matmul_scale * x + bias)
    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton_op("magi2::mhc_sigmoid_affine_fwd", mutates_args={})
def _triton_sigmoid_affine_forward(
    x: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    matmul_scale: float,
    sigmoid_scale: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Launch the Triton sigmoid affine transform for connection gates."""
    x = x.contiguous()
    alpha = alpha.contiguous()
    bias = bias.contiguous()
    out = torch.empty_like(x, dtype=out_dtype)
    block_size = 256
    grid = (triton.cdiv(x.numel(), block_size),)
    wrap_triton(_sigmoid_affine_fwd_kernel)[grid](
        x,
        alpha,
        bias,
        out,
        x.numel(),
        matmul_scale,
        sigmoid_scale,
        x.shape[-1],
        block_size,
    )
    return out


def _sigmoid_affine(
    x: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    matmul_scale: float,
    sigmoid_scale: float,
    out_dtype: torch.dtype,
    backend: str,
) -> torch.Tensor:
    """Apply the sigmoid affine gate with the selected execution backend."""
    if _use_triton_backend(x, backend):
        return _triton_sigmoid_affine_forward(
            x, alpha, bias, matmul_scale, sigmoid_scale, out_dtype
        )
    return (
        sigmoid_scale * torch.sigmoid(alpha * matmul_scale * x + bias.unsqueeze(0))
    ).to(out_dtype)


@triton.jit
def _sinkhorn_knopp_affine_fwd_kernel(
    x_ptr,
    alpha_ptr,
    bias_ptr,
    out_ptr,
    scale,
    sk_eps,
    stride_xb,
    stride_xr,
    stride_xc,
    stride_ob,
    stride_or,
    stride_oc,
    n_batch,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    num_iters: tl.constexpr,
    block_batch: tl.constexpr,
    block_r: tl.constexpr,
    block_c: tl.constexpr,
):
    """Apply affine logits and Sinkhorn normalization with Triton."""
    batch_pid = tl.program_id(0)
    batch_offsets = batch_pid * block_batch + tl.arange(0, block_batch)
    batch_mask = batch_offsets < n_batch
    row_offsets = tl.arange(0, block_r)
    col_offsets = tl.arange(0, block_c)
    mask = (
        batch_mask[:, None, None]
        & (row_offsets[None, :, None] < n_rows)
        & (col_offsets[None, None, :] < n_cols)
    )

    x = tl.load(
        x_ptr
        + batch_offsets[:, None, None] * stride_xb
        + row_offsets[None, :, None] * stride_xr
        + col_offsets[None, None, :] * stride_xc,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    alpha = tl.load(alpha_ptr).to(tl.float32)
    bias = tl.load(
        bias_ptr + row_offsets[:, None] * stride_xr + col_offsets[None, :] * stride_xc,
        mask=(row_offsets[:, None] < n_rows) & (col_offsets[None, :] < n_cols),
        other=0.0,
    ).to(tl.float32)

    h = alpha * scale * x + bias
    h_max = tl.max(tl.max(tl.where(mask, h, -float("inf")), axis=2), axis=1)
    h_max = tl.where(h_max == -float("inf"), 0.0, h_max)
    m = tl.exp(h - h_max[:, None, None])
    for _ in range(num_iters):
        col_sum = tl.sum(m, axis=1)
        m = m / (col_sum[:, None, :] + sk_eps)
        row_sum = tl.sum(m, axis=2)
        m = m / (row_sum[:, :, None] + sk_eps)

    tl.store(
        out_ptr
        + batch_offsets[:, None, None] * stride_ob
        + row_offsets[None, :, None] * stride_or
        + col_offsets[None, None, :] * stride_oc,
        m.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton_op("magi2::mhc_sinkhorn_knopp_affine_fwd", mutates_args={})
def _triton_sinkhorn_knopp_affine_forward(
    x: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    matmul_scale: float,
    num_sk_iters: int,
    sk_eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Launch Triton Sinkhorn normalization for residual mixing matrices."""
    x = x.contiguous()
    alpha = alpha.contiguous()
    bias = bias.contiguous()
    t, n_rows, n_cols = x.shape
    out = torch.empty_like(x, dtype=out_dtype)
    block_r = triton.next_power_of_2(n_rows)
    block_c = triton.next_power_of_2(n_cols)
    block_batch = max(1, 256 // (block_r * block_c))
    grid = (triton.cdiv(t, block_batch),)
    wrap_triton(_sinkhorn_knopp_affine_fwd_kernel)[grid](
        x,
        alpha,
        bias,
        out,
        matmul_scale,
        sk_eps,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        t,
        n_rows,
        n_cols,
        num_sk_iters,
        block_batch,
        block_r,
        block_c,
    )
    return out


def _sinkhorn_knopp(h: torch.Tensor, num_iters: int, eps: float) -> torch.Tensor:
    m = torch.exp(h - h.amax(dim=(-2, -1), keepdim=True))
    for _ in range(num_iters):
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
    return m


def _sinkhorn_knopp_affine(
    x: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    matmul_scale: float,
    num_sk_iters: int,
    sk_eps: float,
    out_dtype: torch.dtype,
    backend: str,
) -> torch.Tensor:
    """Build residual mixing matrices with the selected execution backend."""
    if _use_triton_backend(x, backend):
        return _triton_sinkhorn_knopp_affine_forward(
            x, alpha, bias, matmul_scale, num_sk_iters, sk_eps, out_dtype
        )
    h = alpha * matmul_scale * x.to(torch.float32) + bias.unsqueeze(0).to(torch.float32)
    return _sinkhorn_knopp(h, num_sk_iters, sk_eps).to(out_dtype)


@triton.jit
def _apply_hpre_fwd_kernel(
    h_pre_ptr,
    x_ptr,
    out_ptr,
    hpre_stride_t,
    hpre_stride_n,
    x_stride_t,
    x_stride_n,
    x_stride_c,
    out_stride_t,
    out_stride_c,
    n_stream: tl.constexpr,
    hidden_size: tl.constexpr,
    block_size_c: tl.constexpr,
):
    """Reduce multi-stream features with pre-connection weights in Triton."""
    t = tl.program_id(0)
    n_offsets = tl.arange(0, n_stream)
    c_offsets_base = tl.arange(0, block_size_c)
    hpre = tl.load(h_pre_ptr + t * hpre_stride_t + n_offsets * hpre_stride_n).to(
        tl.float32
    )[:, None]

    for c_start in tl.range(0, hidden_size, block_size_c):
        c_offsets = c_start + c_offsets_base
        c_mask = c_offsets < hidden_size
        x = tl.load(
            x_ptr
            + t * x_stride_t
            + n_offsets[:, None] * x_stride_n
            + c_offsets[None, :] * x_stride_c,
            mask=c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        out = tl.sum(hpre * x, axis=0)
        tl.store(
            out_ptr + t * out_stride_t + c_offsets * out_stride_c, out, mask=c_mask
        )


@triton_op("magi2::mhc_apply_hpre_fwd", mutates_args={})
def _triton_apply_hpre_forward(h_pre: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Launch Triton reduction of multi-stream features into one stream."""
    _, n_stream, hidden_size = x.shape
    out = torch.empty((x.size(0), hidden_size), device=x.device, dtype=x.dtype)
    block_size_c = min(1024, triton.next_power_of_2(max(1, hidden_size // 2)))
    grid = (x.size(0),)
    wrap_triton(_apply_hpre_fwd_kernel)[grid](
        h_pre,
        x,
        out,
        h_pre.stride(0),
        h_pre.stride(1),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        n_stream,
        hidden_size,
        block_size_c,
        num_warps=4,
    )
    return out


def _apply_hpre(h_pre: torch.Tensor, x: torch.Tensor, backend: str) -> torch.Tensor:
    if _use_triton_backend(x, backend):
        return _triton_apply_hpre_forward(h_pre, x)
    return torch.einsum("tn,tnc->tc", h_pre, x)


@triton.jit
def _hyper_connect_fwd_kernel(
    res_ptr,
    out_ptr,
    h_post_ptr,
    h_res_ptr,
    output_ptr,
    stride_res_t,
    stride_res_n,
    stride_res_c,
    stride_out_t,
    stride_out_c,
    stride_h_post_t,
    stride_h_post_n,
    stride_h_res_t,
    stride_h_res_n,
    stride_h_res_n2,
    stride_output_t,
    stride_output_n,
    stride_output_c,
    n_stream: tl.constexpr,
    hidden_size: tl.constexpr,
    block_size_c: tl.constexpr,
):
    """Combine residual streams with post-connection weights in Triton."""
    t = tl.program_id(0)
    i_offsets = tl.arange(0, n_stream)
    j_offsets = tl.arange(0, n_stream)
    c_offsets_base = tl.arange(0, block_size_c)

    h_post = tl.load(h_post_ptr + t * stride_h_post_t + i_offsets * stride_h_post_n).to(
        tl.float32
    )
    h_res = tl.load(
        h_res_ptr
        + t * stride_h_res_t
        + i_offsets[:, None] * stride_h_res_n
        + j_offsets[None, :] * stride_h_res_n2
    ).to(tl.float32)

    for c_start in tl.range(0, hidden_size, block_size_c):
        c_offsets = c_start + c_offsets_base
        c_mask = c_offsets < hidden_size
        out_row = tl.load(
            out_ptr + t * stride_out_t + c_offsets * stride_out_c,
            mask=c_mask,
            other=0.0,
        ).to(tl.float32)

        for i in tl.static_range(0, n_stream):
            acc = tl.zeros([block_size_c], dtype=tl.float32)
            for j in tl.static_range(0, n_stream):
                h_res_ij = tl.sum(
                    tl.where(
                        (tl.arange(0, n_stream)[:, None] == i)
                        & (tl.arange(0, n_stream)[None, :] == j),
                        h_res,
                        0.0,
                    )
                )
                res_row = tl.load(
                    res_ptr
                    + t * stride_res_t
                    + j * stride_res_n
                    + c_offsets * stride_res_c,
                    mask=c_mask,
                    other=0.0,
                ).to(tl.float32)
                acc += h_res_ij * res_row

            h_post_i = tl.sum(tl.where(tl.arange(0, n_stream) == i, h_post, 0.0))
            acc += h_post_i * out_row
            tl.store(
                output_ptr
                + t * stride_output_t
                + i * stride_output_n
                + c_offsets * stride_output_c,
                acc.to(output_ptr.dtype.element_ty),
                mask=c_mask,
            )


@triton_op("magi2::mhc_hyper_connect_fwd", mutates_args={})
def _triton_hyper_connect_forward(
    res: torch.Tensor, out: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor
) -> torch.Tensor:
    """Launch Triton residual-stream mixing and output injection."""
    _, n_stream, hidden_size = res.shape
    output = torch.empty_like(res)
    block_size_c = 256
    grid = (res.size(0),)
    wrap_triton(_hyper_connect_fwd_kernel)[grid](
        res,
        out,
        h_post,
        h_res,
        output,
        res.stride(0),
        res.stride(1),
        res.stride(2),
        out.stride(0),
        out.stride(1),
        h_post.stride(0),
        h_post.stride(1),
        h_res.stride(0),
        h_res.stride(1),
        h_res.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        n_stream,
        hidden_size,
        block_size_c,
        num_warps=4,
    )
    return output


def _hyper_connect(
    res: torch.Tensor,
    out: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    """Mix residual streams and inject one transformer output stream."""
    if _use_triton_backend(res, backend):
        return _triton_hyper_connect_forward(res, out, h_post, h_res)
    out_mstream = torch.einsum("tn,tc->tnc", h_post, out)
    mixed_res = torch.einsum("tij,tjc->tic", h_res, res)
    return mixed_res + out_mstream


class MHCHandler:
    """High-performance mHC helper with local Triton inference kernels."""

    def __init__(
        self,
        n_stream: int,
        hidden_size: int,
        num_sk_iters: int = 20,
        sk_eps: float = 1e-12,
        dtype: torch.dtype = torch.float32,
        backend: str | None = None,
    ) -> None:
        """Configure multi-stream hyper-connection transforms and backend."""
        self.n = n_stream
        self.hidden_size = hidden_size
        self.num_sk_iters = num_sk_iters
        self.sk_eps = sk_eps
        self.dtype = dtype
        self.backend = backend or os.environ.get("MAGI2_MHC_BACKEND", "triton")
        self.matmul_scale = 1.0 / math.sqrt(float(n_stream * hidden_size))

    def flatten_mstream(self, x: torch.Tensor) -> torch.Tensor:
        self._check_mstream_shape(x)
        return x.view(x.size(0), -1)

    def apply_norm_and_compute_h_(
        self,
        x: torch.Tensor,
        norm_fn: Callable[[torch.Tensor], torch.Tensor],
        phi_fused: torch.Tensor,
    ) -> MHCTensorTuple:
        """Normalize streams and project pre-, post-, and residual parameters."""
        self._check_flatten_mstream_shape(x)
        h_fused = torch.matmul(norm_fn(x).to(self.dtype), phi_fused)
        h_pre, h_post, h_res = torch.split(
            h_fused, [self.n, self.n, self.n * self.n], dim=-1
        )
        return h_pre, h_post, h_res.view(-1, self.n, self.n)

    def compute_and_apply_hpre(
        self,
        x: torch.Tensor,
        abh_pre: MHCTensorTuple,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Compute pre-connection gates and reduce residual streams."""
        self._check_mstream_shape(x)
        alpha, bias, h_pre_ = abh_pre
        dtype = out_dtype or x.dtype
        h_pre = _sigmoid_affine(
            h_pre_,
            alpha,
            bias,
            self.matmul_scale,
            sigmoid_scale=1.0,
            out_dtype=dtype,
            backend=self.backend,
        )
        return _apply_hpre(h_pre, x, self.backend)

    def compute_hpost_and_hres(
        self,
        abh_post: MHCTensorTuple,
        abh_res: MHCTensorTuple,
        out_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute output gates and residual-stream mixing matrices."""
        alpha_post, bias_post, h_post_ = abh_post
        alpha_res, bias_res, h_res_ = abh_res
        dtype = out_dtype or h_post_.dtype
        h_post = _sigmoid_affine(
            h_post_,
            alpha_post,
            bias_post,
            self.matmul_scale,
            sigmoid_scale=2.0,
            out_dtype=dtype,
            backend=self.backend,
        )
        h_res = _sinkhorn_knopp_affine(
            h_res_,
            alpha_res,
            bias_res,
            self.matmul_scale,
            self.num_sk_iters,
            self.sk_eps,
            dtype,
            self.backend,
        )
        return h_post, h_res

    def apply_hpost_and_hres(
        self,
        res: torch.Tensor,
        out: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
    ) -> torch.Tensor:
        self._check_mstream_shape(res)
        self._check_sstream_shape(out)
        return _hyper_connect(res, out, h_post, h_res, self.backend)

    def _check_sstream_shape(self, x: torch.Tensor) -> None:
        if x.dim() != 2 or x.size(-1) != self.hidden_size:
            raise ValueError(
                f"Expected single-stream shape (T, {self.hidden_size}), got {tuple(x.shape)}"
            )

    def _check_mstream_shape(self, x: torch.Tensor) -> None:
        if x.dim() != 3 or x.size(1) != self.n or x.size(-1) != self.hidden_size:
            raise ValueError(
                f"Expected multi-stream shape (T, {self.n}, {self.hidden_size}), got {tuple(x.shape)}"
            )

    def _check_flatten_mstream_shape(self, x: torch.Tensor) -> None:
        expected = self.n * self.hidden_size
        if x.dim() != 2 or x.size(-1) != expected:
            raise ValueError(
                f"Expected flattened multi-stream shape (T, {expected}), got {tuple(x.shape)}"
            )


# ============================================================
# modeling
# ============================================================

os.environ.setdefault("MAGI_ATTENTION_WORKSPACE_BASE", "/tmp")


@dataclass
class VarlenHandler:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Encode scalar diffusion positions with sinusoidal frequencies."""
    position = position.to(torch.float32) * 1000.0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=position.device)
        / half
    )
    args = position[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _attention_sinks_for_cp(
    attn_sinks: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if attn_sinks is None or psm.get_world_size("cp") <= 1:
        return attn_sinks
    return torch.chunk(attn_sinks, chunks=psm.get_world_size("cp"), dim=-1)[
        dist.get_rank(psm.get_parallel_group("cp"))
    ].contiguous()


def _fa3_varlen_func_with_sink_inference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softcap: float,
    sink: Optional[torch.Tensor],
    sink_layout: str,
    deterministic: bool = False,
) -> torch.Tensor:
    """Run variable-length FlashAttention 3 and correct attention-sink outputs."""
    if torch.is_grad_enabled():
        return fa3_varlen_func_with_sink(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softcap=softcap,
            sink=sink,
            sink_layout=sink_layout,
            deterministic=deterministic
            or os.environ.get("MAGI2_DETERMINISTIC", "0") == "1",
        )

    softmax_scale = q.shape[-1] ** -0.5
    out, softmax_lse, *_ = flash_attn_3_cuda.fwd(
        q,
        k,
        v,
        None,  # k_new
        None,  # v_new
        None,  # qv
        None,  # out
        cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(),
        None,  # cu_seqlens_k_new
        None,  # seqused_q
        None,  # seqused_k
        max_seqlen_q,
        max_seqlen_k,
        None,  # page_table
        None,  # kv_batch_idx
        None,  # leftpad_k
        None,  # rotary_cos
        None,  # rotary_sin
        None,  # seqlens_rotary
        None,  # q_descale
        None,  # k_descale
        None,  # v_descale
        softmax_scale,
        False,  # causal
        -1,  # window_size_left
        -1,  # window_size_right
        0,  # attention_chunk
        softcap,
        True,  # rotary_interleaved
        None,  # scheduler_metadata
        1,  # num_splits
        None,  # pack_gqa
        0,  # sm_margin
    )
    out, _ = FA3VarlenFuncWithSink.correct_out_lse_with_sink(
        out=out, lse=softmax_lse, sink=sink, sink_layout=sink_layout
    )
    return out


def _flash_attn_with_sink_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    cp_split_sizes: list[int],
    attn_softcap: float,
    attn_sinks: Optional[torch.Tensor],
) -> torch.Tensor:
    """Execute sink attention with context-parallel head and sequence exchange."""
    q, k, v = q.squeeze(0).bfloat16(), k.squeeze(0).bfloat16(), v.squeeze(0).bfloat16()
    if attn_sinks is not None and attn_sinks.numel() == 0:
        attn_sinks = None
    attn_sinks = _attention_sinks_for_cp(attn_sinks)

    if psm.get_world_size("cp") > 1:
        q, k, v = batch_scatter_head_gather_seqlen(
            [q, k, v], cp_split_sizes, psm.get_parallel_group("cp")
        )
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    if cu_seqlens_q is not None:
        out = _fa3_varlen_func_with_sink_inference(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softcap=attn_softcap,
            sink=attn_sinks,
            sink_layout="sh",
            deterministic=os.environ.get("MAGI2_DETERMINISTIC", "0") == "1",
        )
    else:
        out = fa3_func_with_sink(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            softcap=attn_softcap,
            sink=attn_sinks.unsqueeze(0) if attn_sinks is not None else None,
            sink_layout="sh",
            deterministic=os.environ.get("MAGI2_DETERMINISTIC", "0") == "1",
        )
        out = out.squeeze(0)

    if psm.get_world_size("cp") > 1:
        out = scatter_seqlen_gather_head(
            out, cp_split_sizes, psm.get_parallel_group("cp"), async_op=False
        )
        out = rearrange(
            out, "(cp sq) hn hd -> sq (cp hn) hd", cp=psm.get_world_size("cp")
        )
    return out


def _fa3_with_sink_cp_meta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_sinks: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    cp_split_sizes: torch.Tensor,
    attn_softcap: float,
) -> torch.Tensor:
    """Describe the compiled context-parallel attention output tensor."""
    return torch.empty_like(q, dtype=torch.bfloat16).squeeze(0)


@magi_register_custom_op(
    "infra::magi2_fa3_with_sink_cp",
    infer_output_meta_fn=_fa3_with_sink_cp_meta,
    is_subgraph_boundary=True,
)
def magi2_fa3_with_sink_cp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_sinks: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    cp_split_sizes: torch.Tensor,
    attn_softcap: float,
) -> torch.Tensor:
    """FA3 attention with sink tokens and context parallelism for MAGI-2.

    max_seqlen_q/k are derived from cu_seqlens inside this boundary op instead
    of being passed as int args, because MagiCompile specializes Python ints as
    constants — see test_boundary_int_args_are_specialized.

    cp_split_sizes is passed as a Tensor (created outside compiled code) so its
    values flow dynamically through the FX graph.
    """
    if cu_seqlens_q is not None:
        max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
        max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
    else:
        max_seqlen_q = q.shape[0]
        max_seqlen_k = k.shape[0]
    return _flash_attn_with_sink_impl(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        cp_split_sizes=cp_split_sizes.tolist(),
        attn_softcap=attn_softcap,
        attn_sinks=attn_sinks,
    )


def flash_attn_with_sink(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    varlen_handler: VarlenHandler,
    cp_split_sizes: torch.Tensor,
    attn_softcap: float,
    attn_sinks: Optional[torch.Tensor],
) -> torch.Tensor:
    """Invoke sink attention using packed variable-length sequence metadata."""
    sinks_tensor = (
        attn_sinks if attn_sinks is not None else torch.empty(0, device=q.device)
    )
    return magi2_fa3_with_sink_cp(
        q,
        k,
        v,
        sinks_tensor,
        cu_seqlens_q=varlen_handler.cu_seqlens_q,
        cu_seqlens_k=varlen_handler.cu_seqlens_k,
        cp_split_sizes=cp_split_sizes,
        attn_softcap=attn_softcap,
    )


@dataclass
class AttentionConfig:
    hidden_size: int
    num_heads_q: int
    num_heads_kv: int
    head_dim: int
    params_dtype: torch.dtype
    num_modality: int
    num_layers: int
    attn_softcap: float = -1.0
    sink_token_num: int = 1


class Attention(nn.Module):
    def __init__(self, config: AttentionConfig):
        """Build modality-aware projections, normalization, and attention sinks."""
        super().__init__()
        self.config = config
        self.pre_norm = MultiModalityRMSNorm(
            config.hidden_size, eps=1e-6, num_modality=config.num_modality
        )

        self.head_dim = config.head_dim
        self.num_heads_q = config.num_heads_q
        self.num_heads_kv = config.num_heads_kv
        self.q_size = self.num_heads_q * self.head_dim
        self.kv_size = self.num_heads_kv * self.head_dim

        self.gating_total_size = self.num_heads_q

        out_dim = self.q_size + self.kv_size * 2
        self.linear_g = create_linear(
            config.hidden_size,
            self.gating_total_size,
            num_experts=config.num_modality,
            bias=False,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
        )

        self.linear_qkv = create_linear(
            config.hidden_size,
            out_dim,
            num_experts=config.num_modality,
            bias=False,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
        )
        self.linear_proj = create_linear(
            self.q_size,
            config.hidden_size,
            bias=False,
            num_experts=config.num_modality,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
        )

        self.sinks = nn.Parameter(
            torch.empty(
                max(1, config.sink_token_num), config.num_heads_q, dtype=torch.float32
            )
        )

        self.q_norm = MultiModalityRMSNorm(
            config.head_dim, num_modality=config.num_modality, out_dtype=torch.float32
        )
        self.k_norm = MultiModalityRMSNorm(
            config.head_dim, num_modality=config.num_modality, out_dtype=torch.float32
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        varlen_handler: VarlenHandler,
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """Apply modality-aware sink attention to packed token features."""
        x = self.pre_norm(hidden_states, modality_dispatcher=modality_dispatcher)
        g = self.linear_g(x, modality_dispatcher=modality_dispatcher)
        qkv = self.linear_qkv(x, modality_dispatcher=modality_dispatcher)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=1)

        q = q.view(hidden_states.size(0), self.num_heads_q, self.head_dim).clone()
        k = k.view(hidden_states.size(0), self.num_heads_kv, self.head_dim).clone()
        v = v.view(hidden_states.size(0), self.num_heads_kv, self.head_dim)
        g = g.view(hidden_states.size(0), self.num_heads_q, 1).clone()

        q = self.q_norm(q, modality_dispatcher=modality_dispatcher)
        k = self.k_norm(k, modality_dispatcher=modality_dispatcher)

        q = modality_dispatcher._inv_permute(q).unsqueeze(0)
        k = modality_dispatcher._inv_permute(k).unsqueeze(0)
        v = modality_dispatcher._inv_permute(v).unsqueeze(0)

        sin_emb, cos_emb = rope.tensor_split(2, -1)
        q = apply_rotary_emb_torch(q, cos_emb, sin_emb)
        k = apply_rotary_emb_torch(k, cos_emb, sin_emb)
        out = flash_attn_with_sink(
            q,
            k,
            v,
            varlen_handler,
            cp_split_sizes,
            self.config.attn_softcap,
            self.sinks,
        )

        out = modality_dispatcher._permute(out)
        out = out * torch.sigmoid(g)
        out = out.reshape(-1, self.q_size).to(torch.bfloat16)
        return self.linear_proj(out, modality_dispatcher=modality_dispatcher)


@dataclass
class MLPConfig:
    hidden_size: int
    intermediate_size: int
    activation_type: MLPActivationType
    params_dtype: torch.dtype
    num_modality: int = 1
    num_layers: int = 1
    gated_act: bool = False


class MLP(nn.Module):
    def __init__(self, config: MLPConfig) -> None:
        """Build modality-aware gated feed-forward projections."""
        super().__init__()
        self.config = config
        self.pre_norm = MultiModalityRMSNorm(
            config.hidden_size, num_modality=config.num_modality
        )
        intermediate_size_up = (
            config.intermediate_size * 2
            if config.gated_act
            else config.intermediate_size
        )
        self.up_gate_proj = create_linear(
            config.hidden_size,
            intermediate_size_up,
            bias=False,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
            num_experts=config.num_modality,
        )
        self.down_proj = create_linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
            num_experts=config.num_modality,
        )
        self.activation_func = create_activation_func(config.activation_type)

    def forward(
        self, x: torch.Tensor, modality_dispatcher: ModalityDispatcher
    ) -> torch.Tensor:
        x = self.pre_norm(x, modality_dispatcher=modality_dispatcher)
        x = self.up_gate_proj(x, modality_dispatcher=modality_dispatcher)
        x = self.activation_func(x)
        x = self.down_proj(x, modality_dispatcher=modality_dispatcher)
        return x


@dataclass
class CoreMultiHeadMoEConfig:
    hidden_size: int
    num_heads: int
    num_experts: int
    top_k: int
    expert_intermediate_size: int
    num_layers: int
    params_dtype: torch.dtype
    score_func: str = "sigmoid"
    route_norm: bool = True
    route_scale: float = 1.0
    recompute_once_in_bwd: bool = True


_magi2_moe_registry: dict[int, "CoreMultiHeadMoE"] = {}


class CoreMultiHeadMoE(nn.Module):
    def __init__(self, config: CoreMultiHeadMoEConfig) -> None:
        """Build expert-parallel multi-head mixture-of-experts routing state."""
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self._moe_key = len(_magi2_moe_registry)
        _magi2_moe_registry[self._moe_key] = self
        self.config = config
        self.num_heads = config.num_heads
        self.num_experts = config.num_experts
        self.topk = config.top_k
        self.d_head = config.hidden_size // config.num_heads
        self.d_expert = config.expert_intermediate_size
        self.flatten_num_experts = self.num_heads * self.num_experts
        self.ep_size = psm.get_world_size("ep")

        # When num_heads is not divisible by ep_size, pad to next multiple.
        # Ranks beyond the real head range hold dummy (zero) weights and
        # participate in all-to-all but produce discarded output.
        if self.ep_size > 1 and self.num_heads % self.ep_size != 0:
            import math

            self.padded_num_heads = (
                math.ceil(self.num_heads / self.ep_size) * self.ep_size
            )
            self.ep_pad_heads = self.padded_num_heads - self.num_heads
            ep_rank = psm.get_local_rank("ep")
            real_heads_end = self.num_heads
            my_head_start = ep_rank * (self.padded_num_heads // self.ep_size)
            self.has_real_moe_heads = my_head_start < real_heads_end
        else:
            self.padded_num_heads = self.num_heads
            self.ep_pad_heads = 0
            self.has_real_moe_heads = True

        if self.ep_pad_heads > 0:
            self.register_buffer(
                "_ep_pad_zeros",
                torch.zeros(
                    1, self.ep_pad_heads, self.d_head, dtype=config.params_dtype
                ),
                persistent=False,
            )

        self.local_num_heads = self.padded_num_heads // self.ep_size
        self.local_flatten_num_experts = self.local_num_heads * self.num_experts

        self.gate = nn.Parameter(
            torch.empty(
                self.local_flatten_num_experts, self.d_head, dtype=torch.float32
            )
        )
        self.W_gate = nn.Parameter(
            torch.empty(
                self.local_flatten_num_experts,
                self.d_head,
                self.d_expert,
                dtype=config.params_dtype,
            )
        )
        self.W_up = nn.Parameter(
            torch.empty(
                self.local_flatten_num_experts,
                self.d_head,
                self.d_expert,
                dtype=config.params_dtype,
            )
        )
        self.W_down = nn.Parameter(
            torch.empty(
                self.local_flatten_num_experts,
                self.d_expert,
                self.d_head,
                dtype=config.params_dtype,
            )
        )
        self.router = nn.Module()
        self.router.register_buffer(
            "expert_bias",
            torch.zeros(self.local_flatten_num_experts, dtype=torch.float32),
        )
        self.router.register_buffer(
            "expert_bias_ema",
            torch.zeros(self.local_flatten_num_experts, dtype=torch.float32),
        )

        if not self.has_real_moe_heads:
            with torch.no_grad():
                self.gate.zero_()
                self.W_gate.zero_()
                self.W_up.zero_()
                self.W_down.zero_()

    def _route(self, x_heads: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-k experts and probabilities for each token head."""
        num_heads = x_heads.size(1)
        gate = self.gate.view(num_heads, self.num_experts, self.d_head).float()
        router_logits = torch.einsum("shd,hed->hse", x_heads.float(), gate)
        expert_bias = self.router.expert_bias.view(num_heads, self.num_experts)
        topk_probs, topk_indices = _mh_moe_compute_topk(
            router_logits=router_logits,
            top_k=self.topk,
            score_func=self.config.score_func,
            expert_bias=expert_bias,
            route_norm=self.config.route_norm,
        )
        topk_probs = topk_probs * self.config.route_scale
        return topk_probs, topk_indices

    def _flash_forward(
        self,
        x_heads: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Execute sorted fused expert computation for routed token heads."""
        gather_ids, probs, expert_offsets = _mh_moe_global_sort(
            topk_probs=topk_probs,
            topk_indices=topk_indices,
            num_experts=self.num_experts,
        )
        return _mh_moe_fwd(
            x=x_heads,
            gather_ids=gather_ids,
            probs=probs,
            expert_offsets=expert_offsets,
            W_gate=self.W_gate,
            W_up=self.W_up,
            W_down=self.W_down,
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Run expert-parallel routing and fused experts across token heads."""
        x_heads = x.view(-1, self.num_heads, self.d_head)

        if self.ep_pad_heads > 0:
            x_heads = torch.cat(
                [x_heads, self._ep_pad_zeros.expand(x_heads.shape[0], -1, -1)], dim=1
            )

        if self.ep_size > 1:
            ep_group = psm.get_parallel_group("ep")
            if ep_group is None:
                raise RuntimeError(
                    f"MAGI-2 EP group is not initialized but ep_size={self.ep_size}"
                )
            x_heads = ep_dispatch(x_heads, ep_group=ep_group)

        if self.has_real_moe_heads:
            topk_probs, topk_indices = self._route(x_heads)
            out = self._flash_forward(x_heads, topk_probs, topk_indices)
        else:
            out = torch.zeros_like(x_heads)

        if self.ep_size > 1:
            out = ep_undispatch(out, ep_group=ep_group)

        if self.ep_pad_heads > 0:
            out = out[:, : self.num_heads, :]

        return out.reshape(-1, self.num_heads * self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return magi2_mh_moe(x, self._moe_key)


def _magi2_mh_moe_meta(x: torch.Tensor, moe_key: int) -> torch.Tensor:
    moe_module: CoreMultiHeadMoE = _magi2_moe_registry[moe_key]
    if moe_module.ep_pad_heads > 0:
        padded_stride = moe_module.padded_num_heads * moe_module.d_head
        return torch.empty_strided(
            x.shape, (padded_stride, 1), dtype=x.dtype, device=x.device
        )
    return torch.empty_like(x)


@magi_register_custom_op(
    "infra::magi2_mh_moe",
    infer_output_meta_fn=_magi2_mh_moe_meta,
    is_subgraph_boundary=True,
)
def magi2_mh_moe(x: torch.Tensor, moe_key: int) -> torch.Tensor:
    """MoE forward with EP dispatch/undispatch for MAGI-2.

    Module is looked up from _magi2_moe_registry by a stable int key assigned
    during __init__ (monotonic counter, not id(self)), so Dynamo's int
    specialization produces deterministic guards that survive cache reuse
    across process restarts.
    """
    moe_module: CoreMultiHeadMoE = _magi2_moe_registry[moe_key]
    return moe_module._forward_impl(x)


@dataclass
class MultiHeadMoELayerConfig:
    hidden_size: int
    num_modality: int
    num_layers: int
    activation_type: MLPActivationType
    params_dtype: torch.dtype
    shared_expert_intermediate_sizes: int
    modality_specific_expert_intermediate_sizes: int
    shared_expert_params_dtype: torch.dtype
    split_merge_intermediate_size: Optional[int]
    moe_config: MoEConfig


class MultiHeadMoELayer(nn.Module):
    def __init__(self, config: MultiHeadMoELayerConfig) -> None:
        """Build routed and shared expert branches for one transformer layer."""
        super().__init__()
        self.config = config
        moe_hidden_size = config.split_merge_intermediate_size or config.hidden_size
        self.moe_mlp = CoreMultiHeadMoE(
            CoreMultiHeadMoEConfig(
                hidden_size=moe_hidden_size,
                num_heads=config.moe_config.num_heads,
                num_experts=config.moe_config.num_experts,
                top_k=config.moe_config.top_k,
                expert_intermediate_size=config.moe_config.expert_intermediate_size,
                num_layers=config.num_layers,
                params_dtype=config.params_dtype,
                score_func=config.moe_config.score_func,
                route_norm=config.moe_config.route_norm,
                route_scale=config.moe_config.route_scale,
            )
        )
        self.is_swiglu = config.activation_type in [MLPActivationType.SWIGLU7]
        self.activation_func = create_activation_func(config.activation_type)
        self.shared_expert_fc1, self.shared_expert_fc2 = (
            self._create_shared_expert_pair(
                config, config.shared_expert_intermediate_sizes, num_modality=1
            )
        )
        (
            self.modality_specific_shared_expert_fc1,
            self.modality_specific_shared_expert_fc2,
        ) = self._create_shared_expert_pair(
            config,
            config.modality_specific_expert_intermediate_sizes,
            num_modality=config.num_modality,
        )
        self.pre_norm = MultiModalityRMSNorm(
            config.hidden_size, num_modality=config.num_modality
        )
        mid = config.split_merge_intermediate_size or config.hidden_size
        self.split_linear = create_linear(
            config.hidden_size,
            mid,
            num_layers=config.num_layers,
            dtype=config.shared_expert_params_dtype,
            bias=False,
        )
        self.merge_linear = create_linear(
            mid,
            config.hidden_size,
            num_layers=config.num_layers,
            dtype=config.shared_expert_params_dtype,
            bias=False,
        )

    @staticmethod
    def _create_shared_expert_pair(
        config: MultiHeadMoELayerConfig, intermediate_size: int, num_modality: int = 1
    ) -> tuple[nn.Module, nn.Module]:
        """Build input and output projections for one shared expert branch."""
        factor = 2 if config.activation_type in [MLPActivationType.SWIGLU7] else 1
        fc1 = create_linear(
            config.hidden_size,
            intermediate_size * factor,
            num_layers=config.num_layers,
            dtype=config.shared_expert_params_dtype,
            num_experts=num_modality,
            bias=False,
        )
        fc2 = create_linear(
            intermediate_size,
            config.hidden_size,
            num_layers=config.num_layers,
            dtype=config.shared_expert_params_dtype,
            num_experts=num_modality,
            bias=False,
        )
        return fc1, fc2

    def _shared_expert_forward(
        self, norm_output: torch.Tensor, modality_dispatcher: ModalityDispatcher
    ) -> torch.Tensor:
        """Run global and modality-specific shared expert branches."""
        x1 = self.shared_expert_fc1(norm_output)
        x2 = self.modality_specific_shared_expert_fc1(
            norm_output, modality_dispatcher=modality_dispatcher
        )
        x = torch.concat([x1, x2], dim=-1)
        x = self.activation_func(x)
        x1, x2 = x.split(
            [
                self.config.shared_expert_intermediate_sizes,
                self.config.modality_specific_expert_intermediate_sizes,
            ],
            dim=-1,
        )
        x1 = self.shared_expert_fc2(x1)
        x2 = self.modality_specific_shared_expert_fc2(
            x2.contiguous(), modality_dispatcher=modality_dispatcher
        )
        return x1 + x2

    def forward(
        self, hidden_states: torch.Tensor, modality_dispatcher: ModalityDispatcher
    ) -> torch.Tensor:
        norm_output = self.pre_norm(
            hidden_states, modality_dispatcher=modality_dispatcher
        )
        moe_input = self.split_linear(norm_output)
        moe_out = self.moe_mlp(moe_input)
        moe_out = self.merge_linear(moe_out)
        return moe_out + self._shared_expert_forward(norm_output, modality_dispatcher)


@dataclass
class AdapterConfig:
    hidden_size: int
    num_attention_heads: int
    text_in_channels: int
    video_in_channels: int
    audio_in_channels: int
    virtual_width_factor: int = 1


class PreAdapter(nn.Module):
    def __init__(self, config: AdapterConfig) -> None:
        """Build modality-specific input projections and rotary embeddings."""
        super().__init__()
        self.config = config
        self.adapter_dim = config.hidden_size * config.virtual_width_factor
        self.video_embedder = nn.Linear(
            config.video_in_channels, self.adapter_dim, bias=True, dtype=torch.float32
        )
        self.text_embedder = nn.Linear(
            config.text_in_channels, self.adapter_dim, bias=True, dtype=torch.float32
        )
        self.audio_embedder = nn.Linear(
            config.audio_in_channels, self.adapter_dim, bias=True, dtype=torch.float32
        )
        self.rope = ElementWiseFourierEmbed(
            config.hidden_size // config.num_attention_heads,
            in_pixels=False,
            learnable=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        video_idx: torch.Tensor,
        audio_idx: torch.Tensor,
        text_idx: torch.Tensor,
        time_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project packed modality tokens and build their rotary embeddings."""
        rope = self.rope(coords_mapping)
        output_x = torch.zeros(
            x.shape[0], self.adapter_dim, device=x.device, dtype=torch.float32
        )
        if text_idx.numel() > 0:
            output_x.index_copy_(
                0,
                text_idx,
                self.text_embedder(
                    x.index_select(0, text_idx)[:, : self.config.text_in_channels]
                ),
            )
        if audio_idx.numel() > 0:
            output_x.index_copy_(
                0,
                audio_idx,
                self.audio_embedder(
                    x.index_select(0, audio_idx)[:, : self.config.audio_in_channels]
                ),
            )
        if video_idx.numel() > 0:
            output_x.index_copy_(
                0,
                video_idx,
                self.video_embedder(
                    x.index_select(0, video_idx)[:, : self.config.video_in_channels]
                ),
            )
        return output_x, rope


class PostAdapter(nn.Module):
    def __init__(self, config: AdapterConfig) -> None:
        """Build modality-specific output normalization and projections."""
        super().__init__()
        self.config = config
        self.adapter_dim = config.hidden_size * config.virtual_width_factor
        self.final_norm_video = MultiModalityRMSNorm(self.adapter_dim)
        self.final_norm_audio = MultiModalityRMSNorm(self.adapter_dim)
        self.final_linear_video = nn.Linear(
            self.adapter_dim, config.video_in_channels, bias=False, dtype=torch.float32
        )
        self.final_linear_audio = nn.Linear(
            self.adapter_dim, config.audio_in_channels, bias=False, dtype=torch.float32
        )
        self.final_out_dim = max(config.video_in_channels, config.audio_in_channels)

    def forward(
        self, x: torch.Tensor, video_idx: torch.Tensor, audio_idx: torch.Tensor
    ) -> torch.Tensor:
        """Project packed video and audio features into channel outputs."""
        x_out = torch.zeros(
            x.shape[0], self.final_out_dim, device=x.device, dtype=torch.float32
        )
        if video_idx.numel() > 0:
            x_video = self.final_norm_video(
                x.index_select(0, video_idx).to(self.final_norm_video.weight.dtype)
            )
            x_video = self.final_linear_video(x_video).to(torch.float32)
            x_out[:, : self.config.video_in_channels].index_copy_(0, video_idx, x_video)
        if audio_idx.numel() > 0:
            x_audio = self.final_norm_audio(
                x.index_select(0, audio_idx).to(self.final_norm_audio.weight.dtype)
            )
            x_audio = self.final_linear_audio(x_audio).to(torch.float32)
            x_out[:, : self.config.audio_in_channels].index_copy_(0, audio_idx, x_audio)
        return x_out


class TransformerLayer(nn.Module):
    def __init__(self, model_config: ModelConfig, layer_idx: int) -> None:
        """Build attention, expert, and multi-stream connections for one layer."""
        super().__init__()
        num_modality = 3 if layer_idx in model_config.mm_layers else 1
        self.config_mhc = model_config.mhc_config
        self.attention = Attention(
            AttentionConfig(
                hidden_size=model_config.hidden_size,
                num_heads_q=model_config.num_heads_q,
                num_heads_kv=model_config.num_heads_kv,
                head_dim=model_config.head_dim,
                params_dtype=model_config.params_dtype,
                num_modality=num_modality,
                num_layers=model_config.num_layers,
                attn_softcap=model_config.attn_softcap,
                sink_token_num=model_config.attn_sinks.sink_token_num,
            )
        )
        activation_type = MLPActivationType(
            model_config.layer_activation_types.get(
                layer_idx, model_config.activation_type
            )
        )
        gated_act = activation_type in [MLPActivationType.SWIGLU7]
        if layer_idx in model_config.moe_config.moe_layers:
            moe_cfg = model_config.moe_config
            self.mlp = MultiHeadMoELayer(
                MultiHeadMoELayerConfig(
                    hidden_size=model_config.hidden_size,
                    num_modality=3,
                    num_layers=model_config.num_layers,
                    activation_type=activation_type,
                    params_dtype=model_config.params_dtype,
                    shared_expert_intermediate_sizes=moe_cfg.shared_expert_intermediate_size,
                    modality_specific_expert_intermediate_sizes=moe_cfg.modality_specific_expert_intermediate_size,
                    shared_expert_params_dtype=torch.bfloat16,
                    split_merge_intermediate_size=moe_cfg.split_merge_intermediate_size,
                    moe_config=moe_cfg,
                )
            )
        else:
            intermediate_size = (
                int(model_config.hidden_size * model_config.intermediate_factor * 2 / 3)
                // 128
                * 128
                if gated_act
                else int(model_config.hidden_size * model_config.intermediate_factor)
            )
            self.mlp = MLP(
                MLPConfig(
                    hidden_size=model_config.hidden_size,
                    intermediate_size=intermediate_size,
                    activation_type=activation_type,
                    params_dtype=model_config.params_dtype,
                    num_modality=num_modality,
                    num_layers=model_config.num_layers,
                    gated_act=gated_act,
                )
            )
        self._init_mhc(model_config, num_modality)

    def _init_mhc(self, model_config: ModelConfig, num_modality: int) -> None:
        """Allocate hyper-connection parameters for attention and feed-forward stages."""
        n = self.config_mhc.num_stream
        c = model_config.hidden_size
        dtype = torch.float32
        self.mhc_alpha_pre_attn = nn.Parameter(
            torch.full((1,), float(self.config_mhc.alpha_init), dtype=dtype)
        )
        self.mhc_alpha_post_attn = nn.Parameter(
            torch.full((1,), float(self.config_mhc.alpha_init), dtype=dtype)
        )
        self.mhc_alpha_res_attn = nn.Parameter(
            torch.full((1,), float(self.config_mhc.alpha_init), dtype=dtype)
        )
        self.mhc_alpha_pre_mlp = nn.Parameter(
            torch.full((1,), float(self.config_mhc.alpha_init), dtype=dtype)
        )
        self.mhc_alpha_post_mlp = nn.Parameter(
            torch.full((1,), float(self.config_mhc.alpha_init), dtype=dtype)
        )
        self.mhc_alpha_res_mlp = nn.Parameter(
            torch.full((1,), float(self.config_mhc.alpha_init), dtype=dtype)
        )
        self.mhc_bias_pre_attn = nn.Parameter(torch.empty(n, dtype=dtype))
        self.mhc_bias_post_attn = nn.Parameter(torch.empty(n, dtype=dtype))
        self.mhc_bias_pre_mlp = nn.Parameter(torch.empty(n, dtype=dtype))
        self.mhc_bias_post_mlp = nn.Parameter(torch.empty(n, dtype=dtype))
        self.mhc_bias_res_attn = nn.Parameter(torch.empty(n, n, dtype=dtype))
        self.mhc_bias_res_mlp = nn.Parameter(torch.empty(n, n, dtype=dtype))
        self.mhc_phi_fused_attn = nn.Parameter(
            torch.empty(n * c, n + n + (n * n), dtype=dtype)
        )
        self.mhc_phi_fused_mlp = nn.Parameter(
            torch.empty(n * c, n + n + (n * n), dtype=dtype)
        )
        self.mhc_norm = MultiModalityRMSNorm(
            c * n, num_modality=num_modality, out_dtype=dtype
        )
        self.mhc_handler = MHCHandler(
            n_stream=n, hidden_size=c, num_sk_iters=20, sk_eps=1e-12, dtype=dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        varlen_handler: VarlenHandler,
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """Run attention and feed-forward stages through multi-stream connections."""
        n = self.config_mhc.num_stream
        s = hidden_states.reshape(hidden_states.shape[0], n, -1)
        assert s.is_contiguous()

        h_post_attn_, h_res_attn_, attn_in = self._forward_mhc_pre_attn(
            s, modality_dispatcher
        )
        attn_out = self.attention(
            attn_in, rope, varlen_handler, modality_dispatcher, cp_split_sizes
        )

        s_, h_post_mlp_, h_res_mlp_, mlp_in = self._forward_mhc_attn_to_mlp(
            s, attn_out, h_post_attn_, h_res_attn_, modality_dispatcher
        )
        mlp_out = self.mlp(mlp_in, modality_dispatcher)

        s = self._forward_mhc_post_mlp(s_, mlp_out, h_post_mlp_, h_res_mlp_)
        return s.reshape(s.shape[0], -1)

    def _forward_mhc_pre_attn(
        self, s: torch.Tensor, modality_dispatcher: ModalityDispatcher
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute pre-attention stream gates and the attention input."""
        s_flat = self.mhc_handler.flatten_mstream(s)
        h_pre_attn_, h_post_attn_, h_res_attn_ = (
            self.mhc_handler.apply_norm_and_compute_h_(
                x=s_flat,
                norm_fn=partial(self.mhc_norm, modality_dispatcher=modality_dispatcher),
                phi_fused=self.mhc_phi_fused_attn,
            )
        )
        attn_in = self.mhc_handler.compute_and_apply_hpre(
            x=s,
            abh_pre=(self.mhc_alpha_pre_attn, self.mhc_bias_pre_attn, h_pre_attn_),
            out_dtype=s.dtype,
        )
        return h_post_attn_, h_res_attn_, attn_in

    def _forward_mhc_attn_to_mlp(
        self,
        s: torch.Tensor,
        attn_out: torch.Tensor,
        h_post_attn_: torch.Tensor,
        h_res_attn_: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mix attention output into residual streams and prepare the MLP input."""
        h_post_attn, h_res_attn = self.mhc_handler.compute_hpost_and_hres(
            abh_post=(self.mhc_alpha_post_attn, self.mhc_bias_post_attn, h_post_attn_),
            abh_res=(self.mhc_alpha_res_attn, self.mhc_bias_res_attn, h_res_attn_),
            out_dtype=s.dtype,
        )

        s_ = self.mhc_handler.apply_hpost_and_hres(
            res=s, out=attn_out, h_post=h_post_attn, h_res=h_res_attn
        )
        s_flat = self.mhc_handler.flatten_mstream(s_)
        h_pre_mlp_, h_post_mlp_, h_res_mlp_ = (
            self.mhc_handler.apply_norm_and_compute_h_(
                x=s_flat,
                norm_fn=partial(self.mhc_norm, modality_dispatcher=modality_dispatcher),
                phi_fused=self.mhc_phi_fused_mlp,
            )
        )
        mlp_in = self.mhc_handler.compute_and_apply_hpre(
            x=s_,
            abh_pre=(self.mhc_alpha_pre_mlp, self.mhc_bias_pre_mlp, h_pre_mlp_),
            out_dtype=s.dtype,
        )
        return s_, h_post_mlp_, h_res_mlp_, mlp_in

    def _forward_mhc_post_mlp(
        self,
        s_: torch.Tensor,
        mlp_out: torch.Tensor,
        h_post_mlp_: torch.Tensor,
        h_res_mlp_: torch.Tensor,
    ) -> torch.Tensor:
        """Mix the feed-forward output back into residual streams."""
        h_post_mlp, h_res_mlp = self.mhc_handler.compute_hpost_and_hres(
            abh_post=(self.mhc_alpha_post_mlp, self.mhc_bias_post_mlp, h_post_mlp_),
            abh_res=(self.mhc_alpha_res_mlp, self.mhc_bias_res_mlp, h_res_mlp_),
            out_dtype=s_.dtype,
        )
        return self.mhc_handler.apply_hpost_and_hres(
            res=s_, out=mlp_out, h_post=h_post_mlp, h_res=h_res_mlp
        )


@magi_compile(
    dynamic_arg_dims={
        "x": 0,
        "rope": 0,
        "varlen_handler.cu_seqlens_q": 0,
        "varlen_handler.cu_seqlens_k": 0,
        "modality_dispatcher.modality_mapping": 0,
        "modality_dispatcher.permuted_modality_mapping": 0,
        "modality_dispatcher.permute_mapping": 0,
        "modality_dispatcher.inv_permute_mapping": 0,
    }
)
class TransformerBlock(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(model_config, i) for i in range(model_config.num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        varlen_handler: VarlenHandler,
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """Apply every transformer layer to context-parallel packed tokens."""
        x = x.to(torch.bfloat16)
        x = modality_dispatcher._permute(x)
        for layer in self.layers:
            x = layer(x, rope, varlen_handler, modality_dispatcher, cp_split_sizes)
        x = modality_dispatcher._inv_permute(x)
        return x


def _initialize_magi2_skip_load_weights(module: nn.Module) -> None:
    """Initialize placeholder parameters when checkpoint loading is skipped."""
    with torch.no_grad():
        for name, submodule in module.named_modules():
            if isinstance(submodule, CustomGroupedLinear):
                if name.endswith(".split_linear"):
                    submodule.weight.fill_(1e-3)
                else:
                    submodule.weight.zero_()
                if submodule.bias is not None:
                    submodule.bias.zero_()
            elif isinstance(submodule, MultiModalityRMSNorm):
                submodule.weight.zero_()
            elif isinstance(submodule, Attention):
                submodule.sinks.zero_()
            elif isinstance(submodule, CoreMultiHeadMoE):
                nn.init.normal_(submodule.gate, mean=0.0, std=0.02)
                submodule.W_gate.zero_()
                submodule.W_up.zero_()
                submodule.W_down.zero_()
                submodule.router.expert_bias.zero_()
                submodule.router.expert_bias_ema.zero_()
            elif isinstance(submodule, TransformerLayer):
                for param_name, parameter in submodule.named_parameters(recurse=False):
                    if param_name.startswith(("mhc_bias_", "mhc_phi_fused_")):
                        parameter.zero_()


def _validate_inference_model_config(model_config: ModelConfig) -> None:
    if not model_config.mhc_config.enable:
        raise ValueError("MAGI-2 inference expects mhc_config.enable=true.")
    if not model_config.attn_gating.enable:
        raise ValueError(
            "MAGI-2 inference expects scalar attention gating to be enabled."
        )
    if not model_config.attn_sinks.enable:
        raise ValueError("MAGI-2 inference expects sh-layout attention sinks.")


class Transformer(BaseDiT):
    _config = Magi2PreviewVideoConfig()
    _fsdp_shard_conditions = _config._fsdp_shard_conditions
    _compile_conditions = _config._compile_conditions
    _supported_attention_backends = _config._supported_attention_backends
    param_names_mapping = _config.param_names_mapping
    reverse_param_names_mapping = _config.reverse_param_names_mapping
    lora_param_names_mapping = _config.lora_param_names_mapping

    def __init__(
        self,
        config: Magi2PreviewVideoConfig,
        hf_config: dict | None = None,
        ep_size: int = 1,
        **kwargs,
    ) -> None:
        """Construct the production MAGI-2 preview transformer with official module names."""
        del kwargs
        super().__init__(config=config, hf_config=hf_config or {})
        model_config = config.arch_config
        model_config.params_dtype = _resolve_params_dtype(model_config.params_dtype)
        _validate_inference_model_config(model_config)
        self.config = model_config
        self.ep_size = ep_size
        self.hidden_size = model_config.hidden_size
        self.num_attention_heads = model_config.num_heads_q
        self.num_channels_latents = model_config.num_channels_latents
        adapter_config = AdapterConfig(
            hidden_size=model_config.hidden_size,
            num_attention_heads=model_config.num_heads_q,
            text_in_channels=model_config.text_in_channels,
            video_in_channels=model_config.video_in_channels,
            audio_in_channels=model_config.audio_in_channels,
            virtual_width_factor=model_config.mhc_config.num_stream,
        )
        self.pre_adapter = PreAdapter(adapter_config)
        self.post_adapter = PostAdapter(adapter_config)
        self.block = TransformerBlock(model_config)
        if env_is_true("SKIP_LOAD_MODEL"):
            self.initialize_for_skip_load()
        self.__post_init__()

    def initialize_for_skip_load(self) -> None:
        _initialize_magi2_skip_load_weights(self)

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        modality_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        time_token_sequence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transform packed video, audio, text, and time-token features."""
        x = ulysses_scheduler().dispatch(x)
        coords_mapping = ulysses_scheduler().dispatch(coords_mapping)
        modality_mapping = ulysses_scheduler().dispatch(modality_mapping)
        if time_token_sequence is not None:
            time_token_sequence = ulysses_scheduler().dispatch(time_token_sequence)
        cp_split_sizes = ulysses_scheduler().cp_split_sizes
        cp_split_sizes_t = torch.tensor(
            cp_split_sizes, device=x.device, dtype=torch.long
        )

        time_mask = modality_mapping == Modality.TIME
        modality_mapping = modality_mapping.clone()
        modality_mapping[time_mask] = Modality.TEXT
        modality_dispatcher = ModalityDispatcher(modality_mapping, 3)
        video_idx = (modality_mapping == Modality.VIDEO).nonzero().flatten()
        audio_idx = (modality_mapping == Modality.AUDIO).nonzero().flatten()
        text_idx = (modality_mapping == Modality.TEXT).nonzero().flatten()
        time_idx = time_mask.nonzero().flatten()

        x, rope = self.pre_adapter(
            x, coords_mapping, video_idx, audio_idx, text_idx, time_idx
        )
        if time_token_sequence is not None and time_token_sequence.shape[-1] > 0:
            x[:, : time_token_sequence.shape[-1]] = time_token_sequence.to(x.dtype)

        x = self.block(
            x,
            rope,
            varlen_handler=varlen_handler,
            modality_dispatcher=modality_dispatcher,
            cp_split_sizes=cp_split_sizes_t,
        )
        x_out = self.post_adapter(x, video_idx, audio_idx)
        x_out = ulysses_scheduler().undispatch(x_out)
        return x_out


# -----------------------------------------------------------------------------
# MoE routing
# -----------------------------------------------------------------------------

"""Top-k routing and global sort for multi-head MoE.

Multi-head MoE routing with pure-torch implementations.
Sufficient for inference; training may benefit from fused CUDA/Triton variants.
"""

from typing import Literal

import torch


# -----------------------------------------------------------------------------
# MoE expert-parallel dispatch
# -----------------------------------------------------------------------------

import torch

# ==========================================
# EP Dispatch Autograd Functions
# ==========================================


def ep_dispatch(x: torch.Tensor, ep_group: dist.ProcessGroup) -> torch.Tensor:
    """
    Dispatch tokens to the Rank with the specified Head.

    Input tensor has shape ``(S, H, D)``.
    Output tensor has shape ``(S * ep_size, H / ep_size, D)``.
    """
    ep_size = dist.get_world_size(ep_group) if ep_group is not None else 1
    if ep_size == 1:
        return x

    S, H, D = x.shape
    assert H % ep_size == 0, (
        f"Number of heads H ({H}) must be divisible by ep_size ({ep_size})"
    )

    x = x.contiguous().view(S, ep_size, H // ep_size, D)
    x = x.permute(1, 0, 2, 3).contiguous()

    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=ep_group)

    return out.view(ep_size * S, H // ep_size, D)



def ep_undispatch(x: torch.Tensor, ep_group: dist.ProcessGroup) -> torch.Tensor:
    """
    Undispatch tokens back to the original Rank and concatenate back to the complete Head.

    Input tensor has shape ``(S * ep_size, H / ep_size, D)``.
    Output tensor has shape ``(S, H, D)``.
    """
    ep_size = dist.get_world_size(ep_group) if ep_group is not None else 1
    if ep_size == 1:
        return x

    total_S, H_per_ep, D = x.shape
    assert total_S % ep_size == 0, (
        f"Total sequence length ({total_S}) must be divisible by ep_size ({ep_size})"
    )

    S = total_S // ep_size

    x = x.contiguous().view(ep_size, S, H_per_ep, D)

    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=ep_group)

    out = out.permute(1, 0, 2, 3).contiguous()
    return out.view(S, H_per_ep * ep_size, D)


# -----------------------------------------------------------------------------
# MoE Triton forward
# -----------------------------------------------------------------------------

"""Plain multi-head MoE forward implementation (inference only)."""


import torch
import triton

# ═══════════════════════════════════════════════════════════════════════════
#  SwiGLU7 device function
# ═══════════════════════════════════════════════════════════════════════════

SWIGLU7_ALPHA = tl.constexpr(1.702)
SWIGLU7_LIMIT = tl.constexpr(7.0)
SWIGLU7_BIAS = tl.constexpr(1.0)


@triton.jit
def swiglu7_fwd_func(gate, up, out_dtype: tl.constexpr):
    """Clamped, weighted, biased SwiGLU (GPT-OSS variant). Returns hidden only.

    hidden = swish(clamp(gate)) * (clamp(up) + 1), with a fast sigmoid swish.
    """
    gate_clamped = tl.minimum(gate, SWIGLU7_LIMIT)
    up_clamped = tl.maximum(tl.minimum(up, SWIGLU7_LIMIT), -SWIGLU7_LIMIT)

    sig = tl.sigmoid(SWIGLU7_ALPHA * gate_clamped)
    swish = gate_clamped * sig
    swiglu = swish * (up_clamped + SWIGLU7_BIAS)

    return swiglu.to(out_dtype)


# ═══════════════════════════════════════════════════════════════════════════
#  Expert-id binary search
# ═══════════════════════════════════════════════════════════════════════════


@triton.jit
def binary_search_expert_id(
    cu_expert_tile_counts_ptr,
    tile_id,
    NUM_EXPERTS: tl.constexpr,
    LOG2_NUM_EXPERTS: tl.constexpr,
):
    """Return the expert id owning ``tile_id`` given the cumulative tile counts."""
    lo = 0
    hi = NUM_EXPERTS
    for _ in tl.static_range(0, LOG2_NUM_EXPERTS + 1):
        mid = (lo + hi + 1) // 2
        below = tl.load(cu_expert_tile_counts_ptr + mid) <= tile_id
        lo = tl.where(below, mid, lo)
        hi = tl.where(below, hi, mid - 1)
    return lo


Magi2PreviewDiT = Transformer
EntryClass = Magi2PreviewDiT
