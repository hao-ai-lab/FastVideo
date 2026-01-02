# SPDX-License-Identifier: Apache-2.0
# SLA (Sparse-Linear Attention) backend for FastVideo
# Adapted from TurboDiffusion SLA implementation
#
# Copyright (c) 2025 by SLA team.
# Citation:
# @article{zhang2025sla,
#   title={SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention},
#   author={Jintao Zhang and Haoxu Wang and Kai Jiang and Shuo Yang and Kaiwen Zheng and
#           Haocheng Xi and Ziteng Wang and Hongzhou Zhu and Min Zhao and Ion Stoica and
#           Joseph E. Gonzalez and Jun Zhu and Jianfei Chen},
#   journal={arXiv preprint arXiv:2509.24006},
#   year={2025}
# }

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from fastvideo.attention.backends.sla_kernels import _attention, get_block_map
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class SLAAttentionBackend(AttentionBackend):
    """Sparse-Linear Attention backend."""

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "SLA_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SLAAttentionImpl"]:
        return SLAAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["SLAAttentionMetadata"]:
        return SLAAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["SLAAttentionMetadataBuilder"]:
        return SLAAttentionMetadataBuilder


@dataclass
class SLAAttentionMetadata(AttentionMetadata):
    """Metadata for SLA attention."""
    current_timestep: int
    topk_ratio: float = 0.5  # Ratio of key blocks to attend to


class SLAAttentionMetadataBuilder(AttentionMetadataBuilder):
    """Builder for SLA attention metadata."""

    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(
        self,
        current_timestep: int,
        topk_ratio: float = 0.5,
        **kwargs: dict[str, Any],
    ) -> SLAAttentionMetadata:
        return SLAAttentionMetadata(
            current_timestep=current_timestep,
            topk_ratio=topk_ratio,
        )


class SLAAttentionImpl(AttentionImpl, nn.Module):
    """SLA attention implementation with learnable linear projection.
    
    This implementation combines sparse attention with linear attention,
    using a learnable projection to blend the outputs. The sparse attention
    uses a block-sparse pattern determined by QK similarity.
    
    Args:
        num_heads: Number of attention heads
        head_size: Dimension of each head
        topk_ratio: Ratio of key blocks to attend to (0-1), default 0.5
        feature_map: Feature map for linear attention ('softmax', 'elu', 'relu')
        BLKQ: Query block size for sparse attention
        BLKK: Key block size for sparse attention
        use_bf16: Whether to use bfloat16 for computation
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool = False,
        softmax_scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
        # SLA-specific parameters
        topk_ratio: float = 0.5,
        feature_map: str = "softmax",
        BLKQ: int = 64,
        BLKK: int = 64,
        use_bf16: bool = True,
        **extra_impl_args,
    ) -> None:
        nn.Module.__init__(self)
        
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale if softmax_scale else head_size**-0.5
        self.causal = causal
        self.prefix = prefix
        
        # SLA-specific config
        self.topk_ratio = topk_ratio
        self.BLKQ = BLKQ
        self.BLKK = BLKK
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        
        # Learnable linear projection for combining sparse + linear attention
        self.proj_l = nn.Linear(head_size, head_size, dtype=torch.float32)
        
        # Feature map for linear attention
        if feature_map == "elu":
            self.feature_map_q = lambda x: F.elu(x) + 1
            self.feature_map_k = lambda x: F.elu(x) + 1
        elif feature_map == "relu":
            self.feature_map_q = F.relu
            self.feature_map_k = F.relu
        elif feature_map == "softmax":
            self.feature_map_q = lambda x: F.softmax(x, dim=-1)
            self.feature_map_k = lambda x: F.softmax(x, dim=-1)
        else:
            raise ValueError(f"Unknown feature map: {feature_map}")
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize projection weights to zero for residual-like behavior."""
        with torch.no_grad():
            nn.init.zeros_(self.proj_l.weight)
            nn.init.zeros_(self.proj_l.bias)  # type: ignore[arg-type]

    def _calc_linear_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Compute linear attention: (Q @ K^T @ V) / normalizer.
        
        Args:
            q: Query tensor (B, H, L, D) after feature map
            k: Key tensor (B, H, L, D) after feature map
            v: Value tensor (B, H, L, D)
            
        Returns:
            Linear attention output (B, H, L, D)
        """
        kvsum = k.transpose(-1, -2) @ v  # (B, H, D, D)
        ksum = torch.sum(k, dim=-2, keepdim=True)  # (B, H, 1, D)
        return (q @ kvsum) / (1e-5 + (q * ksum).sum(dim=-1, keepdim=True))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Forward pass for SLA attention.
        
        Input tensors are in FastVideo format: (B, L, H, D)
        Internally converted to SLA format: (B, H, L, D)
        
        Args:
            query: Query tensor (B, L, H, D)
            key: Key tensor (B, L, H, D)
            value: Value tensor (B, L, H, D)
            attn_metadata: Attention metadata
            
        Returns:
            Output tensor (B, L, H, D)
        """
        original_dtype = query.dtype

        # print("running sla")
        
        # Convert from FastVideo format (B, L, H, D) to SLA format (B, H, L, D)
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        
        # Get topk ratio from metadata if available
        topk_ratio = self.topk_ratio
        if hasattr(attn_metadata, 'topk_ratio'):
            topk_ratio = attn_metadata.topk_ratio  # type: ignore[union-attr]
        
        # Compute block-sparse attention pattern
        sparse_map, lut, real_topk = get_block_map(
            q, k, topk_ratio=topk_ratio, BLKQ=self.BLKQ, BLKK=self.BLKK
        )
        
        # Convert to compute dtype
        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)
        
        # Sparse attention
        o_s = _attention.apply(
            q, k, v, sparse_map, lut, real_topk, self.BLKQ, self.BLKK
        )
        
        # Linear attention with feature maps
        q_linear = self.feature_map_q(q).contiguous().to(self.dtype)
        k_linear = self.feature_map_k(k).contiguous().to(self.dtype)
        o_l = self._calc_linear_attention(q_linear, k_linear, v)
        
        # Project linear attention output and combine
        with torch.amp.autocast('cuda', dtype=self.dtype):
            o_l = self.proj_l(o_l)
        
        # Combine sparse and linear outputs
        output = (o_s + o_l).to(original_dtype)
        
        # Convert back to FastVideo format (B, L, H, D)
        output = output.transpose(1, 2)
        
        return output
