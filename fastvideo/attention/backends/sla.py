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
        # SLA-specific parameters - matched to TurboDiffusion defaults
        topk_ratio: float = 0.1,  # TurboDiffusion uses topk=0.1
        feature_map: str = "softmax",
        BLKQ: int = 128,  # TurboDiffusion uses BLKQ=128
        BLKK: int = 64,   # TurboDiffusion uses BLKK=64
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

        print("running sla in fv")
        
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


# Check if spas_sage_attn is available for SageSLA
SAGESLA_ENABLED = True
try:
    import spas_sage_attn._qattn as qattn
    import spas_sage_attn._fused as fused
    from spas_sage_attn.utils import get_vanilla_qk_quant, block_map_lut_triton
except ImportError:
    SAGESLA_ENABLED = False

SAGE2PP_ENABLED = True
try:
    from spas_sage_attn._qattn import qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
except ImportError:
    SAGE2PP_ENABLED = False


class SageSLAAttentionBackend(AttentionBackend):
    """Quantized Sparse-Linear Attention backend using SageAttention kernels."""

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "SAGE_SLA_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SageSLAAttentionImpl"]:
        return SageSLAAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["SLAAttentionMetadata"]:
        return SLAAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["SLAAttentionMetadataBuilder"]:
        return SLAAttentionMetadataBuilder


def _get_cuda_arch(device_index: int) -> str:
    """Get CUDA architecture string for the given device."""
    major, minor = torch.cuda.get_device_capability(device_index)
    return f"sm{major}{minor}"


class SageSLAAttentionImpl(AttentionImpl, nn.Module):
    """SageSLA attention implementation using quantized SageAttention kernels.
    
    This uses INT8 quantization for Q/K and FP8 for V to achieve better performance
    while maintaining accuracy. Requires spas_sage_attn package.
    
    Args:
        num_heads: Number of attention heads
        head_size: Dimension of each head (must be 64 or 128)
        topk_ratio: Ratio of key blocks to attend to (0-1), default 0.5
        feature_map: Feature map for linear attention ('softmax', 'elu', 'relu')
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
        # SageSLA-specific parameters
        topk_ratio: float = 0.5,
        feature_map: str = "softmax",
        use_bf16: bool = True,
        **extra_impl_args,
    ) -> None:
        nn.Module.__init__(self)
        
        if not SAGESLA_ENABLED:
            raise ImportError(
                "SageSLA requires spas_sage_attn. "
                "Install with: pip install git+https://github.com/thu-ml/SpargeAttn.git"
            )
        
        assert head_size in [64, 128], f"SageSLA requires head_size in [64, 128], got {head_size}"
        
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale if softmax_scale else head_size**-0.5
        self.causal = causal
        self.prefix = prefix
        
        # SageSLA-specific config
        self.topk_ratio = topk_ratio
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
        """Compute linear attention: (Q @ K^T @ V) / normalizer."""
        kvsum = k.transpose(-1, -2) @ v
        ksum = torch.sum(k, dim=-2, keepdim=True)
        return (q @ kvsum) / (1e-5 + (q * ksum).sum(dim=-1, keepdim=True))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Forward pass for SageSLA attention with quantized kernels.
        
        Input tensors are in FastVideo format: (B, L, H, D)
        
        Args:
            query: Query tensor (B, L, H, D)
            key: Key tensor (B, L, H, D)
            value: Value tensor (B, L, H, D)
            attn_metadata: Attention metadata
            
        Returns:
            Output tensor (B, L, H, D)
        """
        original_dtype = query.dtype
        
        # Convert from FastVideo format (B, L, H, D) to SLA format (B, H, L, D)
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        
        # Get topk ratio from metadata if available
        topk_ratio = self.topk_ratio
        if hasattr(attn_metadata, 'topk_ratio'):
            topk_ratio = attn_metadata.topk_ratio  # type: ignore[union-attr]
        
        # Determine block sizes based on GPU architecture
        arch = _get_cuda_arch(q.device.index)
        if arch == "sm90":
            BLKQ, BLKK = 64, 128
        else:
            BLKQ, BLKK = 128, 64
        
        # Compute block-sparse attention pattern
        sparse_map, lut, real_topk = get_block_map(
            q, k, topk_ratio=topk_ratio, BLKQ=BLKQ, BLKK=BLKK
        )
        
        # Convert to compute dtype
        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)
        
        # ========== SPARGE QUANTIZED ATTENTION ==========
        km = k.mean(dim=-2, keepdim=True)
        headdim = q.size(-1)
        scale = 1.0 / (headdim ** 0.5)
        
        # Quantize Q, K to INT8
        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km, BLKQ, BLKK)
        lut_triton, valid_block_num = block_map_lut_triton(sparse_map)
        
        # Quantize V to FP8
        b, h_kv, kv_len, head_dim = v.shape
        padded_len = (kv_len + 127) // 128 * 128
        v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
        fused.transpose_pad_permute_cuda(v, v_transposed_permutted, 1)
        v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
        v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
        fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 2.25, 1)
        
        # Sparse attention with quantized kernels
        o_s = torch.empty_like(q)
        if arch == "sm90":
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_sm90(
                q_int8, k_int8, v_fp8, o_s, lut_triton, valid_block_num, 
                q_scale, k_scale, v_scale, 1, False, 1, scale
            )
        else:
            pvthreshold = torch.full((q.shape[-3],), 1e6, dtype=torch.float32, device=q.device)
            if SAGE2PP_ENABLED:
                qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
                    q_int8, k_int8, v_fp8, o_s, lut_triton, valid_block_num, pvthreshold,
                    q_scale, k_scale, v_scale, 1, False, 1, scale, 0
                )
            else:
                qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
                    q_int8, k_int8, v_fp8, o_s, lut_triton, valid_block_num, pvthreshold,
                    q_scale, k_scale, v_scale, 1, False, 1, scale, 0
                )
        # ========== END SPARGE ==========
        
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
