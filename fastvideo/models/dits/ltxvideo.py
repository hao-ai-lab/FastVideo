# Copyright 2025 The Genmo team and The HuggingFace Team.
# All rights reserved.
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

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# FastVideo optimized imports
from fastvideo.layers.layernorm import RMSNorm, FP32LayerNorm, LayerNormScaleShift, ScaleResidualLayerNormScaleShift
from fastvideo.layers.layernorm import ScaleResidual
from fastvideo.layers.activation import NewGELU
from fastvideo.layers.vocab_parallel_embedding import VocabParallelEmbedding, UnquantizedEmbeddingMethod
from fastvideo.configs.models.dits import LTXVideoConfig
from fastvideo.layers.visual_embedding import TimestepEmbedder
from fastvideo.layers.linear import ReplicatedLinear, ColumnParallelLinear, RowParallelLinear, QKVParallelLinear
from fastvideo.platforms import AttentionBackendEnum, current_platform

from diffusers.models.attention import FeedForward


#from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.normalization import AdaLayerNormSingle

from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
# from ..attention import FeedForward
from fastvideo.attention import DistributedAttention, LocalAttention
#from diffusers.attention_processor import Attention
from fastvideo.models.dits.base import CachableDiT

from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
# from ..normalization import AdaLayerNormSingle

from fastvideo.layers.linear import ReplicatedLinear


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



class LTXVideoAttentionProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the LTX model. It applies a normalization layer and rotary embedding on the query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "LTXVideoAttentionProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: LocalAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # TODO: Optimize with fused QKV projection for better performance
        # Current implementation uses separate projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # # Fused QKV projection example
        # from fastvideo.layers.linear import QKVParallelLinear
        # self.qkv_proj = QKVParallelLinear(
        #     hidden_size=hidden_size,
        #     head_size=attention_head_dim,
        #     total_num_heads=num_attention_heads,
        #     bias=True
        # )

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # TODO: Consider local attention patterns for long sequences
        # TODO: Add distributed attention support for multi-GPU setups
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # TODO: Add output dimension parallelism here
        #example
        # # Output dimension parallelism
        # from fastvideo.layers.linear import ColumnParallelLinear
        # self.q_proj = ColumnParallelLinear(
        #     input_size=hidden_size,
        #     output_size=head_size * num_heads,
        #     bias=bias,
        #     gather_output=False
        # )
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

        


class FastVideoLTXRotaryPosEmbed(nn.Module):
    """FastVideo optimized rotary position embedding for LTX model."""
    
    def __init__(
        self,
        dim: int,
        base_num_frames: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        patch_size: int = 1,
        patch_size_t: int = 1,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.base_num_frames = base_num_frames
        self.base_height = base_height
        self.base_width = base_width
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.theta = theta

    def _prepare_video_coords(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        rope_interpolation_scale: Tuple[torch.Tensor, float, float],
        device: torch.device,
    ) -> torch.Tensor:
        # Add defaults based on base dimensions if None
        num_frames = num_frames or self.base_num_frames
        height = height or (self.base_height // self.patch_size)
        width = width or (self.base_width // self.patch_size)
        print(f"num_frames {num_frames} height{height} width {width}")
        # Always compute rope in fp32
        grid_h = torch.arange(height, dtype=torch.float32, device=device)
        grid_w = torch.arange(width, dtype=torch.float32, device=device)
        grid_f = torch.arange(num_frames, dtype=torch.float32, device=device)
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

        if rope_interpolation_scale is not None:
            grid[:, 0:1] = grid[:, 0:1] * rope_interpolation_scale[0] * self.patch_size_t / self.base_num_frames
            grid[:, 1:2] = grid[:, 1:2] * rope_interpolation_scale[1] * self.patch_size / self.base_height
            grid[:, 2:3] = grid[:, 2:3] * rope_interpolation_scale[2] * self.patch_size / self.base_width

        grid = grid.flatten(2, 4).transpose(1, 2)

        return grid

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        rope_interpolation_scale: Optional[Tuple[torch.Tensor, float, float]] = None,
        video_coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.size(0)

        if video_coords is None:
            grid = self._prepare_video_coords(
                batch_size,
                num_frames,
                height,
                width,
                rope_interpolation_scale=rope_interpolation_scale,
                device=hidden_states.device,
            )
        else:
            grid = torch.stack(
                [
                    video_coords[:, 0] / self.base_num_frames,
                    video_coords[:, 1] / self.base_height,
                    video_coords[:, 2] / self.base_width,
                ],
                dim=-1,
            )

        start = 1.0
        end = self.theta
        freqs = self.theta ** torch.linspace(
            math.log(start, self.theta),
            math.log(end, self.theta),
            self.dim // 6,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        freqs = freqs * math.pi / 2.0
        freqs = freqs * (grid.unsqueeze(-1) * 2 - 1)
        freqs = freqs.transpose(-1, -2).flatten(2)

        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        if self.dim % 6 != 0:
            cos_padding = torch.ones_like(cos_freqs[:, :, : self.dim % 6])
            sin_padding = torch.zeros_like(cos_freqs[:, :, : self.dim % 6])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs

# class LTXVideoLocalAttention(nn.Module):
#     def __init__(
#         self,
#         query_dim: int,
#         heads: int,
#         dim_head: int,
#         kv_heads: Optional[int] = None,
#         bias: bool = True,
#         cross_attention_dim: Optional[int] = None,
#         out_bias: bool = True,
#         qk_norm: Optional[str] = None,
#         eps: float = 1e-6,
#     ):
#         super().__init__()
        
#         self.inner_dim = heads * dim_head
#         self.cross_attention_dim = cross_attention_dim
#         self.heads = heads
#         self.dim_head = dim_head
        
#         # Handle self-attention vs cross-attention projections
#         if cross_attention_dim is None:  # Self-attention case
#             # Use QKVParallelLinear for fused Q, K, V projections
#             self.qkv_proj = QKVParallelLinear(
#                 hidden_size=query_dim,
#                 head_size=dim_head,
#                 total_num_heads=heads,
#                 total_num_kv_heads=kv_heads or heads,
#                 bias=bias
#             )
#             # No separate to_q, to_k, to_v for self-attention
#             self.to_q = None
#             self.to_k = None  
#             self.to_v = None
#         else:  # Cross-attention case
#             # Keep separate projections for cross-attention
#             self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
#             self.to_k = nn.Linear(cross_attention_dim, self.inner_dim, bias=bias)
#             self.to_v = nn.Linear(cross_attention_dim, self.inner_dim, bias=bias)
#             self.qkv_proj = None
        
#         # Output projection (same for both cases)
#         self.to_out = nn.ModuleList([
#             nn.Linear(self.inner_dim, query_dim, bias=out_bias),
#             nn.Dropout(0.0)
#         ])
        
#         # Norms (same for both cases)
#         if qk_norm == "rms_norm_across_heads":
#             norm_eps = 1e-5 
#             norm_elementwise_affine = True
#             self.norm_q = torch.nn.RMSNorm(self.inner_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
#             self.norm_k = torch.nn.RMSNorm(self.inner_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
#         else:
#             self.norm_q = None
#             self.norm_k = None

#     def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
#         if encoder_hidden_states is None:
#             encoder_hidden_states = hidden_states
        
#         batch_size, seq_len, _ = hidden_states.shape

#         # Handle Q, K, V projection based on attention type
#         if self.qkv_proj is not None:  # Self-attention case
#             # Use fused QKV projection
#             qkv_output, _ = self.qkv_proj(hidden_states)
#             # Split the output into Q, K, V
#             q_size = self.heads * self.dim_head
#             k_size = self.heads * self.dim_head  # Assuming same size for simplicity
#             v_size = self.heads * self.dim_head
            
#             query = qkv_output[..., :q_size]
#             key = qkv_output[..., q_size:q_size + k_size]
#             value = qkv_output[..., q_size + k_size:q_size + k_size + v_size]
#         else:  # Cross-attention case
#             # Use separate projections
#             query = self.to_q(hidden_states)
#             key = self.to_k(encoder_hidden_states)
#             value = self.to_v(encoder_hidden_states)

#         # Rest of the forward method remains the same
#         if hasattr(self, 'norm_q') and self.norm_q is not None:
#             query = self.norm_q(query)
#             key = self.norm_k(key)

#         if image_rotary_emb is not None:
#             query = apply_rotary_emb(query, image_rotary_emb)
#             key = apply_rotary_emb(key, image_rotary_emb)

#         # Reshape and attention computation
#         query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
#         key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)  
#         value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)

#         hidden_states = F.scaled_dot_product_attention(
#             query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#         )
#         hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

#         hidden_states = self.to_out[0](hidden_states)
#         hidden_states = self.to_out[1](hidden_states)

#         return hidden_states

class LTXVideoLocalAttention(nn.Module):
    """Wrapper for LocalAttention"""
    
    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        kv_heads: Optional[int] = None,
        bias: bool = True,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        qk_norm: Optional[str] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        
        self.inner_dim = heads * dim_head
        self.cross_attention_dim = cross_attention_dim
        self.heads = heads
        self.dim_head = dim_head
        
        #Projections
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim or query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim or query_dim, self.inner_dim, bias=bias)
        # self.to_out = nn.ModuleList([
        #     nn.Linear(self.inner_dim, query_dim, bias=out_bias),
        #     nn.Dropout(0.0)
        # ])
        self.to_out = nn.ModuleList([
            RowParallelLinear(
                input_size=self.inner_dim,
                output_size=query_dim,
                bias=out_bias,
                input_is_parallel=True,  # Input comes from parallel attention computation
                reduce_results=True      # All-reduce to gather full result
            ),
            nn.Dropout(0.0)
        ])

        #self.to_out = 
                # self.out_proj = RowParallelLinear(
        #     input_size=head_size * num_heads,
        #     output_size=hidden_size,
        #     bias=bias,
        #     input_is_parallel=True
        # )



        # #QKVParallelLinear for self-attention:
        # if cross_attention_dim is None:  # Self-attention
        #     self.qkv_proj = QKVParallelLinear(
        #         hidden_size=query_dim,
        #         head_size=dim_head,
        #         total_num_heads=heads,
        #         bias=bias
        #     )
        # else:  # Cross-attention - keep separate for different input dims
        #     self.to_q = ColumnParallelLinear(query_dim, self.inner_dim, bias=bias, gather_output=False)
        #     self.to_k = ColumnParallelLinear(cross_attention_dim, self.inner_dim, bias=bias, gather_output=False)
        #     self.to_v = ColumnParallelLinear(cross_attention_dim, self.inner_dim, bias=bias, gather_output=False)
        

        # Replace output projection:
        # self.to_out[0] = nn.Linear(self.inner_dim, query_dim, bias=out_bias)
        # self.to_out = nn.ModuleList([
        #     RowParallelLinear(self.inner_dim, query_dim, bias=out_bias, input_is_parallel=True),
        #     nn.Dropout(0.0)
        # ])
        

        # self.to_out = nn.ModuleList([
        #     nn.Linear(self.inner_dim, query_dim, bias=out_bias),
        #     nn.Dropout(0.0)
        # ])
    
        
        # Norms
        if qk_norm == "rms_norm_across_heads":
            norm_eps = 1e-5 
            norm_elementwise_affine = True
            self.norm_q = torch.nn.RMSNorm(self.inner_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
            self.norm_k = torch.nn.RMSNorm(self.inner_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

        else:
            self.norm_q = None
            self.norm_k = None
        

        self.attention = LocalAttention(
            num_heads=heads,
            head_size=dim_head,
            num_kv_heads=kv_heads or heads,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN,
                                          AttentionBackendEnum.TORCH_SDPA),
        )

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        batch_size, seq_len, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        if hasattr(self, 'norm_q'):
            query = self.norm_q(query)
            key = self.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Use your original working approach temporarily
        query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)  
        value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

        hidden_states, _ = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)  # Dropout
        return hidden_states

class GELU(nn.Module):
    def __init__(self, dim_in, dim_out, approximate="tanh", bias=True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def forward(self, x):
        x = self.proj(x)
        if self.approximate == "tanh":
            x = F.gelu(x, approximate="tanh")
        else:
            x = F.gelu(x)
        return x


class FastVideoFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        bias: bool = True,
    ):
        super().__init__()
        inner_dim = int(dim * mult)  # 8192
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu-approximate":
            self.net = nn.ModuleList([
                GELU(dim, inner_dim, approximate="tanh", bias=bias),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out, bias=bias)
            ])

        # TODO: fix below
        # if activation_fn == "gelu-approximate":
        #     # Replace the Linear layers with parallel versions:
        #     self.net = nn.ModuleList([
        #         # First layer: Column parallel (split output features)
        #         ColumnParallelLinear(dim, inner_dim, bias=bias, gather_output=False),
        #         GELU(dim, inner_dim, approximate="tanh", bias=bias),
        #         nn.Dropout(dropout),
        #         # Second layer: Row parallel (split input features, gather output)
        #         RowParallelLinear(inner_dim, dim_out, bias=bias, input_is_parallel=True)
        #     ])
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

@maybe_allow_in_graph
class FastVideoLTXTransformerBlock(nn.Module):
    r"""
    FastVideo Transformer block for LTX model.
    
    TODO: describe the parts I changed for FastVideo
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ):
        super().__init__()

        # Use FastVideo RMSNorm
        self.norm1 = RMSNorm(dim, eps=eps, has_weight=elementwise_affine)
        
        # Self-attention using the wrapper
        self.attn1 = LTXVideoLocalAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            kv_heads=num_attention_heads,  # LTX doesn't use GQA
            bias=attention_bias,
            cross_attention_dim=None,  # Self-attention
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            eps=1e-5,
        )

        self.norm2 = RMSNorm(dim, eps=eps, has_weight=elementwise_affine)

        
        # Cross-attention using wrapper
        self.attn2 = LTXVideoLocalAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            kv_heads=num_attention_heads,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            eps=1e-5,
        )

        self.ff = FastVideoFeedForward(dim, activation_fn=activation_fn)


        # Scale-shift table for adaptive layer norm
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        
        # First self-attention block with scale/shift modulation
        norm_hidden_states = self.norm1(hidden_states)

        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None] + temb.reshape(batch_size, temb.size(1), num_ada_params, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        # Self-attention - the wrapper handles all projections and normalization
        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,  # None for self-attention
            image_rotary_emb=image_rotary_emb,
        )
        
        # Gated residual connection
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        # Cross-attention block
        attn_hidden_states = self.attn2(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            image_rotary_emb=None,  # No rotary embeddings for cross-attention
        )
        hidden_states = hidden_states + attn_hidden_states
        
        # Feed-forward block with scale/shift modulation
        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_hidden_states)
        
        # Gated residual connection
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states

@maybe_allow_in_graph
class LTXVideoTransformer3DModel(CachableDiT):
    r"""
    FastVideo optimized Transformer model for video-like data used in LTX.
    
    Key optimizations:
    - RMSNorm for normalization layers
    - QuickGELU activation functions
    - Fused scale/shift operations where possible
    - Prepared for distributed attention and dimension parallelism
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["norm"]
    _repeated_blocks = ["FastVideoLTXTransformerBlock"]

    _fsdp_shard_conditions = LTXVideoConfig()._fsdp_shard_conditions
    param_names_mapping = LTXVideoConfig().param_names_mapping


    def __init__(
        self,
        config: LTXVideoConfig,
        hf_config: Optional[Dict] = None,
        **kwargs
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)
        
        # Handle both config object and kwargs
        if config is not None:
            # Extract parameters from config
            if hasattr(config, 'arch_config'):
                # FastVideo style config
                arch_config = config.arch_config
                in_channels = getattr(arch_config, 'in_channels', 128)
                out_channels = getattr(arch_config, 'out_channels', 128)
                patch_size = getattr(arch_config, 'patch_size', 1)
                patch_size_t = getattr(arch_config, 'patch_size_t', 1)
                num_attention_heads = getattr(arch_config, 'num_attention_heads', 32)
                attention_head_dim = getattr(arch_config, 'attention_head_dim', 64)
                cross_attention_dim = getattr(arch_config, 'cross_attention_dim', 2048)
                num_layers = getattr(arch_config, 'num_layers', 28)
                activation_fn = getattr(arch_config, 'activation_fn', 'gelu-approximate')
                qk_norm = getattr(arch_config, 'qk_norm', 'rms_norm_across_heads')
                norm_elementwise_affine = getattr(arch_config, 'norm_elementwise_affine', False)
                norm_eps = getattr(arch_config, 'norm_eps', 1e-6)
                caption_channels = getattr(arch_config, 'caption_channels', 4096)
                attention_bias = getattr(arch_config, 'attention_bias', True)
                attention_out_bias = getattr(arch_config, 'attention_out_bias', True)
            else:
                # Try to get from hf_config if provided
                if hf_config:
                    in_channels = hf_config.get('in_channels', 128)
                    out_channels = hf_config.get('out_channels', 128)
                    patch_size = hf_config.get('patch_size', 1)
                    patch_size_t = hf_config.get('patch_size_t', 1)
                    num_attention_heads = hf_config.get('num_attention_heads', 32)
                    attention_head_dim = hf_config.get('attention_head_dim', 64)
                    cross_attention_dim = hf_config.get('cross_attention_dim', 2048)
                    num_layers = hf_config.get('num_layers', 28)
                    activation_fn = hf_config.get('activation_fn', 'gelu-approximate')
                    qk_norm = hf_config.get('qk_norm', 'rms_norm_across_heads')
                    norm_elementwise_affine = hf_config.get('norm_elementwise_affine', False)
                    norm_eps = hf_config.get('norm_eps', 1e-6)
                    caption_channels = hf_config.get('caption_channels', 4096)
                    attention_bias = hf_config.get('attention_bias', True)
                    attention_out_bias = hf_config.get('attention_out_bias', True)
                else:
                    # Default values
                    raise ValueError("Either config or hf_config must be provided")
        else:
            # Use kwargs
            in_channels = kwargs.get('in_channels', 128)
            out_channels = kwargs.get('out_channels', 128)
            # anythig else from kwargs??
        
        out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim

        # TODO: Add input dimension parallelism for distributed training
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        
        # Use AdaLayerNormSingle for time embedding (keep original for compatibility)
        self.time_embed = AdaLayerNormSingle(inner_dim, use_additional_conditions=False)


        # Caption projection
        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)

        # FastVideo optimized rotary position embedding
        self.rope = FastVideoLTXRotaryPosEmbed(
            dim=inner_dim,
            base_num_frames=20,
            base_height=2048,
            base_width=2048,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            theta=10000.0,
        )


        # FastVideo optimized transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                FastVideoLTXTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                )
                for _ in range(num_layers)
            ]
        )

        # Using FastVideo FP32LayerNorm for output normalization
        self.norm_out = FP32LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)

        # # TODO: Add output dimension parallelism for distributed training
        # example:
        # # Output dimension parallelism
        # from fastvideo.layers.linear import ColumnParallelLinear
        # self.q_proj = ColumnParallelLinear(
        #     input_size=hidden_size,
        #     output_size=head_size * num_heads,
        #     bias=bias,
        #     gather_output=False
        # )
        self.proj_out = nn.Linear(inner_dim, out_channels)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        rope_interpolation_scale: Optional[Union[Tuple[float, float, float], torch.Tensor]] = None,
        video_coords: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        image_rotary_emb = self.rope(hidden_states, num_frames, height, width, rope_interpolation_scale, video_coords)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        batch_size = hidden_states.size(0)
        
        # TODO: Add input dimension parallelism here
        # example
        # # Input dimension parallelism
        # from fastvideo.layers.linear import RowParallelLinear
        # self.out_proj = RowParallelLinear(
        #     input_size=head_size * num_heads,
        #     output_size=hidden_size,
        #     bias=bias,
        #     input_is_parallel=True
        # )
        hidden_states = self.proj_in(hidden_states)

        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    encoder_attention_mask,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                )

        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        # FP32 normalization for better numerical stability
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        
        # TODO: Add output dimension parallelism here
        # # Output dimension parallelism
        # from fastvideo.layers.linear import ColumnParallelLinear
        # self.q_proj = ColumnParallelLinear(
        #     input_size=hidden_size,
        #     output_size=head_size * num_heads,
        #     bias=bias,
        #     gather_output=False
        # )

        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


def apply_rotary_emb(x, freqs):
    """Apply rotary embeddings to input tensors."""
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)  # [B, S, C // 2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out
# when i use newgelu, passes and same output difference as this version, just the intermediate ones difer a lot but its not relative big diff 
# INFO 08-27 17:43:12 [test_ltxvideo.py:331] Max Diff: 0.03125
# INFO 08-27 17:43:12 [test_ltxvideo.py:332] Mean Diff: 0.0026702880859375

# #copy pasted from stepvideo, move it to a shared file and use this instead?
# class AdaLayerNormSingle(nn.Module):
#     r"""
#         Norm layer adaptive layer norm single (adaLN-single).

#         As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

#         Parameters:
#             embedding_dim (`int`): The size of each embedding vector.
#             use_additional_conditions (`bool`): To use additional conditions for normalization or not.
#     """

#     def __init__(self, embedding_dim: int, time_step_rescale=1000):
#         super().__init__()

#         self.emb = TimestepEmbedder(embedding_dim)

#         self.silu = nn.SiLU()
#         self.linear = ReplicatedLinear(embedding_dim,
#                                        6 * embedding_dim,
#                                        bias=True)

#         self.time_step_rescale = time_step_rescale  ## timestep usually in [0, 1], we rescale it to [0,1000] for stability

#     def forward(
#         self,
#         timestep: torch.Tensor,
#         added_cond_kwargs: dict[str, torch.Tensor] | None = None,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         embedded_timestep = self.emb(timestep * self.time_step_rescale)

#         out, _ = self.linear(self.silu(embedded_timestep))

#         return out, embedded_timestep
