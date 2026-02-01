# # SPDX-License-Identifier: Apache-2.0
# from typing import Any

# import torch
# import torch.nn as nn
# from einops import rearrange

# from fastvideo.attention import LocalAttention
# from fastvideo.configs.models.dits import FluxConfig
# from fastvideo.layers.activation import get_act_fn
# from fastvideo.layers.layernorm import FP32LayerNorm, RMSNorm
# from fastvideo.layers.linear import ReplicatedLinear
# from fastvideo.layers.mlp import MLP
# from fastvideo.layers.rotary_embedding import get_1d_rotary_pos_embed
# from fastvideo.models.dits.base import BaseDiT
# from fastvideo.platforms import AttentionBackendEnum


# def timestep_embedding(t: torch.Tensor,
#                        dim: int,
#                        max_period: int = 10000,
#                        time_factor: float = 1000.0) -> torch.Tensor:
#     t = time_factor * t
#     half = dim // 2
#     freqs = torch.exp(-torch.log(torch.tensor(max_period, dtype=torch.float32)) *
#                       torch.arange(start=0, end=half, dtype=torch.float32) /
#                       half).to(t.device)
#     args = t[:, None].float() * freqs[None]
#     embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#     if dim % 2:
#         embedding = torch.cat(
#             [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#     if torch.is_floating_point(t):
#         embedding = embedding.to(t)
#     return embedding


# class MLPEmbedder(nn.Module):

#     def __init__(self, in_dim: int, hidden_dim: int, dtype: torch.dtype | None):
#         super().__init__()
#         self.in_layer = ReplicatedLinear(in_dim,
#                                          hidden_dim,
#                                          bias=True,
#                                          params_dtype=dtype)
#         self.act = get_act_fn("silu")
#         self.out_layer = ReplicatedLinear(hidden_dim,
#                                           hidden_dim,
#                                           bias=True,
#                                           params_dtype=dtype)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x, _ = self.in_layer(x)
#         x = self.act(x)
#         x, _ = self.out_layer(x)
#         return x


# class QKNorm(nn.Module):

#     def __init__(self, dim: int, dtype: torch.dtype | None):
#         super().__init__()
#         self.query_norm = RMSNorm(dim, eps=1e-6, dtype=dtype)
#         self.key_norm = RMSNorm(dim, eps=1e-6, dtype=dtype)

#     def forward(self, q: torch.Tensor, k: torch.Tensor,
#                 v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         q = self.query_norm(q)
#         k = self.key_norm(k)
#         return q.to(v.dtype), k.to(v.dtype)


# class SelfAttention(nn.Module):

#     def __init__(self,
#                  dim: int,
#                  num_heads: int,
#                  qkv_bias: bool,
#                  dtype: torch.dtype | None,
#                  supported_attention_backends: tuple[AttentionBackendEnum,
#                                                      ...]):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads

#         self.qkv = ReplicatedLinear(dim,
#                                     dim * 3,
#                                     bias=qkv_bias,
#                                     params_dtype=dtype)
#         self.norm = QKNorm(head_dim, dtype=dtype)
#         self.proj = ReplicatedLinear(dim,
#                                      dim,
#                                      bias=True,
#                                      params_dtype=dtype)

#         self.attn = LocalAttention(
#             num_heads=num_heads,
#             head_size=head_dim,
#             supported_attention_backends=supported_attention_backends,
#         )

#     def forward(self, x: torch.Tensor,
#                 freqs_cis: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
#         qkv, _ = self.qkv(x)
#         q, k, v = rearrange(qkv,
#                             "b l (k h d) -> k b l h d",
#                             k=3,
#                             h=self.num_heads)
#         q, k = self.norm(q, k, v)
#         attn = self.attn(q, k, v, freqs_cis=freqs_cis)
#         attn = attn.reshape(x.shape[0], x.shape[1], -1)
#         out, _ = self.proj(attn)
#         return out


# class Modulation(nn.Module):

#     def __init__(self, dim: int, double: bool, dtype: torch.dtype | None):
#         super().__init__()
#         self.is_double = double
#         self.multiplier = 6 if double else 3
#         self.lin = ReplicatedLinear(dim,
#                                     self.multiplier * dim,
#                                     bias=True,
#                                     params_dtype=dtype)

#     def forward(self, vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
#                                                   torch.Tensor, torch.Tensor,
#                                                   torch.Tensor, torch.Tensor]:
#         out, _ = self.lin(torch.nn.functional.silu(vec))
#         chunks = out[:, None, :].chunk(self.multiplier, dim=-1)
#         return chunks  # shift/scale/gate tuples


# class DoubleStreamBlock(nn.Module):

#     def __init__(self,
#                  hidden_size: int,
#                  num_heads: int,
#                  mlp_ratio: float,
#                  qkv_bias: bool,
#                  dtype: torch.dtype | None,
#                  supported_attention_backends: tuple[AttentionBackendEnum,
#                                                      ...],
#                  prefix: str = ""):
#         super().__init__()
#         mlp_hidden_dim = int(hidden_size * mlp_ratio)
#         self.num_heads = num_heads
#         self.hidden_size = hidden_size

#         self.img_mod = Modulation(hidden_size, double=True, dtype=dtype)
#         self.img_norm1 = FP32LayerNorm(hidden_size,
#                                        elementwise_affine=False,
#                                        eps=1e-6)
#         self.img_attn = SelfAttention(hidden_size,
#                                       num_heads,
#                                       qkv_bias=qkv_bias,
#                                       dtype=dtype,
#                                       supported_attention_backends=
#                                       supported_attention_backends)
#         self.img_norm2 = FP32LayerNorm(hidden_size,
#                                        elementwise_affine=False,
#                                        eps=1e-6)
#         self.img_mlp = MLP(hidden_size,
#                            mlp_hidden_dim,
#                            bias=True,
#                            act_type="gelu_tanh",
#                            dtype=dtype,
#                            prefix=f"{prefix}.img_mlp")

#         self.txt_mod = Modulation(hidden_size, double=True, dtype=dtype)
#         self.txt_norm1 = FP32LayerNorm(hidden_size,
#                                        elementwise_affine=False,
#                                        eps=1e-6)
#         self.txt_attn = SelfAttention(hidden_size,
#                                       num_heads,
#                                       qkv_bias=qkv_bias,
#                                       dtype=dtype,
#                                       supported_attention_backends=
#                                       supported_attention_backends)
#         self.txt_norm2 = FP32LayerNorm(hidden_size,
#                                        elementwise_affine=False,
#                                        eps=1e-6)
#         self.txt_mlp = MLP(hidden_size,
#                            mlp_hidden_dim,
#                            bias=True,
#                            act_type="gelu_tanh",
#                            dtype=dtype,
#                            prefix=f"{prefix}.txt_mlp")

#     def forward(
#         self,
#         img: torch.Tensor,
#         txt: torch.Tensor,
#         vec: torch.Tensor,
#         freqs_cis: tuple[torch.Tensor, torch.Tensor],
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2 = self.img_mod(
#             vec)
#         txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = self.txt_mod(
#             vec)

#         img_mod = self.img_norm1(img)
#         img_mod = (1 + img_scale1) * img_mod + img_shift1
#         txt_mod = self.txt_norm1(txt)
#         txt_mod = (1 + txt_scale1) * txt_mod + txt_shift1

#         qkv_img, _ = self.img_attn.qkv(img_mod)
#         img_q, img_k, img_v = rearrange(qkv_img,
#                                         "b l (k h d) -> k b l h d",
#                                         k=3,
#                                         h=self.num_heads)
#         img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

#         qkv_txt, _ = self.txt_attn.qkv(txt_mod)
#         txt_q, txt_k, txt_v = rearrange(qkv_txt,
#                                         "b l (k h d) -> k b l h d",
#                                         k=3,
#                                         h=self.num_heads)
#         txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

#         q = torch.cat((txt_q, img_q), dim=1)
#         k = torch.cat((txt_k, img_k), dim=1)
#         v = torch.cat((txt_v, img_v), dim=1)

#         attn = self.img_attn.attn(q, k, v, freqs_cis=freqs_cis)
#         txt_attn = attn[:, :txt.shape[1]]
#         img_attn = attn[:, txt.shape[1]:]

#         img = img + img_gate1 * self.img_attn.proj(img_attn.reshape(
#             img.shape[0], img.shape[1], -1))[0]
#         img = img + img_gate2 * self.img_mlp((1 + img_scale2) *
#                                              self.img_norm2(img) +
#                                              img_shift2)

#         txt = txt + txt_gate1 * self.txt_attn.proj(txt_attn.reshape(
#             txt.shape[0], txt.shape[1], -1))[0]
#         txt = txt + txt_gate2 * self.txt_mlp((1 + txt_scale2) *
#                                              self.txt_norm2(txt) +
#                                              txt_shift2)
#         return img, txt


# class SingleStreamBlock(nn.Module):

#     def __init__(self,
#                  hidden_size: int,
#                  num_heads: int,
#                  mlp_ratio: float,
#                  dtype: torch.dtype | None,
#                  supported_attention_backends: tuple[AttentionBackendEnum,
#                                                      ...],
#                  prefix: str = ""):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

#         self.linear1 = ReplicatedLinear(hidden_size,
#                                         hidden_size * 3 + self.mlp_hidden_dim,
#                                         bias=True,
#                                         params_dtype=dtype)
#         self.linear2 = ReplicatedLinear(hidden_size + self.mlp_hidden_dim,
#                                         hidden_size,
#                                         bias=True,
#                                         params_dtype=dtype)
#         self.norm = QKNorm(hidden_size // num_heads, dtype=dtype)
#         self.pre_norm = FP32LayerNorm(hidden_size,
#                                       elementwise_affine=False,
#                                       eps=1e-6)
#         self.mlp_act = get_act_fn("gelu_tanh")
#         self.modulation = Modulation(hidden_size, double=False, dtype=dtype)

#         self.attn = LocalAttention(
#             num_heads=num_heads,
#             head_size=hidden_size // num_heads,
#             supported_attention_backends=supported_attention_backends,
#         )

#     def forward(self, x: torch.Tensor, vec: torch.Tensor,
#                 freqs_cis: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
#         mod_shift, mod_scale, mod_gate = self.modulation(vec)[:3]
#         x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift
#         linear1_out, _ = self.linear1(x_mod)
#         qkv, mlp = torch.split(
#             linear1_out, [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
#         q, k, v = rearrange(qkv,
#                             "b l (k h d) -> k b l h d",
#                             k=3,
#                             h=self.num_heads)
#         q, k = self.norm(q, k, v)
#         attn = self.attn(q, k, v, freqs_cis=freqs_cis)
#         attn = attn.reshape(x.shape[0], x.shape[1], -1)
#         out, _ = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=-1))
#         return x + mod_gate * out


# class LastLayer(nn.Module):

#     def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
#         super().__init__()
#         self.norm_final = nn.LayerNorm(hidden_size,
#                                        elementwise_affine=False,
#                                        eps=1e-6)
#         self.linear = nn.Linear(hidden_size,
#                                patch_size * patch_size * out_channels,
#                                bias=True)
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 2 * hidden_size, bias=True),
#         )

#     def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
#         shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
#         x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
#         x = self.linear(x)
#         return x


# def _build_freqs_from_ids(ids: torch.Tensor, axes_dim: list[int],
#                           theta: float) -> tuple[torch.Tensor, torch.Tensor]:
#     ids_0 = ids[0]  # [S, A]
#     cos_list = []
#     sin_list = []
#     for i, dim in enumerate(axes_dim):
#         cos, sin = get_1d_rotary_pos_embed(dim, ids_0[:, i], theta=theta)
#         cos_list.append(cos)
#         sin_list.append(sin)
#     cos = torch.cat(cos_list, dim=1)
#     sin = torch.cat(sin_list, dim=1)
#     return cos, sin


# def _build_ids_from_grid(height: int, width: int, n_axes: int,
#                          device: torch.device) -> torch.Tensor:
#     grid_y = torch.arange(height, device=device)
#     grid_x = torch.arange(width, device=device)
#     yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
#     yy = yy.reshape(-1)
#     xx = xx.reshape(-1)
#     if n_axes == 1:
#         ids = torch.arange(height * width, device=device)[:, None]
#     elif n_axes >= 2:
#         extra = []
#         if n_axes > 2:
#             extra = [torch.zeros_like(yy) for _ in range(n_axes - 2)]
#         ids = torch.stack([yy, xx, *extra], dim=-1)
#     return ids


# class FluxTransformer2DModel(BaseDiT):
#     _fsdp_shard_conditions = FluxConfig()._fsdp_shard_conditions
#     _compile_conditions = FluxConfig()._compile_conditions
#     _supported_attention_backends = FluxConfig()._supported_attention_backends
#     param_names_mapping = FluxConfig().param_names_mapping
#     reverse_param_names_mapping = FluxConfig().reverse_param_names_mapping
#     lora_param_names_mapping = FluxConfig().lora_param_names_mapping

#     def __init__(self, config: FluxConfig, hf_config: dict[str, Any]):
#         super().__init__(config=config, hf_config=hf_config)
#         dtype = getattr(config, "dtype", None)

#         self.in_channels = config.in_channels
#         self.out_channels = config.out_channels
#         self.hidden_size = config.hidden_size
#         self.num_attention_heads = config.num_attention_heads
#         self.num_channels_latents = config.num_channels_latents

#         self.vec_in_dim = config.pooled_projection_dim
#         self.context_in_dim = config.joint_attention_dim
#         self.axes_dim = list(config.rope_axes_dim)
#         self.theta = config.rope_theta
#         self.guidance_embed = config.guidance_embeds
#         self.mlp_ratio = config.mlp_ratio
#         self.qkv_bias = getattr(config, "qkv_bias", False)

#         self.img_in = ReplicatedLinear(self.in_channels,
#                                        self.hidden_size,
#                                        bias=True,
#                                        params_dtype=dtype)
#         self.time_in = MLPEmbedder(in_dim=256,
#                                    hidden_dim=self.hidden_size,
#                                    dtype=dtype)
#         self.vector_in = MLPEmbedder(in_dim=self.vec_in_dim,
#                                      hidden_dim=self.hidden_size,
#                                      dtype=dtype)
#         self.guidance_in = (MLPEmbedder(
#             in_dim=256, hidden_dim=self.hidden_size, dtype=dtype)
#                             if self.guidance_embed else nn.Identity())
#         self.txt_in = ReplicatedLinear(self.context_in_dim,
#                                        self.hidden_size,
#                                        bias=True,
#                                        params_dtype=dtype)

#         self.double_blocks = nn.ModuleList([
#             DoubleStreamBlock(
#                 self.hidden_size,
#                 self.num_attention_heads,
#                 mlp_ratio=self.mlp_ratio,
#                 qkv_bias=self.qkv_bias,
#                 dtype=dtype,
#                 supported_attention_backends=self._supported_attention_backends,
#                 prefix=f"{config.prefix}.double_blocks.{i}",
#             ) for i in range(config.num_layers)
#         ])

#         self.single_blocks = nn.ModuleList([
#             SingleStreamBlock(
#                 self.hidden_size,
#                 self.num_attention_heads,
#                 mlp_ratio=self.mlp_ratio,
#                 dtype=dtype,
#                 supported_attention_backends=self._supported_attention_backends,
#                 prefix=f"{config.prefix}.single_blocks.{i}",
#             ) for i in range(config.num_single_layers)
#         ])

#         self.final_layer = LastLayer(self.hidden_size,
#                                      patch_size=1,
#                                      out_channels=self.out_channels)

#         self.__post_init__()

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         encoder_hidden_states: torch.Tensor | list[torch.Tensor],
#         timestep: torch.LongTensor,
#         encoder_hidden_states_2: torch.Tensor | None = None,
#         img_ids: torch.Tensor | None = None,
#         txt_ids: torch.Tensor | None = None,
#         guidance: torch.Tensor | None = None,
#         **kwargs,
#     ) -> torch.Tensor:
#         if hidden_states.ndim != 5:
#             raise ValueError(
#                 "FluxTransformer2DModel expects hidden_states with shape [B, C, T, H, W]"
#             )

#         img = rearrange(hidden_states, "b c t h w -> b (t h w) c")
#         txt = encoder_hidden_states
#         if isinstance(txt, list):
#             txt = txt[0]

#         y = encoder_hidden_states_2
#         if y is None:
#             y = torch.zeros(txt.shape[0],
#                             self.vec_in_dim,
#                             device=txt.device,
#                             dtype=txt.dtype)

#         img, _ = self.img_in(img)
#         vec = self.time_in(timestep_embedding(timestep, 256))
#         if self.guidance_embed:
#             if guidance is None:
#                 raise ValueError(
#                     "Guidance value is required for guidance-distilled Flux.")
#             vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
#         vec = vec + self.vector_in(y)
#         txt, _ = self.txt_in(txt)

#         bsz, txt_len, _ = txt.shape
#         _, img_len, _ = img.shape
#         if txt_ids is None:
#             txt_ids = torch.zeros(bsz,
#                                   txt_len,
#                                   len(self.axes_dim),
#                                   device=txt.device)
#         if img_ids is None:
#             _, _, _, h, w = hidden_states.shape
#             ids = _build_ids_from_grid(h, w, len(self.axes_dim), txt.device)
#             img_ids = ids.unsqueeze(0).expand(bsz, -1, -1)

#         ids = torch.cat((txt_ids, img_ids), dim=1)
#         freqs_cis = _build_freqs_from_ids(ids, self.axes_dim, self.theta)

#         for block in self.double_blocks:
#             img, txt = block(img=img, txt=txt, vec=vec, freqs_cis=freqs_cis)

#         img = torch.cat((txt, img), 1)
#         for block in self.single_blocks:
#             img = block(img, vec=vec, freqs_cis=freqs_cis)
#         img = img[:, txt.shape[1]:, ...]

#         img = self.final_layer(img, vec)
#         img = rearrange(img, "b (t h w) c -> b c t h w", t=1, h=h, w=w)
#         return img

# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.attention import AttentionModuleMixin, FeedForward
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)
from torch.nn import LayerNorm as LayerNorm

from fastvideo.attention import LocalAttention
from fastvideo.configs.models.dits.flux import FluxConfig
from fastvideo.layers.layernorm import RMSNorm
from fastvideo.layers.linear import ColumnParallelLinear
from fastvideo.layers.rotary_embedding import get_1d_rotary_pos_embed, _apply_rotary_emb
from fastvideo.logger import init_logger
from fastvideo.models.dits.base import CachableDiT

logger = init_logger(__name__)  # pylint: disable=invalid-name


def _apply_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: nn.Module,
    k_norm: nn.Module,
    value_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = q_norm(q).to(value_dtype)
    k = k_norm(k).to(value_dtype)
    return q, k


def _apply_rotary_emb_2d(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return _apply_rotary_emb(x, cos, sin, is_neox_style=False)


def _get_qkv_projections(
    attn: "FluxAttention", hidden_states, encoder_hidden_states=None
):
    query, _ = attn.to_q(hidden_states)
    key, _ = attn.to_k(hidden_states)
    value, _ = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query, _ = attn.add_q_proj(encoder_hidden_states)
        encoder_key, _ = attn.add_k_proj(encoder_hidden_states)
        encoder_value, _ = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


class FluxAttention(torch.nn.Module, AttentionModuleMixin):
    def __init__(
        self,
        query_dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        context_pre_only: Optional[bool] = None,
        pre_only: bool = False,
        supported_attention_backends=None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * num_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else num_heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        self.to_q = ColumnParallelLinear(
            query_dim, self.inner_dim, bias=bias, gather_output=True
        )
        self.to_k = ColumnParallelLinear(
            query_dim, self.inner_dim, bias=bias, gather_output=True
        )
        self.to_v = ColumnParallelLinear(
            query_dim, self.inner_dim, bias=bias, gather_output=True
        )

        if not self.pre_only:
            self.to_out = torch.nn.ModuleList([])
            self.to_out.append(
                ColumnParallelLinear(
                    self.inner_dim, self.out_dim, bias=out_bias, gather_output=True
                )
            )
            if dropout != 0.0:
                self.to_out.append(torch.nn.Dropout(dropout))

        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)
            self.add_q_proj = ColumnParallelLinear(
                added_kv_proj_dim,
                self.inner_dim,
                bias=added_proj_bias,
                gather_output=True,
            )
            self.add_k_proj = ColumnParallelLinear(
                added_kv_proj_dim,
                self.inner_dim,
                bias=added_proj_bias,
                gather_output=True,
            )
            self.add_v_proj = ColumnParallelLinear(
                added_kv_proj_dim,
                self.inner_dim,
                bias=added_proj_bias,
                gather_output=True,
            )
            self.to_add_out = ColumnParallelLinear(
                self.inner_dim, query_dim, bias=out_bias, gather_output=True
            )

        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        freqs_cis=None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        query, key, value, encoder_query, encoder_key, encoder_value = (
            _get_qkv_projections(self, x, encoder_hidden_states)
        )

        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))
        query, key = _apply_qk_norm(
            q=query,
            k=key,
            q_norm=self.norm_q,
            k_norm=self.norm_k,
            value_dtype=value.dtype,
        )

        if self.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (self.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (self.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (self.heads, -1))

            encoder_query, encoder_key = _apply_qk_norm(
                q=encoder_query,
                k=encoder_key,
                q_norm=self.norm_added_q,
                k_norm=self.norm_added_k,
                value_dtype=encoder_value.dtype,
            )

            bsz, seq_len, _, _ = query.shape
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if freqs_cis is not None:
            cos, sin = freqs_cis
            # cos/sin are expected to be shaped [seq_len, head_dim/2]
            query = _apply_rotary_emb_2d(query, cos, sin)
            key = _apply_rotary_emb_2d(key, cos, sin)

        x = self.attn(query, key, value)
        x = x.flatten(2, 3)
        x = x.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, x = x.split_with_sizes(
                [
                    encoder_hidden_states.shape[1],
                    x.shape[1] - encoder_hidden_states.shape[1],
                ],
                dim=1,
            )
            x, _ = self.to_out[0](x)
            if len(self.to_out) == 2:
                x = self.to_out[1](x)
            encoder_hidden_states, _ = self.to_add_out(encoder_hidden_states)

            return x, encoder_hidden_states
        else:
            return x


class FluxSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        supported_attention_backends=None,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = ColumnParallelLinear(
            dim, self.mlp_hidden_dim, bias=True, gather_output=True
        )
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = ColumnParallelLinear(
            dim + self.mlp_hidden_dim, dim, bias=True, gather_output=True
        )

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            num_heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        proj_hidden_states, _ = self.proj_mlp(norm_hidden_states)
        mlp_hidden_states = self.act_mlp(proj_hidden_states)
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            x=norm_hidden_states,
            freqs_cis=freqs_cis,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        proj_out, _ = self.proj_out(hidden_states)
        hidden_states = gate * proj_out
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        return encoder_hidden_states, hidden_states


class FluxTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        supported_attention_backends=None,
    ):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            num_heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            eps=eps,
            supported_attention_backends=supported_attention_backends,
        )

        self.norm2 = LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ff = MLP(
            input_dim=dim, mlp_hidden_dim=dim * 4, output_dim=dim, act_type="gelu"
        )
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ff_context = MLP(
            input_dim=dim, mlp_hidden_dim=dim * 4, output_dim=dim, act_type="gelu"
        )

        self.ff_context = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )

        joint_attention_kwargs = joint_attention_kwargs or {}
        # Attention.
        attention_outputs = self.attn(
            x=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            freqs_cis=freqs_cis,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output
        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
            + c_shift_mlp[:, None]
        )

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        )
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxPosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = ids.float()
        cos_out = []
        sin_out = []
        for i, dim in enumerate(self.axes_dim):
            cos, sin = get_1d_rotary_pos_embed(
                dim,
                pos[:, i],
                theta=self.theta,
                dtype=torch.float32,
                device=pos.device,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class FluxTransformer2DModel(CachableDiT):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/
    """

    param_names_mapping = FluxConfig().arch_config.param_names_mapping

    def __init__(self, config: FluxConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)
        self.config = config.arch_config

        self.out_channels = (
            getattr(self.config, "out_channels", None) or self.config.in_channels
        )
        self.inner_dim = (
            self.config.num_attention_heads * self.config.attention_head_dim
        )

        self.hidden_size = self.inner_dim
        self.num_attention_heads = self.config.num_attention_heads
        self.num_channels_latents = self.config.num_channels_latents

        self.rotary_emb = FluxPosEmbed(
            theta=self.config.rope_theta, axes_dim=self.config.rope_axes_dim
        )

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings
            if self.config.guidance_embeds
            else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=self.config.pooled_projection_dim,
        )

        self.context_embedder = ColumnParallelLinear(
            self.config.joint_attention_dim,
            self.inner_dim,
            bias=True,
            gather_output=True,
        )
        self.x_embedder = ColumnParallelLinear(
            self.config.in_channels, self.inner_dim, bias=True, gather_output=True
        )
        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    supported_attention_backends=self.config._supported_attention_backends,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    supported_attention_backends=self.config._supported_attention_backends,
                )
                for _ in range(self.config.num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = ColumnParallelLinear(
            self.inner_dim,
            self.config.patch_size * self.config.patch_size * self.out_channels,
            bias=True,
            gather_output=True,
        )

        self.layer_names = [
            "transformer_blocks",
            "single_transformer_blocks",
        ]

        self.__post_init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        guidance: torch.Tensor = None,
        freqs_cis: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            guidance (`torch.Tensor`):
                Guidance embeddings.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        """
        if (
            joint_attention_kwargs is not None
            and joint_attention_kwargs.get("scale", None) is not None
        ):
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )
        hidden_states, _ = self.x_embedder(hidden_states)

        # Only pass guidance to time_text_embed if the model supports it
        if self.config.guidance_embeds and guidance is not None:
            temb = self.time_text_embed(timestep, guidance, pooled_projections)
        else:
            temb = self.time_text_embed(timestep, pooled_projections)

        encoder_hidden_states, _ = self.context_embedder(encoder_hidden_states)

        if (
            joint_attention_kwargs is not None
            and "ip_adapter_image_embeds" in joint_attention_kwargs
        ):
            ip_adapter_image_embeds = joint_attention_kwargs.pop(
                "ip_adapter_image_embeds"
            )
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                freqs_cis=freqs_cis,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                freqs_cis=freqs_cis,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        hidden_states = self.norm_out(hidden_states, temb)

        output, _ = self.proj_out(hidden_states)

        return output


EntryClass = FluxTransformer2DModel
