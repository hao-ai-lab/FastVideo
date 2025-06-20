# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from fastvideo.v1.attention import DistributedAttention, LocalAttention
from fastvideo.v1.configs.models.dits.cosmos import CosmosConfig
from fastvideo.v1.forward_context import get_forward_context
from fastvideo.v1.layers.layernorm import RMSNorm
from fastvideo.v1.layers.linear import ReplicatedLinear
from fastvideo.v1.layers.mlp import MLP
from fastvideo.v1.layers.rotary_embedding import apply_rotary_emb
from fastvideo.v1.layers.visual_embedding import Timesteps
from fastvideo.v1.models.dits.base import BaseDiT
from fastvideo.v1.platforms import AttentionBackendEnum


class CosmosPatchEmbed(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 patch_size: Tuple[int, int, int],
                 bias: bool = True) -> None:
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Linear(in_channels * patch_size[0] * patch_size[1] *
                              patch_size[2],
                              out_channels,
                              bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        hidden_states = hidden_states.reshape(batch_size, num_channels,
                                              num_frames // p_t, p_t,
                                              height // p_h, p_h, width // p_w,
                                              p_w)
        hidden_states = hidden_states.permute(0, 2, 4, 6, 1, 3, 5,
                                              7).flatten(4, 7)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class CosmosTimestepEmbedding(nn.Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.linear_1(timesteps)
        emb = self.activation(emb)
        emb = self.linear_2(emb)
        return emb


class CosmosEmbedding(nn.Module):

    def __init__(self, embedding_dim: int, condition_dim: int) -> None:
        super().__init__()

        self.time_proj = Timesteps(embedding_dim,
                                   flip_sin_to_cos=True,
                                   downscale_freq_shift=0.0)
        self.t_embedder = CosmosTimestepEmbedding(embedding_dim, condition_dim)
        self.norm = RMSNorm(embedding_dim, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor,
                timestep: torch.LongTensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep).type_as(hidden_states)
        temb = self.t_embedder(timesteps_proj)
        embedded_timestep = self.norm(timesteps_proj)
        return temb, embedded_timestep


class CosmosAdaLayerNorm(nn.Module):

    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.embedding_dim = in_features

        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(in_features,
                                 elementwise_affine=False,
                                 eps=1e-6)
        self.linear_1 = nn.Linear(in_features, hidden_features, bias=False)
        self.linear_2 = nn.Linear(hidden_features, 2 * in_features, bias=False)

    def forward(self,
                hidden_states: torch.Tensor,
                embedded_timestep: torch.Tensor,
                temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        embedded_timestep = self.activation(embedded_timestep)
        embedded_timestep = self.linear_1(embedded_timestep)
        embedded_timestep = self.linear_2(embedded_timestep)

        if temb is not None:
            embedded_timestep = embedded_timestep + temb[..., :2 *
                                                         self.embedding_dim]

        shift, scale = embedded_timestep.chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)

        if embedded_timestep.ndim == 2:
            shift, scale = (x.unsqueeze(1) for x in (shift, scale))

        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states


class CosmosAdaLayerNormZero(nn.Module):

    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(in_features,
                                 elementwise_affine=False,
                                 eps=1e-6)
        self.activation = nn.SiLU()

        if hidden_features is None:
            self.linear_1 = nn.Identity()
        else:
            self.linear_1 = nn.Linear(in_features, hidden_features, bias=False)

        self.linear_2 = nn.Linear(
            hidden_features if hidden_features is not None else in_features,
            3 * in_features,
            bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embedded_timestep = self.activation(embedded_timestep)
        embedded_timestep = self.linear_1(embedded_timestep)
        embedded_timestep = self.linear_2(embedded_timestep)

        if temb is not None:
            embedded_timestep = embedded_timestep + temb

        shift, scale, gate = embedded_timestep.chunk(3, dim=-1)
        hidden_states = self.norm(hidden_states)

        if embedded_timestep.ndim == 2:
            shift, scale, gate = (x.unsqueeze(1) for x in (shift, scale, gate))

        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states, gate


class CosmosSelfAttention(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 qk_norm=True,
                 eps=1e-6,
                 supported_attention_backends: Optional[Tuple[
                     AttentionBackendEnum, ...]] = None,
                 prefix: str = "") -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.to_q = ReplicatedLinear(dim, dim, bias=False)
        self.to_k = ReplicatedLinear(dim, dim, bias=False)
        self.to_v = ReplicatedLinear(dim, dim, bias=False)
        self.to_out = ReplicatedLinear(dim, dim, bias=False)
        self.norm_q = RMSNorm(self.head_dim,
                              eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(self.head_dim,
                              eps=eps) if qk_norm else nn.Identity()

        # Attention mechanism
        self.attn = DistributedAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=prefix)

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None) -> torch.Tensor:

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Get QKV
        query, _ = self.to_q(hidden_states)
        key, _ = self.to_k(encoder_hidden_states)
        value, _ = self.to_v(encoder_hidden_states)

        # Reshape for multi-head attention
        query = query.unflatten(2, (self.num_heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (self.num_heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (self.num_heads, -1)).transpose(1, 2)

        # Apply normalization
        if self.norm_q is not None:
            query = self.norm_q.forward_native(query)
        if self.norm_k is not None:
            key = self.norm_k.forward_native(key)

        # Apply RoPE if provided
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query,
                                     image_rotary_emb,
                                     use_real=True,
                                     use_real_unbind_dim=-2)
            key = apply_rotary_emb(key,
                                   image_rotary_emb,
                                   use_real=True,
                                   use_real_unbind_dim=-2)

        # Attention computation
        attn_output, _ = self.attn(query, key, value)
        # attn_output = attn_output.flatten(2)
        attn_output = attn_output.transpose(1, 2).flatten(2, 3).type_as(query)

        # Output projection
        attn_output, _ = self.to_out(attn_output)
        return attn_output


class CosmosCrossAttention(nn.Module):

    def __init__(self,
                 dim: int,
                 cross_attention_dim: int,
                 num_heads: int,
                 qk_norm=True,
                 eps=1e-6,
                 supported_attention_backends: Optional[Tuple[
                     AttentionBackendEnum, ...]] = None,
                 prefix: str = "") -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.cross_attention_dim = cross_attention_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.to_q = ReplicatedLinear(dim, dim, bias=False)
        self.to_k = ReplicatedLinear(cross_attention_dim, dim, bias=False)
        self.to_v = ReplicatedLinear(cross_attention_dim, dim, bias=False)
        self.to_out = ReplicatedLinear(dim, dim, bias=False)
        self.norm_q = RMSNorm(self.head_dim,
                              eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(self.head_dim,
                              eps=eps) if qk_norm else nn.Identity()

        # Attention mechanism
        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=supported_attention_backends)

    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        # Get QKV
        query, _ = self.to_q(hidden_states)
        key, _ = self.to_k(encoder_hidden_states)
        value, _ = self.to_v(encoder_hidden_states)

        # Reshape for multi-head attention
        query = query.unflatten(2, (self.num_heads, -1))
        key = key.unflatten(2, (self.num_heads, -1))
        value = value.unflatten(2, (self.num_heads, -1))

        # Apply normalization
        if self.norm_q is not None:
            query = self.norm_q.forward_native(query)
        if self.norm_k is not None:
            key = self.norm_k.forward_native(key)

        # Attention computation
        attn_output = self.attn(query, key, value)
        attn_output = attn_output.flatten(2, 3).type_as(query)

        # Output projection
        attn_output, _ = self.to_out(attn_output)
        return attn_output


class CosmosTransformerBlock(nn.Module):

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        mlp_ratio: float = 4.0,
        adaln_lora_dim: int = 256,
        qk_norm: str = "rms_norm",
        out_bias: bool = False,
        supported_attention_backends: Optional[Tuple[AttentionBackendEnum,
                                                     ...]] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = CosmosAdaLayerNormZero(in_features=hidden_size,
                                            hidden_features=adaln_lora_dim)
        self.attn1 = CosmosSelfAttention(
            dim=hidden_size,
            num_heads=num_attention_heads,
            qk_norm=(qk_norm == "rms_norm"),
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn1")

        self.norm2 = CosmosAdaLayerNormZero(in_features=hidden_size,
                                            hidden_features=adaln_lora_dim)
        self.attn2 = CosmosCrossAttention(
            dim=hidden_size,
            cross_attention_dim=cross_attention_dim,
            num_heads=num_attention_heads,
            qk_norm=(qk_norm == "rms_norm"),
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn2")

        self.norm3 = CosmosAdaLayerNormZero(in_features=hidden_size,
                                            hidden_features=adaln_lora_dim)
        self.ff = MLP(hidden_size,
                      int(hidden_size * mlp_ratio),
                      act_type="gelu",
                      bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        extra_pos_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if extra_pos_emb is not None:
            hidden_states = hidden_states + extra_pos_emb

        # 1. Self Attention
        norm_hidden_states, gate = self.norm1(hidden_states, embedded_timestep,
                                              temb)
        attn_output = self.attn1(norm_hidden_states,
                                 image_rotary_emb=image_rotary_emb)
        hidden_states = hidden_states + gate * attn_output

        # 2. Cross Attention
        norm_hidden_states, gate = self.norm2(hidden_states, embedded_timestep,
                                              temb)
        attn_output = self.attn2(norm_hidden_states,
                                 encoder_hidden_states=encoder_hidden_states,
                                 attention_mask=attention_mask)
        hidden_states = hidden_states + gate * attn_output

        # 3. Feed Forward
        norm_hidden_states, gate = self.norm3(hidden_states, embedded_timestep,
                                              temb)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate * ff_output

        return hidden_states


class CosmosRotaryPosEmbed(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            max_size: Tuple[int, int, int] = (128, 240, 240),
            patch_size: Tuple[int, int, int] = (1, 2, 2),
            base_fps: int = 24,
            rope_scale: Tuple[float, float, float] = (2.0, 1.0, 1.0),
    ) -> None:
        super().__init__()

        self.max_size = [
            size // patch for size, patch in zip(max_size, patch_size)
        ]
        self.patch_size = patch_size
        self.base_fps = base_fps

        self.dim_h = hidden_size // 6 * 2
        self.dim_w = hidden_size // 6 * 2
        self.dim_t = hidden_size - self.dim_h - self.dim_w

        self.h_ntk_factor = rope_scale[1]**(self.dim_h / (self.dim_h - 2))
        self.w_ntk_factor = rope_scale[2]**(self.dim_w / (self.dim_w - 2))
        self.t_ntk_factor = rope_scale[0]**(self.dim_t / (self.dim_t - 2))

    def forward(self,
                hidden_states: torch.Tensor,
                fps: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        pe_size = [
            num_frames // self.patch_size[0], height // self.patch_size[1],
            width // self.patch_size[2]
        ]
        device = hidden_states.device

        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor

        seq = torch.arange(max(self.max_size),
                           device=device,
                           dtype=torch.float32)
        dim_h_range = (
            torch.arange(0, self.dim_h, 2, device=device,
                         dtype=torch.float32)[:(self.dim_h // 2)] / self.dim_h)
        dim_w_range = (
            torch.arange(0, self.dim_w, 2, device=device,
                         dtype=torch.float32)[:(self.dim_w // 2)] / self.dim_w)
        dim_t_range = (
            torch.arange(0, self.dim_t, 2, device=device,
                         dtype=torch.float32)[:(self.dim_t // 2)] / self.dim_t)
        h_spatial_freqs = 1.0 / (h_theta**dim_h_range)
        w_spatial_freqs = 1.0 / (w_theta**dim_w_range)
        temporal_freqs = 1.0 / (t_theta**dim_t_range)

        emb_h = torch.outer(seq[:pe_size[1]],
                            h_spatial_freqs)[None, :, None, :].repeat(
                                pe_size[0], 1, pe_size[2], 1)
        emb_w = torch.outer(seq[:pe_size[2]],
                            w_spatial_freqs)[None, None, :, :].repeat(
                                pe_size[0], pe_size[1], 1, 1)

        # Apply sequence scaling in temporal dimension
        if fps is None:
            # Images
            emb_t = torch.outer(seq[:pe_size[0]], temporal_freqs)
        else:
            # Videos
            emb_t = torch.outer(seq[:pe_size[0]] / fps * self.base_fps,
                                temporal_freqs)

        emb_t = emb_t[:, None, None, :].repeat(1, pe_size[1], pe_size[2], 1)
        freqs = torch.cat([emb_t, emb_h, emb_w] * 2, dim=-1).flatten(0,
                                                                     2).float()
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin


class CosmosLearnablePositionalEmbed(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        max_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.max_size = [
            size // patch for size, patch in zip(max_size, patch_size)
        ]
        self.patch_size = patch_size
        self.eps = eps

        self.pos_emb_t = nn.Parameter(torch.zeros(self.max_size[0],
                                                  hidden_size))
        self.pos_emb_h = nn.Parameter(torch.zeros(self.max_size[1],
                                                  hidden_size))
        self.pos_emb_w = nn.Parameter(torch.zeros(self.max_size[2],
                                                  hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        pe_size = [
            num_frames // self.patch_size[0], height // self.patch_size[1],
            width // self.patch_size[2]
        ]

        emb_t = self.pos_emb_t[:pe_size[0]][None, :, None, None, :].repeat(
            batch_size, 1, pe_size[1], pe_size[2], 1)
        emb_h = self.pos_emb_h[:pe_size[1]][None, None, :, None, :].repeat(
            batch_size, pe_size[0], 1, pe_size[2], 1)
        emb_w = self.pos_emb_w[:pe_size[2]][None, None, None, :, :].repeat(
            batch_size, pe_size[0], pe_size[1], 1, 1)
        emb = emb_t + emb_h + emb_w
        emb = emb.flatten(1, 3)

        norm = torch.linalg.vector_norm(emb,
                                        dim=-1,
                                        keepdim=True,
                                        dtype=torch.float32)
        norm = torch.add(self.eps,
                         norm,
                         alpha=np.sqrt(norm.numel() / emb.numel()))
        return (emb / norm).type_as(hidden_states)


class CosmosTransformer3DModel(BaseDiT):
    _fsdp_shard_conditions = CosmosConfig()._fsdp_shard_conditions
    _compile_conditions = CosmosConfig()._compile_conditions
    _supported_attention_backends = CosmosConfig()._supported_attention_backends
    _param_names_mapping = CosmosConfig()._param_names_mapping
    _lora_param_names_mapping = CosmosConfig()._lora_param_names_mapping

    def __init__(self, config: CosmosConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.max_size = config.max_size
        self.rope_scale = config.rope_scale
        self.concat_padding_mask = config.concat_padding_mask
        self.extra_pos_embed_type = config.extra_pos_embed_type

        # 1. Patch Embedding
        patch_embed_in_channels = config.in_channels + 1 if config.concat_padding_mask else config.in_channels
        self.patch_embed = CosmosPatchEmbed(patch_embed_in_channels,
                                            inner_dim,
                                            config.patch_size,
                                            bias=False)

        # 2. Positional Embedding
        self.rope = CosmosRotaryPosEmbed(hidden_size=config.attention_head_dim,
                                         max_size=config.max_size,
                                         patch_size=config.patch_size,
                                         rope_scale=config.rope_scale)

        self.learnable_pos_embed = None
        if config.extra_pos_embed_type == "learnable":
            self.learnable_pos_embed = CosmosLearnablePositionalEmbed(
                hidden_size=inner_dim,
                max_size=config.max_size,
                patch_size=config.patch_size,
            )

        # 3. Time Embedding
        self.time_embed = CosmosEmbedding(inner_dim, inner_dim)

        # 4. Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            CosmosTransformerBlock(
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                cross_attention_dim=config.text_embed_dim,
                mlp_ratio=config.mlp_ratio,
                adaln_lora_dim=config.adaln_lora_dim,
                qk_norm=config.qk_norm,
                out_bias=False,
                supported_attention_backends=self._supported_attention_backends,
                prefix=f"{config.prefix}.transformer_blocks.{i}",
            ) for i in range(config.num_layers)
        ])

        # 5. Output norm & projection
        self.norm_out = CosmosAdaLayerNorm(inner_dim, config.adaln_lora_dim)
        self.proj_out = nn.Linear(inner_dim,
                                  config.out_channels *
                                  math.prod(config.patch_size),
                                  bias=False)

        self.gradient_checkpointing = False

        # For TeaCache
        self.previous_e0_even = None
        self.previous_e0_odd = None
        self.previous_residual_even = None
        self.previous_residual_odd = None
        self.is_even = True
        self.should_calc_even = True
        self.should_calc_odd = True
        self.accumulated_rel_l1_distance_even = 0
        self.accumulated_rel_l1_distance_odd = 0
        self.cnt = 0
        self.__post_init__()

    def forward(self,
                hidden_states: torch.Tensor,
                timestep: torch.Tensor,
                encoder_hidden_states: Union[torch.Tensor, List[torch.Tensor]],
                attention_mask: Optional[torch.Tensor] = None,
                fps: Optional[int] = None,
                condition_mask: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        forward_batch = get_forward_context().forward_batch
        enable_teacache = forward_batch is not None and forward_batch.enable_teacache

        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        # 1. Concatenate padding mask if needed & prepare attention mask
        if condition_mask is not None:
            hidden_states = torch.cat([hidden_states, condition_mask], dim=1)

        if self.concat_padding_mask and padding_mask is not None:
            from torchvision import transforms
            padding_mask = transforms.functional.resize(
                padding_mask,
                list(hidden_states.shape[-2:]),
                interpolation=transforms.InterpolationMode.NEAREST)
            hidden_states = torch.cat([
                hidden_states,
                padding_mask.unsqueeze(2).repeat(batch_size, 1, num_frames, 1,
                                                 1)
            ],
                                      dim=1)
            # # Resize padding mask to match hidden states spatial dimensions
            # padding_mask_resized = F.interpolate(
            #     padding_mask.float().unsqueeze(1),
            #     size=(height, width),
            #     mode='nearest'
            # ).squeeze(1)
            # hidden_states = torch.cat(
            #     [hidden_states, padding_mask_resized.unsqueeze(1).unsqueeze(2).repeat(1, 1, num_frames, 1, 1)], dim=1
            # )

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(
                1)  # [B, 1, 1, S]

        # 2. Generate positional embeddings
        image_rotary_emb = self.rope(hidden_states, fps=fps)
        extra_pos_emb = self.learnable_pos_embed(
            hidden_states) if self.extra_pos_embed_type == "learnable" else None

        # 3. Patchify input
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.flatten(
            1, 3)  # [B, T, H, W, C] -> [B, THW, C]

        # 4. Timestep embeddings
        if timestep.ndim == 1:
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
        elif timestep.ndim == 5:
            assert timestep.shape == (batch_size, 1, num_frames, 1, 1), (
                f"Expected timestep to have shape [B, 1, T, 1, 1], but got {timestep.shape}"
            )
            timestep = timestep.flatten()
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
            # We can do this because num_frames == post_patch_num_frames, as p_t is 1
            temb, embedded_timestep = (
                x.view(batch_size, post_patch_num_frames, 1, 1,
                       -1).expand(-1, -1, post_patch_height, post_patch_width,
                                  -1).flatten(1, 3)
                for x in (temb, embedded_timestep)
            )  # [BT, C] -> [B, T, 1, 1, C] -> [B, T, H, W, C] -> [B, THW, C]
        else:
            raise ValueError(f"Unsupported timestep shape: {timestep.shape}")

        # 6. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.transformer_blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    embedded_timestep,
                    temb,
                    image_rotary_emb,
                    extra_pos_emb,
                    attention_mask,
                )
        else:
            for block in self.transformer_blocks:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    embedded_timestep=embedded_timestep,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    extra_pos_emb=extra_pos_emb,
                    attention_mask=attention_mask,
                )

        # 7. Output norm & projection & unpatchify
        hidden_states = self.norm_out(hidden_states, embedded_timestep, temb)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.unflatten(2, (p_h, p_w, p_t, -1))
        hidden_states = hidden_states.unflatten(
            1, (post_patch_num_frames, post_patch_height, post_patch_width))
        # NOTE: The permutation order here is not the inverse operation of what happens when patching as usually expected.
        # It might be a source of confusion to the reader, but this is correct
        hidden_states = hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return hidden_states
