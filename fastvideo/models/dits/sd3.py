# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.attention import DistributedAttention
from fastvideo.configs.models import DiTConfig
from fastvideo.forward_context import get_forward_context, set_forward_context
from fastvideo.layers.layernorm import RMSNorm
from fastvideo.layers.visual_embedding import Timesteps
from fastvideo.models.dits.base import BaseDiT
from fastvideo.platforms import AttentionBackendEnum


@dataclass
class SD3Transformer2DModelOutput:
    sample: torch.Tensor


def _chunked_feed_forward(
    ff: nn.Module,
    hidden_states: torch.Tensor,
    chunk_dim: int,
    chunk_size: int,
) -> torch.Tensor:
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"hidden_states.shape[{chunk_dim}] ({hidden_states.shape[chunk_dim]})"
            f" must be divisible by chunk_size ({chunk_size})"
        )
    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    return torch.cat(
        [ff(h) for h in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )


def _get_1d_sincos_pos_embed_from_grid(
    embed_dim: int,
    pos: torch.Tensor,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    if dtype is None:
        dtype = torch.float32 if pos.device.type == "mps" else torch.float64

    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=dtype)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)

    pos = pos.reshape(-1)
    out = torch.outer(pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1)


def _get_2d_sincos_pos_embed_from_grid(
    embed_dim: int,
    grid: torch.Tensor,
) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1)


def _get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int | tuple[int, int],
    interpolation_scale: float = 1.0,
    base_size: int = 16,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = (
        torch.arange(grid_size[0], device=device, dtype=torch.float32)
        / (grid_size[0] / base_size)
        / interpolation_scale
    )
    grid_w = (
        torch.arange(grid_size[1], device=device, dtype=torch.float32)
        / (grid_size[1] / base_size)
        / interpolation_scale
    )
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    return _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


class SD3PatchEmbed(nn.Module):
    """2D patch embedding with SD3 positional embedding cropping behavior."""

    def __init__(
        self,
        height: int = 224,
        width: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        layer_norm: bool = False,
        flatten: bool = True,
        bias: bool = True,
        interpolation_scale: float = 1.0,
        pos_embed_type: str | None = "sincos",
        pos_embed_max_size: int | None = None,
    ) -> None:
        super().__init__()

        num_patches = (height // patch_size) * (width // patch_size)
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            bias=bias,
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False,
                                     eps=1e-6)
        else:
            self.norm = None

        self.patch_size = patch_size
        self.height = height // patch_size
        self.width = width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale

        if pos_embed_max_size:
            grid_size = pos_embed_max_size
        else:
            grid_size = int(num_patches**0.5)

        if pos_embed_type is None:
            self.pos_embed = None
        elif pos_embed_type == "sincos":
            pos_embed = _get_2d_sincos_pos_embed(
                embed_dim,
                grid_size,
                base_size=self.base_size,
                interpolation_scale=self.interpolation_scale,
            )
            persistent = True if pos_embed_max_size else False
            self.register_buffer(
                "pos_embed",
                pos_embed.float().unsqueeze(0),
                persistent=persistent,
            )
        else:
            raise ValueError(f"Unsupported pos_embed_type: {pos_embed_type}")

    def cropped_pos_embed(self, height: int, width: int) -> torch.Tensor:
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping")

        height = height // self.patch_size
        width = width // self.patch_size

        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than pos_embed_max_size "
                f"({self.pos_embed_max_size})"
            )
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width ({width}) cannot be greater than pos_embed_max_size "
                f"({self.pos_embed_max_size})"
            )

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2

        spatial = self.pos_embed.reshape(
            1,
            self.pos_embed_max_size,
            self.pos_embed_max_size,
            -1,
        )
        spatial = spatial[:, top:top + height, left:left + width, :]
        return spatial.reshape(1, -1, spatial.shape[-1])

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
        else:
            height = latent.shape[-2] // self.patch_size
            width = latent.shape[-1] // self.patch_size

        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)

        if self.layer_norm:
            latent = self.norm(latent)

        if self.pos_embed is None:
            return latent.to(latent.dtype)

        if self.pos_embed_max_size:
            pos_embed = self.cropped_pos_embed(height, width)
        else:
            if self.height != height or self.width != width:
                pos_embed = _get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                    device=latent.device,
                )
                pos_embed = pos_embed.float().unsqueeze(0)
            else:
                pos_embed = self.pos_embed

        return (latent + pos_embed).to(latent.dtype)


class SD3TimestepEmbedding(nn.Module):

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
    ) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        if act_fn != "silu":
            raise ValueError(f"Unsupported act_fn: {act_fn}")
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class SD3TextProjection(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int | None = None,
        act_fn: str = "silu",
    ) -> None:
        super().__init__()

        if out_features is None:
            out_features = hidden_size

        self.linear_1 = nn.Linear(
            in_features=in_features,
            out_features=hidden_size,
            bias=True,
        )
        if act_fn != "silu":
            raise ValueError(f"Unsupported act_fn: {act_fn}")
        self.act_1 = nn.SiLU()
        self.linear_2 = nn.Linear(
            in_features=hidden_size,
            out_features=out_features,
            bias=True,
        )

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class CombinedTimestepTextProjEmbeddings(nn.Module):

    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.timestep_embedder = SD3TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
            act_fn="silu",
        )
        self.text_embedder = SD3TextProjection(
            pooled_projection_dim,
            embedding_dim,
            act_fn="silu",
        )

    def forward(
        self,
        timestep: torch.Tensor,
        pooled_projection: torch.Tensor,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + pooled_projections


class SD35AdaLayerNormZeroX(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        norm_type: str = "layer_norm",
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 9 * embedding_dim, bias=bias)

        if norm_type != "layer_norm":
            raise ValueError(
                f"Unsupported norm_type ({norm_type}); expected layer_norm"
            )
        self.norm = nn.LayerNorm(embedding_dim,
                                 elementwise_affine=False,
                                 eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        emb = self.linear(self.silu(emb))
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp,
         shift_msa2, scale_msa2,
         gate_msa2) = emb.chunk(9, dim=1)

        norm_hidden_states = self.norm(hidden_states)
        hidden_states = norm_hidden_states * (
            1 + scale_msa[:, None]) + shift_msa[:, None]
        norm_hidden_states2 = norm_hidden_states * (
            1 + scale_msa2[:, None]) + shift_msa2[:, None]
        return (
            hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            norm_hidden_states2,
            gate_msa2,
        )


class SD3AdaLayerNormZero(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        norm_type: str = "layer_norm",
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)

        if norm_type != "layer_norm":
            raise ValueError(
                f"Unsupported norm_type ({norm_type}); expected layer_norm"
            )
        self.norm = nn.LayerNorm(embedding_dim,
                                 elementwise_affine=False,
                                 eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            emb.chunk(6, dim=1))
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class SD3AdaLayerNormContinuous(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "layer_norm",
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(
            conditioning_embedding_dim,
            embedding_dim * 2,
            bias=bias,
        )

        if norm_type != "layer_norm":
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.norm = nn.LayerNorm(
            embedding_dim,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

    def forward(
        self,
        x: torch.Tensor,
        conditioning_embedding: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class SD3GELU(nn.Module):

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        approximate: str = "none",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return F.gelu(hidden_states, approximate=self.approximate)


class SD3FeedForward(nn.Module):

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if inner_dim is None:
            inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim

        if activation_fn == "gelu-approximate":
            act_fn: nn.Module = SD3GELU(
                dim,
                inner_dim,
                approximate="tanh",
                bias=bias,
            )
        elif activation_fn == "gelu":
            act_fn = SD3GELU(dim, inner_dim, approximate="none", bias=bias)
        else:
            raise ValueError(f"Unsupported activation_fn: {activation_fn}")

        self.net = nn.ModuleList()
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


def _build_qk_norm(
    qk_norm: str | None,
    dim_head: int,
    eps: float,
) -> tuple[nn.Module | None, nn.Module | None]:
    if qk_norm is None:
        return None, None
    if qk_norm == "layer_norm":
        return (
            nn.LayerNorm(dim_head, eps=eps, elementwise_affine=True),
            nn.LayerNorm(dim_head, eps=eps, elementwise_affine=True),
        )
    if qk_norm == "rms_norm":
        return (
            RMSNorm(dim_head, eps=eps, has_weight=True),
            RMSNorm(dim_head, eps=eps, has_weight=True),
        )
    raise ValueError(f"Unsupported qk_norm: {qk_norm}")


class SD3Attention(nn.Module):
    """Joint self-attention used in SD3 blocks."""

    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        out_dim: int,
        added_kv_proj_dim: int | None = None,
        context_pre_only: bool | None = None,
        qk_norm: str | None = None,
        eps: float = 1e-6,
        supported_attention_backends: tuple[AttentionBackendEnum,
                                            ...] | None = None,
    ) -> None:
        super().__init__()

        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = out_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=True)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=True)

        self.norm_q, self.norm_k = _build_qk_norm(qk_norm, dim_head, eps)

        self.add_q_proj: nn.Linear | None = None
        self.add_k_proj: nn.Linear | None = None
        self.add_v_proj: nn.Linear | None = None
        self.norm_added_q: nn.Module | None = None
        self.norm_added_k: nn.Module | None = None
        if added_kv_proj_dim is not None:
            self.add_q_proj = nn.Linear(
                added_kv_proj_dim,
                self.inner_dim,
                bias=True,
            )
            self.add_k_proj = nn.Linear(
                added_kv_proj_dim,
                self.inner_dim,
                bias=True,
            )
            self.add_v_proj = nn.Linear(
                added_kv_proj_dim,
                self.inner_dim,
                bias=True,
            )
            self.norm_added_q, self.norm_added_k = _build_qk_norm(
                qk_norm,
                dim_head,
                eps,
            )

        self.to_out = nn.ModuleList([
            nn.Linear(self.inner_dim, out_dim, bias=True),
            nn.Dropout(0.0),
        ])

        if context_pre_only is not None and not context_pre_only:
            self.to_add_out = nn.Linear(self.inner_dim, out_dim, bias=True)
        else:
            self.to_add_out = None

        self.attn = DistributedAttention(
            num_heads=heads,
            head_size=dim_head,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | torch.Tensor:
        batch_size = hidden_states.shape[0]
        residual_seq_len = hidden_states.shape[1]

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.view(batch_size, -1, self.heads, self.head_dim)
        key = key.view(batch_size, -1, self.heads, self.head_dim)
        value = value.view(batch_size, -1, self.heads, self.head_dim)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        if encoder_hidden_states is not None:
            if (self.add_q_proj is None or self.add_k_proj is None
                    or self.add_v_proj is None):
                raise RuntimeError(
                    "encoder_hidden_states provided but attention does not have"
                    " added projections"
                )

            encoder_query = self.add_q_proj(encoder_hidden_states)
            encoder_key = self.add_k_proj(encoder_hidden_states)
            encoder_value = self.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.view(
                batch_size,
                -1,
                self.heads,
                self.head_dim,
            )
            encoder_key = encoder_key.view(
                batch_size,
                -1,
                self.heads,
                self.head_dim,
            )
            encoder_value = encoder_value.view(
                batch_size,
                -1,
                self.heads,
                self.head_dim,
            )

            if self.norm_added_q is not None:
                encoder_query = self.norm_added_q(encoder_query)
            if self.norm_added_k is not None:
                encoder_key = self.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)

        attn_output, _ = self.attn(query, key, value)
        attn_output = attn_output.reshape(batch_size, -1,
                                          self.heads * self.head_dim)

        if encoder_hidden_states is not None:
            hidden_out = attn_output[:, :residual_seq_len]
            context_out = attn_output[:, residual_seq_len:]
            if self.to_add_out is not None:
                context_out = self.to_add_out(context_out)
        else:
            hidden_out = attn_output
            context_out = None

        hidden_out = self.to_out[0](hidden_out)
        hidden_out = self.to_out[1](hidden_out)

        if context_out is not None:
            return hidden_out, context_out
        return hidden_out


class SD3JointTransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        context_pre_only: bool = False,
        qk_norm: str | None = None,
        use_dual_attention: bool = False,
        supported_attention_backends: tuple[AttentionBackendEnum,
                                            ...] | None = None,
    ) -> None:
        super().__init__()

        self.use_dual_attention = use_dual_attention
        self.context_pre_only = context_pre_only

        if use_dual_attention:
            self.norm1 = SD35AdaLayerNormZeroX(dim)
        else:
            self.norm1 = SD3AdaLayerNormZero(dim)

        if context_pre_only:
            self.norm1_context = SD3AdaLayerNormContinuous(
                dim,
                dim,
                elementwise_affine=False,
                eps=1e-6,
                bias=True,
                norm_type="layer_norm",
            )
        else:
            self.norm1_context = SD3AdaLayerNormZero(dim)

        self.attn = SD3Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            added_kv_proj_dim=dim,
            out_dim=dim,
            context_pre_only=context_pre_only,
            qk_norm=qk_norm,
            eps=1e-6,
            supported_attention_backends=supported_attention_backends,
        )

        if use_dual_attention:
            self.attn2 = SD3Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                out_dim=dim,
                added_kv_proj_dim=None,
                context_pre_only=None,
                qk_norm=qk_norm,
                eps=1e-6,
                supported_attention_backends=supported_attention_backends,
            )
        else:
            self.attn2 = None

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = SD3FeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu-approximate",
        )

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim,
                                              elementwise_affine=False,
                                              eps=1e-6)
            self.ff_context = SD3FeedForward(
                dim=dim,
                dim_out=dim,
                activation_fn="gelu-approximate",
            )
        else:
            self.norm2_context = None
            self.ff_context = None

        self._chunk_size: int | None = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: int | None, dim: int = 0):
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        del joint_attention_kwargs

        if self.use_dual_attention:
            (
                norm_hidden_states,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                norm_hidden_states2,
                gate_msa2,
            ) = self.norm1(hidden_states, emb=temb)
        else:
            (
                norm_hidden_states,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self.norm1(hidden_states, emb=temb)

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(
                encoder_hidden_states,
                temb,
            )
        else:
            (
                norm_encoder_hidden_states,
                c_gate_msa,
                c_shift_mlp,
                c_scale_mlp,
                c_gate_mlp,
            ) = self.norm1_context(encoder_hidden_states, emb=temb)

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
        )

        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        if self.use_dual_attention:
            assert self.attn2 is not None
            attn_output2 = self.attn2(hidden_states=norm_hidden_states2)
            hidden_states = hidden_states + gate_msa2.unsqueeze(1) * attn_output2

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(
                self.ff,
                norm_hidden_states,
                self._chunk_dim,
                self._chunk_size,
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            assert context_attn_output is not None
            assert self.norm2_context is not None
            assert self.ff_context is not None

            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (
                1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

            if self._chunk_size is not None:
                context_ff_output = _chunked_feed_forward(
                    self.ff_context,
                    norm_encoder_hidden_states,
                    self._chunk_dim,
                    self._chunk_size,
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)

            encoder_hidden_states = encoder_hidden_states + (
                c_gate_mlp.unsqueeze(1) * context_ff_output)

        return encoder_hidden_states, hidden_states


class SD3Transformer2DModel(BaseDiT):
    """FastVideo-native SD3 Transformer2DModel."""

    _fsdp_shard_conditions = [
        lambda n, m: "transformer_blocks" in n and n.split(".")[-1].isdigit(),
    ]
    _compile_conditions = _fsdp_shard_conditions
    param_names_mapping: dict[str, Any] = {}
    reverse_param_names_mapping: dict[str, Any] = {}
    lora_param_names_mapping: dict[str, Any] = {}
    _supported_attention_backends = (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    )

    def __init__(self, config: DiTConfig, hf_config: dict[str, Any], **kwargs):
        del kwargs
        super().__init__(config=config, hf_config=hf_config)

        self.fastvideo_config = config
        self.hf_config = hf_config

        arch = config.arch_config

        self.out_channels = arch.out_channels
        self.inner_dim = arch.num_attention_heads * arch.attention_head_dim
        self.patch_size = arch.patch_size
        self.num_layers = arch.num_layers

        self.hidden_size = self.inner_dim
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.in_channels

        self.pos_embed = SD3PatchEmbed(
            height=arch.sample_size,
            width=arch.sample_size,
            patch_size=arch.patch_size,
            in_channels=arch.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=arch.pos_embed_max_size,
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=arch.pooled_projection_dim,
        )
        self.context_embedder = nn.Linear(arch.joint_attention_dim,
                                          arch.caption_projection_dim)

        dual_layers = getattr(arch, "dual_attention_layers", ())
        dual_layers = tuple(dual_layers) if isinstance(dual_layers,
                                                       list) else dual_layers

        self.transformer_blocks = nn.ModuleList([
            SD3JointTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=arch.num_attention_heads,
                attention_head_dim=arch.attention_head_dim,
                context_pre_only=i == arch.num_layers - 1,
                qk_norm=arch.qk_norm,
                use_dual_attention=i in dual_layers,
                supported_attention_backends=self._supported_attention_backends,
            ) for i in range(arch.num_layers)
        ])

        self.norm_out = SD3AdaLayerNormContinuous(
            self.inner_dim,
            self.inner_dim,
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            norm_type="layer_norm",
        )
        self.proj_out = nn.Linear(
            self.inner_dim,
            arch.patch_size * arch.patch_size * self.out_channels,
            bias=True,
        )

        self.gradient_checkpointing = False
        self.__post_init__()

    def enable_forward_chunking(
        self,
        chunk_size: int | None = None,
        dim: int = 0,
    ) -> None:
        if dim not in [0, 1]:
            raise ValueError(f"dim must be 0 or 1, got {dim}")

        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk: int,
                                      chunk_dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk, dim=chunk_dim)
            for child in module.children():
                fn_recursive_feed_forward(child, chunk, chunk_dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def disable_forward_chunking(self):

        def fn_recursive_feed_forward(module: torch.nn.Module):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=None, dim=0)
            for child in module.children():
                fn_recursive_feed_forward(child)

        for module in self.children():
            fn_recursive_feed_forward(module)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        pooled_projections: torch.Tensor | None = None,
        timestep: torch.LongTensor | None = None,
        block_controlnet_hidden_states: list[torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        skip_layers: list[int] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | SD3Transformer2DModelOutput:
        del kwargs

        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must be provided")
        if pooled_projections is None:
            raise ValueError("pooled_projections must be provided")
        if timestep is None:
            raise ValueError("timestep must be provided")

        if timestep.dim() == 0:
            timestep = timestep[None]
        if timestep.dim() > 1:
            timestep = timestep.reshape(-1)
        if timestep.shape[0] == 1 and hidden_states.shape[0] > 1:
            timestep = timestep.expand(hidden_states.shape[0])

        try:
            get_forward_context()
            forward_context = nullcontext()
        except AssertionError:
            timestep_val = int(timestep[0].item()) if timestep.numel() > 0 else 0
            forward_context = set_forward_context(
                current_timestep=timestep_val,
                attn_metadata=None,
            )

        with forward_context:
            height, width = hidden_states.shape[-2:]

            hidden_states = self.pos_embed(hidden_states)
            temb = self.time_text_embed(timestep, pooled_projections)
            encoder_hidden_states = self.context_embedder(encoder_hidden_states)

            for index_block, block in enumerate(self.transformer_blocks):
                is_skip = bool(
                    skip_layers is not None and index_block in skip_layers)
                if is_skip:
                    continue

                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

                if (block_controlnet_hidden_states is not None
                        and not block.context_pre_only):
                    interval_control = len(self.transformer_blocks) / len(
                        block_controlnet_hidden_states)
                    hidden_states = hidden_states + block_controlnet_hidden_states[
                        int(index_block / interval_control)]

            hidden_states = self.norm_out(hidden_states, temb)
            hidden_states = self.proj_out(hidden_states)

            patch_size = self.patch_size
            height = height // patch_size
            width = width // patch_size

            hidden_states = hidden_states.reshape(
                hidden_states.shape[0],
                height,
                width,
                patch_size,
                patch_size,
                self.out_channels,
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                hidden_states.shape[0],
                self.out_channels,
                height * patch_size,
                width * patch_size,
            )

            if not return_dict:
                return (output, )

            return SD3Transformer2DModelOutput(sample=output)


EntryClass = SD3Transformer2DModel
