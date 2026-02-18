# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.attention import LocalAttention
from fastvideo.configs.models.dits import Kandinsky5VideoConfig
from fastvideo.layers.layernorm import LayerNormScaleShift
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.models.dits.base import BaseDiT
from fastvideo.platforms import AttentionBackendEnum


def _build_rotary_freqs(dim: int, max_period: float) -> torch.Tensor:
    return torch.exp(-math.log(max_period) * torch.arange(
        start=0, end=dim, dtype=torch.float32) / dim)


def local_patching(x: torch.Tensor, shape: tuple[int, int, int, int],
                   group_size: tuple[int, int, int],
                   dim: int = 0) -> torch.Tensor:
    batch_size, duration, height, width = shape
    g1, g2, g3 = group_size
    x = x.reshape(
        *x.shape[:dim],
        duration // g1,
        g1,
        height // g2,
        g2,
        width // g3,
        g3,
        *x.shape[dim + 3:],
    )
    x = x.permute(
        *range(len(x.shape[:dim])),
        dim,
        dim + 2,
        dim + 4,
        dim + 1,
        dim + 3,
        dim + 5,
        *range(dim + 6, len(x.shape)),
    )
    x = x.flatten(dim, dim + 2).flatten(dim + 1, dim + 3)
    return x


def local_merge(x: torch.Tensor, shape: tuple[int, int, int, int],
                group_size: tuple[int, int, int],
                dim: int = 0) -> torch.Tensor:
    batch_size, duration, height, width = shape
    g1, g2, g3 = group_size
    x = x.reshape(
        *x.shape[:dim],
        duration // g1,
        height // g2,
        width // g3,
        g1,
        g2,
        g3,
        *x.shape[dim + 2:],
    )
    x = x.permute(
        *range(len(x.shape[:dim])),
        dim,
        dim + 3,
        dim + 1,
        dim + 4,
        dim + 2,
        dim + 5,
        *range(dim + 6, len(x.shape)),
    )
    x = x.flatten(dim, dim + 1).flatten(dim + 1, dim + 2).flatten(
        dim + 2, dim + 3)
    return x


def fractal_flatten(x: torch.Tensor,
                    rope: torch.Tensor,
                    shape: tuple[int, int, int, int],
                    block_mask: bool = False):
    if block_mask:
        pixel_size = 8
        x = local_patching(x, shape, (1, pixel_size, pixel_size), dim=1)
        rope = local_patching(rope, shape, (1, pixel_size, pixel_size), dim=1)
        x = x.flatten(1, 2)
        rope = rope.flatten(1, 2)
    else:
        x = x.flatten(1, 3)
        rope = rope.flatten(1, 3)
    return x, rope


def fractal_unflatten(x: torch.Tensor,
                      shape: tuple[int, int, int, int],
                      block_mask: bool = False):
    if block_mask:
        pixel_size = 8
        x = x.reshape(x.shape[0], -1, pixel_size**2, *x.shape[2:])
        x = local_merge(x, shape, (1, pixel_size, pixel_size), dim=1)
    else:
        x = x.reshape(*shape, *x.shape[2:])
    return x


class Kandinsky5TimeEmbeddings(nn.Module):

    def __init__(self, model_dim: int, time_dim: int,
                 max_period: float = 10000.0):
        super().__init__()
        assert model_dim % 2 == 0
        self.model_dim = model_dim
        self.max_period = max_period
        self.freqs = _build_rotary_freqs(self.model_dim // 2, self.max_period)
        self.in_layer = ReplicatedLinear(model_dim, time_dim, bias=True)
        self.activation = nn.SiLU()
        self.out_layer = ReplicatedLinear(time_dim, time_dim, bias=True)

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        args = torch.outer(time, self.freqs.to(device=time.device))
        time_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        time_embed, _ = self.in_layer(time_embed)
        time_embed = self.activation(time_embed)
        time_embed, _ = self.out_layer(time_embed)
        return time_embed


class Kandinsky5TextEmbeddings(nn.Module):

    def __init__(self, text_dim: int, model_dim: int):
        super().__init__()
        self.in_layer = ReplicatedLinear(text_dim, model_dim, bias=True)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, text_embed: torch.Tensor) -> torch.Tensor:
        text_embed, _ = self.in_layer(text_embed)
        return self.norm(text_embed).type_as(text_embed)


class Kandinsky5VisualEmbeddings(nn.Module):

    def __init__(self, visual_dim: int, model_dim: int,
                 patch_size: tuple[int, int, int]):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = ReplicatedLinear(math.prod(patch_size) * visual_dim,
                                         model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, duration, height, width, dim = x.shape
        x = (x.view(
            batch_size,
            duration // self.patch_size[0],
            self.patch_size[0],
            height // self.patch_size[1],
            self.patch_size[1],
            width // self.patch_size[2],
            self.patch_size[2],
            dim,
        ).permute(0, 1, 3, 5, 2, 4, 6, 7).flatten(4, 7))
        x, _ = self.in_layer(x)
        return x


class Kandinsky5RoPE1D(nn.Module):

    def __init__(self,
                 dim: int,
                 max_pos: int = 1024,
                 max_period: float = 10000.0):
        super().__init__()
        self.max_period = max_period
        self.dim = dim
        self.max_pos = max_pos
        freq = _build_rotary_freqs(dim // 2, max_period)
        pos = torch.arange(max_pos, dtype=freq.dtype)
        self.register_buffer("args", torch.outer(pos, freq), persistent=False)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        args = self.args[pos]
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class Kandinsky5RoPE3D(nn.Module):

    def __init__(self,
                 axes_dims: tuple[int, int, int],
                 max_pos: tuple[int, int, int] = (128, 128, 128),
                 max_period: float = 10000.0):
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period

        for i, (axes_dim, ax_max_pos) in enumerate(
                zip(axes_dims, max_pos, strict=True)):
            freq = _build_rotary_freqs(axes_dim // 2, max_period)
            pos = torch.arange(ax_max_pos, dtype=freq.dtype)
            self.register_buffer(f"args_{i}",
                                 torch.outer(pos, freq),
                                 persistent=False)

    def forward(self,
                shape: tuple[int, int, int, int],
                pos: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                | list[torch.Tensor],
                scale_factor: tuple[float, float,
                                    float] = (1.0, 1.0, 1.0)):
        batch_size, duration, height, width = shape
        args_t = self.args_0[pos[0]] / scale_factor[0]
        args_h = self.args_1[pos[1]] / scale_factor[1]
        args_w = self.args_2[pos[2]] / scale_factor[2]

        args = torch.cat(
            [
                args_t.view(1, duration, 1, 1, -1).repeat(
                    batch_size, 1, height, width, 1),
                args_h.view(1, 1, height, 1, -1).repeat(
                    batch_size, duration, 1, width, 1),
                args_w.view(1, 1, 1, width, -1).repeat(
                    batch_size, duration, height, 1, 1),
            ],
            dim=-1,
        )
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class Kandinsky5Modulation(nn.Module):

    def __init__(self, time_dim: int, model_dim: int, num_params: int):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = ReplicatedLinear(time_dim,
                                          num_params * model_dim,
                                          bias=True)
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        x, _ = self.out_layer(x)
        return x


def _apply_rotary(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
    x_out = (rope * x_).sum(dim=-1)
    return x_out.reshape(*x.shape).to(torch.bfloat16)


class Kandinsky5Attention(nn.Module):

    def __init__(
        self,
        num_channels: int,
        head_dim: int,
        supported_attention_backends: tuple[AttentionBackendEnum, ...] | None,
        prefix: str = "",
    ):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim

        self.to_query = ReplicatedLinear(num_channels,
                                         num_channels,
                                         bias=True,
                                         prefix=f"{prefix}.to_query")
        self.to_key = ReplicatedLinear(num_channels,
                                       num_channels,
                                       bias=True,
                                       prefix=f"{prefix}.to_key")
        self.to_value = ReplicatedLinear(num_channels,
                                         num_channels,
                                         bias=True,
                                         prefix=f"{prefix}.to_value")
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)
        self.out_layer = ReplicatedLinear(num_channels,
                                          num_channels,
                                          bias=True,
                                          prefix=f"{prefix}.out_layer")
        self.local_attention = LocalAttention(
            num_heads=self.num_heads,
            head_size=head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        sparse_params: dict[str, Any] | None = None,
        rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, _ = self.to_query(hidden_states)

        if encoder_hidden_states is not None:
            key, _ = self.to_key(encoder_hidden_states)
            value, _ = self.to_value(encoder_hidden_states)
            shape, cond_shape = query.shape[:-1], key.shape[:-1]
            query = query.reshape(*shape, self.num_heads, -1)
            key = key.reshape(*cond_shape, self.num_heads, -1)
            value = value.reshape(*cond_shape, self.num_heads, -1)
        else:
            key, _ = self.to_key(hidden_states)
            value, _ = self.to_value(hidden_states)
            shape = query.shape[:-1]
            query = query.reshape(*shape, self.num_heads, -1)
            key = key.reshape(*shape, self.num_heads, -1)
            value = value.reshape(*shape, self.num_heads, -1)

        query = self.query_norm(query.float()).type_as(query)
        key = self.key_norm(key.float()).type_as(key)

        if rotary_emb is not None:
            query = _apply_rotary(query, rotary_emb).type_as(query)
            key = _apply_rotary(key, rotary_emb).type_as(key)

        if sparse_params is not None:
            raise NotImplementedError(
                "Sparse attention is not yet supported for Kandinsky5 in FastVideo."
            )

        try:
            hidden_states = self.local_attention(query, key, value)
        except AssertionError as exc:
            # LocalAttention requires pipeline forward context. Standalone
            # parity tests call the model directly, so fallback to Torch SDPA.
            if "Forward context is not set" not in str(exc):
                raise
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            hidden_states = F.scaled_dot_product_attention(query,
                                                           key,
                                                           value,
                                                           attn_mask=None,
                                                           is_causal=False)
            hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.flatten(2)

        hidden_states, _ = self.out_layer(hidden_states)
        return hidden_states


class Kandinsky5FeedForward(nn.Module):

    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        self.mlp = MLP(dim, ff_dim, bias=False, act_type="gelu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Kandinsky5OutLayer(nn.Module):

    def __init__(self, model_dim: int, time_dim: int, visual_dim: int,
                 patch_size: tuple[int, int, int]):
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Kandinsky5Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim,
                                 eps=1e-5,
                                 elementwise_affine=False)
        self.out_layer = ReplicatedLinear(model_dim,
                                          math.prod(patch_size) * visual_dim,
                                          bias=True)

    def forward(self, visual_embed: torch.Tensor, text_embed: torch.Tensor,
                time_embed: torch.Tensor) -> torch.Tensor:
        shift, scale = torch.chunk(
            self.modulation(time_embed).unsqueeze(dim=1), 2, dim=-1)
        visual_embed = (
            self.norm(visual_embed.float()) *
            (scale.float()[:, None, None] + 1.0) +
            shift.float()[:, None, None]).type_as(visual_embed)

        x, _ = self.out_layer(visual_embed)

        batch_size, duration, height, width, _ = x.shape
        x = (x.view(
            batch_size,
            duration,
            height,
            width,
            -1,
            self.patch_size[0],
            self.patch_size[1],
            self.patch_size[2],
        ).permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(1, 2).flatten(
            2, 3).flatten(3, 4))
        return x


class Kandinsky5TransformerEncoderBlock(nn.Module):

    def __init__(self, model_dim: int, time_dim: int, ff_dim: int,
                 head_dim: int,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...]
                 | None = None,
                 prefix: str = ""):
        super().__init__()
        self.text_modulation = Kandinsky5Modulation(time_dim, model_dim, 6)

        self.self_attention_norm = LayerNormScaleShift(
            model_dim,
            norm_type="layer",
            eps=1e-5,
            elementwise_affine=False,
            dtype=torch.float32,
            compute_dtype=torch.float32)
        self.self_attention = Kandinsky5Attention(
            model_dim,
            head_dim,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.self_attention")

        self.feed_forward_norm = LayerNormScaleShift(
            model_dim,
            norm_type="layer",
            eps=1e-5,
            elementwise_affine=False,
            dtype=torch.float32,
            compute_dtype=torch.float32)
        self.feed_forward = Kandinsky5FeedForward(model_dim, ff_dim)

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor,
                rope: torch.Tensor) -> torch.Tensor:
        self_attn_params, ff_params = torch.chunk(
            self.text_modulation(time_embed).unsqueeze(dim=1), 2, dim=-1)
        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = self.self_attention_norm(x.float(),
                                       shift=shift,
                                       scale=scale,
                                       convert_modulation_dtype=True).type_as(x)
        out = self.self_attention(out, rotary_emb=rope)
        x = (x.float() + gate.float() * out.float()).type_as(x)

        ff_shift, ff_scale, ff_gate = torch.chunk(ff_params, 3, dim=-1)
        out = self.feed_forward_norm(x.float(),
                                     shift=ff_shift,
                                     scale=ff_scale,
                                     convert_modulation_dtype=True).type_as(x)
        out = self.feed_forward(out)
        x = (x.float() + ff_gate.float() * out.float()).type_as(x)

        return x


class Kandinsky5TransformerDecoderBlock(nn.Module):

    def __init__(self, model_dim: int, time_dim: int, ff_dim: int,
                 head_dim: int,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...]
                 | None = None,
                 prefix: str = ""):
        super().__init__()
        self.visual_modulation = Kandinsky5Modulation(time_dim, model_dim, 9)

        self.self_attention_norm = LayerNormScaleShift(
            model_dim,
            norm_type="layer",
            eps=1e-5,
            elementwise_affine=False,
            dtype=torch.float32,
            compute_dtype=torch.float32)
        self.self_attention = Kandinsky5Attention(
            model_dim,
            head_dim,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.self_attention")

        self.cross_attention_norm = LayerNormScaleShift(
            model_dim,
            norm_type="layer",
            eps=1e-5,
            elementwise_affine=False,
            dtype=torch.float32,
            compute_dtype=torch.float32)
        self.cross_attention = Kandinsky5Attention(
            model_dim,
            head_dim,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.cross_attention")

        self.feed_forward_norm = LayerNormScaleShift(
            model_dim,
            norm_type="layer",
            eps=1e-5,
            elementwise_affine=False,
            dtype=torch.float32,
            compute_dtype=torch.float32)
        self.feed_forward = Kandinsky5FeedForward(model_dim, ff_dim)

    def forward(self, visual_embed: torch.Tensor, text_embed: torch.Tensor,
                time_embed: torch.Tensor, rope: torch.Tensor,
                sparse_params: dict[str, Any] | None) -> torch.Tensor:
        self_attn_params, cross_attn_params, ff_params = torch.chunk(
            self.visual_modulation(time_embed).unsqueeze(dim=1), 3, dim=-1)

        self_shift, self_scale, self_gate = torch.chunk(self_attn_params,
                                                        3,
                                                        dim=-1)
        visual_out = self.self_attention_norm(
            visual_embed.float(),
            shift=self_shift,
            scale=self_scale,
            convert_modulation_dtype=True,
        ).type_as(visual_embed)
        visual_out = self.self_attention(visual_out,
                                         rotary_emb=rope,
                                         sparse_params=sparse_params)
        visual_embed = (visual_embed.float() +
                        self_gate.float() * visual_out.float()).type_as(
                            visual_embed)

        cross_shift, cross_scale, cross_gate = torch.chunk(cross_attn_params,
                                                           3,
                                                           dim=-1)
        visual_out = self.cross_attention_norm(
            visual_embed.float(),
            shift=cross_shift,
            scale=cross_scale,
            convert_modulation_dtype=True,
        ).type_as(visual_embed)
        visual_out = self.cross_attention(
            visual_out, encoder_hidden_states=text_embed)
        visual_embed = (visual_embed.float() +
                        cross_gate.float() * visual_out.float()).type_as(
                            visual_embed)

        ff_shift, ff_scale, ff_gate = torch.chunk(ff_params, 3, dim=-1)
        visual_out = self.feed_forward_norm(
            visual_embed.float(),
            shift=ff_shift,
            scale=ff_scale,
            convert_modulation_dtype=True,
        ).type_as(visual_embed)
        visual_out = self.feed_forward(visual_out)
        visual_embed = (visual_embed.float() +
                        ff_gate.float() * visual_out.float()).type_as(
                            visual_embed)

        return visual_embed


@dataclass
class Kandinsky5TransformerOutput:
    sample: torch.Tensor


class Kandinsky5Transformer3DModel(BaseDiT):
    """
    Native FastVideo implementation of Kandinsky5 Transformer.
    """

    _fsdp_shard_conditions = Kandinsky5VideoConfig()._fsdp_shard_conditions
    _compile_conditions = Kandinsky5VideoConfig()._compile_conditions
    param_names_mapping = Kandinsky5VideoConfig().param_names_mapping
    reverse_param_names_mapping = Kandinsky5VideoConfig(
    ).reverse_param_names_mapping
    lora_param_names_mapping = Kandinsky5VideoConfig().lora_param_names_mapping
    _supported_attention_backends = Kandinsky5VideoConfig(
    )._supported_attention_backends

    def __init__(self, config: Kandinsky5VideoConfig,
                 hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)
        arch = config.arch_config

        head_dim = sum(arch.axes_dims)
        self.in_visual_dim = arch.in_visual_dim
        self.model_dim = arch.model_dim
        self.patch_size = arch.patch_size
        self.visual_cond = arch.visual_cond
        self.attention_type = arch.attention_type

        visual_embed_dim = (2 * arch.in_visual_dim +
                            1) if arch.visual_cond else arch.in_visual_dim

        self.time_embeddings = Kandinsky5TimeEmbeddings(
            arch.model_dim, arch.time_dim)
        self.text_embeddings = Kandinsky5TextEmbeddings(
            arch.in_text_dim, arch.model_dim)
        self.pooled_text_embeddings = Kandinsky5TextEmbeddings(
            arch.in_text_dim2, arch.time_dim)
        self.visual_embeddings = Kandinsky5VisualEmbeddings(
            visual_embed_dim, arch.model_dim, arch.patch_size)

        self.text_rope_embeddings = Kandinsky5RoPE1D(head_dim)
        self.visual_rope_embeddings = Kandinsky5RoPE3D(arch.axes_dims)

        self.text_transformer_blocks = nn.ModuleList([
            Kandinsky5TransformerEncoderBlock(arch.model_dim, arch.time_dim,
                                              arch.ff_dim,
                                              head_dim,
                                              self._supported_attention_backends,
                                              prefix=f"{config.prefix}.text_transformer_blocks.{i}")
            for i in range(arch.num_text_blocks)
        ])
        self.visual_transformer_blocks = nn.ModuleList([
            Kandinsky5TransformerDecoderBlock(arch.model_dim, arch.time_dim,
                                              arch.ff_dim,
                                              head_dim,
                                              self._supported_attention_backends,
                                              prefix=f"{config.prefix}.visual_transformer_blocks.{i}")
            for i in range(arch.num_visual_blocks)
        ])

        self.out_layer = Kandinsky5OutLayer(arch.model_dim, arch.time_dim,
                                            arch.out_visual_dim,
                                            arch.patch_size)
        self.gradient_checkpointing = False

        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        self.__post_init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor]
        | None = None,
        guidance=None,
        pooled_projections: torch.Tensor | None = None,
        visual_rope_pos: tuple[torch.Tensor, torch.Tensor,
                               torch.Tensor] | list[torch.Tensor] | None = None,
        text_rope_pos: torch.Tensor | None = None,
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
        sparse_params: dict[str, Any] | None = None,
        return_dict: bool = False,
        **kwargs,
    ) -> torch.Tensor | Kandinsky5TransformerOutput:
        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = encoder_hidden_states[0]

        if pooled_projections is None:
            if encoder_hidden_states_image is None:
                raise ValueError(
                    "pooled_projections must be provided for Kandinsky5.")
            pooled_projections = encoder_hidden_states_image
        if isinstance(pooled_projections, list):
            pooled_projections = pooled_projections[0]

        if visual_rope_pos is None or text_rope_pos is None:
            raise ValueError(
                "visual_rope_pos and text_rope_pos are required for Kandinsky5."
            )

        x = hidden_states
        text_embed = self.text_embeddings(encoder_hidden_states)
        time_embed = self.time_embeddings(timestep)
        time_embed = time_embed + self.pooled_text_embeddings(pooled_projections)
        visual_embed = self.visual_embeddings(x)
        text_rope = self.text_rope_embeddings(text_rope_pos).unsqueeze(dim=0)

        for text_transformer_block in self.text_transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                text_embed = torch.utils.checkpoint.checkpoint(
                    text_transformer_block,
                    text_embed,
                    time_embed,
                    text_rope,
                    use_reentrant=False)
            else:
                text_embed = text_transformer_block(text_embed, time_embed,
                                                    text_rope)

        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(visual_shape, visual_rope_pos,
                                                  scale_factor)
        to_fractal = sparse_params[
            "to_fractal"] if sparse_params is not None else False
        visual_embed, visual_rope = fractal_flatten(visual_embed, visual_rope,
                                                    visual_shape,
                                                    block_mask=to_fractal)

        for visual_transformer_block in self.visual_transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                visual_embed = torch.utils.checkpoint.checkpoint(
                    visual_transformer_block,
                    visual_embed,
                    text_embed,
                    time_embed,
                    visual_rope,
                    sparse_params,
                    use_reentrant=False)
            else:
                visual_embed = visual_transformer_block(
                    visual_embed,
                    text_embed,
                    time_embed,
                    visual_rope,
                    sparse_params,
                )

        visual_embed = fractal_unflatten(visual_embed,
                                         visual_shape,
                                         block_mask=to_fractal)
        x = self.out_layer(visual_embed, text_embed, time_embed)

        if return_dict:
            return Kandinsky5TransformerOutput(sample=x)
        return x

    def materialize_non_persistent_buffers(self, device: torch.device,
                                           dtype: torch.dtype | None = None
                                           ) -> None:
        if isinstance(self.time_embeddings.freqs,
                      torch.Tensor) and self.time_embeddings.freqs.is_meta:
            self.time_embeddings.freqs = _build_rotary_freqs(
                self.time_embeddings.model_dim // 2,
                self.time_embeddings.max_period).to(device=device)

        if isinstance(self.text_rope_embeddings.args,
                      torch.Tensor) and self.text_rope_embeddings.args.is_meta:
            freq = _build_rotary_freqs(self.text_rope_embeddings.dim // 2,
                                       self.text_rope_embeddings.max_period).to(
                                           device=device)
            pos = torch.arange(self.text_rope_embeddings.max_pos,
                               dtype=freq.dtype,
                               device=device)
            self.text_rope_embeddings._buffers["args"] = torch.outer(pos, freq)

        for i, (axes_dim,
                ax_max_pos) in enumerate(
                    zip(self.visual_rope_embeddings.axes_dims,
                        self.visual_rope_embeddings.max_pos,
                        strict=True)):
            name = f"args_{i}"
            buf = getattr(self.visual_rope_embeddings, name, None)
            if isinstance(buf, torch.Tensor) and buf.is_meta:
                freq = _build_rotary_freqs(
                    axes_dim // 2, self.visual_rope_embeddings.max_period).to(
                        device=device)
                pos = torch.arange(ax_max_pos, dtype=freq.dtype, device=device)
                self.visual_rope_embeddings._buffers[name] = torch.outer(
                    pos, freq)
