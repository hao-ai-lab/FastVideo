# SPDX-License-Identifier: Apache-2.0
"""Native FastVideo implementation of the Kandinsky6 T2VA/IT2VA transformer.

Kandinsky6 extends Kandinsky5's DiT (fastvideo/models/dits/kandinsky5.py)
with a second, parallel "audio" tower and a fused video<->audio decoder
block, so this file mirrors kandinsky5.py's structure closely for every
shared piece (time/text/visual embeddings, RoPE, modulation, feed-forward,
NABLA sparse attention) and only adds what's new: the audio tower and
Kandinsky6FusedTransformerDecoderBlock's cross-modal attention, ported from
the reference diffusers.models.transformers.transformer_kandinsky6 module.

Video and audio latents are ordinary batched tensors -- `[B, T, H, W, C]` for
video and `[B, A, D]` for audio -- consistent with Kandinsky5 and the rest of
FastVideo's pipeline/stage system. `Kandinsky6RoPE3D.forward` takes an explicit
`shape` tuple rather than deriving T/H/W from the position tensors' lengths.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.attention import LocalAttention
from fastvideo.attention.backends.nabla import CAN_USE_FLEX_ATTN, flex_attention, nablaT_v2
from fastvideo.configs.models.dits import Kandinsky6VideoAudioConfig
from fastvideo.layers.layernorm import LayerNormScaleShift
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.layers.quantization import QuantizationConfig
from fastvideo.logger import init_logger
from fastvideo.models.dits.base import BaseDiT
from fastvideo.platforms import AttentionBackendEnum

logger = init_logger(__name__)

if not CAN_USE_FLEX_ATTN:
    logger.warning("torch.nn.attention.flex_attention is unavailable in this PyTorch build; "
                   "Kandinsky6 NABLA sparse attention cannot be used.")

FRACTAL_PIXEL_SIZE = 8
_ARCH_CONFIG_DEFAULTS = Kandinsky6VideoAudioConfig().arch_config

# Modulation submodules whose maths diffusers keeps fp32 in the distill checkpoint's
# `from_pretrained(torch_dtype=bf16)` load, matched by exact dotted-path component (not substring) against
# `visual_modulation`/`text_modulation` (the block-level AdaLN modules) or `modulation` (the plain attribute name of
# `Kandinsky6OutLayer`/`Kandinsky6OutLayerAudio`'s modulation submodule). The differently-named `va_modulation`/
# `av_modulation` cross-modal gates and the `*_time_embeddings` towers are fp32 in the raw checkpoint file too, but
# diffusers' exact-component match does not keep them fp32, so they are deliberately not listed here.
_FP32_MODULATION_COMPONENTS = frozenset({"visual_modulation", "text_modulation", "modulation"})


def _build_rotary_freqs(dim: int, max_period: float) -> torch.Tensor:
    return torch.exp(-math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim)


def local_patching(x: torch.Tensor,
                   shape: tuple[int, int, int, int],
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


def local_merge(x: torch.Tensor,
                shape: tuple[int, int, int, int],
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
    x = x.flatten(dim, dim + 1).flatten(dim + 1, dim + 2).flatten(dim + 2, dim + 3)
    return x


def fractal_flatten(x: torch.Tensor, rope: torch.Tensor, shape: tuple[int, int, int, int], block_mask: bool = False):
    if block_mask:
        pixel_size = FRACTAL_PIXEL_SIZE
        x = local_patching(x, shape, (1, pixel_size, pixel_size), dim=1)
        rope = local_patching(rope, shape, (1, pixel_size, pixel_size), dim=1)
        x = x.flatten(1, 2)
        rope = rope.flatten(1, 2)
    else:
        x = x.flatten(1, 3)
        rope = rope.flatten(1, 3)
    return x, rope


def fractal_unflatten(x: torch.Tensor, shape: tuple[int, int, int, int], block_mask: bool = False):
    if block_mask:
        pixel_size = FRACTAL_PIXEL_SIZE
        x = x.reshape(x.shape[0], -1, pixel_size**2, *x.shape[2:])
        x = local_merge(x, shape, (1, pixel_size, pixel_size), dim=1)
    else:
        x = x.reshape(*shape, *x.shape[2:])
    return x


class Kandinsky6TimeEmbeddings(nn.Module):

    def __init__(self, model_dim: int, time_dim: int, max_period: float = 10000.0):
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


class Kandinsky6TextEmbeddings(nn.Module):
    """Linear + LayerNorm projection, reused for text tokens and audio latents."""

    def __init__(self, in_dim: int, model_dim: int):
        super().__init__()
        self.in_layer = ReplicatedLinear(in_dim, model_dim, bias=True)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.in_layer(x)
        return self.norm(x).type_as(x)


class Kandinsky6VisualEmbeddings(nn.Module):

    def __init__(self, visual_dim: int, model_dim: int, patch_size: tuple[int, int, int]):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = ReplicatedLinear(math.prod(patch_size) * visual_dim, model_dim)

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


class Kandinsky6RoPE1D(nn.Module):
    """1D rotary embedding for text and audio sequences."""

    def __init__(self, dim: int, max_pos: int = 2048, max_period: float = 10000.0, freqs_scaling: float = 1.0):
        super().__init__()
        self.max_period = max_period
        self.dim = dim
        self.max_pos = max_pos
        self.freqs_scaling = freqs_scaling
        freq = _build_rotary_freqs(dim // 2, max_period) * freqs_scaling
        pos = torch.arange(max_pos, dtype=freq.dtype)
        self.register_buffer("args", torch.outer(pos, freq), persistent=False)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        args = self.args[pos]
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class Kandinsky6RoPE3D(nn.Module):
    """3D rotary embedding for video spatial-temporal (T, H, W) tokens."""

    def __init__(self,
                 axes_dims: tuple[int, int, int],
                 max_pos: tuple[int, int, int] = (128, 128, 128),
                 max_period: float = 10000.0):
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period

        for i, (axes_dim, ax_max_pos) in enumerate(zip(axes_dims, max_pos, strict=True)):
            freq = _build_rotary_freqs(axes_dim // 2, max_period)
            pos = torch.arange(ax_max_pos, dtype=freq.dtype)
            self.register_buffer(f"args_{i}", torch.outer(pos, freq), persistent=False)

    def forward(self,
                shape: tuple[int, int, int, int],
                pos: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                | list[torch.Tensor],
                scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0)):
        batch_size, duration, height, width = shape
        args_t = self.args_0[pos[0]] / scale_factor[0]
        args_h = self.args_1[pos[1]] / scale_factor[1]
        args_w = self.args_2[pos[2]] / scale_factor[2]

        args = torch.cat(
            [
                args_t.view(1, duration, 1, 1, -1).repeat(batch_size, 1, height, width, 1),
                args_h.view(1, 1, height, 1, -1).repeat(batch_size, duration, 1, width, 1),
                args_w.view(1, 1, 1, width, -1).repeat(batch_size, duration, height, 1, 1),
            ],
            dim=-1,
        )
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class Kandinsky6Modulation(nn.Module):

    def __init__(self, time_dim: int, model_dim: int, num_params: int):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = ReplicatedLinear(time_dim, num_params * model_dim, bias=True)
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
    return x_out.reshape(*x.shape).to(x.dtype)


class Kandinsky6Attention(nn.Module):
    """Self- or cross-attention. ``kv_dim`` lets K/V come from a
    differently-sized stream (the video<->audio cross-modal attentions)."""

    def __init__(
        self,
        num_channels: int,
        head_dim: int,
        supported_attention_backends: tuple[AttentionBackendEnum, ...] | None,
        prefix: str = "",
        kv_dim: int | None = None,
        use_nabla: bool = False,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim
        kv_dim = kv_dim or num_channels

        self.to_query = ReplicatedLinear(num_channels,
                                         num_channels,
                                         bias=True,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.to_query")
        self.to_key = ReplicatedLinear(kv_dim,
                                       num_channels,
                                       bias=True,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.to_key")
        self.to_value = ReplicatedLinear(kv_dim,
                                         num_channels,
                                         bias=True,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.to_value")
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)
        self.out_layer = ReplicatedLinear(num_channels,
                                          num_channels,
                                          bias=True,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.out_layer")
        self.local_attention = LocalAttention(
            num_heads=self.num_heads,
            head_size=head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )
        # Only the video self-attention gets a second NABLA-backed attention
        # layer; audio self-attention and every cross-attention are dense.
        self.nabla_attention = None
        if use_nabla:
            self.nabla_attention = LocalAttention(
                num_heads=self.num_heads,
                head_size=head_dim,
                causal=False,
                supported_attention_backends=supported_attention_backends,
                default_backend=AttentionBackendEnum.NABLA_ATTN,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        sparse_params: dict[str, Any] | None = None,
        rotary_emb: torch.Tensor | None = None,
        rotary_emb_kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, _ = self.to_query(hidden_states)

        kv_source = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        key, _ = self.to_key(kv_source)
        value, _ = self.to_value(kv_source)

        shape, kv_shape = query.shape[:-1], key.shape[:-1]
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*kv_shape, self.num_heads, -1)
        value = value.reshape(*kv_shape, self.num_heads, -1)

        query = self.query_norm(query.float()).type_as(query)
        key = self.key_norm(key.float()).type_as(key)

        if rotary_emb is not None:
            query = _apply_rotary(query, rotary_emb).type_as(query)
        kv_rope = rotary_emb_kv if rotary_emb_kv is not None else (rotary_emb if encoder_hidden_states is None else None)
        if kv_rope is not None:
            key = _apply_rotary(key, kv_rope).type_as(key)

        if sparse_params is not None:
            if self.nabla_attention is None:
                raise RuntimeError("sparse_params passed to an attention layer built without use_nabla; "
                                   "this checkpoint/config combination is inconsistent.")
            try:
                hidden_states = self.nabla_attention(query, key, value)
            except AssertionError as exc:
                # Standalone parity tests call the model without a pipeline
                # forward context; run the NABLA kernel directly.
                if "Forward context is not set" not in str(exc):
                    raise
                attn_mask = nablaT_v2(query, key, sparse_params["sta_mask"], thr=sparse_params["P"])
                hidden_states = flex_attention(
                    query=query.transpose(1, 2),
                    key=key.transpose(1, 2),
                    value=value.transpose(1, 2),
                    block_mask=attn_mask,
                ).transpose(1, 2)
        else:
            try:
                hidden_states = self.local_attention(query, key, value)
            except AssertionError as exc:
                # LocalAttention requires pipeline forward context. Standalone
                # parity tests call the model directly, so fall back to SDPA.
                if "Forward context is not set" not in str(exc):
                    raise

                query_shape = query.shape[:-2]
                key_shape = key.shape[:-2]
                query = query.reshape(query_shape[0], -1, self.num_heads, query.shape[-1]).transpose(1, 2)
                key = key.reshape(key_shape[0], -1, self.num_heads, key.shape[-1]).transpose(1, 2)
                value = value.reshape(key_shape[0], -1, self.num_heads, value.shape[-1]).transpose(1, 2)
                hidden_states = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=None,
                    is_causal=False,
                )
                hidden_states = hidden_states.transpose(1, 2).reshape(*query_shape, self.num_heads, -1)

        hidden_states = hidden_states.flatten(-2, -1)
        hidden_states, _ = self.out_layer(hidden_states)
        return hidden_states


class Kandinsky6FeedForward(nn.Module):

    def __init__(self, dim: int, ff_dim: int, prefix: str = "", quant_config: QuantizationConfig | None = None):
        super().__init__()
        self.mlp = MLP(dim, ff_dim, bias=False, act_type="gelu", quant_config=quant_config, prefix=f"{prefix}.mlp")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Kandinsky6OutLayer(nn.Module):
    """Projects visual hidden states back to packed latent patches."""

    def __init__(self, model_dim: int, time_dim: int, visual_dim: int, patch_size: tuple[int, int, int]):
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Kandinsky6Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, eps=1e-5, elementwise_affine=False)
        self.out_layer = ReplicatedLinear(model_dim, math.prod(patch_size) * visual_dim, bias=True)

    def forward(self, visual_embed: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        shift, scale = torch.chunk(self.modulation(time_embed).unsqueeze(dim=1), 2, dim=-1)
        visual_embed = (self.norm(visual_embed.float()) * (scale.float()[:, None, None] + 1.0) +
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
        ).permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(1, 2).flatten(2, 3).flatten(3, 4))
        return x


class Kandinsky6OutLayerAudio(nn.Module):
    """Projects audio hidden states back to audio latent channels."""

    def __init__(self, model_dim: int, time_dim: int, audio_dim: int):
        super().__init__()
        self.modulation = Kandinsky6Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, eps=1e-5, elementwise_affine=False)
        self.out_layer = ReplicatedLinear(model_dim, audio_dim, bias=True)

    def forward(self, audio_embed: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        shift, scale = torch.chunk(self.modulation(time_embed).unsqueeze(dim=1), 2, dim=-1)
        x = (self.norm(audio_embed.float()) * (scale.float() + 1.0) + shift.float()).type_as(audio_embed)
        # Normalized a second time after the scale/shift affine, before the
        # output projection, as in the Diffusers model; the weights expect it.
        x = self.norm(x.float()).type_as(audio_embed)
        out, _ = self.out_layer(x)
        return out


class Kandinsky6TransformerEncoderBlock(nn.Module):
    """Text-only self-attention + feed-forward block (video/audio text towers)."""

    def __init__(self,
                 model_dim: int,
                 time_dim: int,
                 ff_dim: int,
                 head_dim: int,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...]
                 | None = None,
                 prefix: str = "",
                 quant_config: QuantizationConfig | None = None):
        super().__init__()
        self.text_modulation = Kandinsky6Modulation(time_dim, model_dim, 6)

        self.self_attention_norm = LayerNormScaleShift(model_dim,
                                                       norm_type="layer",
                                                       eps=1e-5,
                                                       elementwise_affine=False,
                                                       dtype=torch.float32,
                                                       compute_dtype=torch.float32)
        self.self_attention = Kandinsky6Attention(model_dim,
                                                  head_dim,
                                                  supported_attention_backends=supported_attention_backends,
                                                  prefix=f"{prefix}.self_attention",
                                                  quant_config=quant_config)

        self.feed_forward_norm = LayerNormScaleShift(model_dim,
                                                     norm_type="layer",
                                                     eps=1e-5,
                                                     elementwise_affine=False,
                                                     dtype=torch.float32,
                                                     compute_dtype=torch.float32)
        self.feed_forward = Kandinsky6FeedForward(model_dim,
                                                  ff_dim,
                                                  prefix=f"{prefix}.feed_forward",
                                                  quant_config=quant_config)

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        self_attn_params, ff_params = torch.chunk(self.text_modulation(time_embed).unsqueeze(dim=1), 2, dim=-1)
        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = self.self_attention_norm(x.float(), shift=shift, scale=scale, convert_modulation_dtype=True).type_as(x)
        out = self.self_attention(out, rotary_emb=rope)
        x = (x.float() + gate.float() * out.float()).type_as(x)

        ff_shift, ff_scale, ff_gate = torch.chunk(ff_params, 3, dim=-1)
        out = self.feed_forward_norm(x.float(), shift=ff_shift, scale=ff_scale,
                                     convert_modulation_dtype=True).type_as(x)
        out = self.feed_forward(out)
        x = (x.float() + ff_gate.float() * out.float()).type_as(x)

        return x


class Kandinsky6TransformerDecoderBlock(nn.Module):
    """Self-attention + text cross-attention + feed-forward block.

    Used standalone for plain (non-multimodal) T2V/I2V-parity checkpoints,
    and as the ``videoT``/``audioT`` sub-block inside
    ``Kandinsky6FusedTransformerDecoderBlock`` for T2VA/IT2VA.
    """

    def __init__(self,
                 model_dim: int,
                 time_dim: int,
                 ff_dim: int,
                 head_dim: int,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...]
                 | None = None,
                 prefix: str = "",
                 use_nabla: bool = False,
                 quant_config: QuantizationConfig | None = None):
        super().__init__()
        self.visual_modulation = Kandinsky6Modulation(time_dim, model_dim, 9)

        self.self_attention_norm = LayerNormScaleShift(model_dim,
                                                       norm_type="layer",
                                                       eps=1e-5,
                                                       elementwise_affine=False,
                                                       dtype=torch.float32,
                                                       compute_dtype=torch.float32)
        self.self_attention = Kandinsky6Attention(model_dim,
                                                  head_dim,
                                                  supported_attention_backends=supported_attention_backends,
                                                  prefix=f"{prefix}.self_attention",
                                                  use_nabla=use_nabla,
                                                  quant_config=quant_config)

        self.cross_attention_norm = LayerNormScaleShift(model_dim,
                                                        norm_type="layer",
                                                        eps=1e-5,
                                                        elementwise_affine=False,
                                                        dtype=torch.float32,
                                                        compute_dtype=torch.float32)
        self.cross_attention = Kandinsky6Attention(model_dim,
                                                   head_dim,
                                                   supported_attention_backends=supported_attention_backends,
                                                   prefix=f"{prefix}.cross_attention",
                                                   quant_config=quant_config)

        self.feed_forward_norm = LayerNormScaleShift(model_dim,
                                                     norm_type="layer",
                                                     eps=1e-5,
                                                     elementwise_affine=False,
                                                     dtype=torch.float32,
                                                     compute_dtype=torch.float32)
        self.feed_forward = Kandinsky6FeedForward(model_dim,
                                                  ff_dim,
                                                  prefix=f"{prefix}.feed_forward",
                                                  quant_config=quant_config)

    def forward(self, visual_embed: torch.Tensor, text_embed: torch.Tensor, time_embed: torch.Tensor,
                rope: torch.Tensor | None, sparse_params: dict[str, Any] | None) -> torch.Tensor:
        self_attn_params, cross_attn_params, ff_params = torch.chunk(
            self.visual_modulation(time_embed).unsqueeze(dim=1), 3, dim=-1)

        self_shift, self_scale, self_gate = torch.chunk(self_attn_params, 3, dim=-1)
        visual_out = self.self_attention_norm(
            visual_embed.float(),
            shift=self_shift,
            scale=self_scale,
            convert_modulation_dtype=True,
        ).type_as(visual_embed)
        visual_out = self.self_attention(visual_out, rotary_emb=rope, sparse_params=sparse_params)
        visual_embed = (visual_embed.float() + self_gate.float() * visual_out.float()).type_as(visual_embed)

        cross_shift, cross_scale, cross_gate = torch.chunk(cross_attn_params, 3, dim=-1)
        visual_out = self.cross_attention_norm(
            visual_embed.float(),
            shift=cross_shift,
            scale=cross_scale,
            convert_modulation_dtype=True,
        ).type_as(visual_embed)
        visual_out = self.cross_attention(visual_out, encoder_hidden_states=text_embed)
        visual_embed = (visual_embed.float() + cross_gate.float() * visual_out.float()).type_as(visual_embed)

        ff_shift, ff_scale, ff_gate = torch.chunk(ff_params, 3, dim=-1)
        visual_out = self.feed_forward_norm(
            visual_embed.float(),
            shift=ff_shift,
            scale=ff_scale,
            convert_modulation_dtype=True,
        ).type_as(visual_embed)
        visual_out = self.feed_forward(visual_out)
        visual_embed = (visual_embed.float() + ff_gate.float() * visual_out.float()).type_as(visual_embed)

        return visual_embed


def _apply_gate_sum(x: torch.Tensor, out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (x.float() + gate.float() * out.float()).type_as(x)


def _apply_scale_shift(norm: nn.LayerNorm, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return (norm(x.float()) * (scale.float() + 1.0) + shift.float()).type_as(x)


class Kandinsky6FusedTransformerDecoderBlock(nn.Module):
    """Joint video+audio decoder block: per-modality self/text-cross
    attention plus a dedicated bidirectional video<->audio cross-attention,
    all independently AdaLN-modulated. Ported from
    diffusers.models.transformers.transformer_kandinsky6.Kandinsky6FusedTransformerDecoderBlock.
    """

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        model_dim_a: int,
        time_dim_a: int,
        ff_dim_a: int,
        head_dim_a: int,
        supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
        prefix: str = "",
        use_nabla: bool = False,
        ca_rope: bool = False,
        cross_gates: bool = False,
        fix_modulation: bool = False,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.videoT = Kandinsky6TransformerDecoderBlock(model_dim,
                                                        time_dim,
                                                        ff_dim,
                                                        head_dim,
                                                        supported_attention_backends,
                                                        prefix=f"{prefix}.videoT",
                                                        use_nabla=use_nabla,
                                                        quant_config=quant_config)
        self.audioT = Kandinsky6TransformerDecoderBlock(model_dim_a,
                                                        time_dim_a,
                                                        ff_dim_a,
                                                        head_dim_a,
                                                        supported_attention_backends,
                                                        prefix=f"{prefix}.audioT",
                                                        use_nabla=False,
                                                        quant_config=quant_config)
        self.va_cross_attention = Kandinsky6Attention(model_dim,
                                                      head_dim,
                                                      supported_attention_backends=supported_attention_backends,
                                                      prefix=f"{prefix}.va_cross_attention",
                                                      kv_dim=model_dim_a,
                                                      quant_config=quant_config)
        self.av_cross_attention = Kandinsky6Attention(model_dim_a,
                                                      head_dim_a,
                                                      supported_attention_backends=supported_attention_backends,
                                                      prefix=f"{prefix}.av_cross_attention",
                                                      kv_dim=model_dim,
                                                      quant_config=quant_config)
        self.va_modulation = Kandinsky6Modulation(
            time_dim,
            model_dim if not cross_gates else model_dim * 2 + model_dim_a,
            1 if cross_gates else 3,
        )
        self.av_modulation = Kandinsky6Modulation(
            time_dim_a,
            model_dim_a if not cross_gates else model_dim_a * 2 + model_dim,
            1 if cross_gates else 3,
        )
        self.va_normalization = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.av_normalization = nn.LayerNorm(model_dim_a, elementwise_affine=False)
        self.ca_rope = ca_rope
        self.cross_gates = cross_gates
        self.fix_modulation = fix_modulation
        self.model_dim = model_dim
        self.model_dim_a = model_dim_a

    def forward(
        self,
        vis: torch.Tensor | None,
        aud: torch.Tensor | None,
        text_v: torch.Tensor,
        text_a: torch.Tensor | None,
        time_embed: tuple[torch.Tensor, torch.Tensor | None],
        vis_rope: torch.Tensor | None,
        aud_rope: torch.Tensor | None,
        sparse_params: dict[str, Any] | None,
        va_gate_scale: float = 1.0,
        av_gate_scale: float = 1.0,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """``vis``/``aud`` may each be ``None`` (matches the diffusers reference's guarded
        ``Kandinsky6FusedTransformerDecoderBlock``): every stage below only runs for a present
        modality, and the video<->audio cross-modal mixing only runs when both are present."""
        t_v, t_a = time_embed
        gate_v = vis_out_t = None

        if vis is not None:
            sa_p, ca_p, ff_p = torch.chunk(self.videoT.visual_modulation(t_v).unsqueeze(dim=1), 3, dim=-1)
            shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
            vis = _apply_gate_sum(
                vis,
                self.videoT.self_attention(
                    self.videoT.self_attention_norm(vis.float(), shift=shift, scale=scale,
                                                    convert_modulation_dtype=True).type_as(vis),
                    rotary_emb=vis_rope,
                    sparse_params=sparse_params,
                ),
                gate,
            )
            shift, scale, gate_v = torch.chunk(ca_p, 3, dim=-1)
            vis_pre_ca = self.videoT.cross_attention_norm(vis.float(), shift=shift, scale=scale,
                                                          convert_modulation_dtype=True).type_as(vis)
            vis_out_t = self.videoT.cross_attention(vis_pre_ca, encoder_hidden_states=text_v)

        if aud is not None:
            sa_p, ca_p, ff_p_a = torch.chunk(self.audioT.visual_modulation(t_a).unsqueeze(dim=1), 3, dim=-1)
            shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
            aud = _apply_gate_sum(
                aud,
                self.audioT.self_attention(
                    self.audioT.self_attention_norm(aud.float(), shift=shift, scale=scale,
                                                    convert_modulation_dtype=True).type_as(aud),
                    rotary_emb=aud_rope,
                ),
                gate,
            )
            shift, scale, gate_a = torch.chunk(ca_p, 3, dim=-1)
            aud_pre_ca = self.audioT.cross_attention_norm(aud.float(), shift=shift, scale=scale,
                                                           convert_modulation_dtype=True).type_as(aud)
            aud_out_t = self.audioT.cross_attention(aud_pre_ca, encoder_hidden_states=text_a)
            aud = _apply_gate_sum(aud, aud_out_t, gate_a)

            if vis is not None:
                t_va_mod = t_a if not self.fix_modulation else t_v
                t_av_mod = t_v if not self.fix_modulation else t_a
                va_params = self.va_modulation(t_va_mod).unsqueeze(dim=1)
                av_params = self.av_modulation(t_av_mod).unsqueeze(dim=1)
                if self.cross_gates:
                    va_shift, va_scale, va_gate = torch.split(
                        va_params, [self.model_dim, self.model_dim, self.model_dim_a], dim=-1)
                    av_shift, av_scale, av_gate = torch.split(
                        av_params, [self.model_dim_a, self.model_dim_a, self.model_dim], dim=-1)
                else:
                    va_shift, va_scale, va_gate = torch.chunk(va_params, 3, dim=-1)
                    av_shift, av_scale, av_gate = torch.chunk(av_params, 3, dim=-1)

                vis = _apply_gate_sum(vis, vis_out_t, gate_v)
                vis_for_va = _apply_scale_shift(self.va_normalization, vis, va_scale, va_shift)
                aud_for_av = _apply_scale_shift(self.av_normalization, aud, av_scale, av_shift)
                rq_v = vis_rope if self.ca_rope else None
                rk_a = aud_rope if self.ca_rope else None
                vis_from_aud = self.va_cross_attention(vis_for_va, encoder_hidden_states=aud_pre_ca, rotary_emb=rq_v,
                                                       rotary_emb_kv=rk_a)
                aud_from_vis = self.av_cross_attention(aud_for_av, encoder_hidden_states=vis_pre_ca, rotary_emb=rk_a,
                                                       rotary_emb_kv=rq_v)
                vis = _apply_gate_sum(vis, vis_from_aud, (va_gate if not self.cross_gates else av_gate) * va_gate_scale)
                aud = _apply_gate_sum(aud, aud_from_vis, (av_gate if not self.cross_gates else va_gate) * av_gate_scale)
        elif vis is not None:
            vis = _apply_gate_sum(vis, vis_out_t, gate_v)

        if vis is not None:
            shift, scale, gate = torch.chunk(ff_p, 3, dim=-1)
            vis = _apply_gate_sum(
                vis,
                self.videoT.feed_forward(
                    self.videoT.feed_forward_norm(vis.float(), shift=shift, scale=scale,
                                                  convert_modulation_dtype=True).type_as(vis)),
                gate,
            )
        if aud is not None:
            shift, scale, gate = torch.chunk(ff_p_a, 3, dim=-1)
            aud = _apply_gate_sum(
                aud,
                self.audioT.feed_forward(
                    self.audioT.feed_forward_norm(aud.float(), shift=shift, scale=scale,
                                                  convert_modulation_dtype=True).type_as(aud)),
                gate,
            )
        return vis, aud


@dataclass
class Kandinsky6TransformerOutput:
    sample: torch.Tensor | tuple[torch.Tensor, torch.Tensor]


class Kandinsky6Transformer3DModel(BaseDiT):
    """Native FastVideo implementation of the Kandinsky6 T2VA/IT2VA transformer."""

    _fsdp_shard_conditions = _ARCH_CONFIG_DEFAULTS._fsdp_shard_conditions
    _compile_conditions = _ARCH_CONFIG_DEFAULTS._compile_conditions
    param_names_mapping = _ARCH_CONFIG_DEFAULTS.param_names_mapping
    reverse_param_names_mapping = _ARCH_CONFIG_DEFAULTS.reverse_param_names_mapping
    lora_param_names_mapping = _ARCH_CONFIG_DEFAULTS.lora_param_names_mapping
    _supported_attention_backends = _ARCH_CONFIG_DEFAULTS._supported_attention_backends

    def __init__(self, config: Kandinsky6VideoAudioConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)
        arch = config.arch_config
        quant_config = config.quant_config
        self.quant_config = quant_config

        head_dim = sum(arch.axes_dims)
        head_dim_a = sum(arch.axes_dims_a)
        self.in_visual_dim = arch.in_visual_dim
        self.in_audio_dim = arch.in_audio_dim
        self.model_dim = arch.model_dim
        self.patch_size = arch.patch_size
        self.visual_cond = arch.visual_cond
        self.attention_engine = arch.attention_engine
        self.visual_token_type_num_embeddings = arch.visual_token_type_num_embeddings
        self.scale_factor = tuple(float(v) for v in arch.scale_factor)

        visual_embed_dim = (2 * arch.in_visual_dim + 1) if arch.visual_cond else arch.in_visual_dim

        self.visual_embeddings = Kandinsky6VisualEmbeddings(visual_embed_dim, arch.model_dim, arch.patch_size)
        if self.visual_token_type_num_embeddings > 0:
            self.visual_token_type_embeddings = nn.Embedding(self.visual_token_type_num_embeddings, arch.model_dim)
        self.visual_rope_embeddings = Kandinsky6RoPE3D(arch.axes_dims)
        self.out_layer = Kandinsky6OutLayer(arch.model_dim, arch.time_dim, arch.out_visual_dim, arch.patch_size)

        use_nabla = arch.attention_engine == "nabla"

        # The model is always the joint video+audio DiT; a video-only call passes
        # hidden_states_audio=None instead of using a different model shape.

        # Registered before the (much smaller) text towers below: enable_layerwise_offload
        # (fastvideo/hooks/layerwise_offload.py) hooks only the first top-level nn.ModuleList in
        # registration order, and with dit_layerwise_offload=True (the default) that has to be these
        # blocks rather than the 4-entry video_text_transformer_blocks.
        self.visual_transformer_blocks = nn.ModuleList([
            Kandinsky6FusedTransformerDecoderBlock(
                arch.model_dim,
                arch.time_dim,
                arch.ff_dim,
                head_dim,
                arch.model_dim_a,
                arch.time_dim_a,
                arch.ff_dim_a,
                head_dim_a,
                self._supported_attention_backends,
                prefix=f"{config.prefix}.visual_transformer_blocks.{i}",
                use_nabla=use_nabla,
                ca_rope=arch.ca_rope,
                cross_gates=arch.cross_gates,
                fix_modulation=arch.fix_modulation,
                quant_config=quant_config) for i in range(arch.num_visual_blocks)
        ])

        self.audio_embeddings = Kandinsky6TextEmbeddings(arch.in_audio_dim, arch.model_dim_a)
        self.audio_rope_embeddings = Kandinsky6RoPE1D(head_dim_a, freqs_scaling=arch.audio_freqs_scaling)
        self.audio_out_layer = Kandinsky6OutLayerAudio(arch.model_dim_a, arch.time_dim_a, arch.out_audio_dim or arch.in_audio_dim)

        for tower_prefix, model_dim, time_dim, hd in (
            ("video", arch.model_dim, arch.time_dim, head_dim),
            ("audio", arch.model_dim_a, arch.time_dim_a, head_dim_a),
        ):
            setattr(self, f"{tower_prefix}_time_embeddings", Kandinsky6TimeEmbeddings(model_dim, time_dim))
            setattr(self, f"{tower_prefix}_text_embeddings", Kandinsky6TextEmbeddings(arch.in_text_dim, model_dim))
            setattr(self, f"{tower_prefix}_pooled_text_embeddings",
                    Kandinsky6TextEmbeddings(arch.in_text_dim2, time_dim))
            setattr(self, f"{tower_prefix}_text_rope_embeddings", Kandinsky6RoPE1D(hd))
            setattr(
                self, f"{tower_prefix}_text_transformer_blocks",
                nn.ModuleList([
                    Kandinsky6TransformerEncoderBlock(
                        model_dim,
                        time_dim,
                        arch.ff_dim if tower_prefix == "video" else arch.ff_dim_a,
                        hd,
                        self._supported_attention_backends,
                        prefix=f"{config.prefix}.{tower_prefix}_text_transformer_blocks.{i}",
                        quant_config=quant_config) for i in range(arch.num_text_blocks)
                ]))

        self.gradient_checkpointing = False
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        self.__post_init__()

    def _get_parameter_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        """The distill checkpoint's safetensors file stores its modulation and time-embedding tensors as F32;
        diffusers' ``from_pretrained(torch_dtype=bf16)`` keeps only the exact-component modulation matches fp32 (see
        ``_FP32_MODULATION_COMPONENTS``) and rounds the rest -- including ``va_modulation``/``av_modulation`` and the
        ``*_time_embeddings`` towers -- to bf16. Mirrors ``Kandinsky6SRTransformer3DModel._get_parameter_dtype``.
        """
        if any(component in _FP32_MODULATION_COMPONENTS for component in name.split(".")):
            return torch.float32
        return default_dtype

    def _time_embed(self, prefix: str, time: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        pooled_embeddings = getattr(self, f"{prefix}_pooled_text_embeddings")
        time_embeddings = getattr(self, f"{prefix}_time_embeddings")
        return time_embeddings(time) + pooled_embeddings(pooled)

    def _encode_text(self, prefix: str, text_embed: torch.Tensor, pooled: torch.Tensor, time: torch.Tensor,
                     text_rope_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        text_embeddings = getattr(self, f"{prefix}_text_embeddings")
        rope_embeddings = getattr(self, f"{prefix}_text_rope_embeddings")
        blocks = getattr(self, f"{prefix}_text_transformer_blocks")
        te = text_embeddings(text_embed)
        tm = self._time_embed(prefix, time, pooled)
        # Video and audio each own a text RoPE table sized to their own
        # head_dim (they can differ), so the position indices are looked up
        # per-tower rather than sharing one precomputed rope tensor.
        text_rope = rope_embeddings(text_rope_pos).unsqueeze(dim=0)
        for block in blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                te = torch.utils.checkpoint.checkpoint(block, te, tm, text_rope, use_reentrant=False)
            else:
                te = block(te, tm, text_rope)
        return te, tm

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        pooled_projections: torch.Tensor | None = None,
        hidden_states_audio: torch.Tensor | None = None,
        audio_timestep: torch.Tensor | None = None,
        visual_rope_pos: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | list[torch.Tensor] | None = None,
        audio_rope_pos: torch.Tensor | None = None,
        text_rope_pos: torch.Tensor | None = None,
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
        sparse_params: dict[str, Any] | None = None,
        visual_token_type_ids: torch.Tensor | None = None,
        va_gate_scale: float = 1.0,
        av_gate_scale: float = 1.0,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | Kandinsky6TransformerOutput:
        if pooled_projections is None:
            if encoder_hidden_states_image is None:
                raise ValueError("pooled_projections must be provided for Kandinsky6.")
            pooled_projections = encoder_hidden_states_image
        if visual_rope_pos is None or text_rope_pos is None:
            raise ValueError("visual_rope_pos and text_rope_pos are required for Kandinsky6.")
        if visual_token_type_ids is not None and self.visual_token_type_num_embeddings == 0:
            raise ValueError("visual_token_type_ids requires visual_token_type_num_embeddings > 0.")

        x_video = hidden_states
        x_audio = hidden_states_audio

        # A video-only call (x_audio is None) runs the fused blocks with aud=None.
        video_te, video_tm = self._encode_text("video", encoder_hidden_states, pooled_projections, timestep,
                                               text_rope_pos)

        visual_embed = self.visual_embeddings(x_video)
        if getattr(self, "visual_token_type_embeddings", None) is not None and visual_token_type_ids is not None:
            type_embed = self.visual_token_type_embeddings(visual_token_type_ids)
            visual_embed = visual_embed + type_embed[:, :, None, None, :]
        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(visual_shape, visual_rope_pos, scale_factor)
        # The NABLA sparse mask (inside the fused block's video self-attention) is built for
        # fractal token order, so flatten/unflatten with the same `to_fractal` flag -- a no-op
        # when to_fractal is False.
        to_fractal = sparse_params["to_fractal"] if sparse_params is not None else False
        visual_embed, visual_rope = fractal_flatten(visual_embed, visual_rope, visual_shape, block_mask=to_fractal)

        audio_te = audio_tm = audio_embed = audio_rope = None
        if x_audio is not None:
            audio_timestep = audio_timestep if audio_timestep is not None else timestep
            audio_rope_pos = audio_rope_pos if audio_rope_pos is not None else torch.arange(
                x_audio.shape[1], device=x_audio.device)
            audio_te, audio_tm = self._encode_text("audio", encoder_hidden_states, pooled_projections,
                                                   audio_timestep, text_rope_pos)
            audio_embed = self.audio_embeddings(x_audio)
            audio_rope = self.audio_rope_embeddings(audio_rope_pos).unsqueeze(dim=0)

        for block in self.visual_transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                visual_embed, audio_embed = torch.utils.checkpoint.checkpoint(
                    block,
                    visual_embed,
                    audio_embed,
                    video_te,
                    audio_te,
                    (video_tm, audio_tm),
                    visual_rope,
                    audio_rope,
                    sparse_params,
                    va_gate_scale,
                    av_gate_scale,
                    use_reentrant=False,
                )
            else:
                visual_embed, audio_embed = block(visual_embed, audio_embed, video_te, audio_te,
                                                  (video_tm, audio_tm), visual_rope, audio_rope, sparse_params,
                                                  va_gate_scale, av_gate_scale)

        visual_embed = fractal_unflatten(visual_embed, visual_shape, block_mask=to_fractal)
        video_out = self.out_layer(visual_embed, video_tm)
        result: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        if audio_embed is not None:
            audio_out = self.audio_out_layer(audio_embed, audio_tm)
            result = (video_out, audio_out)
        else:
            result = video_out

        if return_dict:
            return Kandinsky6TransformerOutput(sample=result)
        return result

    def materialize_non_persistent_buffers(self, device: torch.device, dtype: torch.dtype | None = None) -> None:
        for i, (axes_dim, ax_max_pos) in enumerate(
                zip(self.visual_rope_embeddings.axes_dims, self.visual_rope_embeddings.max_pos, strict=True)):
            name = f"args_{i}"
            buf = getattr(self.visual_rope_embeddings, name, None)
            if isinstance(buf, torch.Tensor) and buf.is_meta:
                freq = _build_rotary_freqs(axes_dim // 2, self.visual_rope_embeddings.max_period).to(device=device)
                pos = torch.arange(ax_max_pos, dtype=freq.dtype, device=device)
                self.visual_rope_embeddings._buffers[name] = torch.outer(pos, freq)

        rope1d_modules: list[Kandinsky6RoPE1D] = [
            self.video_text_rope_embeddings,
            self.audio_text_rope_embeddings,
            self.audio_rope_embeddings,
        ]
        time_embeds = [self.video_time_embeddings, self.audio_time_embeddings]

        for rope1d in rope1d_modules:
            if isinstance(rope1d.args, torch.Tensor) and rope1d.args.is_meta:
                freq = _build_rotary_freqs(rope1d.dim // 2, rope1d.max_period).to(device=device) * rope1d.freqs_scaling
                pos = torch.arange(rope1d.max_pos, dtype=freq.dtype, device=device)
                rope1d._buffers["args"] = torch.outer(pos, freq)

        for time_embed in time_embeds:
            if isinstance(time_embed.freqs, torch.Tensor) and time_embed.freqs.is_meta:
                time_embed.freqs = _build_rotary_freqs(time_embed.model_dim // 2,
                                                        time_embed.max_period).to(device=device)


EntryClass = Kandinsky6Transformer3DModel
