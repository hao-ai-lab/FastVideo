# SPDX-License-Identifier: Apache-2.0
"""FastVideo-native Z-Image transformer.

Z-Image attends over padded variable-length image/text streams and requires a
key-padding mask. FastVideo's distributed attention wrappers do not expose that
mask contract yet, so this implementation uses torch SDPA and is SP=1 only.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from fastvideo.configs.models.dits.zimage import ZImageDiTConfig
from fastvideo.distributed.parallel_state import get_sp_world_size, model_parallel_is_initialized
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.models.dits.base import BaseDiT
from fastvideo.platforms import AttentionBackendEnum


def _linear(layer: ReplicatedLinear, x: torch.Tensor) -> torch.Tensor:
    return layer(x)[0]


def _prepare_attention_mask(attention_mask: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim == 2:
        attention_mask = attention_mask[:, None, None, :]
    if attention_mask.dtype == torch.bool:
        additive_mask = torch.zeros_like(attention_mask, dtype=dtype)
        additive_mask.masked_fill_(~attention_mask, float("-inf"))
        return additive_mask
    return attention_mask


class TimestepEmbedder(nn.Module):

    def __init__(
        self,
        out_size: int,
        mid_size: int,
        frequency_embedding_size: int,
        max_period: int,
    ) -> None:
        super().__init__()
        self.mlp = nn.ModuleList([
            ReplicatedLinear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            ReplicatedLinear(mid_size, out_size, bias=True),
        ])
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, self.max_period)
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        return _linear(self.mlp[2], self.mlp[1](_linear(self.mlp[0], t_freq)))


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class FeedForward(nn.Module):

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = ReplicatedLinear(dim, hidden_dim, bias=False)
        self.w2 = ReplicatedLinear(hidden_dim, dim, bias=False)
        self.w3 = ReplicatedLinear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _linear(self.w2, F.silu(_linear(self.w1, x)) * _linear(self.w3, x))


def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        x_out = torch.view_as_real(x * freqs_cis.unsqueeze(2)).flatten(3)
        return x_out.type_as(x_in)


class ZImageAttention(nn.Module):

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, qk_norm: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads

        self.to_q = ReplicatedLinear(dim, n_heads * self.head_dim, bias=False)
        self.to_k = ReplicatedLinear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_v = ReplicatedLinear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_out = nn.ModuleList([ReplicatedLinear(n_heads * self.head_dim, dim, bias=False)])

        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else None
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = _linear(self.to_q, hidden_states).unflatten(-1, (self.n_heads, -1))
        key = _linear(self.to_k, hidden_states).unflatten(-1, (self.n_kv_heads, -1))
        value = _linear(self.to_v, hidden_states).unflatten(-1, (self.n_kv_heads, -1))

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)
        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        mask = _prepare_attention_mask(attention_mask, query.dtype)
        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).contiguous()
        return _linear(self.to_out[0], hidden_states.flatten(2, 3).to(query.dtype))


class ZImageTransformerBlock(nn.Module):

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        adaln_embed_dim: int,
        modulation: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.layer_id = layer_id
        self.modulation = modulation

        self.attention = ZImageAttention(dim, n_heads, n_kv_heads, qk_norm, norm_eps)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        if modulation:
            self.adaLN_modulation = nn.ModuleList(
                [ReplicatedLinear(min(dim, adaln_embed_dim), 4 * dim, bias=True)])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = _linear(
                self.adaLN_modulation[0], adaln_input).unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            attn_out = self.attention(
                self.attention_norm1(x),
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class FinalLayer(nn.Module):

    def __init__(self, hidden_size: int, out_channels: int, adaln_embed_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = ReplicatedLinear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.ModuleList([
            nn.SiLU(),
            ReplicatedLinear(min(hidden_size, adaln_embed_dim), hidden_size, bias=True),
        ])

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + _linear(self.adaLN_modulation[1], self.adaLN_modulation[0](c))
        return _linear(self.linear, self.norm_final(x) * scale.unsqueeze(1))


class RopeEmbedder:

    def __init__(self, theta: float, axes_dims: tuple[int, ...], axes_lens: tuple[int, ...]) -> None:
        if len(axes_dims) != len(axes_lens):
            raise ValueError("RoPE axes require matching dimensions and lengths")
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.freqs_cis: list[torch.Tensor] | None = None

    @staticmethod
    def precompute_freqs_cis(dim: tuple[int, ...], end: tuple[int, ...], theta: float) -> list[torch.Tensor]:
        with torch.device("cpu"):
            freqs_cis = []
            for axis_dim, axis_end in zip(dim, end):
                freqs = 1.0 / (theta**(torch.arange(0, axis_dim, 2, dtype=torch.float64) / axis_dim))
                timestep = torch.arange(axis_end, dtype=torch.float64)
                angles = torch.outer(timestep, freqs).float()
                freqs_cis.append(torch.polar(torch.ones_like(angles), angles).to(torch.complex64))
            return freqs_cis

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or ids.shape[-1] != len(self.axes_dims):
            raise ValueError("RoPE ids must have shape [sequence, number_of_axes]")
        if self.freqs_cis is None:
            self.freqs_cis = [
                freqs.to(ids.device)
                for freqs in self.precompute_freqs_cis(self.axes_dims, self.axes_lens, self.theta)
            ]
        elif self.freqs_cis[0].device != ids.device:
            self.freqs_cis = [freqs.to(ids.device) for freqs in self.freqs_cis]
        return torch.cat([self.freqs_cis[i][ids[:, i]] for i in range(len(self.axes_dims))], dim=-1)


class ZImageTransformer2DModel(BaseDiT):
    _default_config = ZImageDiTConfig()
    _fsdp_shard_conditions = _default_config.arch_config._fsdp_shard_conditions
    _compile_conditions = _default_config.arch_config._compile_conditions
    _supported_attention_backends = (AttentionBackendEnum.TORCH_SDPA, )
    param_names_mapping = _default_config.arch_config.param_names_mapping
    reverse_param_names_mapping = _default_config.arch_config.reverse_param_names_mapping

    def __init__(self, config: ZImageDiTConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)
        arch = config.arch_config

        self.in_channels = arch.in_channels
        self.out_channels = arch.in_channels
        self.all_patch_size = tuple(arch.all_patch_size)
        self.all_f_patch_size = tuple(arch.all_f_patch_size)
        self.dim = arch.dim
        self.n_heads = arch.n_heads
        self.rope_theta = arch.rope_theta
        self.t_scale = arch.t_scale
        self.seq_multi_of = arch.seq_multi_of

        self.hidden_size = arch.dim
        self.num_attention_heads = arch.n_heads
        self.num_channels_latents = arch.in_channels

        self.all_x_embedder = nn.ModuleDict({
            f"{patch_size}-{f_patch_size}": ReplicatedLinear(
                f_patch_size * patch_size * patch_size * arch.in_channels, arch.dim, bias=True)
            for patch_size, f_patch_size in zip(self.all_patch_size, self.all_f_patch_size)
        })
        self.all_final_layer = nn.ModuleDict({
            f"{patch_size}-{f_patch_size}": FinalLayer(
                arch.dim,
                patch_size * patch_size * f_patch_size * self.out_channels,
                arch.adaln_embed_dim,
            )
            for patch_size, f_patch_size in zip(self.all_patch_size, self.all_f_patch_size)
        })

        block_kwargs = {
            "dim": arch.dim,
            "n_heads": arch.n_heads,
            "n_kv_heads": arch.n_kv_heads,
            "norm_eps": arch.norm_eps,
            "qk_norm": arch.qk_norm,
            "adaln_embed_dim": arch.adaln_embed_dim,
        }
        self.noise_refiner = nn.ModuleList([
            ZImageTransformerBlock(1000 + layer_id, modulation=True, **block_kwargs)
            for layer_id in range(arch.n_refiner_layers)
        ])
        self.context_refiner = nn.ModuleList([
            ZImageTransformerBlock(layer_id, modulation=False, **block_kwargs)
            for layer_id in range(arch.n_refiner_layers)
        ])
        self.t_embedder = TimestepEmbedder(
            min(arch.dim, arch.adaln_embed_dim),
            mid_size=arch.timestep_mid_size,
            frequency_embedding_size=arch.frequency_embedding_size,
            max_period=arch.max_period,
        )
        self.cap_embedder = nn.ModuleList([
            RMSNorm(arch.cap_feat_dim, eps=arch.norm_eps),
            ReplicatedLinear(arch.cap_feat_dim, arch.dim, bias=True),
        ])
        self.x_pad_token = nn.Parameter(torch.empty((1, arch.dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, arch.dim)))
        self.layers = nn.ModuleList([
            ZImageTransformerBlock(layer_id, modulation=True, **block_kwargs) for layer_id in range(arch.n_layers)
        ])
        self.axes_dims = tuple(arch.axes_dims)
        self.axes_lens = tuple(arch.axes_lens)
        self.rope_embedder = RopeEmbedder(arch.rope_theta, self.axes_dims, self.axes_lens)
        self.__post_init__()

    def unpatchify(
        self,
        x: list[torch.Tensor],
        size: list[tuple[int, int, int]],
        patch_size: int,
        f_patch_size: int,
    ) -> list[torch.Tensor]:
        patch_height = patch_width = patch_size
        patch_frames = f_patch_size
        if len(x) != len(size):
            raise ValueError("output batch and original sizes must have equal length")
        for i, (frames, height, width) in enumerate(size):
            original_length = (frames // patch_frames) * (height // patch_height) * (width // patch_width)
            x[i] = (x[i][:original_length].view(
                frames // patch_frames,
                height // patch_height,
                width // patch_width,
                patch_frames,
                patch_height,
                patch_width,
                self.out_channels,
            ).permute(6, 0, 3, 1, 4, 2, 5).reshape(self.out_channels, frames, height, width))
        return x

    @staticmethod
    def create_coordinate_grid(
        size: tuple[int, int, int],
        start: tuple[int, int, int] | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        start = start or (0, ) * len(size)
        axes = [
            torch.arange(axis_start, axis_start + span, dtype=torch.int32, device=device)
            for axis_start, span in zip(start, size)
        ]
        return torch.stack(torch.meshgrid(axes, indexing="ij"), dim=-1)

    def patchify_and_embed(
        self,
        all_image: list[torch.Tensor],
        all_cap_feats: list[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[tuple[int, int, int]],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        patch_height = patch_width = patch_size
        patch_frames = f_patch_size
        device = all_image[0].device

        image_out = []
        image_sizes = []
        image_pos_ids = []
        image_pad_masks = []
        cap_pos_ids = []
        cap_pad_masks = []
        cap_feats_out = []

        for image, cap_feat in zip(all_image, all_cap_feats):
            cap_length = len(cap_feat)
            cap_padding = (-cap_length) % self.seq_multi_of
            cap_pos_ids.append(
                self.create_coordinate_grid(
                    (cap_length + cap_padding, 1, 1),
                    start=(1, 0, 0),
                    device=device,
                ).flatten(0, 2))
            cap_pad_masks.append(
                torch.cat([
                    torch.zeros(cap_length, dtype=torch.bool, device=device),
                    torch.ones(cap_padding, dtype=torch.bool, device=device),
                ]) if cap_padding else torch.zeros(cap_length, dtype=torch.bool, device=device))
            cap_feats_out.append(
                torch.cat([cap_feat, cap_feat[-1:].repeat(cap_padding, 1)]) if cap_padding else cap_feat)

            channels, frames, height, width = image.size()
            image_sizes.append((frames, height, width))
            frame_tokens, height_tokens, width_tokens = (
                frames // patch_frames,
                height // patch_height,
                width // patch_width,
            )
            image = image.view(
                channels,
                frame_tokens,
                patch_frames,
                height_tokens,
                patch_height,
                width_tokens,
                patch_width,
            )
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(
                frame_tokens * height_tokens * width_tokens,
                patch_frames * patch_height * patch_width * channels,
            )

            image_length = len(image)
            image_padding = (-image_length) % self.seq_multi_of
            original_pos_ids = self.create_coordinate_grid(
                (frame_tokens, height_tokens, width_tokens),
                start=(cap_length + cap_padding + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            if image_padding:
                padding_pos_ids = self.create_coordinate_grid((1, 1, 1), device=device).flatten(0, 2).repeat(
                    image_padding, 1)
                image_pos_ids.append(torch.cat([original_pos_ids, padding_pos_ids]))
            else:
                image_pos_ids.append(original_pos_ids)
            image_pad_masks.append(
                torch.cat([
                    torch.zeros(image_length, dtype=torch.bool, device=device),
                    torch.ones(image_padding, dtype=torch.bool, device=device),
                ]) if image_padding else torch.zeros(image_length, dtype=torch.bool, device=device))
            image_out.append(
                torch.cat([image, image[-1:].repeat(image_padding, 1)]) if image_padding else image)

        return (
            image_out,
            cap_feats_out,
            image_sizes,
            image_pos_ids,
            cap_pos_ids,
            image_pad_masks,
            cap_pad_masks,
        )

    @staticmethod
    def _attention_mask(lengths: list[int], device: torch.device) -> torch.Tensor:
        mask = torch.zeros((len(lengths), max(lengths)), dtype=torch.bool, device=device)
        for i, length in enumerate(lengths):
            mask[i, :length] = True
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        patch_size: int = 2,
        f_patch_size: int = 1,
        **kwargs,
    ) -> tuple[list[torch.Tensor], dict]:
        del encoder_hidden_states_image, guidance, kwargs
        if model_parallel_is_initialized() and get_sp_world_size() != 1:
            raise NotImplementedError(
                "Z-Image masked SDPA does not support sequence parallelism; run with sp_size=1")
        if patch_size not in self.all_patch_size or f_patch_size not in self.all_f_patch_size:
            raise ValueError(f"unsupported patch sizes: spatial={patch_size}, temporal={f_patch_size}")
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = list(hidden_states.unbind(0))
        if isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = list(encoder_hidden_states.unbind(0))

        device = hidden_states[0].device
        timestep_embedding = self.t_embedder(timestep * self.t_scale)
        (
            hidden_states,
            encoder_hidden_states,
            image_sizes,
            image_pos_ids,
            cap_pos_ids,
            image_inner_pad_masks,
            cap_inner_pad_masks,
        ) = self.patchify_and_embed(hidden_states, encoder_hidden_states, patch_size, f_patch_size)

        image_lengths = [len(item) for item in hidden_states]
        if not all(length % self.seq_multi_of == 0 for length in image_lengths):
            raise ValueError("padded image sequence lengths must be aligned")
        hidden_states = torch.cat(hidden_states)
        hidden_states = _linear(self.all_x_embedder[f"{patch_size}-{f_patch_size}"], hidden_states)
        adaln_input = timestep_embedding.type_as(hidden_states)
        hidden_states[torch.cat(image_inner_pad_masks)] = self.x_pad_token
        hidden_states = list(hidden_states.split(image_lengths))
        image_freqs_cis = list(
            self.rope_embedder(torch.cat(image_pos_ids)).split([len(item) for item in image_pos_ids]))
        hidden_states = pad_sequence(hidden_states, batch_first=True, padding_value=0.0)
        image_freqs_cis = pad_sequence(image_freqs_cis, batch_first=True, padding_value=0.0)
        image_freqs_cis = image_freqs_cis[:, :hidden_states.shape[1]]
        image_attn_mask = self._attention_mask(image_lengths, device)
        for layer in self.noise_refiner:
            hidden_states = layer(hidden_states, image_attn_mask, image_freqs_cis, adaln_input)

        cap_lengths = [len(item) for item in encoder_hidden_states]
        if not all(length % self.seq_multi_of == 0 for length in cap_lengths):
            raise ValueError("padded caption sequence lengths must be aligned")
        encoder_hidden_states = torch.cat(encoder_hidden_states)
        encoder_hidden_states = _linear(self.cap_embedder[1], self.cap_embedder[0](encoder_hidden_states))
        encoder_hidden_states[torch.cat(cap_inner_pad_masks)] = self.cap_pad_token
        encoder_hidden_states = list(encoder_hidden_states.split(cap_lengths))
        cap_freqs_cis = list(self.rope_embedder(torch.cat(cap_pos_ids)).split([len(item) for item in cap_pos_ids]))
        encoder_hidden_states = pad_sequence(encoder_hidden_states, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        cap_freqs_cis = cap_freqs_cis[:, :encoder_hidden_states.shape[1]]
        cap_attn_mask = self._attention_mask(cap_lengths, device)
        for layer in self.context_refiner:
            encoder_hidden_states = layer(encoder_hidden_states, cap_attn_mask, cap_freqs_cis)

        unified = []
        unified_freqs_cis = []
        for i, (image_length, cap_length) in enumerate(zip(image_lengths, cap_lengths)):
            unified.append(
                torch.cat([hidden_states[i][:image_length], encoder_hidden_states[i][:cap_length]]))
            unified_freqs_cis.append(
                torch.cat([image_freqs_cis[i][:image_length], cap_freqs_cis[i][:cap_length]]))
        unified_lengths = [image_length + cap_length for image_length, cap_length in zip(image_lengths, cap_lengths)]
        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        unified_attn_mask = self._attention_mask(unified_lengths, device)
        for layer in self.layers:
            unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        outputs = self.unpatchify(list(unified.unbind(0)), image_sizes, patch_size, f_patch_size)
        return outputs, {}


EntryClass = ZImageTransformer2DModel
