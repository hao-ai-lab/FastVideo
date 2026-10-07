# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2025 The Helios Team and The HuggingFace Team. All rights reserved.
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
"""Native FastVideo implementation of the Helios history-aware video DiT.

The architecture follows the Apache-2.0 Diffusers Helios implementation while
using FastVideo linear, attention, sharding, and model-loader boundaries.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.attention import DistributedAttention, LocalAttention
from fastvideo.configs.models.dits.helios import HeliosConfig
from fastvideo.distributed.communication_op import (
    sequence_model_parallel_all_gather_with_unpad,
    sequence_model_parallel_shard,
)
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.layers.layernorm import FP32LayerNorm, RMSNorm
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.visual_embedding import Timesteps
from fastvideo.models.dits.base import BaseDiT


def pad_for_3d_conv(value: torch.Tensor, kernel_size: tuple[int, int, int]) -> torch.Tensor:
    _, _, frames, height, width = value.shape
    patch_frames, patch_height, patch_width = kernel_size
    pad_frames = (patch_frames - frames % patch_frames) % patch_frames
    pad_height = (patch_height - height % patch_height) % patch_height
    pad_width = (patch_width - width % patch_width) % patch_width
    return F.pad(
        value,
        (0, pad_width, 0, pad_height, 0, pad_frames),
        mode="replicate",
    )


def center_down_sample_3d(value: torch.Tensor, kernel_size: tuple[int, int, int]) -> torch.Tensor:
    return F.avg_pool3d(value, kernel_size, stride=kernel_size)


def apply_rotary_emb_transposed(hidden_states: torch.Tensor, rotary_emb: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = rotary_emb.unsqueeze(-2).chunk(2, dim=-1)
    output = torch.empty_like(hidden_states)
    output[..., 0::2] = first * cos[..., 0::2] - second * sin[..., 1::2]
    output[..., 1::2] = first * sin[..., 1::2] + second * cos[..., 0::2]
    return output.type_as(hidden_states)


class HeliosOutputNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        self.norm = FP32LayerNorm(dim, eps=eps, elementwise_affine=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        original_context_length: int,
    ) -> torch.Tensor:
        temb = temb[:, -original_context_length:, :]
        shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2).to(hidden_states.device)
        scale = scale.squeeze(2).to(hidden_states.device)
        hidden_states = hidden_states[:, -original_context_length:, :]
        return (self.norm(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)


class HeliosAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        eps: float,
        *,
        is_cross_attention: bool,
        supported_attention_backends,
        quant_config=None,
        prefix: str = "",
        is_amplify_history: bool = False,
        history_scale_mode: str = "per_head",
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.is_cross_attention = is_cross_attention
        self.is_amplify_history = is_amplify_history
        self.history_scale_mode = history_scale_mode

        self.to_q = ReplicatedLinear(
            dim,
            self.inner_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.to_q",
        )
        self.to_k = ReplicatedLinear(
            dim,
            self.inner_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.to_k",
        )
        self.to_v = ReplicatedLinear(
            dim,
            self.inner_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.to_v",
        )
        self.to_out = nn.ModuleList([
            ReplicatedLinear(
                self.inner_dim,
                dim,
                quant_config=quant_config,
                prefix=f"{prefix}.to_out.0",
            ),
            nn.Dropout(0.0),
        ])
        self.norm_q = RMSNorm(self.inner_dim, eps=eps)
        self.norm_k = RMSNorm(self.inner_dim, eps=eps)

        attention_cls = LocalAttention
        if not is_cross_attention and get_sp_world_size() > 1:
            attention_cls = DistributedAttention
        self.attn = attention_cls(
            num_heads=heads,
            head_size=dim_head,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.impl",
        )

        if is_amplify_history:
            if history_scale_mode == "scalar":
                self.history_key_scale = nn.Parameter(torch.ones(1))
            elif history_scale_mode == "per_head":
                self.history_key_scale = nn.Parameter(torch.ones(heads))
            else:
                raise ValueError(f"Unknown history_scale_mode: {history_scale_mode}")
            self.max_scale = 10.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        rotary_emb: torch.Tensor | None = None,
        original_seq_len: int | None = None,
        original_context_length: int | None = None,
    ) -> torch.Tensor:
        key_value_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        query = self.to_q(hidden_states)[0]
        key = self.to_k(key_value_states)[0]
        value = self.to_v(key_value_states)[0]
        query = self.norm_q(query).unflatten(2, (self.heads, self.dim_head))
        key = self.norm_k(key).unflatten(2, (self.heads, self.dim_head))
        value = value.unflatten(2, (self.heads, self.dim_head))

        if rotary_emb is not None:
            query = apply_rotary_emb_transposed(query, rotary_emb)
            key = apply_rotary_emb_transposed(key, rotary_emb)

        if not self.is_cross_attention and self.is_amplify_history and original_context_length is not None:
            history_seq_len = hidden_states.shape[1] - original_context_length
            if history_seq_len > 0:
                scale = 1.0 + torch.sigmoid(self.history_key_scale) * (self.max_scale - 1.0)
                if self.history_scale_mode == "per_head":
                    scale = scale.view(1, 1, -1, 1)
                key = torch.cat(
                    [key[:, :history_seq_len] * scale, key[:, history_seq_len:]],
                    dim=1,
                )

        if isinstance(self.attn, DistributedAttention):
            output = self.attn(query, key, value, original_seq_len)[0]
        else:
            output = self.attn(query, key, value)
        output = output.flatten(2)
        output = self.to_out[0](output)[0]
        return self.to_out[1](output)


class HeliosTimestepEmbedding(nn.Module):

    def __init__(self, frequency_dim: int, dim: int, *, prefix: str = "") -> None:
        super().__init__()
        self.linear_1 = ReplicatedLinear(frequency_dim, dim, prefix=f"{prefix}.linear_1")
        self.act = nn.SiLU()
        self.linear_2 = ReplicatedLinear(dim, dim, prefix=f"{prefix}.linear_2")

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)[0]
        sample = self.act(sample)
        return self.linear_2(sample)[0]


class HeliosTextProjection(nn.Module):

    def __init__(self, input_dim: int, dim: int, *, prefix: str = "") -> None:
        super().__init__()
        self.linear_1 = ReplicatedLinear(input_dim, dim, prefix=f"{prefix}.linear_1")
        self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = ReplicatedLinear(dim, dim, prefix=f"{prefix}.linear_2")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(hidden_states)[0]
        hidden_states = self.act_1(hidden_states)
        return self.linear_2(hidden_states)[0]


class HeliosTimeTextEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.time_embedder = HeliosTimestepEmbedding(time_freq_dim, dim, prefix=f"{prefix}.time_embedder")
        self.act_fn = nn.SiLU()
        self.time_proj = ReplicatedLinear(dim, time_proj_dim, prefix=f"{prefix}.time_proj")
        self.text_embedder = HeliosTextProjection(text_embed_dim, dim, prefix=f"{prefix}.text_embedder")

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        is_return_encoder_hidden_states: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        timestep = self.timesteps_proj(timestep)
        timestep = timestep.to(self.time_embedder.linear_1.weight.dtype)
        temb = self.time_embedder(timestep)
        if encoder_hidden_states is not None:
            temb = temb.type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))[0]
        if encoder_hidden_states is not None and is_return_encoder_hidden_states:
            encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        return temb, timestep_proj, encoder_hidden_states


class HeliosRotaryPosEmbed(nn.Module):

    def __init__(self, rope_dim: tuple[int, int, int], theta: float) -> None:
        super().__init__()
        self.dim_t, self.dim_y, self.dim_x = rope_dim
        self.theta = theta
        self.register_buffer("freqs_base_t", self._get_freqs_base(self.dim_t), persistent=False)
        self.register_buffer("freqs_base_y", self._get_freqs_base(self.dim_y), persistent=False)
        self.register_buffer("freqs_base_x", self._get_freqs_base(self.dim_x), persistent=False)

    def _get_freqs_base(self, dim: int) -> torch.Tensor:
        exponent = torch.arange(0, dim, 2, dtype=torch.float32)[:dim // 2] / dim
        return 1.0 / self.theta**exponent

    def materialize_buffers(self, device: torch.device) -> None:
        self.freqs_base_t = self._get_freqs_base(self.dim_t).to(device)
        self.freqs_base_y = self._get_freqs_base(self.dim_y).to(device)
        self.freqs_base_x = self._get_freqs_base(self.dim_x).to(device)

    @staticmethod
    def _get_frequency_batched(freqs_base: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=positions.device.type, enabled=False):
            freqs = torch.einsum("d,bthw->dbthw", freqs_base, positions)
        freqs = freqs.repeat_interleave(2, dim=0)
        return freqs.cos(), freqs.sin()

    @torch.no_grad()
    def forward(
        self,
        frame_indices: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size, num_frames = frame_indices.shape
        frame_indices = frame_indices.to(device=device, dtype=torch.float32)
        y_coords = torch.arange(height, device=device, dtype=torch.float32)
        x_coords = torch.arange(width, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        grid_t = frame_indices[:, :, None, None].expand(batch_size, num_frames, height, width)
        grid_y = grid_y[None, None].expand(batch_size, num_frames, -1, -1)
        grid_x = grid_x[None, None].expand(batch_size, num_frames, -1, -1)
        cos_t, sin_t = self._get_frequency_batched(self.freqs_base_t, grid_t)
        cos_y, sin_y = self._get_frequency_batched(self.freqs_base_y, grid_y)
        cos_x, sin_x = self._get_frequency_batched(self.freqs_base_x, grid_x)
        result = torch.cat([cos_t, cos_y, cos_x, sin_t, sin_y, sin_x], dim=0)
        return result.permute(1, 0, 2, 3, 4)


class HeliosFeedForwardProject(nn.Module):

    def __init__(self, dim: int, ffn_dim: int, *, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.proj = ReplicatedLinear(dim, ffn_dim, quant_config=quant_config, prefix=f"{prefix}.proj")
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.gelu(self.proj(hidden_states)[0])


class HeliosFeedForward(nn.Module):

    def __init__(self, dim: int, ffn_dim: int, *, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.net = nn.ModuleList([
            HeliosFeedForwardProject(
                dim,
                ffn_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.net.0",
            ),
            nn.Dropout(0.0),
            ReplicatedLinear(
                ffn_dim,
                dim,
                quant_config=quant_config,
                prefix=f"{prefix}.net.2",
            ),
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.net[0](hidden_states)
        hidden_states = self.net[1](hidden_states)
        return self.net[2](hidden_states)[0]


class HeliosTransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str,
        cross_attn_norm: bool,
        eps: float,
        guidance_cross_attn: bool,
        supported_attention_backends,
        *,
        quant_config=None,
        prefix: str = "",
        is_amplify_history: bool = False,
        history_scale_mode: str = "per_head",
    ) -> None:
        super().__init__()
        if qk_norm != "rms_norm_across_heads":
            raise ValueError(f"Unsupported Helios qk_norm: {qk_norm}")
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = HeliosAttention(
            dim,
            num_heads,
            dim // num_heads,
            eps,
            is_cross_attention=False,
            supported_attention_backends=supported_attention_backends,
            quant_config=quant_config,
            prefix=f"{prefix}.attn1",
            is_amplify_history=is_amplify_history,
            history_scale_mode=history_scale_mode,
        )
        self.attn2 = HeliosAttention(
            dim,
            num_heads,
            dim // num_heads,
            eps,
            is_cross_attention=True,
            supported_attention_backends=supported_attention_backends,
            quant_config=quant_config,
            prefix=f"{prefix}.attn2",
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.ffn = HeliosFeedForward(dim, ffn_dim, quant_config=quant_config, prefix=f"{prefix}.ffn")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.guidance_cross_attn = guidance_cross_attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        current_token_mask: torch.Tensor,
        original_seq_len: int,
        original_context_length: int,
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            c_shift_msa,
            c_scale_msa,
            c_gate_msa,
        ) = (self.scale_shift_table.unsqueeze(0) + temb.float()).chunk(6, dim=2)
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        c_shift_msa = c_shift_msa.squeeze(2)
        c_scale_msa = c_scale_msa.squeeze(2)
        c_gate_msa = c_gate_msa.squeeze(2)

        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            rotary_emb=rotary_emb,
            original_seq_len=original_seq_len,
            original_context_length=(original_context_length if get_sp_world_size() == 1 else None),
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        if self.guidance_cross_attn:
            if get_sp_world_size() == 1:
                history_length = hidden_states.shape[1] - original_context_length
                history_hidden_states, current_hidden_states = hidden_states.split(
                    [history_length, original_context_length],
                    dim=1,
                )
                norm_hidden_states = self.norm2(current_hidden_states.float()).type_as(current_hidden_states)
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                )
                hidden_states = torch.cat(
                    [history_hidden_states, current_hidden_states + attn_output],
                    dim=1,
                )
            else:
                current_mask = current_token_mask.squeeze(-1).bool()
                current_counts = current_mask.sum(dim=1)
                if not torch.equal(current_counts, current_counts[:1].expand_as(current_counts)):
                    raise ValueError("Helios SP shards require equal current-token counts per batch item")
                current_count = int(current_counts[0].item())
                if current_count > 0:
                    current_hidden_states = hidden_states[current_mask].reshape(
                        hidden_states.shape[0],
                        current_count,
                        hidden_states.shape[-1],
                    )
                    norm_hidden_states = self.norm2(current_hidden_states.float()).type_as(current_hidden_states)
                    current_output = self.attn2(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                    )
                    attn_output = torch.zeros_like(hidden_states).masked_scatter(
                        current_mask.unsqueeze(-1).expand_as(hidden_states),
                        current_output,
                    )
                    hidden_states = hidden_states + attn_output
        else:
            norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
            hidden_states = hidden_states + attn_output

        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) +
                              c_shift_msa).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        return (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)


class HeliosTransformer3DModel(BaseDiT):
    _fsdp_shard_conditions = HeliosConfig()._fsdp_shard_conditions
    _compile_conditions = HeliosConfig()._compile_conditions
    _supported_attention_backends = HeliosConfig()._supported_attention_backends
    param_names_mapping = HeliosConfig().param_names_mapping
    reverse_param_names_mapping = HeliosConfig().reverse_param_names_mapping
    lora_param_names_mapping = HeliosConfig().lora_param_names_mapping

    def __init__(self, config: HeliosConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = tuple(config.patch_size)
        self.zero_history_timestep = config.zero_history_timestep
        self.quant_config = config.quant_config

        if config.num_attention_heads % get_sp_world_size() != 0:
            raise ValueError(f"Helios heads ({config.num_attention_heads}) must be divisible by "
                             f"sequence parallel size ({get_sp_world_size()})")
        inner_dim = config.hidden_size
        self.rope = HeliosRotaryPosEmbed(tuple(config.rope_dim), config.rope_theta)
        self.patch_embedding = nn.Conv3d(
            config.in_channels,
            inner_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        if config.has_multi_term_memory_patch:
            self.patch_short = nn.Conv3d(
                config.in_channels,
                inner_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            self.patch_mid = nn.Conv3d(
                config.in_channels,
                inner_dim,
                kernel_size=tuple(2 * value for value in self.patch_size),
                stride=tuple(2 * value for value in self.patch_size),
            )
            self.patch_long = nn.Conv3d(
                config.in_channels,
                inner_dim,
                kernel_size=tuple(4 * value for value in self.patch_size),
                stride=tuple(4 * value for value in self.patch_size),
            )
        self.condition_embedder = HeliosTimeTextEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=config.text_dim,
            prefix=f"{config.prefix}.condition_embedder",
        )
        self.blocks = nn.ModuleList([
            HeliosTransformerBlock(
                inner_dim,
                config.ffn_dim,
                config.num_attention_heads,
                config.qk_norm,
                config.cross_attn_norm,
                config.eps,
                config.guidance_cross_attn,
                self._supported_attention_backends,
                quant_config=config.quant_config,
                prefix=f"{config.prefix}.blocks.{index}",
                is_amplify_history=config.is_amplify_history,
                history_scale_mode=config.history_scale_mode,
            ) for index in range(config.num_layers)
        ])
        self.norm_out = HeliosOutputNorm(inner_dim, config.eps)
        self.proj_out = ReplicatedLinear(
            inner_dim,
            config.out_channels * math.prod(self.patch_size),
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.proj_out",
        )
        self.gradient_checkpointing = False
        self.__post_init__()

    def materialize_non_persistent_buffers(self, device: torch.device, dtype: torch.dtype | None = None) -> None:
        del dtype
        self.rope.materialize_buffers(device)

    @staticmethod
    def _validate_history_pair(
        history: torch.Tensor | None,
        history_indices: torch.Tensor | None,
        history_name: str,
        history_indices_name: str,
    ) -> None:
        if (history is None) != (history_indices is None):
            raise ValueError(f"{history_name} and {history_indices_name} must be provided together")

    def _patch_history(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor,
        history: torch.Tensor | None,
        history_indices: torch.Tensor | None,
        patch: nn.Conv3d,
        rope_height: int | None,
        rope_width: int | None,
        downsample: tuple[int, int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int | None, int | None]:
        if history is None or history_indices is None:
            return hidden_states, rotary_emb, rope_height, rope_width
        if downsample is not None:
            history = pad_for_3d_conv(history, tuple(patch.kernel_size))
        history = patch(history)
        _, _, _, patched_height, patched_width = history.shape
        history = history.flatten(2).transpose(1, 2)
        rope_height = patched_height if rope_height is None else rope_height
        rope_width = patched_width if rope_width is None else rope_width
        history_rotary = self.rope(
            history_indices,
            height=rope_height,
            width=rope_width,
            device=history.device,
        )
        if downsample is not None:
            history_rotary = pad_for_3d_conv(history_rotary, downsample)
            history_rotary = center_down_sample_3d(history_rotary, downsample)
        history_rotary = history_rotary.flatten(2).transpose(1, 2)
        return (
            torch.cat([history, hidden_states], dim=1),
            torch.cat([history_rotary, rotary_emb], dim=1),
            patched_height,
            patched_width,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        indices_hidden_states: torch.Tensor | None = None,
        indices_latents_history_short: torch.Tensor | None = None,
        indices_latents_history_mid: torch.Tensor | None = None,
        indices_latents_history_long: torch.Tensor | None = None,
        latents_history_short: torch.Tensor | None = None,
        latents_history_mid: torch.Tensor | None = None,
        latents_history_long: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        self._validate_history_pair(
            latents_history_short,
            indices_latents_history_short,
            "latents_history_short",
            "indices_latents_history_short",
        )
        self._validate_history_pair(
            latents_history_mid,
            indices_latents_history_mid,
            "latents_history_mid",
            "indices_latents_history_mid",
        )
        self._validate_history_pair(
            latents_history_long,
            indices_latents_history_long,
            "latents_history_long",
            "indices_latents_history_long",
        )
        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = encoder_hidden_states[0]
        batch_size = hidden_states.shape[0]
        patch_frames, patch_height, patch_width = self.patch_size

        hidden_states = self.patch_embedding(hidden_states)
        _, _, post_frames, post_height, post_width = hidden_states.shape
        if indices_hidden_states is None:
            indices_hidden_states = torch.arange(post_frames).unsqueeze(0).expand(batch_size, -1)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        rotary_emb = (self.rope(
            indices_hidden_states,
            height=post_height,
            width=post_width,
            device=hidden_states.device,
        ).flatten(2).transpose(1, 2))
        original_context_length = hidden_states.shape[1]

        hidden_states, rotary_emb, history_height, history_width = self._patch_history(
            hidden_states,
            rotary_emb,
            latents_history_short,
            indices_latents_history_short,
            self.patch_short,
            None,
            None,
        )
        if (latents_history_mid is not None or latents_history_long is not None) and (latents_history_short is None):
            raise ValueError("Helios mid/long history requires short history geometry")
        hidden_states, rotary_emb, _, _ = self._patch_history(
            hidden_states,
            rotary_emb,
            latents_history_mid,
            indices_latents_history_mid,
            self.patch_mid,
            history_height,
            history_width,
            downsample=(2, 2, 2),
        )
        hidden_states, rotary_emb, _, _ = self._patch_history(
            hidden_states,
            rotary_emb,
            latents_history_long,
            indices_latents_history_long,
            self.patch_long,
            history_height,
            history_width,
            downsample=(4, 4, 4),
        )

        history_context_length = hidden_states.shape[1] - original_context_length
        if self.zero_history_timestep:
            timestep_zero = torch.zeros(1, dtype=timestep.dtype, device=timestep.device)
            temb_zero, timestep_proj_zero, _ = self.condition_embedder(
                timestep_zero,
                encoder_hidden_states,
                is_return_encoder_hidden_states=False,
            )
            temb_zero = temb_zero.unsqueeze(1).expand(batch_size, history_context_length, -1)
            timestep_proj_zero = (timestep_proj_zero.unflatten(-1, (6, -1)).view(1, 6, 1, -1).expand(
                batch_size, -1, history_context_length, -1))

        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(timestep, encoder_hidden_states)
        assert encoder_hidden_states is not None
        timestep_proj = timestep_proj.unflatten(-1, (6, -1))
        temb = temb.view(batch_size, 1, -1).expand(batch_size, original_context_length, -1)
        timestep_proj = timestep_proj.view(batch_size, 6, 1, -1).expand(batch_size, 6, original_context_length, -1)
        if self.zero_history_timestep:
            temb = torch.cat([temb_zero, temb], dim=1)
            timestep_proj = torch.cat([timestep_proj_zero, timestep_proj], dim=2)
        timestep_proj = timestep_proj.permute(0, 2, 1, 3)

        current_token_mask = hidden_states.new_zeros(batch_size, hidden_states.shape[1], 1)
        current_token_mask[:, -original_context_length:] = 1
        original_seq_len = hidden_states.shape[1]
        if get_sp_world_size() > 1:
            hidden_states, original_seq_len = sequence_model_parallel_shard(hidden_states, dim=1)
            rotary_emb = sequence_model_parallel_shard(rotary_emb, dim=1)[0]
            timestep_proj = sequence_model_parallel_shard(timestep_proj, dim=1)[0]
            current_token_mask = sequence_model_parallel_shard(current_token_mask, dim=1)[0]

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                current_token_mask,
                original_seq_len,
                original_context_length,
            )

        if get_sp_world_size() > 1:
            hidden_states = sequence_model_parallel_all_gather_with_unpad(hidden_states, original_seq_len, dim=1)
        hidden_states = self.norm_out(hidden_states, temb, original_context_length)
        hidden_states = self.proj_out(hidden_states)[0]
        hidden_states = hidden_states.reshape(
            batch_size,
            post_frames,
            post_height,
            post_width,
            patch_frames,
            patch_height,
            patch_width,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


EntryClass = HeliosTransformer3DModel
