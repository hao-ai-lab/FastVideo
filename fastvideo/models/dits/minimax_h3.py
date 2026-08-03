# SPDX-License-Identifier: Apache-2.0
"""FastVideo-native MiniMax H3 joint audio-video diffusion transformer."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.attention import DistributedAttention
from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3Config
from fastvideo.distributed.communication_op import (
    sequence_model_parallel_all_gather_with_unpad,
    sequence_model_parallel_shard,
)
from fastvideo.distributed.parallel_state import get_sp_world_size, model_parallel_is_initialized
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.quantization import QuantizationConfig
from fastvideo.layers.visual_embedding import Timesteps
from fastvideo.models.dits.base import BaseDiT
from fastvideo.platforms import AttentionBackendEnum


MINIMAX_H3_MODALITY_NUM = 3
_CFG = MiniMaxH3Config()


class MiniMaxH3RotaryPosEmbed(nn.Module):
    """Three-axis rotary frequencies over packed `(t, h, w)` coordinates."""

    def __init__(self, rope_freq_dim: int, rope_theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) / (2 * rope_freq_dim))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = position_ids.to(torch.float32)
        freqs = position_ids.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        freqs_t, freqs_h, freqs_w = freqs.unbind(dim=1)
        freqs = torch.cat((freqs_t, freqs_h, freqs_w), dim=-1)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos(), freqs.sin()


class MiniMaxH3TimestepEmbedding(nn.Module):
    """Linear-SiLU-linear timestep MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.fc_in = ReplicatedLinear(
            input_dim,
            hidden_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc_in",
        )
        self.fc_out = ReplicatedLinear(
            hidden_dim,
            output_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc_out",
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc_in(sample)
        hidden_states, _ = self.fc_out(F.silu(hidden_states))
        return hidden_states


class MiniMaxH3FeedForward(nn.Module):
    """Bias-free H3 SwiGLU with value-first packed halves."""

    def __init__(
        self,
        hidden_size: int,
        ffn_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.fc_in = ReplicatedLinear(
            hidden_size,
            2 * ffn_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc_in",
        )
        self.fc_out = ReplicatedLinear(
            ffn_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc_out",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc_in(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        hidden_states = hidden_states * F.silu(gate)
        hidden_states, _ = self.fc_out(hidden_states)
        return hidden_states


class MiniMaxH3Attention(nn.Module):
    """Full self-attention over one sequence-parallel packed document."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm_eps: float,
        supported_attention_backends: tuple[AttentionBackendEnum, ...],
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.to_q = ReplicatedLinear(
            hidden_size,
            inner_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_q",
        )
        self.to_k = ReplicatedLinear(
            hidden_size,
            inner_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_k",
        )
        self.to_v = ReplicatedLinear(
            hidden_size,
            inner_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_v",
        )
        self.norm_q = nn.RMSNorm(attention_head_dim, eps=qk_norm_eps)
        self.norm_k = nn.RMSNorm(attention_head_dim, eps=qk_norm_eps)
        self.to_out = ReplicatedLinear(
            inner_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_out",
        )
        self.distributed_attention = DistributedAttention(
            num_heads=num_attention_heads,
            head_size=attention_head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=prefix,
        )

    @staticmethod
    def _apply_rotary_emb(
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Rotate the RoPE prefix while preserving the rest of each attention head."""
        cos, sin = rotary_emb
        rotary_dim = cos.shape[-1]
        hidden_states_rotary = hidden_states[..., :rotary_dim]
        hidden_states_pass = hidden_states[..., rotary_dim:]
        cos = cos.to(hidden_states.dtype)[None, :, None, :]
        sin = sin.to(hidden_states.dtype)[None, :, None, :]
        first_half, second_half = hidden_states_rotary.chunk(2, dim=-1)
        hidden_states_rotated = torch.cat((-second_half, first_half), dim=-1)
        hidden_states_rotary = hidden_states_rotary * cos + hidden_states_rotated * sin
        return torch.cat((hidden_states_rotary, hidden_states_pass), dim=-1).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        original_seq_len: int,
    ) -> torch.Tensor:
        query, _ = self.to_q(hidden_states)
        key, _ = self.to_k(hidden_states)
        value, _ = self.to_v(hidden_states)
        query = query.unflatten(-1, (self.num_attention_heads, self.attention_head_dim))
        key = key.unflatten(-1, (self.num_attention_heads, self.attention_head_dim))
        value = value.unflatten(-1, (self.num_attention_heads, self.attention_head_dim))
        query = self.norm_q(query)
        key = self.norm_k(key)
        if rotary_emb is not None:
            query = self._apply_rotary_emb(query, rotary_emb)
            key = self._apply_rotary_emb(key, rotary_emb)

        # H3 rotates only 96/128 channels, which the generic `freqs_cis`
        # branch cannot express. Apply it above, then pass no RoPE here.
        hidden_states, _ = self.distributed_attention(
            query,
            key,
            value,
            original_seq_len=original_seq_len,
            freqs_cis=None,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states, _ = self.to_out(hidden_states)
        return hidden_states


class MiniMaxH3TokenRefinerBlock(nn.Module):
    """Plain pre-norm Transformer block for the projected text stream."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
        supported_attention_backends: tuple[AttentionBackendEnum, ...],
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size,
            num_attention_heads,
            attention_head_dim,
            qk_norm_eps,
            supported_attention_backends,
            quant_config,
            prefix=f"{prefix}.attn",
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = MiniMaxH3FeedForward(
            hidden_size,
            ffn_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.ff",
        )

    def forward(self, hidden_states: torch.Tensor, original_seq_len: int) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), None, original_seq_len)
        return hidden_states + self.ff(self.norm2(hidden_states))


class MiniMaxH3TokenRefiner(nn.Module):
    """Two-block text refiner used before packing the modalities."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        num_layers: int,
        norm_eps: float,
        qk_norm_eps: float,
        final_norm_eps: float,
        supported_attention_backends: tuple[AttentionBackendEnum, ...],
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.refiner_blocks = nn.ModuleList([
            MiniMaxH3TokenRefinerBlock(
                hidden_size,
                num_attention_heads,
                attention_head_dim,
                ffn_dim,
                norm_eps,
                qk_norm_eps,
                supported_attention_backends,
                quant_config,
                prefix=f"{prefix}.refiner_blocks.{index}",
            ) for index in range(num_layers)
        ])
        self.final_norm = nn.RMSNorm(hidden_size, eps=final_norm_eps)

    def forward(self, hidden_states: torch.Tensor, original_seq_len: int) -> torch.Tensor:
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, original_seq_len)
        return self.final_norm(hidden_states)


class MiniMaxH3AdaLayerNormModulation(nn.Module):
    """Produce six modulation tables for every `(timestep, modality)` pair."""

    def __init__(
        self,
        time_embed_dim: int,
        hidden_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = ReplicatedLinear(
            time_embed_dim,
            6 * hidden_size * MINIMAX_H3_MODALITY_NUM,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
        )

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        temb, _ = self.linear(F.silu(temb).to(self.linear.weight.dtype))
        return temb.view(-1, 6 * self.hidden_size).chunk(6, dim=-1)


class MiniMaxH3AdaLayerNormOut(nn.Module):
    """Final RMSNorm with per-timestep row modulation."""

    def __init__(
        self,
        hidden_size: int,
        time_embed_dim: int,
        eps: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.linear = ReplicatedLinear(
            time_embed_dim,
            2 * hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        shift_scale, _ = self.linear(F.silu(temb).to(self.linear.weight.dtype))
        shift, scale = shift_scale.chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )


class MiniMaxH3TransformerBlock(nn.Module):
    """Packed self-attention and feed-forward branches with row-indexed AdaLN."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        time_embed_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
        supported_attention_backends: tuple[AttentionBackendEnum, ...],
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size,
            num_attention_heads,
            attention_head_dim,
            qk_norm_eps,
            supported_attention_backends,
            quant_config,
            prefix=f"{prefix}.attn",
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = MiniMaxH3FeedForward(
            hidden_size,
            ffn_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.ff",
        )
        self.adaln_proj = MiniMaxH3AdaLayerNormModulation(
            time_embed_dim,
            hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.adaln_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)

        residual = hidden_states
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1.0 + scale_msa.index_select(0, adaln_indices)
        ) + shift_msa.index_select(0, adaln_indices)
        attention_output = self.attn(norm_hidden_states, rotary_emb, original_seq_len)
        hidden_states = residual + gate_msa.index_select(0, adaln_indices) * attention_output

        residual = hidden_states
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1.0 + scale_mlp.index_select(0, adaln_indices)
        ) + shift_mlp.index_select(0, adaln_indices)
        feed_forward_output = self.ff(norm_hidden_states)
        return residual + gate_mlp.index_select(0, adaln_indices) * feed_forward_output


class MiniMaxH3Transformer3DModel(BaseDiT):
    """Joint H3 Transformer over one padless text/audio/video document.

    The layout builder validates semantic rows before denoising. Sequence-
    parallel padding is transport-only and `DistributedAttention` trims it
    before attention.
    """

    _fsdp_shard_conditions = _CFG._fsdp_shard_conditions
    _compile_conditions = _CFG._compile_conditions
    _supported_attention_backends = _CFG._supported_attention_backends
    param_names_mapping = _CFG.param_names_mapping
    reverse_param_names_mapping = _CFG.reverse_param_names_mapping
    lora_param_names_mapping = _CFG.lora_param_names_mapping
    _keep_in_fp32_modules = frozenset({
        "proj_in",
        "audio_proj_in",
        "time_embedder",
        "proj_out",
        "audio_proj_out",
        "rope",
    })

    def _get_parameter_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        """Keep the released input, timestep, and output projections in FP32."""
        return torch.float32 if name.split(".", 1)[0] in self._keep_in_fp32_modules else default_dtype

    def __init__(self, config: MiniMaxH3Config, hf_config: dict[str, Any]) -> None:
        super().__init__(config, hf_config)
        arch = config.arch_config
        sp_world_size = get_sp_world_size() if model_parallel_is_initialized() else 1
        if arch.num_attention_heads % sp_world_size:
            raise ValueError(
                f"MiniMax H3 attention heads ({arch.num_attention_heads}) must be divisible by "
                f"sequence parallel size ({sp_world_size})."
            )

        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.in_channels
        self.patch_size = tuple(int(value) for value in arch.patch_size)
        video_patch_dim = arch.in_channels * math.prod(arch.patch_size)

        self.proj_in = ReplicatedLinear(
            video_patch_dim,
            arch.hidden_size,
            bias=True,
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.proj_in",
        )
        self.audio_proj_in = ReplicatedLinear(
            arch.audio_in_channels,
            arch.hidden_size,
            bias=True,
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.audio_proj_in",
        )
        self.context_embedder = ReplicatedLinear(
            arch.text_dim,
            arch.hidden_size,
            bias=True,
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.context_embedder",
        )
        self.time_proj = Timesteps(
            num_channels=arch.freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.time_embedder = MiniMaxH3TimestepEmbedding(
            arch.freq_dim,
            arch.time_embed_hidden_dim,
            arch.time_embed_dim,
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.time_embedder",
        )
        self.rope = MiniMaxH3RotaryPosEmbed(arch.rope_freq_dim, arch.rope_theta)
        self.token_refiner = MiniMaxH3TokenRefiner(
            arch.hidden_size,
            arch.num_attention_heads,
            arch.attention_head_dim,
            arch.ffn_dim,
            arch.num_refiner_layers,
            arch.norm_eps,
            arch.qk_norm_eps,
            arch.final_norm_eps,
            self.supported_attention_backends,
            config.quant_config,
            prefix=f"{config.prefix}.token_refiner",
        )
        self.transformer_blocks = nn.ModuleList([
            MiniMaxH3TransformerBlock(
                arch.hidden_size,
                arch.num_attention_heads,
                arch.attention_head_dim,
                arch.ffn_dim,
                arch.time_embed_dim,
                arch.norm_eps,
                arch.qk_norm_eps,
                self.supported_attention_backends,
                config.quant_config,
                prefix=f"{config.prefix}.transformer_blocks.{index}",
            ) for index in range(arch.num_layers)
        ])
        self.norm_out = MiniMaxH3AdaLayerNormOut(
            arch.hidden_size,
            arch.time_embed_dim,
            arch.final_norm_eps,
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.norm_out",
        )
        self.proj_out = ReplicatedLinear(
            arch.hidden_size,
            video_patch_dim,
            bias=True,
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.proj_out",
        )
        self.audio_proj_out = ReplicatedLinear(
            arch.hidden_size,
            arch.audio_in_channels,
            bias=True,
            quant_config=config.quant_config,
            prefix=f"{config.prefix}.audio_proj_out",
        )
        self.__post_init__()

    def materialize_non_persistent_buffers(
        self,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Rebuild analytic RoPE state after meta-device checkpoint loading."""
        del dtype
        if self.rope.inv_freq.is_meta:
            arch = self.config.arch_config
            inv_freq = 1.0 / (
                arch.rope_theta
                ** (
                    torch.arange(0, 2 * arch.rope_freq_dim, 2, device=device, dtype=torch.float32)
                    / (2 * arch.rope_freq_dim)
                )
            )
            self.rope._buffers["inv_freq"] = inv_freq

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict video and audio velocities from one caller-defined packed layout."""
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"position_ids must have shape (seq_len, 3), got {tuple(position_ids.shape)}.")
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError("token_tags and timestep_indices must both match the packed sequence length.")
        if hidden_states.shape[1] != video_indices.numel():
            raise ValueError("hidden_states row count must match video_indices.")
        if audio_hidden_states.shape[1] != audio_indices.numel():
            raise ValueError("audio_hidden_states row count must match audio_indices.")
        if encoder_hidden_states.shape[1] != text_indices.numel():
            raise ValueError("encoder_hidden_states row count must match text_indices.")

        rotary_emb = self.rope(position_ids)
        video_embeds, _ = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
        audio_embeds, _ = self.audio_proj_in(audio_hidden_states.to(self.audio_proj_in.weight.dtype))
        text_embeds, _ = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype))

        text_original_seq_len = text_embeds.shape[1]
        sp_world_size = get_sp_world_size() if model_parallel_is_initialized() else 1
        if sp_world_size > 1:
            text_embeds, _ = sequence_model_parallel_shard(text_embeds, dim=1)
        text_embeds = self.token_refiner(text_embeds, text_original_seq_len)
        if sp_world_size > 1:
            text_embeds = sequence_model_parallel_all_gather_with_unpad(
                text_embeds,
                text_original_seq_len,
                dim=1,
            )

        packed_hidden_states = text_embeds.new_zeros(
            (text_embeds.shape[0], sequence_length, text_embeds.shape[-1])
        )
        packed_hidden_states = packed_hidden_states.index_copy(1, text_indices, text_embeds)
        packed_hidden_states = packed_hidden_states.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        packed_hidden_states = packed_hidden_states.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(self.time_embedder.fc_in.weight.dtype))
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
        local_timestep_indices = timestep_indices
        original_seq_len = sequence_length

        if sp_world_size > 1:
            packed_hidden_states, _ = sequence_model_parallel_shard(packed_hidden_states, dim=1)
            rotary_cos, _ = sequence_model_parallel_shard(rotary_emb[0], dim=0)
            rotary_sin, _ = sequence_model_parallel_shard(rotary_emb[1], dim=0)
            adaln_indices, _ = sequence_model_parallel_shard(adaln_indices, dim=0)
            local_timestep_indices, _ = sequence_model_parallel_shard(local_timestep_indices, dim=0)
            rotary_emb = (rotary_cos, rotary_sin)

        for block in self.transformer_blocks:
            packed_hidden_states = block(
                packed_hidden_states,
                temb,
                adaln_indices,
                rotary_emb,
                original_seq_len,
            )

        packed_hidden_states = self.norm_out(
            packed_hidden_states,
            temb,
            local_timestep_indices,
        ).to(self.proj_out.weight.dtype)
        video_output, _ = self.proj_out(packed_hidden_states)
        audio_output, _ = self.audio_proj_out(packed_hidden_states)
        if sp_world_size > 1:
            video_output = sequence_model_parallel_all_gather_with_unpad(video_output, original_seq_len, dim=1)
            audio_output = sequence_model_parallel_all_gather_with_unpad(audio_output, original_seq_len, dim=1)
        video_output = video_output.index_select(1, video_indices)
        audio_output = audio_output.index_select(1, audio_indices)

        return video_output, audio_output


EntryClass = MiniMaxH3Transformer3DModel
