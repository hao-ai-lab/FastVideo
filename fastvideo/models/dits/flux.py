# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn

from diffusers.models.embeddings import apply_rotary_emb, get_1d_rotary_pos_embed

from fastvideo.attention import DistributedAttention
from fastvideo.configs.models import DiTConfig
from fastvideo.forward_context import get_forward_context, set_forward_context
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.visual_embedding import Timesteps
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.sd3 import (
    CombinedTimestepTextProjEmbeddings,
    SD3AdaLayerNormContinuous,
    SD3AdaLayerNormZero,
    SD3FeedForward,
    SD3TextProjection,
    SD3TimestepEmbedding,
)
from fastvideo.platforms import AttentionBackendEnum


@dataclass
class FluxTransformer2DModelOutput:
    sample: torch.Tensor


class FluxPosEmbed(nn.Module):
    """1D RoPE axes concatenated per Diffusers `FluxPosEmbed`."""

    def __init__(self, theta: int, axes_dim: list[int]) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_axes = ids.shape[-1]
        cos_out: list[torch.Tensor] = []
        sin_out: list[torch.Tensor] = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class FluxCombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, pooled_projection_dim: int) -> None:
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
        self.guidance_embedder = SD3TimestepEmbedding(
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
        guidance: torch.Tensor,
        pooled_projection: torch.Tensor,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))
        time_guidance_emb = timesteps_emb + guidance_emb
        pooled_projections = self.text_embedder(pooled_projection)
        return time_guidance_emb + pooled_projections


class FluxAdaLayerNormZeroSingle(nn.Module):
    def __init__(self, embedding_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = ReplicatedLinear(embedding_dim, 3 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb, _ = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


class FluxJointAttention(nn.Module):
    """Joint attention: text tokens precede image tokens (Diffusers order)."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
    ) -> None:
        super().__init__()
        self.heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim

        self.norm_q = nn.RMSNorm(attention_head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(attention_head_dim, eps=1e-6)
        self.norm_added_q = nn.RMSNorm(attention_head_dim, eps=1e-6)
        self.norm_added_k = nn.RMSNorm(attention_head_dim, eps=1e-6)

        self.to_q = ReplicatedLinear(dim, self.inner_dim, bias=True)
        self.to_k = ReplicatedLinear(dim, self.inner_dim, bias=True)
        self.to_v = ReplicatedLinear(dim, self.inner_dim, bias=True)
        self.add_q_proj = ReplicatedLinear(dim, self.inner_dim, bias=True)
        self.add_k_proj = ReplicatedLinear(dim, self.inner_dim, bias=True)
        self.add_v_proj = ReplicatedLinear(dim, self.inner_dim, bias=True)

        self.to_out = nn.ModuleList(
            [
                ReplicatedLinear(self.inner_dim, dim, bias=True),
                nn.Dropout(0.0),
            ]
        )
        self.to_add_out = ReplicatedLinear(self.inner_dim, dim, bias=True)

        self.attn = DistributedAttention(
            num_heads=num_attention_heads,
            head_size=attention_head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        text_seq_len = encoder_hidden_states.shape[1]
        img_seq_len = hidden_states.shape[1]

        q, _ = self.to_q(hidden_states)
        k, _ = self.to_k(hidden_states)
        v, _ = self.to_v(hidden_states)
        q = q.view(batch_size, img_seq_len, self.heads, self.head_dim)
        k = k.view(batch_size, img_seq_len, self.heads, self.head_dim)
        v = v.view(batch_size, img_seq_len, self.heads, self.head_dim)
        q = self.norm_q(q)
        k = self.norm_k(k)

        enc_q, _ = self.add_q_proj(encoder_hidden_states)
        enc_k, _ = self.add_k_proj(encoder_hidden_states)
        enc_v, _ = self.add_v_proj(encoder_hidden_states)
        enc_q = enc_q.view(batch_size, text_seq_len, self.heads, self.head_dim)
        enc_k = enc_k.view(batch_size, text_seq_len, self.heads, self.head_dim)
        enc_v = enc_v.view(batch_size, text_seq_len, self.heads, self.head_dim)
        enc_q = self.norm_added_q(enc_q)
        enc_k = self.norm_added_k(enc_k)

        q = torch.cat([enc_q, q], dim=1)
        k = torch.cat([enc_k, k], dim=1)
        v = torch.cat([enc_v, v], dim=1)

        q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
        k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)

        joint_out, _ = self.attn(q, k, v)
        joint_out = joint_out.reshape(batch_size, text_seq_len + img_seq_len, self.inner_dim)

        enc_out = joint_out[:, :text_seq_len]
        img_out = joint_out[:, text_seq_len:]

        img_out, _ = self.to_out[0](img_out)
        img_out = self.to_out[1](img_out)
        enc_out, _ = self.to_add_out(enc_out)
        return img_out, enc_out


class FluxSingleStreamAttention(nn.Module):
    """Self-attention on concatenated text+image sequence (single blocks)."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
    ) -> None:
        super().__init__()
        self.heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim

        self.norm_q = nn.RMSNorm(attention_head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(attention_head_dim, eps=1e-6)
        self.to_q = ReplicatedLinear(dim, self.inner_dim, bias=True)
        self.to_k = ReplicatedLinear(dim, self.inner_dim, bias=True)
        self.to_v = ReplicatedLinear(dim, self.inner_dim, bias=True)
        self.attn = DistributedAttention(
            num_heads=num_attention_heads,
            head_size=attention_head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        q, _ = self.to_q(hidden_states)
        k, _ = self.to_k(hidden_states)
        v, _ = self.to_v(hidden_states)
        q = q.view(batch_size, seq_len, self.heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.heads, self.head_dim)
        q = self.norm_q(q)
        k = self.norm_k(k)
        q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
        k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
        out, _ = self.attn(q, k, v)
        return out.reshape(batch_size, seq_len, self.inner_dim)


class FluxTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = SD3AdaLayerNormZero(dim)
        self.norm1_context = SD3AdaLayerNormZero(dim)
        self.attn = FluxJointAttention(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            supported_attention_backends=supported_attention_backends,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = SD3FeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu-approximate",
        )
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = SD3FeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu-approximate",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del joint_attention_kwargs
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        (norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp) = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + (c_gate_mlp.unsqueeze(1) * context_ff_output)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
    ) -> None:
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm = FluxAdaLayerNormZeroSingle(dim)
        self.proj_mlp = ReplicatedLinear(dim, mlp_hidden_dim, bias=True)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = ReplicatedLinear(dim + mlp_hidden_dim, dim, bias=True)
        self.attn = FluxSingleStreamAttention(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del joint_attention_kwargs
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states)[0])
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)[0]
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        encoder_hidden_states = hidden_states[:, :text_seq_len]
        hidden_states = hidden_states[:, text_seq_len:]
        return encoder_hidden_states, hidden_states


class FluxTransformer2DModel(BaseDiT):
    """FastVideo FLUX transformer; load Diffusers FLUX safetensors 1:1."""

    _fsdp_shard_conditions = [
        lambda n, m: (n.startswith("transformer_blocks.") or n.startswith("single_transformer_blocks."))
        and n.split(".")[-1].isdigit(),
    ]
    _compile_conditions = _fsdp_shard_conditions
    # HF weight names already match this module layout (cf. SGLang regex maps).
    param_names_mapping: dict[str, Any] = {}
    reverse_param_names_mapping: dict[str, Any] = {}
    lora_param_names_mapping: dict[str, Any] = {}
    _supported_attention_backends = (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    )

    def __init__(self, config: DiTConfig, hf_config: dict[str, Any], **kwargs) -> None:
        del kwargs
        super().__init__(config=config, hf_config=hf_config)
        self.fastvideo_config = config
        self.hf_config = hf_config
        arch = config.arch_config

        out_ch = arch.out_channels
        self.out_channels = out_ch if out_ch is not None else arch.in_channels
        self.inner_dim = arch.num_attention_heads * arch.attention_head_dim
        self.hidden_size = self.inner_dim
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.in_channels

        axes_list = list(arch.axes_dims_rope)
        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_list)
        if arch.guidance_embeds:
            self.time_text_embed = FluxCombinedTimestepGuidanceTextProjEmbeddings(
                embedding_dim=self.inner_dim,
                pooled_projection_dim=arch.pooled_projection_dim,
            )
        else:
            self.time_text_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim,
                pooled_projection_dim=arch.pooled_projection_dim,
            )
        self.context_embedder = ReplicatedLinear(arch.joint_attention_dim, self.inner_dim)
        self.x_embedder = ReplicatedLinear(arch.in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=arch.num_attention_heads,
                    attention_head_dim=arch.attention_head_dim,
                    supported_attention_backends=self._supported_attention_backends,
                )
                for _ in range(arch.num_layers)
            ]
        )
        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=arch.num_attention_heads,
                    attention_head_dim=arch.attention_head_dim,
                    supported_attention_backends=self._supported_attention_backends,
                )
                for _ in range(arch.num_single_layers)
            ]
        )

        self.norm_out = SD3AdaLayerNormContinuous(
            self.inner_dim,
            self.inner_dim,
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            norm_type="layer_norm",
        )
        self.proj_out = ReplicatedLinear(
            self.inner_dim,
            arch.patch_size * arch.patch_size * self.out_channels,
            bias=True,
        )
        self.gradient_checkpointing = False
        self.__post_init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        pooled_projections: torch.Tensor | None = None,
        timestep: torch.LongTensor | torch.Tensor | None = None,
        img_ids: torch.Tensor | None = None,
        txt_ids: torch.Tensor | None = None,
        guidance: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        controlnet_block_samples: Any | None = None,
        controlnet_single_block_samples: Any | None = None,
        controlnet_blocks_repeat: bool = False,
        **kwargs: Any,
    ) -> FluxTransformer2DModelOutput | tuple[torch.Tensor, ...]:
        del kwargs
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must be provided")
        if pooled_projections is None:
            raise ValueError("pooled_projections must be provided")
        if timestep is None:
            raise ValueError("timestep must be provided")
        if img_ids is None or txt_ids is None:
            raise ValueError("img_ids and txt_ids must be provided")

        arch = self.fastvideo_config.arch_config
        if arch.guidance_embeds and guidance is None:
            raise ValueError("guidance must be provided when guidance_embeds=True")

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
            ts0 = int(timestep[0].item()) if timestep.numel() > 0 else 0
            forward_context = set_forward_context(current_timestep=ts0, attn_metadata=None)

        with forward_context:
            hidden_states, _ = self.x_embedder(hidden_states)

            ts = timestep.to(hidden_states.dtype) * 1000
            g = None if guidance is None else guidance.to(hidden_states.dtype) * 1000

            if arch.guidance_embeds:
                assert g is not None
                temb = self.time_text_embed(ts, g, pooled_projections)
            else:
                temb = self.time_text_embed(timestep=ts, pooled_projection=pooled_projections)

            encoder_hidden_states, _ = self.context_embedder(encoder_hidden_states)

            if txt_ids.ndim == 3:
                txt_ids = txt_ids[0]
            if img_ids.ndim == 3:
                img_ids = img_ids[0]

            ids = torch.cat((txt_ids, img_ids), dim=0)
            image_rotary_emb = self.pos_embed(ids)

            jkwargs = joint_attention_kwargs or {}

            for idx, block in enumerate(self.transformer_blocks):
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=jkwargs,
                )
                if controlnet_block_samples is not None:
                    interval = len(self.transformer_blocks) / len(controlnet_block_samples)
                    interval = int(math.ceil(interval))
                    if controlnet_blocks_repeat:
                        hidden_states = hidden_states + controlnet_block_samples[idx % len(controlnet_block_samples)]
                    else:
                        hidden_states = hidden_states + controlnet_block_samples[idx // interval]

            for idx, block in enumerate(self.single_transformer_blocks):
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=jkwargs,
                )
                if controlnet_single_block_samples is not None:
                    interval = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                    interval = int(math.ceil(interval))
                    hidden_states = hidden_states + controlnet_single_block_samples[idx // interval]

            hidden_states = self.norm_out(hidden_states, temb)
            output, _ = self.proj_out(hidden_states)

            if not return_dict:
                return (output,)
            return FluxTransformer2DModelOutput(sample=output)


EntryClass = FluxTransformer2DModel
