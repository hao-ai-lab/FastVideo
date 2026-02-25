# SPDX-License-Identifier: Apache-2.0
"""
GEN3C is a video-conditioned diffusion model that uses a 3D cache for camera control.

Key Features:
- Extends Cosmos 2.5 architecture with video conditioning inputs
- condition_video_input_mask: Binary mask indicating conditioning frames
- condition_video_pose: VAE-encoded 3D cache buffers (rendered warped images/masks)
- Augment sigma embedding for conditioning noise augmentation
- 3D RoPE with learnable per-axis positional embeddings

Reference: https://arxiv.org/abs/2503.03751
"""

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from fastvideo.attention import DistributedAttention, LocalAttention
from fastvideo.configs.models.dits.gen3c import Gen3CVideoConfig
from fastvideo.distributed.communication_op import (
    sequence_model_parallel_all_gather_with_unpad,
    sequence_model_parallel_shard,
)
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.distributed.utils import create_attention_mask_for_padding
from fastvideo.forward_context import get_forward_context
from fastvideo.layers.layernorm import RMSNorm
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import apply_rotary_emb
from fastvideo.layers.visual_embedding import Timesteps
from fastvideo.models.dits.base import BaseDiT
from fastvideo.platforms import AttentionBackendEnum


class Gen3CPatchEmbed(nn.Module):
    """
    GEN3C patch embedding - converts video (B, C, T, H, W) to patches (B, T', H', W', D).
    Uses linear projection after rearranging patches.
    
    Input channels include:
    - VAE latent (16 channels)
    - condition_video_input_mask (1 channel)
    - condition_video_pose (frame_buffer_max * 32 channels)
    - padding_mask (1 channel, if concat_padding_mask=True)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]

        self.proj = nn.Linear(self.dim, out_channels, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, C, T, H, W)
        Returns:
            (B, T', H', W', D) where T'=T//pt, H'=H//ph, W'=W//pw
        """
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size

        # Rearrange: b c (t pt) (h ph) (w pw) -> b t h w (c pt ph pw)
        hidden_states = hidden_states.reshape(
            batch_size, num_channels,
            num_frames // p_t, p_t,
            height // p_h, p_h,
            width // p_w, p_w
        )
        hidden_states = hidden_states.permute(0, 2, 4, 6, 1, 3, 5, 7)
        hidden_states = hidden_states.flatten(4, 7)  # Flatten patch dimensions
        
        # Project to model dimension
        hidden_states = self.proj(hidden_states)
        return hidden_states


class Gen3CTimestepEmbedding(nn.Module):
    """
    GEN3C timestep embedding with AdaLN-LoRA support.
    Generates both standard embedding and AdaLN-LoRA parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_adaln_lora: bool = True,
        adaln_lora_dim: int = 256,
    ) -> None:
        super().__init__()
        self.use_adaln_lora = use_adaln_lora
        
        self.linear_1 = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.SiLU()
        
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False)

    def forward(self, sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            emb: Standard embedding (B, D) - the original sinusoidal input
            adaln_lora: AdaLN-LoRA parameters (B, 3D) or None

        Note: When use_adaln_lora=True, the standard embedding is the INPUT
        (sinusoidal timestep embedding), not the processed output. The processed
        output (linear_2) is used exclusively for AdaLN-LoRA parameters.
        This matches the official GEN3C implementation.
        """
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora = emb  # (B, 3D) - full processed embedding for LoRA
            emb_standard = sample  # (B, D) - original input as the standard embedding
        else:
            emb_standard = emb
            adaln_lora = None

        return emb_standard, adaln_lora


class Gen3CEmbedding(nn.Module):
    """
    GEN3C timestep conditioning embedding.
    Generates sinusoidal embeddings and processes them through MLP.
    """

    def __init__(
        self,
        embedding_dim: int,
        condition_dim: int,
        use_adaln_lora: bool = True,
        adaln_lora_dim: int = 256,
    ) -> None:
        super().__init__()

        self.time_proj = Timesteps(embedding_dim, flip_sin_to_cos=True, downscale_freq_shift=0.0)
        self.t_embedder = Gen3CTimestepEmbedding(
            embedding_dim,
            condition_dim,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
        )

    def forward(
        self,
        timestep: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            timestep: (B,) tensor of timesteps
            dtype: Target dtype for embeddings
            
        Returns:
            embedded_timestep: Timestep embedding (B, D)
            adaln_lora: AdaLN-LoRA parameters (B, 3D) or None
        """
        # timestep should be 1D (B,)
        timestep = timestep.flatten()
        timesteps_proj = self.time_proj(timestep).to(dtype)  # (B, D)
        embedded_timestep, adaln_lora = self.t_embedder(timesteps_proj)
        
        return embedded_timestep, adaln_lora


class Gen3CAdaLayerNormZero(nn.Module):
    """
    GEN3C Adaptive Layer Normalization with zero initialization and gate.
    This is a simplified version that expects pre-computed shift/scale/gate parameters.
    """

    def __init__(
        self,
        in_features: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor
            shift: Shift parameter for modulation
            scale: Scale parameter for modulation
            
        Returns:
            normalized_hidden_states: Modulated normalized hidden states
        """
        # Apply layer norm and modulation
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states


class Gen3CSelfAttention(nn.Module):
    """
    GEN3C self-attention with QK normalization and RoPE.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
        supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
    ) -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()

        if supported_attention_backends is None:
            supported_attention_backends = (AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA)
        
        self.attn = DistributedAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix="self_attn"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, S, D) where S = T*H*W
            rope_emb: Tuple of (cos, sin) for RoPE
        """
        # Get QKV
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # Reshape for multi-head attention: (B, S, D) -> (B, S, H, D_h) -> (B, H, S, D_h)
        query = query.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        key = key.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        value = value.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)

        # Apply QK normalization
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Apply RoPE if provided (query/key are now in (B, H, S, D_h) format)
        if rope_emb is not None:
            cos, sin = rope_emb
            query = apply_rotary_emb(query, (cos, sin), use_real=True, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, (cos, sin), use_real=True, use_real_unbind_dim=-2)

        # Attention computation
        query = query.transpose(1, 2)  # (B, H, S, D_h) -> (B, S, H, D_h)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        attn_output, _ = self.attn(query, key, value, attention_mask=attention_mask)
        attn_output = attn_output.flatten(-2, -1)  # (B, S, H*D_h)

        # Output projection
        attn_output = self.to_out(attn_output)
        return attn_output


class Gen3CCrossAttention(nn.Module):
    """
    GEN3C cross-attention for text conditioning.
    """

    def __init__(
        self,
        dim: int,
        cross_attention_dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
        supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
    ) -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.cross_attention_dim = cross_attention_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()

        if supported_attention_backends is None:
            supported_attention_backends = (AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA)
        
        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, S, D)
            encoder_hidden_states: (B, N, D_text)
        """
        # Get QKV
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        # Reshape for multi-head attention
        query = query.unflatten(-1, (self.num_heads, self.head_dim))
        key = key.unflatten(-1, (self.num_heads, self.head_dim))
        value = value.unflatten(-1, (self.num_heads, self.head_dim))

        # Apply QK normalization
        query = self.norm_q(query)
        key = self.norm_k(key)

        attn_output = self.attn(query, key, value)
        attn_output = attn_output.flatten(-2, -1)

        # Output projection
        attn_output = self.to_out(attn_output)
        return attn_output


class Gen3CTransformerBlock(nn.Module):
    """
    GEN3C transformer block with self-attention, cross-attention, and MLP.
    Uses AdaLN-LoRA for conditioning.
    """

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        mlp_ratio: float = 4.0,
        adaln_lora_dim: int = 256,
        use_adaln_lora: bool = True,
        qk_norm: bool = True,
        supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim
        self.use_adaln_lora = use_adaln_lora

        # Layer norms
        self.norm1 = Gen3CAdaLayerNormZero(hidden_size)
        self.norm2 = Gen3CAdaLayerNormZero(hidden_size)
        self.norm3 = Gen3CAdaLayerNormZero(hidden_size)

        # Attention and MLP layers
        self.attn1 = Gen3CSelfAttention(
            dim=hidden_size,
            num_heads=num_attention_heads,
            qk_norm=qk_norm,
            supported_attention_backends=supported_attention_backends,
        )
        self.attn2 = Gen3CCrossAttention(
            dim=hidden_size,
            cross_attention_dim=cross_attention_dim,
            num_heads=num_attention_heads,
            qk_norm=qk_norm,
            supported_attention_backends=supported_attention_backends,
        )
        self.mlp = MLP(hidden_size, int(hidden_size * mlp_ratio), act_type="gelu", bias=False)

        # AdaLN modulation layers
        if use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * hidden_size, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * hidden_size, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=False)
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=False)
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=False)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        affine_emb: torch.Tensor,
        adaln_lora: torch.Tensor | None = None,
        rope_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        extra_pos_emb: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, S, D) where S = T*H*W (sequence may be sharded with SP)
            encoder_hidden_states: (B, N, D_text)
            affine_emb: (B, D) affine embedding
            adaln_lora: (B, 3D) AdaLN-LoRA parameters
            rope_emb: Tuple of (cos, sin) for RoPE
            extra_pos_emb: Optional learnable positional embeddings (B, S, D)
        """
        # Add extra positional embeddings if provided
        if extra_pos_emb is not None:
            hidden_states = hidden_states + extra_pos_emb

        B, S, D = hidden_states.shape

        # Compute modulation parameters
        if self.use_adaln_lora and adaln_lora is not None:
            shift_self_attn, scale_self_attn, gate_self_attn = (
                self.adaln_modulation_self_attn(affine_emb) + adaln_lora
            ).chunk(3, dim=-1)
            shift_cross_attn, scale_cross_attn, gate_cross_attn = (
                self.adaln_modulation_cross_attn(affine_emb) + adaln_lora
            ).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = (
                self.adaln_modulation_mlp(affine_emb) + adaln_lora
            ).chunk(3, dim=-1)
        else:
            shift_self_attn, scale_self_attn, gate_self_attn = self.adaln_modulation_self_attn(
                affine_emb
            ).chunk(3, dim=-1)
            shift_cross_attn, scale_cross_attn, gate_cross_attn = self.adaln_modulation_cross_attn(
                affine_emb
            ).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(affine_emb).chunk(3, dim=-1)

        # Reshape modulation parameters for broadcasting: (B, D) -> (B, 1, D)
        shift_self_attn = shift_self_attn.unsqueeze(1).type_as(hidden_states)
        scale_self_attn = scale_self_attn.unsqueeze(1).type_as(hidden_states)
        gate_self_attn = gate_self_attn.unsqueeze(1).type_as(hidden_states)

        shift_cross_attn = shift_cross_attn.unsqueeze(1).type_as(hidden_states)
        scale_cross_attn = scale_cross_attn.unsqueeze(1).type_as(hidden_states)
        gate_cross_attn = gate_cross_attn.unsqueeze(1).type_as(hidden_states)

        shift_mlp = shift_mlp.unsqueeze(1).type_as(hidden_states)
        scale_mlp = scale_mlp.unsqueeze(1).type_as(hidden_states)
        gate_mlp = gate_mlp.unsqueeze(1).type_as(hidden_states)

        # Self-attention block
        norm_hidden_states = self.norm1(hidden_states, shift_self_attn, scale_self_attn)

        attn_output = self.attn1(
            norm_hidden_states,
            rope_emb=rope_emb,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + gate_self_attn * attn_output

        # Cross-attention block
        norm_hidden_states = self.norm2(hidden_states, shift_cross_attn, scale_cross_attn)

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + gate_cross_attn * attn_output

        # MLP block
        norm_hidden_states = self.norm3(hidden_states, shift_mlp, scale_mlp)

        mlp_output = self.mlp(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * mlp_output

        return hidden_states


class Gen3CRotaryPosEmbed(nn.Module):
    """
    GEN3C 3D Rotary Position Embedding with NTK-aware extrapolation.
    """

    def __init__(
        self,
        hidden_size: int,
        max_size: tuple[int, int, int] = (128, 240, 240),
        patch_size: tuple[int, int, int] = (1, 2, 2),
        base_fps: int = 24,
        rope_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        enable_fps_modulation: bool = True,
    ) -> None:
        super().__init__()

        self.max_size = [size // patch for size, patch in zip(max_size, patch_size, strict=True)]
        self.patch_size = patch_size
        self.base_fps = base_fps
        self.enable_fps_modulation = enable_fps_modulation

        # Split dimensions: 1/3 for T, 1/3 for H, 1/3 for W
        self.dim_h = hidden_size // 6 * 2
        self.dim_w = hidden_size // 6 * 2
        self.dim_t = hidden_size - self.dim_h - self.dim_w

        # NTK-aware extrapolation factors
        self.h_ntk_factor = rope_scale[1] ** (self.dim_h / (self.dim_h - 2))
        self.w_ntk_factor = rope_scale[2] ** (self.dim_w / (self.dim_w - 2))
        self.t_ntk_factor = rope_scale[0] ** (self.dim_t / (self.dim_t - 2))

    def forward(
        self, hidden_states: torch.Tensor, fps: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate 3D RoPE embeddings.
        """
        batch_size, T, H, W, input_dim = hidden_states.shape
        device = hidden_states.device

        # Generate frequency scales with NTK
        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor

        seq = torch.arange(max(self.max_size), device=device, dtype=torch.float32)

        dim_h_range = torch.arange(0, self.dim_h, 2, device=device, dtype=torch.float32)[: (self.dim_h // 2)] / self.dim_h
        dim_w_range = torch.arange(0, self.dim_w, 2, device=device, dtype=torch.float32)[: (self.dim_w // 2)] / self.dim_w
        dim_t_range = torch.arange(0, self.dim_t, 2, device=device, dtype=torch.float32)[: (self.dim_t // 2)] / self.dim_t

        h_spatial_freqs = 1.0 / (h_theta ** dim_h_range)
        w_spatial_freqs = 1.0 / (w_theta ** dim_w_range)
        temporal_freqs = 1.0 / (t_theta ** dim_t_range)

        # Generate positional embeddings
        half_emb_h = torch.outer(seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(seq[:W], w_spatial_freqs)

        if self.enable_fps_modulation and fps is not None:
            half_emb_t = torch.outer(seq[:T] / fps * self.base_fps, temporal_freqs)
        else:
            half_emb_t = torch.outer(seq[:T], temporal_freqs)

        # Broadcast and concatenate embeddings
        emb_t = half_emb_t[:, None, None, :].repeat(1, H, W, 1)
        emb_h = half_emb_h[None, :, None, :].repeat(T, 1, W, 1)
        emb_w = half_emb_w[None, None, :, :].repeat(T, H, 1, 1)

        # Concatenate [t, h, w, t, h, w] for sin/cos pairs
        freqs = torch.cat([emb_t, emb_h, emb_w] * 2, dim=-1)
        freqs = freqs.flatten(0, 2).float()  # (THW, D)

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        return cos, sin


class Gen3CLearnablePositionalEmbed(nn.Module):
    """
    GEN3C learnable absolute positional embeddings.
    """

    def __init__(
        self,
        hidden_size: int,
        max_size: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.max_size = [size // patch for size, patch in zip(max_size, patch_size, strict=True)]
        self.patch_size = patch_size
        self.eps = eps

        self.pos_emb_t = nn.Parameter(torch.zeros(self.max_size[0], hidden_size))
        self.pos_emb_h = nn.Parameter(torch.zeros(self.max_size[1], hidden_size))
        self.pos_emb_w = nn.Parameter(torch.zeros(self.max_size[2], hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, H, W, D)
        Returns:
            pos_emb: (B, T, H, W, D)
        """
        B, T, H, W, D = hidden_states.shape

        emb_t = self.pos_emb_t[:T][None, :, None, None, :].repeat(B, 1, H, W, 1)
        emb_h = self.pos_emb_h[:H][None, None, :, None, :].repeat(B, T, 1, W, 1)
        emb_w = self.pos_emb_w[:W][None, None, None, :, :].repeat(B, T, H, 1, 1)

        emb = emb_t + emb_h + emb_w

        # Normalize
        norm = torch.linalg.vector_norm(emb, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(self.eps, norm, alpha=np.sqrt(norm.numel() / emb.numel()))
        return (emb / norm).type_as(hidden_states)


class Gen3CFinalLayer(nn.Module):
    """
    GEN3C final layer with AdaLN modulation and unpatchification.
    """

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        patch_size: tuple[int, int, int],
        adaln_lora_dim: int = 256,
        use_adaln_lora: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.use_adaln_lora = use_adaln_lora

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # AdaLN modulation
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 2 * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=False),
            )

        # Output projection
        output_dim = out_channels * patch_size[0] * patch_size[1] * patch_size[2]
        self.proj_out = nn.Linear(hidden_size, output_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        affine_emb: torch.Tensor,
        adaln_lora: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, H, W, D)
            affine_emb: (B, D)
            adaln_lora: (B, 3D) or None
        """
        # Generate modulation parameters
        modulation = self.adaln_modulation(affine_emb)

        if self.use_adaln_lora and adaln_lora is not None:
            modulation = modulation + adaln_lora[..., : 2 * self.hidden_size]

        shift, scale = modulation.chunk(2, dim=-1)

        # Apply normalization and modulation
        hidden_states = self.norm(hidden_states)

        # Broadcast over all non-channel dimensions. Supports both:
        # - (B, T, H, W, D)
        # - (B, S, D) when sequence-parallel path is active.
        mod_shape = [hidden_states.shape[0]] + [1] * (hidden_states.dim() - 2) + [self.hidden_size]
        shift = shift.view(*mod_shape)
        scale = scale.view(*mod_shape)

        hidden_states = hidden_states * (1 + scale) + shift

        # Project to output
        hidden_states = self.proj_out(hidden_states)

        return hidden_states


class Gen3CTransformer3DModel(BaseDiT):
    """
    GEN3C DiT - Video-conditioned diffusion transformer with 3D cache support.
    
    Key features:
    - AdaLN-LoRA conditioning
    - 3D RoPE with NTK-aware extrapolation
    - Learnable positional embeddings
    - QK normalization
    - Video conditioning with 3D cache buffers
    - Augment sigma embedding for conditioning noise augmentation
    """
    
    _fsdp_shard_conditions = Gen3CVideoConfig()._fsdp_shard_conditions
    _compile_conditions = Gen3CVideoConfig()._compile_conditions
    param_names_mapping = Gen3CVideoConfig().param_names_mapping
    lora_param_names_mapping = Gen3CVideoConfig().lora_param_names_mapping

    def __init__(self, config: Gen3CVideoConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = inner_dim
        self.num_attention_heads = config.num_attention_heads
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.max_size = config.max_size
        self.rope_scale = config.rope_scale
        self.concat_padding_mask = config.concat_padding_mask
        self.use_adaln_lora = getattr(config, "use_adaln_lora", True)
        self.adaln_lora_dim = getattr(config, "adaln_lora_dim", 256)
        self.extra_pos_embed_type = getattr(config, "extra_pos_embed_type", "learnable")
        self.affine_emb_norm = getattr(config, "affine_emb_norm", True)
        
        # GEN3C-specific
        self.frame_buffer_max = getattr(config, "frame_buffer_max", 2)
        self.buffer_channels = self.frame_buffer_max * 32  # Each buffer: 16 (frame) + 16 (mask)
        self.add_augment_sigma_embedding = getattr(config, "add_augment_sigma_embedding", True)

        # 1. Patch Embedding
        # Input channels: VAE (16) + condition_mask (1) + pose_buffers (64 for 2 buffers) + padding (1)
        patch_embed_in_channels = config.in_channels + 1 + self.buffer_channels
        if config.concat_padding_mask:
            patch_embed_in_channels += 1
        
        self.patch_embed = Gen3CPatchEmbed(
            patch_embed_in_channels, inner_dim, config.patch_size
        )

        # 2. Positional Embeddings
        self.rope = Gen3CRotaryPosEmbed(
            hidden_size=config.attention_head_dim,
            max_size=config.max_size,
            patch_size=config.patch_size,
            rope_scale=config.rope_scale,
            enable_fps_modulation=getattr(config, "rope_enable_fps_modulation", True),
        )

        self.learnable_pos_embed = None
        if self.extra_pos_embed_type == "learnable":
            self.learnable_pos_embed = Gen3CLearnablePositionalEmbed(
                hidden_size=inner_dim,
                max_size=config.max_size,
                patch_size=config.patch_size,
            )

        # 3. Time Embedding
        self.time_embed = Gen3CEmbedding(
            inner_dim,
            inner_dim,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
        )

        # 4. Augment Sigma Embedding (GEN3C-specific)
        if self.add_augment_sigma_embedding:
            self.augment_sigma_embed = Gen3CEmbedding(
                inner_dim,
                inner_dim,
                use_adaln_lora=self.use_adaln_lora,
                adaln_lora_dim=self.adaln_lora_dim,
            )

        # 5. Affine Embedding Normalization
        if self.affine_emb_norm:
            self.affine_norm = RMSNorm(inner_dim, eps=config.eps)
        else:
            self.affine_norm = nn.Identity()

        # 6. Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            Gen3CTransformerBlock(
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                cross_attention_dim=config.text_embed_dim,
                mlp_ratio=config.mlp_ratio,
                adaln_lora_dim=self.adaln_lora_dim,
                use_adaln_lora=self.use_adaln_lora,
                qk_norm=(config.qk_norm == "rms_norm"),
                supported_attention_backends=config._supported_attention_backends,
            )
            for i in range(config.num_layers)
        ])

        # 7. Final Layer
        self.final_layer = Gen3CFinalLayer(
            hidden_size=inner_dim,
            out_channels=config.out_channels,
            patch_size=config.patch_size,
            adaln_lora_dim=self.adaln_lora_dim,
            use_adaln_lora=self.use_adaln_lora,
        )

        self.gradient_checkpointing = False
        self.__post_init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        fps: int | None = None,
        condition_video_input_mask: torch.Tensor | None = None,
        condition_video_pose: torch.Tensor | None = None,
        condition_video_augment_sigma: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, C, T, H, W) latent video (16 channels)
            timestep: (B,) diffusion timesteps
            encoder_hidden_states: (B, N, D_text) text embeddings
            attention_mask: Optional attention mask
            fps: Frames per second
            condition_video_input_mask: (B, 1, T, H, W) conditioning mask for video frames
            condition_video_pose: (B, buffer_channels, T, H, W) VAE-encoded 3D cache buffers
            condition_video_augment_sigma: (B,) sigma for conditioning noise augmentation
            padding_mask: (B, 1, H, W) padding mask
        """
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        # 1. Concatenate condition_video_input_mask
        if condition_video_input_mask is not None:
            hidden_states = torch.cat([hidden_states, condition_video_input_mask], dim=1)
        else:
            # Default: zeros mask (no conditioning)
            condition_video_input_mask = torch.zeros(
                batch_size, 1, num_frames, height, width, 
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            hidden_states = torch.cat([hidden_states, condition_video_input_mask], dim=1)

        # 2. Concatenate condition_video_pose (3D cache buffers)
        if condition_video_pose is not None:
            hidden_states = torch.cat([hidden_states, condition_video_pose], dim=1)
        else:
            # Default: zeros for pose buffers
            pose_zeros = torch.zeros(
                batch_size, self.buffer_channels, num_frames, height, width,
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            hidden_states = torch.cat([hidden_states, pose_zeros], dim=1)

        # 3. Concatenate padding mask if needed
        if self.concat_padding_mask:
            if padding_mask is None:
                padding_mask = torch.ones(
                    batch_size, 1, height, width,
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
            padding_mask = transforms.functional.resize(
                padding_mask,
                list(hidden_states.shape[-2:]),
                interpolation=transforms.InterpolationMode.NEAREST,
            )
            hidden_states = torch.cat(
                [hidden_states, padding_mask.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)],
                dim=1,
            )

        # 4. Patchify input
        p_t, p_h, p_w = self.patch_size
        hidden_states = self.patch_embed(hidden_states)  # (B, T', H', W', D)
        post_patch_num_frames, post_patch_height, post_patch_width = hidden_states.shape[1:4]

        # 5. Generate RoPE embeddings
        rope_emb = self.rope(hidden_states, fps=fps)

        # 6. Generate learnable positional embeddings
        extra_pos_emb = None
        if self.learnable_pos_embed is not None:
            extra_pos_emb = self.learnable_pos_embed(hidden_states)

        # Flatten to sequence representation for transformer blocks.
        hidden_states = hidden_states.flatten(1, 3)  # (B, S, D)
        if extra_pos_emb is not None:
            extra_pos_emb = extra_pos_emb.flatten(1, 3)

        # Sequence parallel sharding with optional padding.
        # This is a no-op when SP world size is 1.
        sp_world_size = get_sp_world_size()
        original_seq_len = hidden_states.shape[1]
        sp_attention_mask = None
        if sp_world_size > 1:
            hidden_states, original_seq_len = sequence_model_parallel_shard(
                hidden_states, dim=1
            )
            if extra_pos_emb is not None:
                extra_pos_emb, _ = sequence_model_parallel_shard(extra_pos_emb, dim=1)
            rope_cos, _ = sequence_model_parallel_shard(rope_emb[0], dim=0)
            rope_sin, _ = sequence_model_parallel_shard(rope_emb[1], dim=0)
            rope_emb = (rope_cos, rope_sin)

            padded_seq_len = hidden_states.shape[1] * sp_world_size
            if padded_seq_len > original_seq_len:
                sp_attention_mask = create_attention_mask_for_padding(
                    seq_len=original_seq_len,
                    padded_seq_len=padded_seq_len,
                    batch_size=batch_size,
                    device=hidden_states.device,
                )

        # 7. Timestep embeddings
        affine_emb, adaln_lora = self.time_embed(timestep, hidden_states.dtype)

        # 8. Augment sigma embedding (GEN3C-specific)
        if self.add_augment_sigma_embedding:
            if condition_video_augment_sigma is None:
                condition_video_augment_sigma = torch.zeros_like(timestep)
            augment_emb, _ = self.augment_sigma_embed(condition_video_augment_sigma, hidden_states.dtype)
            affine_emb = affine_emb + augment_emb

        # 9. Apply affine normalization
        affine_emb = self.affine_norm(affine_emb)

        # 10. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.transformer_blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    affine_emb,
                    adaln_lora,
                    rope_emb,
                    extra_pos_emb,
                    sp_attention_mask,
                )
        else:
            for block in self.transformer_blocks:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    affine_emb=affine_emb,
                    adaln_lora=adaln_lora,
                    rope_emb=rope_emb,
                    extra_pos_emb=extra_pos_emb,
                    attention_mask=sp_attention_mask,
                )

        # 11. Final layer
        hidden_states = self.final_layer(hidden_states, affine_emb, adaln_lora)

        if sp_world_size > 1:
            hidden_states = sequence_model_parallel_all_gather_with_unpad(
                hidden_states, original_seq_len, dim=1
            )

        # 12. Unpatchify: (B, S, P) -> (B, T', H', W', P) -> (B, C, T, H, W)
        hidden_states = hidden_states.unflatten(
            1, (post_patch_num_frames, post_patch_height, post_patch_width)
        )
        hidden_states = hidden_states.unflatten(-1, (p_t, p_h, p_w, self.out_channels))
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        hidden_states = hidden_states.flatten(2, 3).flatten(3, 4).flatten(4, 5)

        return hidden_states
