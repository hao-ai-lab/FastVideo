# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from fastvideo.attention import DistributedAttention, LocalAttention
from fastvideo.configs.models.dits.cosmos2_5 import Cosmos25VideoConfig
from fastvideo.forward_context import get_forward_context
from fastvideo.layers.layernorm import RMSNorm
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import apply_rotary_emb
from fastvideo.layers.visual_embedding import Timesteps
from fastvideo.models.dits.base import BaseDiT
from fastvideo.platforms import AttentionBackendEnum


class Cosmos25PatchEmbed(nn.Module):
    """
    COSMOS 2.5 patch embedding - converts video (B, C, T, H, W) to patches (B, T', H', W', D).
    Uses linear projection after rearranging patches.
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


class Cosmos25TimestepEmbedding(nn.Module):
    """
    COSMOS 2.5 timestep embedding with AdaLN-LoRA support.
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
            emb: Standard embedding (B, T, D)
            adaln_lora: AdaLN-LoRA parameters (B, T, 3D) or None
        """
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora = emb  # (B, T, 3D)
            emb_standard = sample  # Use input as standard embedding
        else:
            emb_standard = emb
            adaln_lora = None

        return emb_standard, adaln_lora


class Cosmos25Embedding(nn.Module):
    """
    COSMOS 2.5 timestep conditioning embedding.
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
        self.t_embedder = Cosmos25TimestepEmbedding(
            embedding_dim,
            condition_dim,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
        )
        self.norm = RMSNorm(embedding_dim, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            timestep: (B, T) tensor of timesteps
            
        Returns:
            embedded_timestep: Normalized timestep embedding (B, T, D)
            adaln_lora: AdaLN-LoRA parameters (B, T, 3D) or None
        """
        # Handle 2D timestep input (B, T) like the official model
        assert timestep.ndim == 2, f"Expected 2D timestep, got {timestep.ndim}D with shape {timestep.shape}"
        B, T = timestep.shape
        
        # Flatten for Timesteps layer which expects 1D, then reshape back
        timestep_flat = timestep.flatten()  # (B*T,)
        timesteps_proj = self.time_proj(timestep_flat).type_as(hidden_states)  # (B*T, D)
        timesteps_proj = timesteps_proj.reshape(B, T, -1)  # (B, T, D)
        
        embedded_timestep, adaln_lora = self.t_embedder(timesteps_proj)
        embedded_timestep = self.norm(embedded_timestep)
        
        return embedded_timestep, adaln_lora


class Cosmos25AdaLayerNormZero(nn.Module):
    """
    COSMOS 2.5 Adaptive Layer Normalization with zero initialization and gate.
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


class Cosmos25SelfAttention(nn.Module):
    """
    COSMOS 2.5 self-attention with QK normalization and RoPE.
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

        # Use DistributedAttention for flexible backend support (torch SDPA / FlashAttention)
        # For single-GPU (non-distributed), use LocalAttention to avoid distributed requirements
        if supported_attention_backends is None:
            supported_attention_backends = (AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA)
        
        # Check if distributed environment is initialized
        try:
            from fastvideo.distributed.parallel_state import model_parallel_is_initialized
            use_distributed = torch.distributed.is_initialized() and model_parallel_is_initialized()
        except:
            use_distributed = False
        
        if use_distributed:
            self.attn = DistributedAttention(
                num_heads=num_heads,
                head_size=self.head_dim,
                causal=False,
                supported_attention_backends=supported_attention_backends,
                prefix="self_attn"
            )
        else:
            self.attn = LocalAttention(
                num_heads=num_heads,
                head_size=self.head_dim,
                causal=False,
                supported_attention_backends=supported_attention_backends,
            )
        self._use_distributed = use_distributed

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
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

        # Attention computation using DistributedAttention or LocalAttention
        # Both expect (B, S, H, D_h), so transpose first
        query = query.transpose(1, 2)  # (B, H, S, D_h) -> (B, S, H, D_h)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        if self._use_distributed:
            attn_output, _ = self.attn(query, key, value)
        else:
            attn_output = self.attn(query, key, value)

        # Reshape back: (B, S, H, D_h) -> (B, S, H*D_h)
        attn_output = attn_output.flatten(-2, -1)

        # Output projection
        attn_output = self.to_out(attn_output)
        return attn_output


class Cosmos25CrossAttention(nn.Module):
    """
    COSMOS 2.5 cross-attention for text conditioning.
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

        # Use DistributedAttention for flexible backend support (torch SDPA / FlashAttention)
        # For single-GPU (non-distributed), use LocalAttention to avoid distributed requirements
        if supported_attention_backends is None:
            supported_attention_backends = (AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA)
        
        # Check if distributed environment is initialized
        try:
            from fastvideo.distributed.parallel_state import model_parallel_is_initialized
            use_distributed = torch.distributed.is_initialized() and model_parallel_is_initialized()
        except:
            use_distributed = False
        
        if use_distributed:
            self.attn = DistributedAttention(
                num_heads=num_heads,
                head_size=self.head_dim,
                causal=False,
                supported_attention_backends=supported_attention_backends,
                prefix="cross_attn"
            )
        else:
            self.attn = LocalAttention(
                num_heads=num_heads,
                head_size=self.head_dim,
                causal=False,
                supported_attention_backends=supported_attention_backends,
            )
        self._use_distributed = use_distributed

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

        # Note: Both DistributedAttention and LocalAttention expect (B, S, H, D_h), which is what we already have
        # No need to transpose to (B, H, S, D_h) format

        # Attention computation using DistributedAttention or LocalAttention
        if self._use_distributed:
            attn_output, _ = self.attn(query, key, value)
        else:
            attn_output = self.attn(query, key, value)

        # Reshape back: (B, S, H, D_h) -> (B, S, H*D_h)
        attn_output = attn_output.flatten(-2, -1)

        # Output projection
        attn_output = self.to_out(attn_output)
        return attn_output


class Cosmos25TransformerBlock(nn.Module):
    """
    COSMOS 2.5 transformer block with self-attention, cross-attention, and MLP.
    Uses AdaLN-LoRA for conditioning.
    Matches the official architecture where modulation parameters are computed once per block.
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

        # Layer norms (no modulation logic inside)
        self.norm1 = Cosmos25AdaLayerNormZero(hidden_size)
        self.norm2 = Cosmos25AdaLayerNormZero(hidden_size)
        self.norm3 = Cosmos25AdaLayerNormZero(hidden_size)

        # Attention and MLP layers
        self.attn1 = Cosmos25SelfAttention(
            dim=hidden_size,
            num_heads=num_attention_heads,
            qk_norm=qk_norm,
            supported_attention_backends=supported_attention_backends,
        )
        self.attn2 = Cosmos25CrossAttention(
            dim=hidden_size,
            cross_attention_dim=cross_attention_dim,
            num_heads=num_attention_heads,
            qk_norm=qk_norm,
            supported_attention_backends=supported_attention_backends,
        )
        self.mlp = MLP(hidden_size, int(hidden_size * mlp_ratio), act_type="gelu", bias=False)

        # AdaLN modulation layers (compute shift/scale/gate for each sub-layer)
        # These match the official model's adaln_modulation_* layers
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
        embedded_timestep: torch.Tensor,
        adaln_lora: torch.Tensor | None = None,
        rope_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        extra_pos_emb: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, H, W, D)
            encoder_hidden_states: (B, N, D_text)
            embedded_timestep: (B, T, D)
            adaln_lora: (B, T, 3D) AdaLN-LoRA parameters
            rope_emb: Tuple of (cos, sin) for RoPE
            extra_pos_emb: Optional learnable positional embeddings
        """
        # Add extra positional embeddings if provided
        if extra_pos_emb is not None:
            hidden_states = hidden_states + extra_pos_emb

        B, T, H, W, D = hidden_states.shape

        # Step 1: Compute ALL modulation parameters once (matches official model)
        if self.use_adaln_lora and adaln_lora is not None:
            shift_self_attn, scale_self_attn, gate_self_attn = (
                self.adaln_modulation_self_attn(embedded_timestep) + adaln_lora
            ).chunk(3, dim=-1)
            shift_cross_attn, scale_cross_attn, gate_cross_attn = (
                self.adaln_modulation_cross_attn(embedded_timestep) + adaln_lora
            ).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = (
                self.adaln_modulation_mlp(embedded_timestep) + adaln_lora
            ).chunk(3, dim=-1)
        else:
            shift_self_attn, scale_self_attn, gate_self_attn = self.adaln_modulation_self_attn(
                embedded_timestep
            ).chunk(3, dim=-1)
            shift_cross_attn, scale_cross_attn, gate_cross_attn = self.adaln_modulation_cross_attn(
                embedded_timestep
            ).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(embedded_timestep).chunk(3, dim=-1)

        # Reshape modulation parameters from (B, T, D) to (B, T, 1, 1, D) for broadcasting
        shift_self_attn = shift_self_attn.unsqueeze(2).unsqueeze(2).type_as(hidden_states)
        scale_self_attn = scale_self_attn.unsqueeze(2).unsqueeze(2).type_as(hidden_states)
        gate_self_attn = gate_self_attn.unsqueeze(2).unsqueeze(2).type_as(hidden_states)

        shift_cross_attn = shift_cross_attn.unsqueeze(2).unsqueeze(2).type_as(hidden_states)
        scale_cross_attn = scale_cross_attn.unsqueeze(2).unsqueeze(2).type_as(hidden_states)
        gate_cross_attn = gate_cross_attn.unsqueeze(2).unsqueeze(2).type_as(hidden_states)

        shift_mlp = shift_mlp.unsqueeze(2).unsqueeze(2).type_as(hidden_states)
        scale_mlp = scale_mlp.unsqueeze(2).unsqueeze(2).type_as(hidden_states)
        gate_mlp = gate_mlp.unsqueeze(2).unsqueeze(2).type_as(hidden_states)

        # Step 2: Self-attention block
        norm_hidden_states = self.norm1(hidden_states, shift_self_attn, scale_self_attn)
        # Flatten for attention: (B, T, H, W, D) -> (B, THW, D)
        norm_hidden_states_flat = norm_hidden_states.flatten(1, 3)
        
        attn_output = self.attn1(norm_hidden_states_flat, rope_emb=rope_emb)
        
        # Reshape back and apply residual
        attn_output = attn_output.unflatten(1, (T, H, W))  # (B, T, H, W, D)
        hidden_states = hidden_states + gate_self_attn * attn_output

        # Step 3: Cross-attention block
        norm_hidden_states = self.norm2(hidden_states, shift_cross_attn, scale_cross_attn)
        norm_hidden_states_flat = norm_hidden_states.flatten(1, 3)
        
        attn_output = self.attn2(
            norm_hidden_states_flat,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        
        attn_output = attn_output.unflatten(1, (T, H, W))
        hidden_states = hidden_states + gate_cross_attn * attn_output

        # Step 4: MLP block
        norm_hidden_states = self.norm3(hidden_states, shift_mlp, scale_mlp)
        norm_hidden_states_flat = norm_hidden_states.flatten(1, 3)
        
        mlp_output = self.mlp(norm_hidden_states_flat)
        
        mlp_output = mlp_output.unflatten(1, (T, H, W))
        hidden_states = hidden_states + gate_mlp * mlp_output

        return hidden_states


class Cosmos25RotaryPosEmbed(nn.Module):
    """
    COSMOS 2.5 3D Rotary Position Embedding with NTK-aware extrapolation.
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

        self.max_size = [size // patch for size, patch in zip(max_size, patch_size, strict=False)]
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
        
        Args:
            hidden_states: (B, T, H, W, D) - patch-embedded features
            fps: Frames per second for temporal scaling
            
        Returns:
            cos, sin: RoPE embeddings (THW, D)
        """
        batch_size, T, H, W, input_dim = hidden_states.shape
        device = hidden_states.device

        # T, H, W are already patch dimensions after patch_embed
        # No need to divide by patch_size

        # Generate frequency scales with NTK
        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor

        seq = torch.arange(max(self.max_size), device=device, dtype=torch.float32)

        # Use self.dim_h/w/t which were set during initialization
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
            # Apply FPS scaling
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

        cos = torch.cos(freqs)  # (THW, D)
        sin = torch.sin(freqs)  # (THW, D)
        
        return cos, sin


class Cosmos25LearnablePositionalEmbed(nn.Module):
    """
    COSMOS 2.5 learnable absolute positional embeddings (optional).
    """

    def __init__(
        self,
        hidden_size: int,
        max_size: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.max_size = [size // patch for size, patch in zip(max_size, patch_size, strict=False)]
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


class Cosmos25FinalLayer(nn.Module):
    """
    COSMOS 2.5 final layer with AdaLN modulation and unpatchification.
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
        self.activation = nn.SiLU()

        if use_adaln_lora:
            self.linear_1 = nn.Linear(hidden_size, adaln_lora_dim, bias=False)
            self.linear_2 = nn.Linear(adaln_lora_dim, 2 * hidden_size, bias=False)
        else:
            self.linear_1 = nn.Identity()
            self.linear_2 = nn.Linear(hidden_size, 2 * hidden_size, bias=False)

        # Output projection
        output_dim = out_channels * patch_size[0] * patch_size[1] * patch_size[2]
        self.proj_out = nn.Linear(hidden_size, output_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        adaln_lora: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, H, W, D)
            embedded_timestep: (B, T, D) or (B, D)
            adaln_lora: (B, T, 3D) or None
        """
        # Generate modulation parameters
        embedded_timestep = self.activation(embedded_timestep)
        embedded_timestep = self.linear_1(embedded_timestep)
        embedded_timestep = self.linear_2(embedded_timestep)

        if self.use_adaln_lora and adaln_lora is not None:
            # Use first 2*hidden_size elements for shift/scale
            embedded_timestep = embedded_timestep + adaln_lora[..., : 2 * self.hidden_size]

        shift, scale = embedded_timestep.chunk(2, dim=-1)

        # Apply normalization and modulation
        hidden_states = self.norm(hidden_states)

        # Reshape for broadcasting if needed
        if embedded_timestep.ndim == 2:
            shift, scale = (x.unsqueeze(1) for x in (shift, scale))
        elif embedded_timestep.ndim == 3 and hidden_states.ndim == 5:
            shift, scale = (x.unsqueeze(2).unsqueeze(2) for x in (shift, scale))

        hidden_states = hidden_states * (1 + scale) + shift

        # Project to output
        hidden_states = self.proj_out(hidden_states)

        return hidden_states


class Cosmos25Transformer3DModel(BaseDiT):
    """
    COSMOS 2.5 DiT - MiniTrainDIT architecture adapted for FastVideo.
    
    Key features:
    - AdaLN-LoRA conditioning
    - 3D RoPE with NTK-aware extrapolation
    - Optional learnable positional embeddings
    - QK normalization
    - Cross-attention projection (optional)
    """
    
    _fsdp_shard_conditions = Cosmos25VideoConfig()._fsdp_shard_conditions
    _compile_conditions = Cosmos25VideoConfig()._compile_conditions
    param_names_mapping = Cosmos25VideoConfig().param_names_mapping
    lora_param_names_mapping = Cosmos25VideoConfig().lora_param_names_mapping

    def __init__(self, config: Cosmos25VideoConfig, hf_config: dict[str, Any]) -> None:
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
        self.extra_pos_embed_type = getattr(config, "extra_pos_embed_type", None)
        self.use_crossattn_projection = getattr(config, "use_crossattn_projection", False)

        # 1. Patch Embedding
        # Account for: VAE channels + condition_mask (1) + padding_mask (1 if concat_padding_mask)
        patch_embed_in_channels = config.in_channels  # Base VAE channels (16)
        patch_embed_in_channels += 1  # Always add 1 for condition_mask
        if config.concat_padding_mask:
            patch_embed_in_channels += 1  # Add 1 for padding_mask
        # Total: 16 + 1 + 1 = 18 (with concat_padding_mask=True)
        
        self.patch_embed = Cosmos25PatchEmbed(
            patch_embed_in_channels, inner_dim, config.patch_size
        )

        # 2. Positional Embeddings
        self.rope = Cosmos25RotaryPosEmbed(
            hidden_size=config.attention_head_dim,
            max_size=config.max_size,
            patch_size=config.patch_size,
            rope_scale=config.rope_scale,
            enable_fps_modulation=getattr(config, "rope_enable_fps_modulation", True),
        )

        self.learnable_pos_embed = None
        if self.extra_pos_embed_type == "learnable":
            self.learnable_pos_embed = Cosmos25LearnablePositionalEmbed(
                hidden_size=inner_dim,
                max_size=config.max_size,
                patch_size=config.patch_size,
            )

        # 3. Time Embedding
        self.time_embed = Cosmos25Embedding(
            inner_dim,
            inner_dim,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
        )

        # 4. Cross-attention projection (optional)
        if self.use_crossattn_projection:
            crossattn_proj_in_channels = getattr(config, "crossattn_proj_in_channels", config.text_embed_dim)
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, config.text_embed_dim, bias=True),
                nn.GELU(),
            )

        # 5. Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            Cosmos25TransformerBlock(
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

        # 6. Final Layer
        self.final_layer = Cosmos25FinalLayer(
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
        condition_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, C, T, H, W) latent video
            timestep: (B,) or (B, T) diffusion timesteps
            encoder_hidden_states: (B, N, D_text) text embeddings
            attention_mask: Optional attention mask
            fps: Frames per second
            condition_mask: (B, 1, T, H, W) conditioning mask
            padding_mask: (B, 1, H, W) padding mask
        """
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        # 1. Concatenate condition mask if provided
        if condition_mask is not None:
            hidden_states = torch.cat([hidden_states, condition_mask], dim=1)

        # 2. Concatenate padding mask if needed
        if self.concat_padding_mask and padding_mask is not None:
            from torchvision import transforms
            padding_mask = transforms.functional.resize(
                padding_mask,
                list(hidden_states.shape[-2:]),
                interpolation=transforms.InterpolationMode.NEAREST,
            )
            hidden_states = torch.cat(
                [hidden_states, padding_mask.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)],
                dim=1,
            )

        # 3. Patchify input
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        
        
        hidden_states = self.patch_embed(hidden_states)  # (B, T', H', W', D)
       

        # 4. Generate RoPE embeddings (after patchify, using patch dimensions)
        rope_emb = self.rope(hidden_states, fps=fps)

        # 5. Generate learnable positional embeddings (if used)
        extra_pos_emb = None
        if self.learnable_pos_embed is not None:
            extra_pos_emb = self.learnable_pos_embed(hidden_states)

        # 6. Timestep embeddings
        # Official model expects timestep in (B, T) format, so ensure it has 2D shape
        if timestep.ndim == 1:
            # Scalar timestep per sample: (B,) -> (B, 1)
            timestep = timestep.unsqueeze(1)
        elif timestep.ndim == 2:
            # Already in (B, T) format
            pass
        else:
            raise ValueError(f"Unsupported timestep shape: {timestep.shape}")
        
        # Now timestep is always (B, T), pass directly to time_embed
        embedded_timestep, adaln_lora = self.time_embed(hidden_states, timestep)

        # 7. Apply cross-attention projection (if used)
        if self.use_crossattn_projection:
            encoder_hidden_states = self.crossattn_proj(encoder_hidden_states)
            
        
        

        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N)

        # 8. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.transformer_blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    embedded_timestep,
                    adaln_lora,
                    rope_emb,
                    extra_pos_emb,
                    attention_mask,
                )
        else:
            for i, block in enumerate(self.transformer_blocks):
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    embedded_timestep=embedded_timestep,
                    adaln_lora=adaln_lora,
                    rope_emb=rope_emb,
                    extra_pos_emb=extra_pos_emb,
                    attention_mask=attention_mask,
                )

        # 9. Final layer - output norm & projection
        hidden_states = self.final_layer(hidden_states, embedded_timestep, adaln_lora)

        # 10. Unpatchify: (B, T', H', W', P) -> (B, C, T, H, W)
        # After unflatten: (B, T', H', W', p_t, p_h, p_w, C) with dims [0,1,2,3,4,5,6,7]
        hidden_states = hidden_states.unflatten(-1, (p_t, p_h, p_w, self.out_channels))
        # Permute to: (B, C, T', p_t, H', p_h, W', p_w)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        # Flatten pairs to get (B, C, T, H, W)
        hidden_states = hidden_states.flatten(2, 3).flatten(3, 4).flatten(4, 5)

        return hidden_states

