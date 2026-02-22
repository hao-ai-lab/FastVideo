# SPDX-License-Identifier: Apache-2.0
"""
Native LongCat Video DiT implementation using FastVideo conventions.
"""

from typing import Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fastvideo.configs.models.dits import LongCatVideoConfig
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.layernorm import RMSNorm, FP32LayerNorm
from fastvideo.layers.activation import get_act_fn
from fastvideo.layers.rotary_embedding_3d import RotaryPositionalEmbedding3D
from fastvideo.attention.layer import DistributedAttention, LocalAttention
from fastvideo.models.dits.base import CachableDiT
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.third_party.longcat_video.block_sparse_attention.bsa_interface import flash_attn_bsa_3d


# ============================================================================
# Embeddings
# ============================================================================

class PatchEmbed3D(nn.Module):
    """
    3D patch embedding using Conv3d.
    """

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 16,
        embed_dim: int = 4096,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            [B, N, C] where N = (T/pt) * (H/ph) * (W/pw)
        """
        # Padding if needed
        _, _, T, H, W = x.shape
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if T % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - T % self.patch_size[0]))

        x = self.proj(x)  # [B, C, T', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x


class TimestepEmbedder(nn.Module):
    """
    Sinusoidal timestep embedding + MLP projection.
    """

    def __init__(
        self,
        frequency_embedding_size: int = 256,
        adaln_tembed_dim: int = 512,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size

        # Use FastVideo's ReplicatedLinear
        self.linear_1 = ReplicatedLinear(
            frequency_embedding_size,
            adaln_tembed_dim,
            bias=True,
            params_dtype=dtype,
        )
        self.act = nn.SiLU()
        self.linear_2 = ReplicatedLinear(
            adaln_tembed_dim,
            adaln_tembed_dim,
            bias=True,
            params_dtype=dtype,
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor, latent_shape: tuple | None = None) -> torch.Tensor:
        """
        Args:
            t: [B] or [B, T] timesteps
            latent_shape: (T, H, W) for temporal expansion
        Returns:
            [B, T, C]
        """
        # Sinusoidal embedding in FP32
        t_freq = self.timestep_embedding(t.flatten(), self.frequency_embedding_size)
        
        # Cast to model dtype before MLP (matching original LongCat)
        # Handle LoRA wrapper if present
        linear_layer = self.linear_1.base_layer if hasattr(self.linear_1, 'base_layer') else self.linear_1
        target_dtype = linear_layer.weight.dtype
        if t_freq.dtype != target_dtype:
            t_freq = t_freq.to(target_dtype)
        
        # MLP projection
        t_emb, _ = self.linear_1(t_freq)
        t_emb = self.act(t_emb)
        t_emb, _ = self.linear_2(t_emb)
        
        # Reshape if needed
        if latent_shape is not None and len(t.shape) > 1:
            B = t.shape[0]
            T = latent_shape[0]
            t_emb = t_emb.reshape(B, T, -1)
        
        return t_emb


class CaptionEmbedder(nn.Module):
    """
    Caption embedding with MLP projection and optional text compaction.
    """

    def __init__(
        self,
        caption_channels: int = 4096,
        hidden_size: int = 4096,
        text_tokens_zero_pad: bool = True,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.text_tokens_zero_pad = text_tokens_zero_pad

        # Two-layer MLP using ReplicatedLinear
        # CRITICAL: Original LongCat uses GELU(approximate="tanh"), NOT SiLU!
        self.linear_1 = ReplicatedLinear(
            caption_channels,
            hidden_size,
            bias=True,
            params_dtype=dtype,
        )
        self.act = nn.GELU(approximate="tanh")  # Match original LongCat
        self.linear_2 = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=True,
            params_dtype=dtype,
        )

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_hidden_states: [B, N_text, C_text] or [B, 1, N_text, C_text]
            encoder_attention_mask: [B, N_text] or [B, 1, 1, N_text]
        Returns:
            y: [B, N_text, C] - standard padded representation (like other models)
        """
        # Handle extra dimension from wrapper
        if len(encoder_hidden_states.shape) == 4:
            encoder_hidden_states = encoder_hidden_states.squeeze(1)

        # Project
        y, _ = self.linear_1(encoder_hidden_states)
        y = self.act(y)
        y, _ = self.linear_2(y)  # [B, N_text, C]
        
        # Handle attention masking - just zero out padded tokens if requested
        if encoder_attention_mask is not None:
            # Remove extra dimensions
            if len(encoder_attention_mask.shape) == 4:
                encoder_attention_mask = encoder_attention_mask.squeeze(1).squeeze(1)
            elif len(encoder_attention_mask.shape) == 3:
                encoder_attention_mask = encoder_attention_mask.squeeze(1)
            
            # Zero out padded tokens if requested
            if self.text_tokens_zero_pad:
                y = y * encoder_attention_mask.unsqueeze(-1)
        
        # Return standard format [B, N_text, C] - no compaction!
        return y


# ============================================================================
# Attention Modules (Placeholders for now)
# ============================================================================

class LongCatSelfAttention(nn.Module):
    """
    Self-attention with 3D RoPE support and optional BSA.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        config: LongCatVideoConfig,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Separate Q/K/V projections (not fused like original)
        self.to_q = ReplicatedLinear(dim, dim, bias=True, params_dtype=dtype)
        self.to_k = ReplicatedLinear(dim, dim, bias=True, params_dtype=dtype)
        self.to_v = ReplicatedLinear(dim, dim, bias=True, params_dtype=dtype)

        # Per-head RMS normalization
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6, dtype=dtype or torch.float32)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6, dtype=dtype or torch.float32)

        # Output projection
        self.to_out = ReplicatedLinear(dim, dim, bias=True, params_dtype=dtype)

        # 3D RoPE
        self.rope_3d = RotaryPositionalEmbedding3D(head_dim=self.head_dim)

        # BSA configuration
        self.enable_bsa = getattr(config, 'enable_bsa', False)
        self.bsa_params = getattr(config, 'bsa_params', None)

        # FastVideo attention backend (used when BSA is disabled)
        self.attn = DistributedAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            supported_attention_backends=config._supported_attention_backends,
        )

    def forward(
        self,
        x: torch.Tensor,              # [B, N, C]
        latent_shape: tuple,          # (T, H, W)
        num_cond_latents: int = 0,    # Number of conditioning latent frames (for I2V)
        return_kv: bool = False,      # Return K/V for caching
        **kwargs
    ) -> torch.Tensor | tuple:
        """
        Forward pass with 3D RoPE and optional BSA.
        
        For I2V mode (num_cond_latents > 0):
        - Conditioned tokens only attend to themselves
        - Noise tokens attend to ALL tokens (cond + noise)
        
        Args:
            return_kv: If True, return (output, (k_cache, v_cache)) for KV caching
        """
        B, N, C = x.shape
        T, H, W = latent_shape

        # Project to Q/K/V
        q, _ = self.to_q(x)
        k, _ = self.to_k(x)
        v, _ = self.to_v(x)

        # Reshape to heads: [B, N, num_heads, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)

        # Per-head RMS normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Save pre-RoPE K/V for cache if requested (before RoPE is applied)
        if return_kv:
            # [B, N, num_heads, head_dim] -> [B, num_heads, N, head_dim]
            k_cache = k.transpose(1, 2).clone()
            v_cache = v.transpose(1, 2).clone()

        # For RoPE: need [B, num_heads, N, head_dim]
        q_rope = q.transpose(1, 2)
        k_rope = k.transpose(1, 2)

        # Apply 3D RoPE
        q_rope, k_rope = self.rope_3d(q_rope, k_rope, grid_size=latent_shape)

        # Transpose back: [B, N, num_heads, head_dim] or [B, H, N, D] for BSA
        q = q_rope.transpose(1, 2)
        k = k_rope.transpose(1, 2)

        # === I2V Split Attention ===
        # For I2V, conditioned tokens and noise tokens are processed separately
        if num_cond_latents > 0:
            # Calculate number of conditioned tokens (cond_latents * spatial_tokens_per_frame)
            num_cond_tokens = num_cond_latents * (N // T)
            
            # Conditioned tokens: only attend to themselves (same seq length, use self.attn)
            q_cond = q[:, :num_cond_tokens].contiguous()
            k_cond = k[:, :num_cond_tokens].contiguous()
            v_cond = v[:, :num_cond_tokens].contiguous()
            out_cond, _ = self.attn(q_cond, k_cond, v_cond)
            
            # Noise tokens: attend to ALL tokens (different seq lengths!)
            # Need to use flash attention directly since q has different length than k/v
            q_noise = q[:, num_cond_tokens:].contiguous()  # [B, N_noise, num_heads, head_dim]
            # k, v are full: [B, N, num_heads, head_dim]
            
            # Transpose for flash attention: [B, num_heads, seq, head_dim]
            q_noise_t = q_noise.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            
            # Use scaled dot product attention (handles different q/kv lengths)
            out_noise_t = torch.nn.functional.scaled_dot_product_attention(
                q_noise_t, k_t, v_t, 
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False
            )  # [B, num_heads, N_noise, head_dim]
            
            # Transpose back: [B, N_noise, num_heads, head_dim]
            out_noise = out_noise_t.transpose(1, 2)
            
            # Merge conditioned and noise outputs
            out = torch.cat([out_cond, out_noise], dim=1)
            
            # Reshape and project out
            out = out.reshape(B, N, C)
            out, _ = self.to_out(out)
            
            if return_kv:
                return out, (k_cache, v_cache)
            return out

        # === Attention: BSA or standard ===
        if self.enable_bsa and T > 1:  # Only use BSA for multi-frame videos
            # BSA expects [B, H, S, D] format
            q_bsa = q.transpose(1, 2).contiguous()  # [B, num_heads, N, head_dim]
            k_bsa = k.transpose(1, 2).contiguous()
            v_bsa = v.transpose(1, 2).contiguous()
            
            # Handle SP split: BSA operates on per-rank spatial dimensions
            # Replicate LongCat's cp_split_hw logic exactly
            from fastvideo.distributed.parallel_state import get_sp_world_size
            sp_size = get_sp_world_size()
            if sp_size > 1:
                # Calculate optimal 2D split (same as LongCat's get_optimal_split)
                factors = []
                for i in range(1, int(sp_size**0.5) + 1):
                    if sp_size % i == 0:
                        factors.append([i, sp_size // i])
                cp_split_hw = min(factors, key=lambda x: abs(x[0] - x[1]))
                
                # Split H and W dimensions by their respective factors
                T_bsa, H_bsa, W_bsa = latent_shape
                assert H_bsa % cp_split_hw[0] == 0 and W_bsa % cp_split_hw[1] == 0, \
                    f"H {H_bsa} must be divisible by {cp_split_hw[0]}, W {W_bsa} must be divisible by {cp_split_hw[1]}"
                H_bsa = H_bsa // cp_split_hw[0]
                W_bsa = W_bsa // cp_split_hw[1]
                latent_shape_bsa = (T_bsa, H_bsa, W_bsa)
            else:
                latent_shape_bsa = latent_shape
            
            # Call BSA with per-rank latent shape
            out = flash_attn_bsa_3d(
                q_bsa, k_bsa, v_bsa,
                latent_shape_q=latent_shape_bsa,
                latent_shape_k=latent_shape_bsa,
                **self.bsa_params
            )  # [B, num_heads, N, head_dim]
            
            # Transpose back: [B, N, num_heads, head_dim]
            out = out.transpose(1, 2)
        else:
            # Standard attention: [B, N, num_heads, head_dim]
            out, _ = self.attn(q, k, v)

        # Reshape and project out
        out = out.reshape(B, N, C)
        out, _ = self.to_out(out)

        if return_kv:
            return out, (k_cache, v_cache)
        return out

    def forward_with_kv_cache(
        self,
        x: torch.Tensor,              # [B, N_noise, C] - only noise tokens
        latent_shape: tuple,          # (T_noise, H, W) - shape for noise only
        num_cond_latents: int,        # Number of conditioning latent frames
        kv_cache: tuple,              # (k_cond, v_cond) - [B, heads, N_cond, head_dim]
    ) -> torch.Tensor:
        """
        Forward using cached K/V from conditioning frames.
        
        x contains only NOISE tokens.
        kv_cache contains pre-computed K/V for CONDITIONING tokens.
        
        CRITICAL: RoPE positions for noise tokens must start AFTER conditioning.
        We achieve this by padding Q with dummy tokens for conditioning positions,
        applying RoPE to the full sequence, then extracting only noise token Q.
        """
        B, N, C = x.shape
        T, H, W = latent_shape

        k_cache, v_cache = kv_cache

        # Handle batch size mismatch (cache might be smaller for CFG)
        # When using CFG, latent_model_input is doubled [neg, pos], but cache is for original batch
        if k_cache.shape[0] != B:
            # Expand cache to match input batch size
            # For CFG: repeat the cache for both negative and positive branches
            repeat_factor = B // k_cache.shape[0]
            k_cache = k_cache.repeat(repeat_factor, 1, 1, 1)
            v_cache = v_cache.repeat(repeat_factor, 1, 1, 1)

        # Project to Q/K/V for noise tokens
        q, _ = self.to_q(x)
        k, _ = self.to_k(x)
        v, _ = self.to_v(x)

        # Reshape to heads: [B, N, num_heads, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)

        # Per-head RMS normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose for RoPE: [B, heads, N, head_dim]
        q_rope = q.transpose(1, 2)
        k_rope = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # CRITICAL: Apply RoPE with correct positional offset
        # Noise frame queries need positions starting from num_cond_latents
        # Following the original LongCat approach:
        # 1. Pad Q with dummy tokens matching k_cache shape
        # 2. Apply RoPE to full sequence (T_cond + T_noise)
        # 3. Extract only the noise portion of Q
        
        # Create dummy Q padding to fill conditioning positions
        # k_cache shape: [B, heads, N_cond, head_dim]
        q_padding = torch.cat([torch.empty_like(k_cache), q_rope], dim=2).contiguous()
        
        # Concatenate cached K with noise K for RoPE
        k_full = torch.cat([k_cache, k_rope], dim=2)
        v_full = torch.cat([v_cache, v], dim=2)
        
        # Apply RoPE to full sequence (includes both cond and noise positions)
        # Grid size: (T_cond + T_noise, H, W)
        full_T = num_cond_latents + T
        q_padding, k_full = self.rope_3d(q_padding, k_full, grid_size=(full_T, H, W))
        
        # Extract only the noise portion of Q (last N tokens)
        q_rope = q_padding[:, :, -N:].contiguous()

        # Run attention: Q_noise attends to full K/V (cond + noise)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_rope, k_full, v_full,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )  # [B, heads, N_noise, head_dim]

        # Transpose back: [B, N_noise, heads, head_dim]
        out = out.transpose(1, 2)
        out = out.reshape(B, N, C)
        out, _ = self.to_out(out)

        return out


class LongCatCrossAttention(nn.Module):
    """
    Cross-attention for text conditioning (standard implementation like other models).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        config: LongCatVideoConfig,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Separate Q/K/V projections
        self.to_q = ReplicatedLinear(dim, dim, bias=True, params_dtype=dtype)
        self.to_k = ReplicatedLinear(dim, dim, bias=True, params_dtype=dtype)
        self.to_v = ReplicatedLinear(dim, dim, bias=True, params_dtype=dtype)

        # Per-head RMS normalization
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6, dtype=dtype or torch.float32)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6, dtype=dtype or torch.float32)

        # Output projection
        self.to_out = ReplicatedLinear(dim, dim, bias=True, params_dtype=dtype)

        # Cross-attention uses LocalAttention (FastVideo standard)
        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=config.arch_config._supported_attention_backends,
        )

    def forward(
        self,
        x: torch.Tensor,              # [B, N_img, C]
        context: torch.Tensor,        # [B, N_text, C]
        latent_shape: tuple = None,   # (T, H, W) - needed for I2V
        num_cond_latents: int = 0,    # Number of conditioning latent frames (for I2V)
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for cross-attention (standard implementation).
        
        Args:
            x: Image tokens [B, N_img, C]
            context: Text tokens [B, N_text, C] (standard padded format)
            latent_shape: (T, H, W) - needed for calculating num_cond_tokens
            num_cond_latents: Number of conditioning latent frames (for I2V)
            
        For I2V mode (num_cond_latents > 0):
        - Conditioned tokens get ZERO cross-attention output
        - Only noise tokens get cross-attention with text
        """
        B, N_img, C = x.shape

        # === I2V: Only noise tokens get cross-attention ===
        if num_cond_latents > 0 and latent_shape is not None:
            T, H, W = latent_shape
            num_cond_tokens = num_cond_latents * (N_img // T)
            
            # Only process noise tokens
            x_noise = x[:, num_cond_tokens:]  # [B, N_noise, C]
            
            # Project Q, K, V for noise tokens only
            q, _ = self.to_q(x_noise)
            k, _ = self.to_k(context)
            v, _ = self.to_v(context)
            
            N_text = context.shape[1]
            N_noise = x_noise.shape[1]

            # Reshape to heads
            q = q.view(B, N_noise, self.num_heads, self.head_dim)
            k = k.view(B, N_text, self.num_heads, self.head_dim)
            v = v.view(B, N_text, self.num_heads, self.head_dim)

            # Per-head RMS normalization
            q = self.q_norm(q)
            k = self.k_norm(k)

            # Run cross-attention
            out_noise = self.attn(q, k, v)  # [B, N_noise, num_heads, head_dim]
            out_noise = out_noise.reshape(B, N_noise, C)
            out_noise, _ = self.to_out(out_noise)
            
            # Conditioned tokens get zero output
            out_cond = torch.zeros(
                (B, num_cond_tokens, C), 
                dtype=out_noise.dtype, 
                device=out_noise.device
            )
            
            # Merge
            out = torch.cat([out_cond, out_noise], dim=1)
            return out

        # === Standard cross-attention ===
        # Project Q, K, V (standard cross-attention like WanVideo/Cosmos)
        q, _ = self.to_q(x)
        k, _ = self.to_k(context)
        v, _ = self.to_v(context)
        
        N_text = context.shape[1]

        # Reshape to heads
        q = q.view(B, N_img, self.num_heads, self.head_dim)
        k = k.view(B, N_text, self.num_heads, self.head_dim)
        v = v.view(B, N_text, self.num_heads, self.head_dim)

        # Per-head RMS normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Run cross-attention using FastVideo's LocalAttention
        # LocalAttention handles different q and k/v sequence lengths automatically
        out = self.attn(q, k, v)  # [B, N_img, num_heads, head_dim]

        # Reshape and project out
        out = out.reshape(B, N_img, C)
        out, _ = self.to_out(out)

        return out


# ============================================================================
# Feed-Forward Network
# ============================================================================

class LongCatSwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward network using FastVideo's ReplicatedLinear.
    
    FFN(x) = down(gate(x) * SiLU(up(x)))
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # Three projections for SwiGLU (no bias as per original)
        self.w1 = ReplicatedLinear(dim, hidden_dim, bias=False, params_dtype=dtype)  # gate
        self.w3 = ReplicatedLinear(dim, hidden_dim, bias=False, params_dtype=dtype)  # up
        self.w2 = ReplicatedLinear(hidden_dim, dim, bias=False, params_dtype=dtype)  # down

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: SiLU(w1(x)) * w3(x) -> w2 (matching original LongCat)
        """
        w1_out, _ = self.w1(x)
        w3_out, _ = self.w3(x)
        combined = self.act(w1_out) * w3_out
        out, _ = self.w2(combined)
        return out


# ============================================================================
# Modulation Utilities
# ============================================================================

def modulate_fp32(norm: nn.Module, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Apply modulation in FP32 for numerical stability.
    
    Converts inputs to FP32 for the modulation operation, then casts back.
    """
    orig_dtype = x.dtype
    
    # Convert to FP32 for numerical stability
    shift_fp32 = shift.float()
    scale_fp32 = scale.float()
    
    # Normalize and modulate in FP32
    x_norm = norm(x.to(torch.float32))
    x_mod = x_norm * (scale_fp32 + 1) + shift_fp32
    
    return x_mod.to(orig_dtype)


# ============================================================================
# Transformer Block
# ============================================================================

class LongCatTransformerBlock(nn.Module):
    """
    Single-stream transformer block with:
    - AdaLN modulation (FP32)
    - Self-attention
    - Cross-attention
    - SwiGLU FFN
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        adaln_tembed_dim: int,
        config: LongCatVideoConfig,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # AdaLN modulation (6 parameters: scale/shift for attn & ffn, gate for residual)
        self.adaln_linear_1 = ReplicatedLinear(
            adaln_tembed_dim,
            6 * hidden_size,
            bias=True,
            params_dtype=dtype,
        )
        self.adaln_act = nn.SiLU()

        # Normalization layers (CRITICAL: Use LayerNorm not RMSNorm like original!)
        # Original LongCat uses LayerNorm_FP32 with elementwise_affine=False
        self.norm_attn = FP32LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.norm_ffn = FP32LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        # Cross-attention norm has elementwise_affine=True (has weight and bias)
        self.norm_cross = FP32LayerNorm(hidden_size, eps=1e-6, elementwise_affine=True)

        # Self-attention
        self.self_attn = LongCatSelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            config=config,
            dtype=dtype,
        )

        # Cross-attention
        self.cross_attn = LongCatCrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            config=config,
            dtype=dtype,
        )

        # SwiGLU FFN
        ffn_hidden_dim = int(hidden_size * mlp_ratio * 2 / 3)
        # Round up to nearest multiple of 256
        ffn_hidden_dim = 256 * ((ffn_hidden_dim + 255) // 256)

        self.ffn = LongCatSwiGLUFFN(
            dim=hidden_size,
            hidden_dim=ffn_hidden_dim,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,              # [B, N, C]
        context: torch.Tensor,        # [B, N_text, C]
        t: torch.Tensor,              # [B, T, C_t]
        latent_shape: tuple,          # (T, H, W)
        num_cond_latents: int = 0,    # Number of conditioning latent frames (for I2V)
        return_kv: bool = False,      # Return K/V for caching
        kv_cache: tuple | None = None,  # Pre-computed K/V cache
        skip_crs_attn: bool = False,  # Skip cross-attention (for cache init)
        **kwargs
    ) -> torch.Tensor | tuple:
        """
        Forward pass with AdaLN modulation.
        
        Args:
            num_cond_latents: For I2V, number of conditioning latent frames.
                              These frames use split attention behavior.
            return_kv: If True, return (x, (k_cache, v_cache))
            kv_cache: Pre-computed K/V from conditioning frames
            skip_crs_attn: If True, skip cross-attention (used during cache init)
        """
        B, N, C = x.shape
        T, H, W = latent_shape
        x_orig_dtype = x.dtype  # Save for later casting

        # === AdaLN Modulation (CRITICAL: FP32 for stability like original) ===
        # Use autocast to compute modulation params in FP32
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            t_mod = self.adaln_act(t)
            mod_params, _ = self.adaln_linear_1(t_mod)
            # Ensure FP32 output (needed when LoRA is applied)
            if mod_params.dtype != torch.float32:
                mod_params = mod_params.float()
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                mod_params.unsqueeze(2).chunk(6, dim=-1)  # [B, T, 1, C]

        # === Self-Attention ===
        x_norm = modulate_fp32(self.norm_attn, x.view(B, T, -1, C), shift_msa, scale_msa)
        x_norm = x_norm.view(B, N, C)

        # Handle KV cache
        if kv_cache is not None:
            # Move cache to device if offloaded
            kv_cache = (kv_cache[0].to(x.device), kv_cache[1].to(x.device))
            attn_out = self.self_attn.forward_with_kv_cache(
                x_norm,
                latent_shape=latent_shape,
                num_cond_latents=num_cond_latents,
                kv_cache=kv_cache,
            )
            kv_cache_new = None  # Don't return cache when using cache
        else:
            attn_result = self.self_attn(
                x_norm, 
                latent_shape=latent_shape,
                num_cond_latents=num_cond_latents,
                return_kv=return_kv,
            )
            if return_kv:
                attn_out, kv_cache_new = attn_result
            else:
                attn_out = attn_result
                kv_cache_new = None

        # Residual with gating (CRITICAL: FP32 like original, then cast back)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            x = x + (gate_msa * attn_out.view(B, T, -1, C)).view(B, N, C)
        x = x.to(x_orig_dtype)

        # === Cross-Attention (skip if requested) ===
        if not skip_crs_attn:
            x_norm_cross = self.norm_cross(x)
            # When using KV cache, no need for num_cond_latents in cross-attn
            cross_num_cond = 0 if kv_cache is not None else num_cond_latents
            cross_out = self.cross_attn(
                x_norm_cross, 
                context,
                latent_shape=latent_shape,
                num_cond_latents=cross_num_cond
            )
            x = x + cross_out

        # === FFN ===
        x_norm_ffn = modulate_fp32(self.norm_ffn, x.view(B, T, -1, C), shift_mlp, scale_mlp)
        x_norm_ffn = x_norm_ffn.view(B, N, C)

        ffn_out = self.ffn(x_norm_ffn)

        # Residual with gating (CRITICAL: FP32 like original, then cast back)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            x = x + (gate_mlp * ffn_out.view(B, T, -1, C)).view(B, N, C)
        x = x.to(x_orig_dtype)

        if return_kv:
            return x, kv_cache_new
        return x


# ============================================================================
# Final Layer
# ============================================================================

class FinalLayer(nn.Module):
    """
    Final output projection with AdaLN modulation.
    """

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        adaln_tembed_dim: int,
        patch_size: tuple[int, int, int],
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # AdaLN for final layer (2 parameters: scale and shift)
        self.adaln_linear = ReplicatedLinear(
            adaln_tembed_dim,
            2 * hidden_size,
            bias=True,
            params_dtype=dtype,
        )
        self.adaln_act = nn.SiLU()

        # CRITICAL: Use LayerNorm not RMSNorm! (matches original)
        self.norm = FP32LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)

        # Output projection
        num_patch = patch_size[0] * patch_size[1] * patch_size[2]
        self.proj = ReplicatedLinear(
            hidden_size,
            num_patch * out_channels,
            bias=True,
            params_dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,  # [B, N, C]
        t: torch.Tensor,  # [B, T, C_t]
        latent_shape: tuple,
    ) -> torch.Tensor:
        """
        Returns: [B, N, out_channels * patch_size^3]
        """
        B, N, C = x.shape
        T, _, _ = latent_shape

        # AdaLN modulation
        t_mod = self.adaln_act(t)
        mod_params, _ = self.adaln_linear(t_mod)
        shift, scale = mod_params.unsqueeze(2).chunk(2, dim=-1)

        # Modulate (converts to FP32 internally for stability)
        x = modulate_fp32(self.norm, x.view(B, T, -1, C), shift, scale)
        x = x.reshape(B, N, C)

        # Project
        x, _ = self.proj(x)

        return x


# ============================================================================
# Main Model
# ============================================================================

class LongCatTransformer3DModel(CachableDiT):
    """
    Native LongCat Video Transformer using FastVideo layers.
    """

    # FSDP sharding: shard at each transformer block
    _fsdp_shard_conditions = [
        lambda n, m: "blocks" in n and n.split(".")[-1].isdigit(),
    ]
    # torch.compile optimization: compile each transformer block for speedup
    _compile_conditions = [
        lambda n, m: "blocks" in n and n.split(".")[-1].isdigit(),
    ]

    # Parameter name mapping (for weight conversion)
    param_names_mapping = {}  # Will be defined in config
    reverse_param_names_mapping = {}
    lora_param_names_mapping = {}

    # Supported attention backends
    _supported_attention_backends = (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    )

    def __init__(self, config: LongCatVideoConfig, hf_config: dict[str, Any]):
        super().__init__(config=config, hf_config=hf_config)

        # Extract architecture parameters
        self.hidden_size = config.hidden_size  # 4096
        self.num_attention_heads = config.num_attention_heads  # 32
        self.depth = config.depth  # 48
        self.mlp_ratio = config.mlp_ratio  # 4
        self.in_channels = config.in_channels  # 16
        self.out_channels = config.out_channels  # 16
        self.num_channels_latents = config.in_channels
        self.patch_size = config.patch_size  # [1, 2, 2]

        # Embeddings
        self.patch_embed = PatchEmbed3D(
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.hidden_size,
        )

        self.time_embedder = TimestepEmbedder(
            frequency_embedding_size=config.frequency_embedding_size,
            adaln_tembed_dim=config.adaln_tembed_dim,
        )

        self.caption_embedder = CaptionEmbedder(
            caption_channels=config.caption_channels,
            hidden_size=self.hidden_size,
            text_tokens_zero_pad=getattr(config, 'text_tokens_zero_pad', True),
        )

        # Transformer blocks (48 blocks)
        self.blocks = nn.ModuleList([
            LongCatTransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_attention_heads,
                mlp_ratio=self.mlp_ratio,
                adaln_tembed_dim=config.adaln_tembed_dim,
                config=config,
            )
            for _ in range(self.depth)
        ])

        # Output projection
        self.final_layer = FinalLayer(
            hidden_size=self.hidden_size,
            out_channels=self.out_channels,
            adaln_tembed_dim=config.adaln_tembed_dim,
            patch_size=self.patch_size,
        )

    def enable_bsa(self):
        """Enable BSA for all self-attention layers."""
        for block in self.blocks:
            block.self_attn.enable_bsa = True
    
    def disable_bsa(self):
        """Disable BSA for all self-attention layers."""
        for block in self.blocks:
            block.self_attn.enable_bsa = False

    def forward(
        self,
        hidden_states: torch.Tensor,           # [B, C, T, H, W]
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],   # [B, N_text, C_text]
        timestep: torch.LongTensor,            # [B] or [B, T]
        encoder_attention_mask: torch.Tensor | None = None,  # [B, N_text]
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance: float | None = None,         # Unused, for API compatibility
        num_cond_latents: int = 0,             # For I2V: number of conditioning latent frames
        # === KV Cache Parameters ===
        return_kv: bool = False,               # If True, return (output, kv_cache_dict)
        kv_cache_dict: dict | None = None,     # Pre-computed {block_idx: (k, v)}
        skip_crs_attn: bool = False,           # Skip cross-attention (for cache init)
        offload_kv_cache: bool = False,        # Move cache to CPU after compute
        **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Forward pass with FastVideo parameter ordering.
        
        NOTE: This follows FastVideo convention:
              (hidden_states, encoder_hidden_states, timestep)
              
        Args:
            num_cond_latents: For I2V, number of conditioning latent frames.
                              These frames are treated as "clean" (timestep=0)
                              and use split attention behavior.
            return_kv: If True, return (output, kv_cache_dict)
            kv_cache_dict: Pre-computed K/V cache {block_idx: (k, v)}
            skip_crs_attn: If True, skip cross-attention (for cache init)
            offload_kv_cache: If True, move cache to CPU after compute
        """
        B, _, T, H, W = hidden_states.shape

        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]

        # Handle list of encoder outputs (take first one)
        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = encoder_hidden_states[0]

        # 1. Patch embedding
        x = self.patch_embed(hidden_states)  # [B, N, C]

        # 2. Timestep embedding
        # Expand timestep from [B] to [B, T] if needed
        if timestep.ndim == 1:
            timestep = timestep.unsqueeze(1).expand(-1, N_t)  # [B, T]
        
        t = self.time_embedder(timestep.flatten(), latent_shape=(N_t, N_h, N_w))
        if t.ndim == 2:
            t = t.reshape(B, N_t, -1)  # [B, T, C_t]

        # 3. Caption embedding (standard format, no compaction)
        context = self.caption_embedder(
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )  # [B, N_text, C]

        # 4. Transformer blocks with optional KV cache
        kv_cache_dict_ret = {} if return_kv else None
        
        for i, block in enumerate(self.blocks):
            # Get cache for this block if available
            block_kv_cache = kv_cache_dict.get(i, None) if kv_cache_dict else None
            
            block_out = block(
                x, context, t,
                latent_shape=(N_t, N_h, N_w),
                num_cond_latents=num_cond_latents,
                return_kv=return_kv,
                kv_cache=block_kv_cache,
                skip_crs_attn=skip_crs_attn,
            )
            
            if return_kv:
                x, kv_cache = block_out
                # Store cache
                if offload_kv_cache:
                    kv_cache_dict_ret[i] = (kv_cache[0].cpu(), kv_cache[1].cpu())
                else:
                    kv_cache_dict_ret[i] = (kv_cache[0].contiguous(), kv_cache[1].contiguous())
            else:
                x = block_out

        # 5. Output projection
        output = self.final_layer(x, t, latent_shape=(N_t, N_h, N_w))

        # Reshape to [B, C_out, T, H, W]
        output = self.unpatchify(output, N_t, N_h, N_w)

        # Cast to float32 for better accuracy (as per original)
        output = output.to(torch.float32)

        if return_kv:
            return output, kv_cache_dict_ret
        return output

    def unpatchify(self, x: torch.Tensor, N_t: int, N_h: int, N_w: int) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] where C = T_p * H_p * W_p * C_out
        Returns:
            [B, C_out, T, H, W]
        """
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        return x

# Entry point for model registry
EntryClass = LongCatTransformer3DModel
