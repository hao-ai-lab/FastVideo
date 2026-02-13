# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 Hugging Face Team and Overworld (original)
# Copyright (C) 2026 FastVideo Contributors (FastVideo port)
"""
Waypoint-1-Small World Model transformer for FastVideo.

This is a port of the Overworld Waypoint-1-Small model to FastVideo's
architecture, maintaining weight compatibility with the official checkpoint.

Reference: https://huggingface.co/Overworld/Waypoint-1-Small

Correctness: Denoising follows the official rectified-flow update
x = x + (sigma_next - sigma_curr) * v (see Overworld denoise.py).
For parity with diffusers/WorldEngine, consider adding a cache pass
(single forward with sigma=0 after denoising) to update KV cache when
generating multiple frames autoregressively.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.attention import DistributedAttention, LocalAttention
from fastvideo.logger import init_logger
from fastvideo.configs.models.dits.waypoint_transformer import (
    WaypointConfig,
)
from fastvideo.models.dits.base import CachableDiT
from fastvideo.platforms import AttentionBackendEnum

# Default config instance used for class-level attributes (matches MatrixGame pattern)
_DEFAULT_WAYPOINT_CONFIG = WaypointConfig()
_DEFAULT_WAYPOINT_ARCH = _DEFAULT_WAYPOINT_CONFIG.arch_config

logger = init_logger(__name__)


# =============================================================================
# Note: CtrlInput is defined in fastvideo/pipelines/basic/waypoint/waypoint_pipeline.py
# to avoid circular imports and keep pipeline-specific code separate
# =============================================================================


# =============================================================================
# Helper Functions
# =============================================================================

def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization without learnable parameters."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def ada_rmsnorm(
    x: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Adaptive RMS normalization; scale/bias [B, N, D] broadcast to [B, L, D]."""
    B, L, D = x.shape
    N = scale.shape[1]
    x = rms_norm(x, eps)
    scale = scale.unsqueeze(2).expand(-1, -1, L // N, -1).reshape(B, L, D)
    bias = bias.unsqueeze(2).expand(-1, -1, L // N, -1).reshape(B, L, D)
    return x * (1 + scale) + bias


def ada_gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Apply gating; gate [B, N, D] broadcast to [B, L, D]."""
    B, L, D = x.shape
    N = gate.shape[1]
    gate = gate.unsqueeze(2).expand(-1, -1, L // N, -1).reshape(B, L, D)
    return x * gate


# =============================================================================
# Basic Building Blocks
# =============================================================================

class MLP(nn.Module):
    """Simple 2-layer MLP with SiLU activation."""
    
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class AdaLN(nn.Module):
    """Adaptive Layer Normalization for output; cond [B, N, D] per-frame scale/shift."""

    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 2 * d_model, bias=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        N = cond.shape[1]
        h = F.silu(cond)
        ab = self.fc(h)
        ab = ab.view(B, N, 1, 2 * D).expand(-1, -1, L // N, -1).reshape(B, L, 2 * D)
        scale, shift = ab.chunk(2, dim=-1)
        return rms_norm(x) * (1 + scale) + shift


class CFG(nn.Module):
    """Classifier-Free Guidance module with null embedding."""
    
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.null_emb = nn.Parameter(torch.zeros(1, 1, d_model))
    
    def forward(self, x: torch.Tensor, is_conditioned: Optional[bool] = None) -> torch.Tensor:
        B, L, _ = x.shape
        null = self.null_emb.expand(B, L, -1)
        
        if self.training or is_conditioned is None:
            if self.dropout == 0.0:
                return x
            drop = torch.rand(B, 1, 1, device=x.device) < self.dropout
            return torch.where(drop, null, x)
        
        return x if is_conditioned else null


# =============================================================================
# Controller Input Embedding
# =============================================================================

class ControllerInputEmbedding(nn.Module):
    """Embeds controller inputs (mouse + buttons + scroll) into model dimension."""
    
    def __init__(self, n_buttons: int, d_model: int, mlp_ratio: int = 4):
        super().__init__()
        # Input: mouse (2) + buttons (n_buttons) + scroll (1) = n_buttons + 3
        self.mlp = MLP(n_buttons + 3, d_model * mlp_ratio, d_model)
    
    def forward(self, mouse: torch.Tensor, button: torch.Tensor, scroll: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mouse: [B, N, 2] mouse velocity
            button: [B, N, n_buttons] button states (one-hot or multi-hot)
            scroll: [B, N, 1] scroll wheel
        Returns:
            [B, N, d_model] control embedding
        """
        x = torch.cat((mouse, button, scroll), dim=-1)
        return self.mlp(x)


# =============================================================================
# Noise Conditioning (Wan-style)
# =============================================================================

class NoiseConditioner(nn.Module):
    """Timestep/noise level conditioner (Wan-style). Matches Overworld: sigma*1000,
    logspace freqs, sin/cos order, unit-variance scale. Frequencies are computed
    in forward() (no buffer) so the model state_dict matches the checkpoint.
    """

    def __init__(self, d_model: int, freq_dim: int = 512, base: float = 10_000.0):
        super().__init__()
        self.freq_dim = freq_dim
        self.base = base
        half = freq_dim // 2
        assert half * 2 == freq_dim
        self.mlp = MLP(freq_dim, d_model * 4, d_model)

    @torch.autocast("cuda", enabled=False)
    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sigma: [B, N] noise levels
        Returns:
            [B, N, d_model] conditioning embedding
        """
        orig_dtype, shape = sigma.dtype, sigma.shape
        s = sigma.reshape(-1).float() * 1000.0
        half = self.freq_dim // 2
        freqs = torch.logspace(
            0, -1, steps=half, base=self.base,
            dtype=torch.float32, device=sigma.device,
        )
        phase = s.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        emb = emb * (2.0 ** 0.5)
        # MLP is in model dtype (e.g. bfloat16); cast emb to match to avoid mat1/mat2 dtype error
        mlp_dtype = next(self.mlp.parameters()).dtype
        emb = self.mlp(emb.to(mlp_dtype))
        return emb.to(orig_dtype).view(*shape, -1)


# =============================================================================
# OrthoRoPE (3-axis: time, height, width) — matches HF Overworld/Waypoint-1-Small
# - Time: geometric spectrum (lang-style), rotates 1/4 head_dim
# - Height/Width: linear spectrum (pixel-style), positions in [-1, 1], 1/8 each
# =============================================================================


def _pixel_frequencies(dim: int, max_freq: float, device: torch.device) -> torch.Tensor:
    """Linear frequency spectrum for spatial RoPE.
    HF: pixel_frequencies(dim) returns [dim//2] freqs; repeat_interleave(2) -> dim.
    """
    return torch.linspace(
        1.0, max_freq / 2, dim // 2, device=device, dtype=torch.float32
    ) * math.pi


def _lang_frequencies(dim: int, device: torch.device) -> torch.Tensor:
    """Geometric frequency spectrum for temporal RoPE.
    HF: lang_frequencies(dim) returns [dim//2] freqs; repeat_interleave(2) -> dim.
    """
    return 10.0 ** (
        -torch.arange(dim // 2, device=device, dtype=torch.float32) / 2
    )


class OrthoRoPE(nn.Module):
    """RoPE matching HF Overworld/Waypoint-1-Small attn.py exactly.
    - Spatial (X/Y): pixel_frequencies(head_dim//8) -> [D/16] freqs, repeat 2 -> D/8 each.
    - Temporal: lang_frequencies(head_dim//4) -> [D/8] freqs, repeat 2 -> D/4.
    - Total: D/8 + D/8 + D/4 = D/2. Positions: linspace(-1+1/W, 1-1/W, W) style.
    """

    def __init__(self, height: int, width: int, n_frames: int, head_dim: int):
        super().__init__()
        self.height = height
        self.width = width
        self.n_frames = n_frames
        self.head_dim = head_dim
        # HF: spatial D/8 each (pixel_frequencies(D/8) -> D/16, repeat 2), temporal D/4
        assert head_dim // 8 + head_dim // 8 + head_dim // 4 == head_dim // 2

    def get_angles(
        self,
        t_pos: torch.Tensor,
        y_pos: torch.Tensor,
        x_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos, sin [B, L, head_dim/2] matching HF _compute_freqs + get_angles."""
        device = t_pos.device
        H, W = self.height, self.width
        head_dim = self.head_dim
        max_freq = min(H, W) * 0.8

        # HF: spatial_freqs = pixel_frequencies(head_dim//8, max_freq)  # [D/16]
        spatial_freqs = _pixel_frequencies(head_dim // 8, max_freq, device)
        # HF: pos_x = linspace(-1+1/W, 1-1/W, W); we have integer x_pos in [0, W-1]
        w1 = max(W - 1, 1)
        h1 = max(H - 1, 1)
        norm_x = (-1.0 + 1.0 / W) + (2.0 - 2.0 / W) * x_pos.float() / w1
        norm_y = (-1.0 + 1.0 / H) + (2.0 - 2.0 / H) * y_pos.float() / h1
        # freqs_x: [B,L,D/16] then repeat_interleave(2) -> [B,L,D/8]
        angle_x = norm_x.unsqueeze(-1) * spatial_freqs.unsqueeze(0).unsqueeze(0)
        angle_x = angle_x.repeat_interleave(2, dim=-1)
        angle_y = norm_y.unsqueeze(-1) * spatial_freqs.unsqueeze(0).unsqueeze(0)
        angle_y = angle_y.repeat_interleave(2, dim=-1)

        # HF: temporal_freqs = lang_frequencies(head_dim//4)  # [D/8]
        temporal_freqs = _lang_frequencies(head_dim // 4, device)
        angle_t = t_pos.float().unsqueeze(-1) * temporal_freqs.unsqueeze(0).unsqueeze(0)
        angle_t = angle_t.repeat_interleave(2, dim=-1)

        angles = torch.cat([angle_x, angle_y, angle_t], dim=-1)
        return angles.cos(), angles.sin()

    @torch.autocast("cuda", enabled=False)
    def forward(
        self, x: torch.Tensor, pos_ids: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply RoPE to ALL of head_dim (matching official OrthoRoPE).

        x: [B, L, n_heads, head_dim]  (or [B, n_heads, L, head_dim])
        cos/sin: [B, L, head_dim/2] with repeat_interleave(2) baked in.

        Official rotation:
            x0, x1 = x.unfold(-1, 2, 2).unbind(-1)   # consecutive pairs
            y0 = x0 * cos - x1 * sin
            y1 = x1 * cos + x0 * sin
            return cat((y0, y1), dim=-1)
        """
        cos, sin = self.get_angles(
            pos_ids["t_pos"],
            pos_ids["y_pos"],
            pos_ids["x_pos"],
        )
        # cos/sin: [B, L, head_dim/2] — one value per consecutive pair.
        # Add head-dim broadcast: [B, L, 1, head_dim/2]
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

        # unfold ALL of head_dim into consecutive pairs
        x_float = x.float()
        x0, x1 = x_float.unfold(-1, 2, 2).unbind(-1)  # each [..., head_dim/2]

        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        return torch.cat((y0, y1), dim=-1).type_as(x)


# =============================================================================
# MLPFusion for Control Conditioning
# =============================================================================

class MLPFusion(nn.Module):
    """Fuses per-frame control conditioning into tokens via MLP."""
    
    def __init__(self, d_model: int):
        super().__init__()
        # Input is concatenation of x and cond, each d_model
        self.mlp = MLP(2 * d_model, d_model, d_model)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] token features
            cond: [B, N, D] per-frame conditioning (N frames)
        Returns:
            [B, L, D] fused features
        """
        B, L, D = x.shape
        N = cond.shape[1]
        tokens_per_frame = L // N
        
        # Split fc1 weights for efficient computation
        Wx, Wc = self.mlp.fc1.weight.chunk(2, dim=1)
        
        # Reshape x to [B, N, tokens_per_frame, D]
        x_reshaped = x.view(B, N, tokens_per_frame, D)
        
        # Compute: fc1_x(x) + fc1_c(cond).unsqueeze(2)
        h = F.linear(x_reshaped, Wx) + F.linear(cond, Wc).unsqueeze(2)
        h = F.silu(h)
        y = F.linear(h, self.mlp.fc2.weight)
        
        return y.flatten(1, 2)


# =============================================================================
# Conditioning Head (AdaLN-style modulation)
# =============================================================================

class CondHead(nn.Module):
    """Per-layer conditioning head producing 6 modulation vectors [B, N, D] each."""

    n_cond = 6  # scale0, bias0, gate0, scale1, bias1, gate1

    def __init__(self, d_model: int, noise_conditioning: str = "wan"):
        super().__init__()
        self.bias_in = (
            nn.Parameter(torch.zeros(d_model)) if noise_conditioning == "wan" else None
        )
        self.cond_proj = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(self.n_cond)
        ])

    def forward(self, cond: torch.Tensor):
        """cond [B, N, D] -> 6 tensors each [B, N, D]."""
        if self.bias_in is not None:
            cond = cond + self.bias_in
        h = F.silu(cond)
        return tuple(proj(h) for proj in self.cond_proj)


# =============================================================================
# Attention Layers
# =============================================================================

class GatedSelfAttention(nn.Module):
    """Gated self-attention with GQA, QK norm, and OrthoRoPE."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        layer_idx: int,
        causal: bool = True,
        gated_attn: bool = True,
        rope: Optional["OrthoRoPE"] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.layer_idx = layer_idx
        self.causal = causal
        self.gated_attn = gated_attn
        self.rope = rope

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn = DistributedAttention(
            num_heads=n_heads,
            head_size=self.head_dim,
            num_kv_heads=n_kv_heads,
            causal=causal,
            supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA,),
        )

        if gated_attn:
            self.gate_proj = nn.Linear(n_heads, n_heads, bias=False)
            nn.init.zeros_(self.gate_proj.weight)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        pos_ids: Optional[dict[str, torch.Tensor]] = None,
        kv_cache=None,
        update_cache: bool = False,
    ) -> torch.Tensor:
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim)

        q, k = rms_norm(q), rms_norm(k)

        if self.rope is not None and pos_ids is not None:
            q = self.rope(q, pos_ids)
            k = self.rope(k, pos_ids)

        # KV cache: prepend cached K/V for cross-frame causal attention
        if kv_cache is not None:
            cache_end = kv_cache["end"]
            if isinstance(cache_end, torch.Tensor):
                cache_end = int(cache_end.item())
            cache_k = kv_cache["k"]  # [B, n_kv_heads, cache_size, head_dim]
            cache_v = kv_cache["v"]
            if cache_end > 0:
                cached_k = cache_k[:, :, :cache_end, :]  # [B, n_kv_heads, end, head_dim]
                cached_v = cache_v[:, :, :cache_end, :]
                k_t = torch.cat([cached_k, k.transpose(1, 2)], dim=2)
                v_t = torch.cat([cached_v, v.transpose(1, 2)], dim=2)
            else:
                k_t = k.transpose(1, 2)  # [B, n_kv_heads, L, head_dim]
                v_t = v.transpose(1, 2)
            q_t = q.transpose(1, 2)  # [B, n_heads, L, head_dim]
            # No mask: spatial tokens are bidirectional; they can all see cache + current
            attn_out = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale,
                enable_gqa=(self.n_kv_heads != self.n_heads),
            )
            attn_out = attn_out.transpose(1, 2)  # [B, L, n_heads, head_dim]
            if update_cache:
                self._write_kv_cache(kv_cache, k, v, L)
        elif self.n_kv_heads != self.n_heads:
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            attn_out = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale,
                enable_gqa=True,
            )
            attn_out = attn_out.transpose(1, 2)
        else:
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            attn_out = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale,
                enable_gqa=(self.n_kv_heads != self.n_heads),
            )
            attn_out = attn_out.transpose(1, 2)
        # Official: gate attn output BEFORE out_proj
        # attn_out is [B, L, n_heads, head_dim] here (already transposed)
        if self.gated_attn:
            gates = torch.sigmoid(
                self.gate_proj(x[..., : self.n_heads])
            )  # [B, L, n_heads]
            attn_out = attn_out * gates.unsqueeze(-1)

        attn_out = attn_out.reshape(B, L, D)
        return self.out_proj(attn_out)

    def _write_kv_cache(
        self,
        kv_cache: dict,
        k: torch.Tensor,
        v: torch.Tensor,
        L: int,
    ) -> None:
        """Write current frame K/V into the layer cache (ring buffer)."""
        cache_k = kv_cache["k"]
        cache_v = kv_cache["v"]
        cache_size = cache_k.shape[2]
        end = kv_cache["end"]
        k_t = k.transpose(1, 2)  # [B, n_kv_heads, L, head_dim]
        v_t = v.transpose(1, 2)
        if end + L <= cache_size:
            cache_k[:, :, end : end + L, :] = k_t
            cache_v[:, :, end : end + L, :] = v_t
            kv_cache["end"] = end + L
        else:
            num_keep = cache_size - L
            cache_k[:, :, :num_keep, :].copy_(
                cache_k[:, :, end - num_keep : end, :]
            )
            cache_v[:, :, :num_keep, :].copy_(
                cache_v[:, :, end - num_keep : end, :]
            )
            cache_k[:, :, num_keep:, :] = k_t
            cache_v[:, :, num_keep:, :] = v_t
            kv_cache["end"] = cache_size


class CrossAttention(nn.Module):
    """Cross-attention for prompt conditioning.
    
    Uses LocalAttention for cross-attention (no sequence parallelism needed).
    Uses context_dim as inner dimension (not d_model), matching checkpoint structure.
    Uses a fixed head_dim=64 and calculates n_heads from context_dim.
    """
    
    def __init__(self, d_model: int, context_dim: int, head_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.context_dim = context_dim
        self.head_dim = head_dim
        # Calculate n_heads from context_dim and head_dim
        assert context_dim % head_dim == 0, f"context_dim {context_dim} must be divisible by head_dim {head_dim}"
        self.n_heads = context_dim // head_dim
        
        # Q: project from d_model to context_dim
        self.q_proj = nn.Linear(d_model, context_dim, bias=False)
        # K, V: project from context_dim to context_dim
        self.k_proj = nn.Linear(context_dim, context_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, context_dim, bias=False)
        # Out: project from context_dim back to d_model
        self.out_proj = nn.Linear(context_dim, d_model, bias=False)

        # Use LocalAttention for cross-attention (local operation, no SP needed)
        self.attn = LocalAttention(
            num_heads=self.n_heads,
            head_size=self.head_dim,
            num_kv_heads=self.n_heads,
            causal=False,
            supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA, ),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        _, S, _ = context.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(context).view(B, S, self.n_heads, self.head_dim)
        v = self.v_proj(context).view(B, S, self.n_heads, self.head_dim)
        q, k = rms_norm(q), rms_norm(k)

        # Official Waypoint cross-attention does NOT mask padding:
        #   out = flex_attention(q, k, v)  — no mask arg.
        # Match that behaviour here.
        attn_out = self.attn(q=q, k=k, v=v)
        attn_out = attn_out.reshape(B, L, self.context_dim)
        return self.out_proj(attn_out)


# =============================================================================
# Transformer Block
# =============================================================================

class WaypointBlock(nn.Module):
    """Single Waypoint transformer block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        mlp_ratio: int,
        layer_idx: int,
        prompt_conditioning: Optional[str],
        prompt_conditioning_period: int,
        prompt_embedding_dim: int,
        ctrl_conditioning_period: int,
        noise_conditioning: str,
        causal: bool = True,
        gated_attn: bool = True,
        rope: Optional[OrthoRoPE] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.attn = GatedSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            layer_idx=layer_idx,
            causal=causal,
            gated_attn=gated_attn,
            rope=rope,
        )
        
        # Feed-forward MLP
        self.mlp = MLP(d_model, d_model * mlp_ratio, d_model)
        
        # Conditioning head (6 modulation vectors)
        self.cond_head = CondHead(d_model, noise_conditioning)
        
        # Optional cross-attention for prompts (every prompt_conditioning_period layers)
        do_prompt_cond = (
            prompt_conditioning is not None
            and layer_idx % prompt_conditioning_period == 0
        )
        self.prompt_cross_attn = (
            CrossAttention(d_model, prompt_embedding_dim)  # Uses head_dim=64 internally
            if do_prompt_cond else None
        )
        
        # Optional MLPFusion for controls (every ctrl_conditioning_period layers)
        do_ctrl_cond = layer_idx % ctrl_conditioning_period == 0
        self.ctrl_mlpfusion = MLPFusion(d_model) if do_ctrl_cond else None
    
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        prompt_emb: Optional[torch.Tensor] = None,
        prompt_pad_mask: Optional[torch.Tensor] = None,
        ctrl_emb: Optional[torch.Tensor] = None,
        pos_emb: Optional[dict[str, torch.Tensor]] = None,
        kv_cache=None,
        update_cache: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] token features
            cond: [B, N, D] noise conditioning
            prompt_emb: [B, P, prompt_dim] prompt embeddings
            prompt_pad_mask: [B, P] prompt padding mask
            ctrl_emb: [B, N, D] control embeddings
            pos_emb: pos_ids dict (t_pos, y_pos, x_pos) for RoPE
            kv_cache: Per-layer KV cache dict for autoregressive generation
            update_cache: If True, write current K/V into cache (cache pass)
        """
        s0, b0, g0, s1, b1, g1 = self.cond_head(cond)

        residual = x
        x = ada_rmsnorm(x, s0, b0)
        x = self.attn(
            x, pos_ids=pos_emb, kv_cache=kv_cache, update_cache=update_cache
        )
        x = ada_gate(x, g0) + residual
        
        # Cross-attention for prompts
        if self.prompt_cross_attn is not None and prompt_emb is not None:
            x = self.prompt_cross_attn(
                rms_norm(x),
                context=rms_norm(prompt_emb),
                context_pad_mask=prompt_pad_mask,
            ) + x
        
        # MLPFusion for controls
        if self.ctrl_mlpfusion is not None and ctrl_emb is not None:
            x = self.ctrl_mlpfusion(rms_norm(x), rms_norm(ctrl_emb)) + x
        
        # MLP with AdaLN
        x = ada_gate(self.mlp(ada_rmsnorm(x, s1, b1)), g1) + x
        
        return x


# =============================================================================
# Main Transformer Stack
# =============================================================================

class WaypointTransformer(nn.Module):
    """Stack of Waypoint transformer blocks with shared OrthoRoPE."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        head_dim = config.d_model // config.n_heads
        # RoPE uses config.height / config.width directly (the PATCH grid
        # dimensions, e.g. 16×16).  The official model builds OrthoRoPE with
        # these same dimensions.  Do NOT multiply by patch size.
        rope = OrthoRoPE(
            height=config.height,
            width=config.width,
            n_frames=config.n_frames,
            head_dim=head_dim,
        )
        self.blocks = nn.ModuleList([
            WaypointBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                mlp_ratio=config.mlp_ratio,
                layer_idx=idx,
                prompt_conditioning=config.prompt_conditioning,
                prompt_conditioning_period=config.prompt_conditioning_period,
                prompt_embedding_dim=config.prompt_embedding_dim,
                ctrl_conditioning_period=config.ctrl_conditioning_period,
                noise_conditioning=config.noise_conditioning,
                causal=config.causal,
                gated_attn=config.gated_attn,
                rope=rope,
            )
            for idx in range(config.n_layers)
        ])

        if config.noise_conditioning in ("dit_air", "wan"):
            ref_proj = self.blocks[0].cond_head.cond_proj
            for blk in self.blocks[1:]:
                for blk_mod, ref_mod in zip(blk.cond_head.cond_proj, ref_proj):
                    blk_mod.weight = ref_mod.weight

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        prompt_emb: Optional[torch.Tensor] = None,
        prompt_pad_mask: Optional[torch.Tensor] = None,
        ctrl_emb: Optional[torch.Tensor] = None,
        pos_emb: Optional[dict[str, torch.Tensor]] = None,
        kv_cache=None,
        update_cache: bool = False,
    ) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x = block(
                x, cond, prompt_emb, prompt_pad_mask, ctrl_emb, pos_emb,
                kv_cache=layer_cache, update_cache=update_cache,
            )
        return x


# =============================================================================
# Main WorldModel
# =============================================================================

def is_blocks(name: str) -> bool:
    """FSDP shard condition for transformer blocks."""
    return ".blocks." in name


class WaypointWorldModel(CachableDiT):
    """
    Waypoint World Model for interactive video generation.
    
    Denoises a frame given:
    - All previous frames (via KV cache)
    - The prompt embedding
    - The controller input embedding (mouse, buttons, scroll)
    - The current noise level
    """
    
    # Required class attributes for CachableDiT (read from default config)
    _fsdp_shard_conditions = _DEFAULT_WAYPOINT_ARCH._fsdp_shard_conditions
    _compile_conditions = []
    param_names_mapping = _DEFAULT_WAYPOINT_ARCH.param_names_mapping
    reverse_param_names_mapping = _DEFAULT_WAYPOINT_ARCH.reverse_param_names_mapping
    lora_param_names_mapping: dict = {}
    
    def __init__(self, config, hf_config: dict = None):
        super().__init__(config=config, hf_config=hf_config or {})
        
        # Required instance attributes for BaseDiT
        self.hidden_size = config.d_model
        self.num_attention_heads = config.n_heads
        self.num_channels_latents = config.channels
        
        # Timestep/noise conditioning
        self.denoise_step_emb = NoiseConditioner(config.d_model)
        
        # Controller input embedding
        self.ctrl_emb = ControllerInputEmbedding(
            config.n_buttons, config.d_model, config.mlp_ratio
        )
        
        # CFG modules
        if config.ctrl_conditioning is not None:
            self.ctrl_cfg = CFG(config.d_model, config.ctrl_cond_dropout)
        else:
            self.ctrl_cfg = None
            
        if config.prompt_conditioning is not None:
            self.prompt_cfg = CFG(config.prompt_embedding_dim, config.prompt_cond_dropout)
        else:
            self.prompt_cfg = None
        
        # Main transformer
        self.transformer = WaypointTransformer(config)
        
        # Patch embedding and output
        self.patch = tuple(config.patch)
        ph, pw = self.patch
        
        self.patchify = nn.Conv2d(
            config.channels,
            config.d_model,
            kernel_size=self.patch,
            stride=self.patch,
            bias=False,
        )
        
        self.unpatchify = nn.Linear(
            config.d_model,
            config.channels * ph * pw,
            bias=True,
        )
        
        self.out_norm = AdaLN(config.d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        frame_timestamp: torch.Tensor,
        prompt_emb: Optional[torch.Tensor] = None,
        prompt_pad_mask: Optional[torch.Tensor] = None,
        mouse: Optional[torch.Tensor] = None,
        button: Optional[torch.Tensor] = None,
        scroll: Optional[torch.Tensor] = None,
        kv_cache=None,
        update_cache: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, C, H, W] - latent frames
            sigma: [B, N] - noise levels
            frame_timestamp: [B, N] - frame indices
            prompt_emb: [B, P, D] - prompt embeddings
            prompt_pad_mask: [B, P] - padding mask for prompts
            mouse: [B, N, 2] - mouse velocity
            button: [B, N, n_buttons] - button states
            scroll: [B, N, 1] - scroll wheel sign
            kv_cache: Optional KV cache for autoregressive generation
            update_cache: If True, write current frame K/V into cache (cache pass)
            
        Returns:
            [B, N, C, H, W] - denoised latent frames
        """
        B, N, C, H, W = x.shape
        ph, pw = self.patch
        
        assert H % ph == 0 and W % pw == 0, f"H={H}, W={W} must be divisible by patch={self.patch}"
        Hp, Wp = H // ph, W // pw
        # Waypoint expects tokens_per_frame = 256 (16*16); latent must match.
        expected_tokens = getattr(
            self.config, "tokens_per_frame", None
        )
        if expected_tokens is not None and Hp * Wp != expected_tokens:
            raise ValueError(
                f"Token layout mismatch: Hp*Wp={Hp * Wp} but "
                f"config.tokens_per_frame={expected_tokens}. "
                "Pipeline must use latent_h,w so that (H//ph)*(W//pw)==tokens_per_frame."
            )

        # Noise conditioning
        cond = self.denoise_step_emb(sigma)  # [B, N, d_model]
        
        # Control embedding
        if button is not None:
            ctrl_emb = self.ctrl_emb(mouse, button, scroll)  # [B, N, d_model]
            if self.ctrl_cfg is not None:
                ctrl_emb = self.ctrl_cfg(ctrl_emb)
        else:
            ctrl_emb = None
        
        # Prompt CFG
        if prompt_emb is not None and self.prompt_cfg is not None:
            prompt_emb = self.prompt_cfg(prompt_emb)
        
        # Patchify: [B, N, C, H, W] -> [B, N*Hp*Wp, d_model]
        x = x.reshape(B * N, C, H, W)
        x = self.patchify(x)  # [B*N, d_model, Hp, Wp]
        x = x.view(B, N, self.config.d_model, Hp, Wp)
        x = x.permute(0, 1, 3, 4, 2).reshape(B, N * Hp * Wp, self.config.d_model)

        # Position IDs for RoPE: t_pos, y_pos, x_pos each [B, L]
        # Official uses grid indices 0..width-1 and 0..height-1 (NOT multiplied
        # by patch size). The OrthoRoPE normalises to [-1,1] internally.
        L = N * Hp * Wp
        idx = torch.arange(Hp * Wp, device=x.device, dtype=torch.long)
        y_xy = idx // Wp       # 0..Hp-1  (matches official idx.div(width))
        x_xy = idx % Wp        # 0..Wp-1  (matches official idx.remainder(width))
        y_pos = (
            y_xy.unsqueeze(0).unsqueeze(0).expand(B, N, -1).reshape(B, L)
        )
        x_pos = (
            x_xy.unsqueeze(0).unsqueeze(0).expand(B, N, -1).reshape(B, L)
        )
        t_pos = (
            frame_timestamp.unsqueeze(2).expand(-1, -1, Hp * Wp).reshape(B, L)
        )
        pos_ids = {"t_pos": t_pos, "y_pos": y_pos, "x_pos": x_pos}

        x = self.transformer(
            x, cond, prompt_emb, prompt_pad_mask, ctrl_emb,
            pos_emb=pos_ids, kv_cache=kv_cache, update_cache=update_cache,
        )

        x = F.silu(self.out_norm(x, cond))
        x = self.unpatchify(x)  # [B, N*Hp*Wp, C*ph*pw]
        
        # Reshape back to image
        x = x.view(B, N, Hp, Wp, C, ph, pw)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, N, C, H, W)
        
        return x
    
    @classmethod
    def from_config(cls, config):
        """Create model from config."""
        return cls(config)

    def retrieve_cached_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """No-op: Waypoint does not use TeaCache; return states unchanged."""
        return hidden_states


# Entry point for model registry
EntryClass = WaypointWorldModel

