# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 Hugging Face Team and Overworld (original)
# Copyright (C) 2026 FastVideo Contributors (FastVideo port)
"""
Waypoint-1-Small World Model transformer for FastVideo.

This is a port of the Overworld Waypoint-1-Small model to FastVideo's
architecture, maintaining weight compatibility with the official checkpoint.

Reference: https://huggingface.co/Overworld/Waypoint-1-Small
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
        noop_mask,
    )

    _FLEX_ATTN_AVAILABLE = True
except ImportError:
    _FLEX_ATTN_AVAILABLE = False

from fastvideo.attention import DistributedAttention, LocalAttention
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.configs.models.dits.waypoint_transformer import (
    WaypointConfig,
)
from fastvideo.models.dits.base import BaseDiT
from fastvideo.platforms import AttentionBackendEnum

_DEFAULT_WAYPOINT_CONFIG = WaypointConfig()
_DEFAULT_WAYPOINT_ARCH = _DEFAULT_WAYPOINT_CONFIG.arch_config

if _FLEX_ATTN_AVAILABLE:
    # Match official numerics: its compiled flex_attention runs the fp32 matmuls
    # in tf32 (PyTorch's default "high" precision).
    torch.backends.cuda.matmul.allow_tf32 = True
    _flex_attention = torch.compile(
        flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
    )
else:
    _flex_attention = None


# =============================================================================
# Note: CtrlInput is defined in fastvideo/pipelines/basic/waypoint/waypoint_pipeline.py
# to avoid circular imports and keep pipeline-specific code separate
# =============================================================================


# =============================================================================
# Helper Functions
# =============================================================================


def rms_norm(x: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1), ), eps=eps)


class AdaLN(nn.Module):

    def __init__(
        self,
        d_model: int,
        bias: bool = False,
        eps: float = 1e-6,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.eps = eps
        self.fc = ReplicatedLinear(
            d_model,
            2 * d_model,
            bias=bias,
            params_dtype=params_dtype,
            prefix=prefix,
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        N = cond.shape[1]
        h = F.silu(cond)
        ab, _ = self.fc(h)
        ab = ab.view(B, N, 1,
                     2 * D).expand(-1, -1, L // N, -1).reshape(B, L, 2 * D)
        scale, shift = ab.chunk(2, dim=-1)
        return rms_norm(x) * (1 + scale) + shift


def ada_rmsnorm(
    x: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    eps: float | None = None,
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
# Basic Building Blocks (uses FastVideo MLP and AdaLN from layers)
# =============================================================================

def _waypoint_mlp(in_dim: int, hidden_dim: int, out_dim: int,
                  bias: bool = False) -> MLP:
    """Create Waypoint-style 2-layer MLP with SiLU."""
    return MLP(
        input_dim=in_dim,
        mlp_hidden_dim=hidden_dim,
        output_dim=out_dim,
        bias=bias,
        act_type="silu",
    )


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
        self.mlp = _waypoint_mlp(n_buttons + 3, d_model * mlp_ratio, d_model)
    
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
        self.mlp = _waypoint_mlp(freq_dim, d_model * 4, d_model)

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
        # This single RoPE is shared across all layers and applied to both q and
        # k, so memoize the angles on the pos_ids dict identity (fresh per
        # model.forward) to avoid recomputing them 2*n_layers times.
        self._angle_cache: tuple[dict, torch.Tensor, torch.Tensor] | None = None

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

        spatial_freqs = _pixel_frequencies(head_dim // 8, max_freq, device)
        # Map integer x/y_pos in [0, W-1]/[0, H-1] to HF's pos = linspace(-1+1/W, 1-1/W, W).
        w1 = max(W - 1, 1)
        h1 = max(H - 1, 1)
        norm_x = (-1.0 + 1.0 / W) + (2.0 - 2.0 / W) * x_pos.float() / w1
        norm_y = (-1.0 + 1.0 / H) + (2.0 - 2.0 / H) * y_pos.float() / h1
        angle_x = norm_x.unsqueeze(-1) * spatial_freqs.unsqueeze(0).unsqueeze(0)
        angle_x = angle_x.repeat_interleave(2, dim=-1)
        angle_y = norm_y.unsqueeze(-1) * spatial_freqs.unsqueeze(0).unsqueeze(0)
        angle_y = angle_y.repeat_interleave(2, dim=-1)

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
        cache = self._angle_cache
        if cache is not None and cache[0] is pos_ids:
            cos, sin = cache[1], cache[2]
        else:
            cos, sin = self.get_angles(
                pos_ids["t_pos"],
                pos_ids["y_pos"],
                pos_ids["x_pos"],
            )
            self._angle_cache = (pos_ids, cos, sin)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

        x_float = x.float()
        x0, x1 = x_float.unfold(-1, 2, 2).unbind(-1)

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
        # Input is the concatenation of x and cond (each d_model).
        self.mlp = _waypoint_mlp(2 * d_model, d_model, d_model)
    
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

        # Split fc_in weights so x and cond are fused without concatenating them.
        Wx, Wc = self.mlp.fc_in.weight.chunk(2, dim=1)
        x_reshaped = x.view(B, N, tokens_per_frame, D)
        h = F.linear(x_reshaped, Wx) + F.linear(cond, Wc).unsqueeze(2)
        h = F.silu(h)
        y = F.linear(h, self.mlp.fc_out.weight)
        
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
            nn.Parameter(torch.zeros(d_model))
            if noise_conditioning == "wan" else None
        )
        self.cond_proj = nn.ModuleList([
            ReplicatedLinear(d_model, d_model, bias=False)
            for _ in range(self.n_cond)
        ])

    def forward(self, cond: torch.Tensor):
        """cond [B, N, D] -> 6 tensors each [B, N, D]."""
        if self.bias_in is not None:
            cond = cond + self.bias_in
        h = F.silu(cond)
        return tuple(proj(h)[0] for proj in self.cond_proj)


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

        self.q_proj = ReplicatedLinear(d_model, d_model, bias=False)
        self.k_proj = ReplicatedLinear(
            d_model, n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = ReplicatedLinear(
            d_model, n_kv_heads * self.head_dim, bias=False
        )
        self.out_proj = ReplicatedLinear(d_model, d_model, bias=False)

        self.attn = DistributedAttention(
            num_heads=n_heads,
            head_size=self.head_dim,
            num_kv_heads=n_kv_heads,
            causal=causal,
            supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA,),
        )
        if gated_attn:
            self.gate_proj = ReplicatedLinear(n_heads, n_heads, bias=False)
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

        q, _ = self.q_proj(x)
        q = q.view(B, L, self.n_heads, self.head_dim)
        k, _ = self.k_proj(x)
        k = k.view(B, L, self.n_kv_heads, self.head_dim)
        v, _ = self.v_proj(x)
        v = v.view(B, L, self.n_kv_heads, self.head_dim)

        q, k = rms_norm(q), rms_norm(k)

        if self.rope is not None and pos_ids is not None:
            q = self.rope(q, pos_ids)
            k = self.rope(k, pos_ids)

        # Use flex_attention when no KV cache (match WanVideo/MatrixGame pattern).
        # With cache, use SDPA directly (DistributedAttention has no cache support).
        if kv_cache is None:
            if _FLEX_ATTN_AVAILABLE and _flex_attention is not None:
                padded_length = math.ceil(L / 128) * 128 - L
                total_len = L + padded_length
                block_mask = create_block_mask(
                    noop_mask,
                    B=None,
                    H=None,
                    Q_LEN=total_len,
                    KV_LEN=total_len,
                    device=q.device,
                    _compile=False,
                )
                q_pad = F.pad(
                    q.transpose(1, 2),
                    (0, 0, 0, padded_length),
                    value=0,
                )
                k_pad = F.pad(
                    k.transpose(1, 2),
                    (0, 0, 0, padded_length),
                    value=0,
                )
                v_pad = F.pad(
                    v.transpose(1, 2),
                    (0, 0, 0, padded_length),
                    value=0,
                )
                attn_out = _flex_attention(
                    q_pad,
                    k_pad,
                    v_pad,
                    block_mask=block_mask,
                    scale=self.scale,
                    enable_gqa=(self.n_kv_heads != self.n_heads),
                )[:, :, :L].transpose(1, 2)
            else:
                if self.n_kv_heads != self.n_heads:
                    rep = self.n_heads // self.n_kv_heads
                    k = k.repeat_interleave(rep, dim=2)
                    v = v.repeat_interleave(rep, dim=2)
                attn_out, _ = self.attn(
                    q, k, v, freqs_cis=None, attention_mask=None
                )
        elif kv_cache is not None:
            frame_t = int(pos_ids["t_pos"][0, 0].item())
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            q_t = q.transpose(1, 2)
            key, val, key_mask = self._upsert_kv_cache(
                kv_cache, k_t, v_t, frame_t, update_cache
            )
            if _FLEX_ATTN_AVAILABLE and _flex_attention is not None:
                # Match official: compiled flex_attention over the ring cache,
                # keyed by the per-slot validity mask (True = attend).
                written = key_mask.reshape(-1)  # [capacity] bool

                def mask_mod(b, h, q_idx, kv_idx):
                    return written[kv_idx]

                block_mask = create_block_mask(
                    mask_mod,
                    B=None,
                    H=None,
                    Q_LEN=q_t.shape[-2],
                    KV_LEN=written.shape[0],
                    device=q_t.device,
                    _compile=False,
                )
                attn_out = _flex_attention(
                    q_t, key, val,
                    block_mask=block_mask,
                    scale=self.scale,
                    enable_gqa=(self.n_kv_heads != self.n_heads),
                )
            else:
                attn_out = F.scaled_dot_product_attention(
                    q_t, key, val,
                    attn_mask=key_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=self.scale,
                    enable_gqa=(self.n_kv_heads != self.n_heads),
                )
            attn_out = attn_out.transpose(1, 2)  # [B, L, n_heads, head_dim]

        # Official gates the attn output BEFORE out_proj.
        if self.gated_attn:
            gates, _ = self.gate_proj(x[..., : self.n_heads])
            gates = torch.sigmoid(gates)  # [B, L, n_heads]
            attn_out = attn_out * gates.unsqueeze(-1)

        attn_out = attn_out.reshape(B, L, D)
        out, _ = self.out_proj(attn_out)
        return out

    def _upsert_kv_cache(
        self,
        kv_cache: dict,
        k: torch.Tensor,
        v: torch.Tensor,
        frame_t: int,
        update_cache: bool,
    ):
        """Per-layer ring KV cache matching official LayerKVCache.upsert.

        Ring of ``num_buckets`` slots (each tpf wide) over ``[0, L)`` plus a tail
        ``[L, L+tpf)`` that always holds the current frame. The current frame's
        queries attend over the whole capacity, masked to written slots minus the
        about-to-be-overwritten bucket on a write step. When not frozen and on a
        write step, the current frame is persisted into its ring bucket.
        """
        cache_k = kv_cache["k"]  # [B, n_kv_heads, capacity, head_dim]
        cache_v = kv_cache["v"]
        written = kv_cache["written"]  # [capacity] bool
        L = kv_cache["L"]
        tpf = kv_cache["tpf"]
        pinned_dilation = kv_cache["pinned_dilation"]
        num_buckets = kv_cache["num_buckets"]
        current_idx = kv_cache["current_idx"]  # [tpf] long, == arange(tpf)+L

        bucket = (frame_t + pinned_dilation - 1) // pinned_dilation
        slot = bucket % num_buckets
        base = slot * tpf
        ring_idx = kv_cache["frame_offsets"] + base  # [tpf] long in [0, L)
        write_step = (frame_t % pinned_dilation) == 0

        cache_k.index_copy_(2, current_idx, k)
        cache_v.index_copy_(2, current_idx, v)

        mask_written = written.clone()
        if write_step:
            mask_written[ring_idx] = False
        key_mask = mask_written.view(1, 1, 1, -1)

        frozen_ref = kv_cache.get("frozen_ref")
        is_frozen = update_cache is False or (frozen_ref is not None and frozen_ref[0])
        # Persist into the ring bucket only on a write step; on a non-write step
        # the destination is the tail (already written just above), so it's a no-op.
        if not is_frozen and write_step:
            cache_k.index_copy_(2, ring_idx, k)
            cache_v.index_copy_(2, ring_idx, v)
            written[ring_idx] = True

        return cache_k, cache_v, key_mask


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
        assert context_dim % head_dim == 0, f"context_dim {context_dim} must be divisible by head_dim {head_dim}"
        self.n_heads = context_dim // head_dim

        self.q_proj = ReplicatedLinear(d_model, context_dim, bias=False)
        self.k_proj = ReplicatedLinear(context_dim, context_dim, bias=False)
        self.v_proj = ReplicatedLinear(context_dim, context_dim, bias=False)
        self.out_proj = ReplicatedLinear(context_dim, d_model, bias=False)

        # LocalAttention: cross-attention is a local op, no sequence parallelism.
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

        q, _ = self.q_proj(x)
        q = q.view(B, L, self.n_heads, self.head_dim)
        k, _ = self.k_proj(context)
        k = k.view(B, S, self.n_heads, self.head_dim)
        v, _ = self.v_proj(context)
        v = v.view(B, S, self.n_heads, self.head_dim)
        q, k = rms_norm(q), rms_norm(k)

        # Official Waypoint cross-attention does NOT mask padding:
        #   out = flex_attention(q, k, v)  — no mask arg.
        # Match that behaviour here.
        attn_out = self.attn(q=q, k=k, v=v)
        attn_out = attn_out.reshape(B, L, self.context_dim)
        out, _ = self.out_proj(attn_out)
        return out


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

        self.mlp = _waypoint_mlp(d_model, d_model * mlp_ratio, d_model)
        # CondHead emits 6 modulation vectors (scale/bias/gate × 2).
        self.cond_head = CondHead(d_model, noise_conditioning)
        
        # Prompt cross-attention / control fusion run only every Nth layer.
        do_prompt_cond = (
            prompt_conditioning is not None
            and layer_idx % prompt_conditioning_period == 0
        )
        self.prompt_cross_attn = (
            CrossAttention(d_model, prompt_embedding_dim)
            if do_prompt_cond else None
        )
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

        if self.prompt_cross_attn is not None and prompt_emb is not None:
            x = self.prompt_cross_attn(
                rms_norm(x),
                context=rms_norm(prompt_emb),
                context_pad_mask=prompt_pad_mask,
            ) + x

        if self.ctrl_mlpfusion is not None and ctrl_emb is not None:
            x = self.ctrl_mlpfusion(rms_norm(x), rms_norm(ctrl_emb)) + x

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


class WaypointWorldModel(BaseDiT):
    """
    Waypoint World Model for interactive video generation.
    
    Denoises a frame given:
    - All previous frames (via KV cache)
    - The prompt embedding
    - The controller input embedding (mouse, buttons, scroll)
    - The current noise level
    """
    
    # Required class attributes for BaseDiT (read from default config)
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

        self.denoise_step_emb = NoiseConditioner(config.d_model)
        self.ctrl_emb = ControllerInputEmbedding(
            config.n_buttons, config.d_model, config.mlp_ratio
        )

        if config.ctrl_conditioning is not None:
            self.ctrl_cfg = CFG(config.d_model, config.ctrl_cond_dropout)
        else:
            self.ctrl_cfg = None

        if config.prompt_conditioning is not None:
            self.prompt_cfg = CFG(config.prompt_embedding_dim, config.prompt_cond_dropout)
        else:
            self.prompt_cfg = None

        self.transformer = WaypointTransformer(config)

        self.patch = tuple(config.patch)
        ph, pw = self.patch
        
        self.patchify = nn.Conv2d(
            config.channels,
            config.d_model,
            kernel_size=self.patch,
            stride=self.patch,
            bias=False,
        )
        
        self.unpatchify = ReplicatedLinear(
            config.d_model,
            config.channels * ph * pw,
            bias=True,
        )

        self.out_norm = AdaLN(config.d_model)

        self.__post_init__()

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

        cond = self.denoise_step_emb(sigma)  # [B, N, d_model]

        if button is not None:
            ctrl_emb = self.ctrl_emb(mouse, button, scroll)  # [B, N, d_model]
            if self.ctrl_cfg is not None:
                ctrl_emb = self.ctrl_cfg(ctrl_emb)
        else:
            ctrl_emb = None

        if prompt_emb is not None and self.prompt_cfg is not None:
            prompt_emb = self.prompt_cfg(prompt_emb)

        # Patchify: [B, N, C, H, W] -> [B, N*Hp*Wp, d_model]
        x = x.reshape(B * N, C, H, W)
        x = self.patchify(x)
        x = x.view(B, N, self.config.d_model, Hp, Wp)
        x = x.permute(0, 1, 3, 4, 2).reshape(B, N * Hp * Wp, self.config.d_model)

        # RoPE position ids use raw grid indices 0..Wp-1 / 0..Hp-1 (NOT scaled by
        # patch size); OrthoRoPE normalises to [-1, 1] internally.
        L = N * Hp * Wp
        idx = torch.arange(Hp * Wp, device=x.device, dtype=torch.long)
        y_xy = idx // Wp
        x_xy = idx % Wp
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
        x, _ = self.unpatchify(x)
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

