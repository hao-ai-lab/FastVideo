# SPDX-License-Identifier: Apache-2.0
"""DaVinci-MagiHuman DiT implementation for FastVideo.

Architecture reference: https://github.com/GAIR-NLP/daVinci-MagiHuman
  - 15B unified single-stream Transformer, 40 layers
  - Sandwich design: layers [0,1,2,3,36,37,38,39] have per-modality (MoE-style)
    QKV/proj/MLP weights; middle 32 layers share a single weight across modalities
  - GQA: 40 query heads, 8 KV heads, head_dim=128
  - Per-head scalar attention gating (sigmoid)
  - GELU7 activation for first 4 layers, SwiGLU7 for remaining 36
  - No cross-attention — all modalities (video, text, audio) share self-attention
  - Flow matching scheduler, shift=5.0
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.attention import DistributedAttention
from fastvideo.layers.layernorm import RMSNorm
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.rotary_embedding import apply_rotary_emb
from fastvideo.models.dits.base import BaseDiT

# ---------------------------------------------------------------------------
# Architecture constants (from inference/common/config.py ModelConfig)
# ---------------------------------------------------------------------------
_MM_LAYERS = [0, 1, 2, 3, 36, 37, 38, 39]  # per-modality (MoE) layers
_GELU7_LAYERS = [0, 1, 2, 3]  # first 4 use GELU7; rest use SwiGLU7
_NUM_LAYERS = 40
_HIDDEN_SIZE = 5120
_HEAD_DIM = 128
_NUM_HEADS_Q = 40  # hidden_size // head_dim
_NUM_HEADS_KV = 8  # num_query_groups
_SWIGLU_INTERMEDIATE = 13652  # int(5120 * 4 * 2/3) // 4 * 4
_GELU_INTERMEDIATE = 20480  # 5120 * 4
_VIDEO_IN_CHANNELS = 192  # z_dim=48, patch=(1,2,2) → 48*1*2*2
_AUDIO_IN_CHANNELS = 64
_TEXT_IN_CHANNELS = 3584  # t5gemma-9b hidden dim
_PATCH_SIZE = (1, 2, 2)
_Z_DIM = 48  # VAE latent channels
_NUM_MODALITIES = 3  # video, text, audio


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DaVinciConfig:
    num_layers: int = _NUM_LAYERS
    hidden_size: int = _HIDDEN_SIZE
    head_dim: int = _HEAD_DIM
    num_heads_q: int = _NUM_HEADS_Q
    num_heads_kv: int = _NUM_HEADS_KV
    video_in_channels: int = _VIDEO_IN_CHANNELS
    audio_in_channels: int = _AUDIO_IN_CHANNELS
    text_in_channels: int = _TEXT_IN_CHANNELS
    patch_size: tuple = _PATCH_SIZE
    z_dim: int = _Z_DIM
    mm_layers: list = field(default_factory=lambda: list(_MM_LAYERS))
    gelu7_layers: list = field(default_factory=lambda: list(_GELU7_LAYERS))
    enable_attn_gating: bool = True


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def _swiglu7(x: torch.Tensor) -> torch.Tensor:
    """Clamped SiLU gate * linear, as in the official daVinci implementation."""
    x = x.float()
    x_gate, x_linear = x[..., ::2], x[..., 1::2]
    x_gate = x_gate.clamp(max=7.0)
    x_linear = x_linear.clamp(-7.0, 7.0)
    return (x_gate * torch.sigmoid(1.702 * x_gate) * (x_linear + 1)).to(torch.bfloat16)


def _gelu7(x: torch.Tensor) -> torch.Tensor:
    """Clamped GELU-style gate (no split), used in first 4 layers."""
    x = x.float()
    x_gate = x.clamp(max=7.0)
    return (x_gate * torch.sigmoid(1.702 * x_gate)).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# RMSNorm — per-modality variant
# For sandwich layers (mm_layers) we store separate norm weights per modality.
# For middle layers a single norm weight applies to all tokens.
# ---------------------------------------------------------------------------

class ModalityRMSNorm(nn.Module):
    """RMSNorm matching official MultiModalityRMSNorm weight layout.

    Weight shape: [dim * num_modalities] — stacked flat, zero-initialized.
    Forward: x * (weight_slice + 1)  (residual parameterization).
    """

    def __init__(self, hidden_size: int, num_modalities: int = 1, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        # Matches official: [dim * num_modality], init=zeros
        self.weight = nn.Parameter(torch.zeros(hidden_size * num_modalities))

    def _rms_norm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        # Residual: x * (w + 1)
        return (x * rms * (w.float() + 1)).to(orig_dtype)

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.num_modalities == 1 or modality_ids is None:
            return self._rms_norm(x, self.weight)
        # Slice expert weights: [dim * num_mod] → chunks of [dim]
        w_chunks = self.weight.chunk(self.num_modalities, dim=0)
        out = torch.empty_like(x)
        for m in range(self.num_modalities):
            mask = modality_ids == m
            if mask.any():
                out[mask] = self._rms_norm(x[mask], w_chunks[m])
        return out


# ---------------------------------------------------------------------------
# Adapter — per-modality input projection + RoPE
# ---------------------------------------------------------------------------

class DaVinciAdapter(nn.Module):
    """Projects each modality to hidden_size and computes RoPE embeddings."""

    def __init__(self, config: DaVinciConfig):
        super().__init__()
        self.video_proj = nn.Linear(config.video_in_channels, config.hidden_size, bias=True)
        self.text_proj = nn.Linear(config.text_in_channels, config.hidden_size, bias=True)
        self.audio_proj = nn.Linear(config.audio_in_channels, config.hidden_size, bias=True)
        # RoPE embed dim = head_dim // 2 (sin+cos split later)
        self.rope_dim = config.head_dim

    def forward(
        self,
        video_tokens: torch.Tensor,   # (N_v, video_in_channels)
        text_tokens: torch.Tensor,    # (N_t, text_in_channels)
        audio_tokens: Optional[torch.Tensor],  # (N_a, audio_in_channels) or None
        video_coords: torch.Tensor,   # (N_v, 3) — t, h, w coords
        text_coords: torch.Tensor,    # (N_t, 3)
        audio_coords: Optional[torch.Tensor],  # (N_a, 3) or None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            hidden_states: (N_total, hidden_size) joint sequence
            modality_ids: (N_total,) int tensor — 0=video, 1=text, 2=audio
            rope: (N_total, head_dim) — precomputed sin/cos for RoPE
        """
        parts = []
        mod_ids = []

        v = self.video_proj(video_tokens)
        parts.append(v)
        mod_ids.append(torch.zeros(v.shape[0], dtype=torch.long, device=v.device))

        t = self.text_proj(text_tokens)
        parts.append(t)
        mod_ids.append(torch.ones(t.shape[0], dtype=torch.long, device=t.device))

        all_coords = [video_coords, text_coords]

        if audio_tokens is not None and audio_coords is not None:
            a = self.audio_proj(audio_tokens)
            parts.append(a)
            mod_ids.append(torch.full((a.shape[0],), 2, dtype=torch.long, device=a.device))
            all_coords.append(audio_coords)

        hidden_states = torch.cat(parts, dim=0)
        modality_ids = torch.cat(mod_ids, dim=0)
        coords = torch.cat(all_coords, dim=0)

        rope = self._build_rope(coords, hidden_states.device, hidden_states.dtype)
        return hidden_states, modality_ids, rope

    def _build_rope(
        self,
        coords: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build RoPE sin/cos from (N, 3) coordinate grid."""
        half_dim = self.rope_dim // 2
        freq = 1.0 / (10000.0 ** (torch.arange(0, half_dim, 2, device=device).float() / half_dim))
        # coords: (N, 3) → use all 3 dims
        angles = coords.float().unsqueeze(-1) * freq  # (N, 3, half_dim//2)
        angles = angles.flatten(1)  # (N, 3 * half_dim//2)
        sin = angles.sin().to(dtype)
        cos = angles.cos().to(dtype)
        return torch.cat([sin, cos], dim=-1)  # (N, rope_dim)


# ---------------------------------------------------------------------------
# Stacked linear for MoE (sandwich) layers
# Mirrors official weight layout: weight shape [num_experts * out, in]
# Using this allows direct weight loading without remapping.
# Middle layers use ReplicatedLinear for LoRA compatibility.
# ---------------------------------------------------------------------------

class MoELinear(nn.Module):
    """Stacked per-modality linear for sandwich layers (num_experts=3).

    Weight layout matches official BaseLinear/NativeMoELinear:
      weight: [num_experts * out_features, in_features]
    Forward dispatches each modality group to its own weight slice.
    """

    def __init__(self, in_features: int, out_features: int, num_experts: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.empty(num_experts * out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_experts * out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        out = torch.empty(x.shape[0], self.out_features, device=x.device, dtype=x.dtype)
        w_chunks = self.weight.chunk(self.num_experts, dim=0)
        b_chunks = self.bias.chunk(self.num_experts) if self.bias is not None else [None] * self.num_experts
        for m in range(self.num_experts):
            mask = modality_ids == m
            if mask.any():
                out[mask] = F.linear(x[mask], w_chunks[m], b_chunks[m])
        return out


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class DaVinciAttention(nn.Module):

    def __init__(self, config: DaVinciConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_mm_layer = layer_idx in config.mm_layers

        q_size = config.num_heads_q * config.head_dim
        kv_size = config.num_heads_kv * config.head_dim
        gating_size = config.num_heads_q if config.enable_attn_gating else 0
        qkv_out = q_size + kv_size * 2 + gating_size

        self.q_size = q_size
        self.kv_size = kv_size
        self.gating_size = gating_size

        self.pre_norm = ModalityRMSNorm(
            config.hidden_size,
            num_modalities=_NUM_MODALITIES if self.is_mm_layer else 1,
        )

        if self.is_mm_layer:
            self.linear_qkv = MoELinear(config.hidden_size, qkv_out, _NUM_MODALITIES, bias=False)
            self.linear_proj = MoELinear(q_size, config.hidden_size, _NUM_MODALITIES, bias=False)
        else:
            self.linear_qkv = ReplicatedLinear(config.hidden_size, qkv_out, bias=False)
            self.linear_proj = ReplicatedLinear(q_size, config.hidden_size, bias=False)

        self.q_norm = ModalityRMSNorm(
            config.head_dim,
            num_modalities=_NUM_MODALITIES if self.is_mm_layer else 1,
        )
        self.k_norm = ModalityRMSNorm(
            config.head_dim,
            num_modalities=_NUM_MODALITIES if self.is_mm_layer else 1,
        )

        self.attn = DistributedAttention(
            num_heads=config.num_heads_q,
            head_size=config.head_dim,
            num_kv_heads=config.num_heads_kv,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        modality_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_norm(hidden_states, modality_ids if self.is_mm_layer else None)

        if self.is_mm_layer:
            qkv = self.linear_qkv(hidden_states, modality_ids)
        else:
            qkv, _ = self.linear_qkv(hidden_states)

        q, k, v, g = torch.split(
            qkv, [self.q_size, self.kv_size, self.kv_size, self.gating_size], dim=-1
        )

        N = q.shape[0]
        q = q.view(N, self.config.num_heads_q, self.config.head_dim)
        k = k.view(N, self.config.num_heads_kv, self.config.head_dim)
        v = v.view(N, self.config.num_heads_kv, self.config.head_dim)

        mod_ids_per_head = modality_ids if self.is_mm_layer else None
        q = self.q_norm(q, mod_ids_per_head)
        k = self.k_norm(k, mod_ids_per_head)

        # Apply RoPE
        sin, cos = rope.chunk(2, dim=-1)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Self-attention via FastVideo DistributedAttention
        attn_out = self.attn(q, k, v)  # (N, num_heads_q, head_dim)

        if self.config.enable_attn_gating:
            g = g.view(N, self.config.num_heads_q, 1)
            attn_out = attn_out * torch.sigmoid(g)

        attn_out = attn_out.reshape(N, self.config.num_heads_q * self.config.head_dim)

        if self.is_mm_layer:
            out = self.linear_proj(attn_out, modality_ids)
        else:
            out, _ = self.linear_proj(attn_out)

        return out


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class DaVinciMLP(nn.Module):

    def __init__(self, config: DaVinciConfig, layer_idx: int):
        super().__init__()
        self.is_mm_layer = layer_idx in config.mm_layers
        self.use_gelu7 = layer_idx in config.gelu7_layers

        if self.use_gelu7:
            intermediate = _GELU_INTERMEDIATE
            up_out = intermediate  # no gate split
        else:
            intermediate = _SWIGLU_INTERMEDIATE
            up_out = intermediate * 2  # gate + linear interleaved

        self.pre_norm = ModalityRMSNorm(
            config.hidden_size,
            num_modalities=_NUM_MODALITIES if self.is_mm_layer else 1,
        )

        if self.is_mm_layer:
            self.up_gate_proj = MoELinear(config.hidden_size, up_out, _NUM_MODALITIES, bias=False)
            self.down_proj = MoELinear(intermediate, config.hidden_size, _NUM_MODALITIES, bias=False)
        else:
            self.up_gate_proj = ReplicatedLinear(config.hidden_size, up_out, bias=False)
            self.down_proj = ReplicatedLinear(intermediate, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x, modality_ids if self.is_mm_layer else None)

        if self.is_mm_layer:
            x = self.up_gate_proj(x, modality_ids)
        else:
            x, _ = self.up_gate_proj(x)

        x = _gelu7(x) if self.use_gelu7 else _swiglu7(x)

        if self.is_mm_layer:
            x = self.down_proj(x, modality_ids)
        else:
            x, _ = self.down_proj(x)

        return x


# ---------------------------------------------------------------------------
# Transformer layer
# ---------------------------------------------------------------------------

class DaVinciTransformerLayer(nn.Module):

    def __init__(self, config: DaVinciConfig, layer_idx: int):
        super().__init__()
        self.attention = DaVinciAttention(config, layer_idx)
        self.mlp = DaVinciMLP(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        modality_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(hidden_states, rope, modality_ids)
        hidden_states = hidden_states + self.mlp(hidden_states, modality_ids)
        return hidden_states


# ---------------------------------------------------------------------------
# Top-level DiT
# ---------------------------------------------------------------------------

class DaVinciDiT(BaseDiT):
    """FastVideo implementation of the DaVinci-MagiHuman DiT."""

    param_names_mapping: list = []
    _fsdp_shard_conditions: list = []
    _compile_conditions: list = []

    def __init__(self, config, hf_config: dict, **kwargs):
        super().__init__(config, hf_config, **kwargs)
        arch = DaVinciConfig()

        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_heads_q
        self.num_channels_latents = arch.z_dim

        self.adapter = DaVinciAdapter(arch)
        self.layers = nn.ModuleList(
            [DaVinciTransformerLayer(arch, i) for i in range(arch.num_layers)]
        )
        self.final_norm_video = ModalityRMSNorm(arch.hidden_size)
        self.final_norm_audio = ModalityRMSNorm(arch.hidden_size)
        self.final_proj_video = nn.Linear(arch.hidden_size, arch.video_in_channels, bias=False)
        self.final_proj_audio = nn.Linear(arch.hidden_size, arch.audio_in_channels, bias=False)

    def _patchify(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert (B, C, T, H, W) latents to (N, C*pt*ph*pw) tokens + coords."""
        B, C, T, H, W = latents.shape
        pt, ph, pw = _PATCH_SIZE
        latents = latents.reshape(B, C, T // pt, pt, H // ph, ph, W // pw, pw)
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4)  # (B, Tp, Hp, Wp, D)
        Tp, Hp, Wp = T // pt, H // ph, W // pw

        # Build coordinate grid (t, h, w) for RoPE
        t_idx = torch.arange(Tp, device=latents.device)
        h_idx = torch.arange(Hp, device=latents.device)
        w_idx = torch.arange(Wp, device=latents.device)
        coords = torch.stack(
            torch.meshgrid(t_idx, h_idx, w_idx, indexing="ij"), dim=-1
        ).reshape(-1, 3).float()

        tokens = latents.reshape(B * Tp * Hp * Wp, -1)
        return tokens, coords

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        audio_hidden_states: Optional[torch.Tensor] = None,
        audio_coords: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: video latents (B, C, T, H, W)
            encoder_hidden_states: text tokens (B, N_t, text_in_channels)
            timestep: not used in daVinci (flow matching is handled externally)
            audio_hidden_states: (B, N_a, audio_in_channels) or None
            audio_coords: (B, N_a, 3) or None
        Returns:
            video latents (B, C, T, H, W)
        """
        B = hidden_states.shape[0]

        # Patchify video
        video_tokens, video_coords = self._patchify(hidden_states)

        # Text tokens: (B, N_t, D) → flatten batch dim for now (B=1 inference)
        text_tokens = encoder_hidden_states.reshape(-1, encoder_hidden_states.shape[-1])
        text_coords = torch.zeros(text_tokens.shape[0], 3, device=text_tokens.device)

        # Audio tokens
        a_tokens = audio_hidden_states.reshape(-1, audio_hidden_states.shape[-1]) if audio_hidden_states is not None else None
        a_coords = audio_coords.reshape(-1, 3) if audio_coords is not None else None

        # Project all modalities and build joint sequence
        x, modality_ids, rope = self.adapter(
            video_tokens, text_tokens, a_tokens,
            video_coords, text_coords, a_coords,
        )

        # Transformer
        for layer in self.layers:
            x = layer(x, rope, modality_ids)

        # Extract video tokens and project back
        video_mask = modality_ids == 0
        x_video = self.final_norm_video(x[video_mask])
        x_video = self.final_proj_video(x_video)

        # Reconstruct spatial layout
        T, H, W = hidden_states.shape[2], hidden_states.shape[3], hidden_states.shape[4]
        pt, ph, pw = _PATCH_SIZE
        Tp, Hp, Wp = T // pt, H // ph, W // pw

        x_video = x_video.reshape(B, Tp, Hp, Wp, _Z_DIM, pt, ph, pw)
        x_video = x_video.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(B, _Z_DIM, T, H, W)
        return x_video