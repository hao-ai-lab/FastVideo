# SPDX-License-Identifier: Apache-2.0
"""FastVideo-native Cosmos3 omni DiT (``Cosmos3VFMTransformer``).

Numerical-parity port of the official ``cosmos_framework`` ``Cosmos3VFMNetwork``
(the ``Qwen3-VL-text`` MoT backbone + the VFM vision / action / sound heads).
The module tree mirrors the published *diffusers* checkpoint layout so a
converter can strict-load with a near-identity ``param_names_mapping``:

* top level: ``embed_tokens`` / ``norm`` / ``norm_moe_gen`` / ``lm_head`` /
  ``proj_in`` / ``proj_out`` / ``time_embedder.linear_{1,2}`` plus the dormant
  ``action_*`` / ``audio_*`` heads (constructed for strict-load parity);
* per layer ``layers.{i}``: dual-pathway ``self_attn`` with understanding
  (``to_{q,k,v}`` / ``to_out``) and generation (``add_{q,k,v}_proj`` /
  ``to_add_out``) projections + per-head QK-norms (``norm_{q,k}`` for und,
  ``norm_added_{q,k}`` for gen), ``mlp`` (und) and ``mlp_moe_gen`` (gen) SwiGLU
  blocks, and four RMSNorms.

The forward replicates the framework contract for the video path:
``patchify -> proj_in -> (+ 3d-rope additive latent pos emb) -> scatter-add
timestep embeds onto noisy patches -> dual-pathway decoder (causal text +
full vision two-way attention, GQA, per-head QK-norm before RoPE, unified
3D-MRoPE) -> und/gen final norms -> proj_out on noisy vision patches ->
unpatchify``.

RoPE is applied with the framework's exact ``rotate_half`` math (contiguous
split-half: ``q * cos + rotate_half(q) * sin``) rather than the interleaved
``apply_rotary_emb`` helper, and attention uses plain SDPA so the whole module
runs on CPU / float32 for parity testing. No diffusers / transformers
model-class imports happen at runtime — the DiT is fully native.

This module also keeps two pure, standalone math utilities used by the Tier-A
scaffold parity tests: the unified-3D mRoPE position-ID generators
(``compute_mrope_position_ids_text`` / ``compute_mrope_position_ids_vision``)
and the batched ``patchify`` / ``unpatchify`` ``[B,C,T,H,W] <-> [B,N,p*p*C]``
roundtrip helpers.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.configs.models.dits.cosmos3 import Cosmos3VideoConfig
from fastvideo.layers.layernorm import RMSNorm
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.visual_embedding import timestep_embedding
from fastvideo.models.dits.base import BaseDiT

EntryClass = ["Cosmos3VFMTransformer"]


# ===========================================================================
# Standalone mRoPE position-ID generators (unified 3D mRoPE)
# ===========================================================================
def compute_mrope_position_ids_text(
    num_tokens: int,
    temporal_offset: int,
) -> tuple[torch.Tensor, int]:
    """Generate 3D mRoPE position IDs for text tokens.

    Text tokens broadcast a single monotonically-increasing position-ID
    sequence across all three (t, h, w) axes.
    """
    ids = torch.arange(num_tokens, dtype=torch.long) + temporal_offset
    mrope_ids = ids.unsqueeze(0).expand(3, -1).contiguous()
    return mrope_ids, temporal_offset + num_tokens


def compute_mrope_position_ids_vision(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    base_temporal_compression_factor: int | None = None,
    enable_fps_modulation: bool = True,
    start_frame_offset: int = 0,
) -> tuple[torch.Tensor, int]:
    """Generate 3D mRoPE position IDs for vision tokens.

    Builds a ``(t, h, w)`` position grid (Qwen3-VL style, spatial indices reset
    per temporal segment) flattened in t-major order. Optionally modulates the
    temporal axis by ``base_fps / tcf * (1 / (fps / tcf))`` so two clips at
    different FPS retain wall-clock-aligned temporal positions.
    """
    fps_modulation = enable_fps_modulation and fps is not None

    if fps_modulation:
        assert fps is not None
        tps = fps / temporal_compression_factor
        effective_base_tcf = (base_temporal_compression_factor
                              if base_temporal_compression_factor is not None else temporal_compression_factor)
        base_tps = base_fps / effective_base_tcf
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        t_index = (((frame_indices + start_frame_offset) / tps * base_tps + temporal_offset).view(-1, 1).expand(
            -1, grid_h * grid_w).flatten())
    else:
        t_index = (torch.arange(grid_t, dtype=torch.long).view(-1, 1).expand(-1, grid_h * grid_w).flatten() +
                   int(temporal_offset) + start_frame_offset)

    h_index = (torch.arange(grid_h, dtype=torch.long).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten())
    w_index = (torch.arange(grid_w, dtype=torch.long).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten())

    if fps_modulation:
        mrope_ids = torch.stack([t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0)
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)

    next_offset = math.floor(mrope_ids.max().item()) + 1
    return mrope_ids, next_offset


# ===========================================================================
# RoPE helpers (match cosmos_framework qwen3_vl rotate_half + apply_rotary_pos_emb)
# ===========================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Contiguous split-half rotation, identical to qwen3_vl.rotate_half."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE exactly as cosmos_framework qwen3_vl.apply_rotary_pos_emb.

    ``q`` / ``k`` are ``[N, heads, head_dim]`` and ``cos`` / ``sin`` are
    ``[N, head_dim]`` (per-token); ``unsqueeze_dim=1`` broadcasts cos/sin over
    the head axis.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Cosmos3TextRotaryEmbedding(nn.Module):
    """Unified 3D-MRoPE rotary embedding (qwen3_vl Cosmos3 flavor).

    Reproduces ``Qwen3VLTextRotaryEmbedding``: ``inv_freq`` from the default
    RoPE init (``1 / theta**(arange(0, head_dim, 2) / head_dim)``), interleaved
    MRoPE mixing of the T/H/W frequency bands per ``mrope_section``, and a final
    ``cat([freqs, freqs])`` so cos/sin are ``[N, head_dim]``.
    """

    def __init__(
        self,
        head_dim: int,
        rope_theta: float,
        mrope_section: list[int],
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = float(rope_theta)
        self.mrope_section = list(mrope_section)
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)
        self.attention_scaling = 1.0  # default RoPE has unit attention scaling

    def _compute_inv_freq(self, device: torch.device | None = None) -> torch.Tensor:
        exponent = torch.arange(0, self.head_dim, 2, dtype=torch.int64, device=device).float() / self.head_dim
        return 1.0 / (self.rope_theta**exponent)

    def reset_inv_freq(self, device: torch.device) -> None:
        """Recompute the non-persistent ``inv_freq`` on ``device`` after a
        meta-device weight load (it is derived from ``rope_theta`` and is not
        part of the checkpoint)."""
        self.register_buffer("inv_freq", self._compute_inv_freq(device), persistent=False)

    def _apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        """freqs: [3, N, head_dim//2] -> [N, head_dim//2] (interleaved T/H/W)."""
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor, device: torch.device,
                dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids: [N] (1D) or [3, N] (mrope) -> (cos, sin) each [N, head_dim]."""
        if position_ids.ndim == 1:
            position_ids = position_ids[None, :].expand(3, -1)  # [3, N]
        position_ids = position_ids.to(device)
        inv_freq = self.inv_freq.to(device).float()  # [head_dim//2]
        inv_freq_expanded = inv_freq[None, :, None].expand(3, -1, 1)  # [3, head_dim//2, 1]
        position_ids_expanded = position_ids[:, None, :].float()  # [3, 1, N]
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # [3, N, head_dim//2]
        freqs = self._apply_interleaved_mrope(freqs)  # [N, head_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [N, head_dim]
        cos = (emb.cos() * self.attention_scaling).to(dtype)
        sin = (emb.sin() * self.attention_scaling).to(dtype)
        return cos, sin


# ===========================================================================
# 3D-RoPE additive latent position embedding (VideoRopePosition3DEmb)
# ===========================================================================
class Cosmos3VideoRopePosition3DEmb(nn.Module):
    """Additive 3D-RoPE-style latent position embedding.

    Mirrors ``cosmos_framework`` ``VideoRopePosition3DEmb`` (used when
    ``position_embedding_type == "3d_rope"``). Produces an additive
    ``[N_vision, head_dim]`` tensor concatenated over per-latent (t, h, w)
    grids. ``enable_fps_modulation`` is supported for completeness but the
    video path keeps it disabled by default.
    """

    def __init__(
        self,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        base_temporal_compression_factor: int = 4,
        temporal_compression_factor: int = 4,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = False,
    ) -> None:
        super().__init__()
        self.base_tps = base_fps / base_temporal_compression_factor
        self.temporal_compression_factor = temporal_compression_factor
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        self.enable_fps_modulation = enable_fps_modulation

        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        self._dim_h = dim_h
        self._dim_t = dim_t

        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[:(dim_h // 2)].float() / dim_h,
            persistent=True,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[:(dim_t // 2)].float() / dim_t,
            persistent=True,
        )
        self.h_ntk_factor = h_extrapolation_ratio**(dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio**(dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio**(dim_t / (dim_t - 2))

    def _generate(self, t: int, h: int, w: int, device: torch.device, fps: torch.Tensor | None,
                  start_frame_offset: int) -> torch.Tensor:
        tps = (fps / self.temporal_compression_factor) if fps is not None else None

        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor

        spatial = self.dim_spatial_range.to(device).float()
        temporal = self.dim_temporal_range.to(device).float()
        h_spatial_freqs = 1.0 / (h_theta**spatial)
        w_spatial_freqs = 1.0 / (w_theta**spatial)
        temporal_freqs = 1.0 / (t_theta**temporal)

        max_needed = max(t, h, w)
        seq = torch.arange(max_needed, device=device, dtype=torch.float32)
        half_emb_h = torch.outer(seq[:h], h_spatial_freqs)  # [h, dim_h/2]
        half_emb_w = torch.outer(seq[:w], w_spatial_freqs)  # [w, dim_w/2]
        frame_indices = seq[:t]  # [t]

        if self.enable_fps_modulation and tps is not None:
            scaled_time = (frame_indices + start_frame_offset) / tps[:1] * self.base_tps
            half_emb_t = torch.outer(scaled_time, temporal_freqs)
        else:
            half_emb_t = torch.outer(frame_indices, temporal_freqs)  # [t, dim_t/2]

        emb_t = half_emb_t[:, None, None, :].expand(t, h, w, -1)
        emb_h = half_emb_h[None, :, None, :].expand(t, h, w, -1)
        emb_w = half_emb_w[None, None, :, :].expand(t, h, w, -1)
        rope = torch.cat([emb_t, emb_h, emb_w] * 2, dim=-1)  # [t, h, w, head_dim]
        return rope.reshape(t * h * w, -1).float()

    def forward(
        self,
        token_shapes: list[tuple[int, int, int]],
        fps: torch.Tensor | None = None,
        start_frame_offset: int = 0,
    ) -> torch.Tensor:
        device = self.dim_spatial_range.device
        out = []
        for i, (t, h, w) in enumerate(token_shapes):
            video_fps = fps[i:i + 1] if fps is not None else None
            out.append(self._generate(t, h, w, device, video_fps, start_frame_offset))
        return torch.cat(out, dim=0)  # [N_vision, head_dim]


# ===========================================================================
# Timestep embedder (DiT-style; matches modeling_utils.TimestepEmbedder)
# ===========================================================================
class Cosmos3TimestepEmbedder(nn.Module):
    """Sinusoidal timestep -> MLP embedder. Checkpoint keys: ``linear_{1,2}``."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.hidden_size = hidden_size
        self.linear_1 = ReplicatedLinear(frequency_embedding_size, hidden_size, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = ReplicatedLinear(hidden_size, hidden_size, bias=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        # Sinusoidal runs in fp32 (timestep_scale makes inputs tiny); cast to the
        # MLP weight dtype (bf16 at inference; no-op in the fp32 parity tests).
        t_freq = t_freq.to(self.linear_1.weight.dtype)
        h, _ = self.linear_1(t_freq)
        h = self.act(h)
        out, _ = self.linear_2(h)
        return out


# ===========================================================================
# SwiGLU MLP (Qwen3-VL-text dense MLP)
# ===========================================================================
class Cosmos3MLP(nn.Module):
    """SwiGLU MLP: ``down(act(gate(x)) * up(x))``. Keys: gate/up/down_proj."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = ReplicatedLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = ReplicatedLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = ReplicatedLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        out, _ = self.down_proj(self.act_fn(gate) * up)
        return out


# ===========================================================================
# Dual-pathway packed two-way attention
# ===========================================================================
def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, is_causal: bool, scale: float) -> torch.Tensor:
    """SDPA on ``[S, heads, head_dim]`` with GQA broadcast. Returns same layout."""
    n_heads = q.shape[1]
    n_kv = k.shape[1]
    q_ = q.permute(1, 0, 2).unsqueeze(0)  # [1, heads, S, head_dim]
    k_ = k.permute(1, 0, 2).unsqueeze(0)
    v_ = v.permute(1, 0, 2).unsqueeze(0)
    if n_kv != n_heads:
        k_ = k_.repeat_interleave(n_heads // n_kv, dim=1)
        v_ = v_.repeat_interleave(n_heads // n_kv, dim=1)
    out = F.scaled_dot_product_attention(q_, k_, v_, is_causal=is_causal, scale=scale)
    return out.squeeze(0).permute(1, 0, 2)  # [S, heads, head_dim]


class Cosmos3DualAttention(nn.Module):
    """Dual-pathway packed attention (understanding + generation).

    Understanding (text) tokens use causal self-attention; generation (vision)
    tokens use full attention where the gen query attends to ALL tokens
    (und ++ gen). GQA with per-head QK-norm applied *before* RoPE. Checkpoint
    keys: und ``to_{q,k,v}`` / ``to_out`` / ``norm_{q,k}``; gen
    ``add_{q,k,v}_proj`` / ``to_add_out`` / ``norm_added_{q,k}``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        eps: float,
        attention_bias: bool,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5

        q_dim = num_attention_heads * head_dim
        kv_dim = num_key_value_heads * head_dim

        # Understanding pathway
        self.to_q = ReplicatedLinear(hidden_size, q_dim, bias=attention_bias)
        self.to_k = ReplicatedLinear(hidden_size, kv_dim, bias=attention_bias)
        self.to_v = ReplicatedLinear(hidden_size, kv_dim, bias=attention_bias)
        self.to_out = ReplicatedLinear(q_dim, hidden_size, bias=attention_bias)
        self.norm_q = RMSNorm(head_dim, eps=eps)
        self.norm_k = RMSNorm(head_dim, eps=eps)

        # Generation pathway
        self.add_q_proj = ReplicatedLinear(hidden_size, q_dim, bias=attention_bias)
        self.add_k_proj = ReplicatedLinear(hidden_size, kv_dim, bias=attention_bias)
        self.add_v_proj = ReplicatedLinear(hidden_size, kv_dim, bias=attention_bias)
        self.to_add_out = ReplicatedLinear(q_dim, hidden_size, bias=attention_bias)
        self.norm_added_q = RMSNorm(head_dim, eps=eps)
        self.norm_added_k = RMSNorm(head_dim, eps=eps)

    def forward(
        self,
        und_seq: torch.Tensor,  # [N_und, hidden]
        gen_seq: torch.Tensor,  # [N_gen, hidden]
        cos_und: torch.Tensor,
        sin_und: torch.Tensor,
        cos_gen: torch.Tensor,
        sin_gen: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_heads = self.num_attention_heads
        n_kv = self.num_key_value_heads
        d = self.head_dim

        q_und, _ = self.to_q(und_seq)
        k_und, _ = self.to_k(und_seq)
        v_und, _ = self.to_v(und_seq)
        q_gen, _ = self.add_q_proj(gen_seq)
        k_gen, _ = self.add_k_proj(gen_seq)
        v_gen, _ = self.add_v_proj(gen_seq)

        q_und = q_und.view(-1, n_heads, d)
        k_und = k_und.view(-1, n_kv, d)
        v_und = v_und.view(-1, n_kv, d)
        q_gen = q_gen.view(-1, n_heads, d)
        k_gen = k_gen.view(-1, n_kv, d)
        v_gen = v_gen.view(-1, n_kv, d)

        # Per-head QK-norm BEFORE RoPE
        q_und = self.norm_q(q_und)
        k_und = self.norm_k(k_und)
        q_gen = self.norm_added_q(q_gen)
        k_gen = self.norm_added_k(k_gen)

        # RoPE (heads-second layout; unsqueeze cos/sin over the head axis at dim=1)
        q_und, k_und = _apply_rotary_pos_emb(q_und, k_und, cos_und, sin_und, unsqueeze_dim=1)
        q_gen, k_gen = _apply_rotary_pos_emb(q_gen, k_gen, cos_gen, sin_gen, unsqueeze_dim=1)

        # Causal self-attention over understanding (text) tokens
        und_out = _sdpa(q_und, k_und, v_und, is_causal=True, scale=self.scaling)
        und_out = und_out.reshape(und_out.shape[0], n_heads * d)

        # Full attention: gen query attends to ALL tokens (und ++ gen)
        all_k = torch.cat([k_und, k_gen], dim=0)
        all_v = torch.cat([v_und, v_gen], dim=0)
        gen_out = _sdpa(q_gen, all_k, all_v, is_causal=False, scale=self.scaling)
        gen_out = gen_out.reshape(gen_out.shape[0], n_heads * d)

        und_out, _ = self.to_out(und_out)
        gen_out, _ = self.to_add_out(gen_out)
        return und_out, gen_out


# ===========================================================================
# Dual-pathway decoder layer
# ===========================================================================
class Cosmos3DecoderLayer(nn.Module):
    """MoT decoder layer: dual-pathway attention + dual SwiGLU MLP, 4 RMSNorms."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        eps: float,
        attention_bias: bool,
    ) -> None:
        super().__init__()
        self.self_attn = Cosmos3DualAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            eps=eps,
            attention_bias=attention_bias,
        )
        self.mlp = Cosmos3MLP(hidden_size, intermediate_size)
        self.mlp_moe_gen = Cosmos3MLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.input_layernorm_moe_gen = RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm_moe_gen = RMSNorm(hidden_size, eps=eps)

    def forward(
        self,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        cos_und: torch.Tensor,
        sin_und: torch.Tensor,
        cos_gen: torch.Tensor,
        sin_gen: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-attention norm
        und_norm = self.input_layernorm(und_seq)
        gen_norm = self.input_layernorm_moe_gen(gen_seq)

        und_attn, gen_attn = self.self_attn(und_norm, gen_norm, cos_und, sin_und, cos_gen, sin_gen)
        und_res = und_seq + und_attn
        gen_res = gen_seq + gen_attn

        # Pre-MLP norm + dual SwiGLU
        und_ln = self.post_attention_layernorm(und_res)
        gen_ln = self.post_attention_layernorm_moe_gen(gen_res)
        und_out = und_res + self.mlp(und_ln)
        gen_out = gen_res + self.mlp_moe_gen(gen_ln)
        return und_out, gen_out


# ===========================================================================
# Domain-aware linear (dormant action head; per-domain weight/bias embeddings)
# ===========================================================================
class _DomainAwareLinear(nn.Module):
    """One weight/bias pair per embodiment domain (matches framework keys)."""

    def __init__(self, input_size: int, output_size: int, num_domains: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.num_domains = int(num_domains)
        self.fc = nn.Embedding(self.num_domains, self.output_size * self.input_size)
        self.bias = nn.Embedding(self.num_domains, self.output_size)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        if domain_id.ndim == 0:
            domain_id = domain_id.unsqueeze(0)
        domain_id = domain_id.to(device=x.device, dtype=torch.long).reshape(-1)
        weight = self.fc(domain_id).view(domain_id.shape[0], self.input_size, self.output_size)
        bias = self.bias(domain_id).view(domain_id.shape[0], self.output_size)
        if x.ndim == 2:
            return torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
        return torch.bmm(x, weight) + bias.unsqueeze(1)


# ===========================================================================
# Full DiT
# ===========================================================================
class Cosmos3VFMTransformer(BaseDiT):
    """FastVideo-native Cosmos3 omni DiT.

    Mirrors the published checkpoint's transformer key surface and the
    ``cosmos_framework`` ``Cosmos3VFMNetwork`` forward math for the video path.
    Runs on CPU / float32 (plain SDPA + native-math RoPE).
    """

    _fsdp_shard_conditions = Cosmos3VideoConfig().arch_config._fsdp_shard_conditions
    _compile_conditions = Cosmos3VideoConfig().arch_config._compile_conditions
    param_names_mapping = Cosmos3VideoConfig().arch_config.param_names_mapping
    reverse_param_names_mapping: dict = {}

    def __init__(self, config: Cosmos3VideoConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)
        arch = config.arch_config

        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_key_value_heads = arch.num_key_value_heads
        self.head_dim = arch.head_dim
        self.num_hidden_layers = arch.num_hidden_layers
        self.intermediate_size = arch.intermediate_size
        self.vocab_size = arch.vocab_size
        self.rms_norm_eps = arch.rms_norm_eps
        self.attention_bias = arch.attention_bias

        # VAE / patch geometry
        self.latent_patch_size = arch.latent_patch_size
        self.latent_channel = arch.latent_channel
        # Alias kept for the standalone patchify/unpatchify helpers + scaffold tests.
        self.latent_channel_size = arch.latent_channel
        self.patch_latent_dim = arch.patch_latent_dim
        self.num_channels_latents = arch.latent_channel
        self.timestep_scale = arch.timestep_scale
        self.position_embedding_type = arch.position_embedding_type

        # ---- Backbone ----
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList([
            Cosmos3DecoderLayer(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                intermediate_size=self.intermediate_size,
                eps=self.rms_norm_eps,
                attention_bias=self.attention_bias,
            ) for _ in range(self.num_hidden_layers)
        ])
        self.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.norm_moe_gen = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # Backbone RoPE (unified 3D-MRoPE)
        self.rotary_emb = Cosmos3TextRotaryEmbedding(
            head_dim=self.head_dim,
            rope_theta=arch.rope_theta,
            mrope_section=arch.mrope_section,
        )

        # ---- Vision head ----
        self.proj_in = ReplicatedLinear(self.patch_latent_dim, self.hidden_size, bias=True)
        self.proj_out = ReplicatedLinear(self.hidden_size, self.patch_latent_dim, bias=True)
        self.time_embedder = Cosmos3TimestepEmbedder(self.hidden_size)

        # Additive latent position embedding only for the legacy 3d_rope variant.
        self.latent_pos_embed: Cosmos3VideoRopePosition3DEmb | None = None
        if self.position_embedding_type == "3d_rope":
            self.latent_pos_embed = Cosmos3VideoRopePosition3DEmb(
                head_dim=self.hidden_size,
                len_h=getattr(arch, "max_latent_h", 32),
                len_w=getattr(arch, "max_latent_w", 32),
                len_t=getattr(arch, "max_latent_t", 32),
                base_fps=int(arch.base_fps),
                base_temporal_compression_factor=arch.temporal_compression_factor,
                temporal_compression_factor=arch.temporal_compression_factor,
                enable_fps_modulation=arch.enable_fps_modulation,
            )

        # ---- Dormant action head (constructed for strict-load parity) ----
        if getattr(arch, "action_gen", False):
            self.action_dim = arch.action_dim
            self.num_embodiment_domains = arch.num_embodiment_domains
            self.action_proj_in = _DomainAwareLinear(self.action_dim, self.hidden_size, self.num_embodiment_domains)
            self.action_proj_out = _DomainAwareLinear(self.hidden_size, self.action_dim, self.num_embodiment_domains)
            self.action_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))

        # ---- Dormant audio / sound head (constructed for strict-load parity) ----
        if getattr(arch, "sound_gen", False):
            self.sound_dim = arch.sound_dim
            self.audio_proj_in = ReplicatedLinear(self.sound_dim, self.hidden_size, bias=True)
            self.audio_proj_out = ReplicatedLinear(self.hidden_size, self.sound_dim, bias=True)
            self.audio_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))

        self.__post_init__()

    # ------------------------------------------------------------------
    # Standalone batched patchify / unpatchify (scaffold-test contract)
    # ------------------------------------------------------------------
    def _pad_to_patch_size(self, h: int, w: int) -> tuple[int, int, int, int]:
        """Return ``(hp, wp, H_padded, W_padded)`` for ``latent_patch_size`` padding."""
        p = self.latent_patch_size
        h_padded = ((h + p - 1) // p) * p
        w_padded = ((w + p - 1) // p) * p
        return h_padded // p, w_padded // p, h_padded, w_padded

    def patchify(self, latents: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        """``[B, C, t, h, w] -> [B, t*hp*wp, p*p*C]`` (zero-pad h/w to patch multiples)."""
        batch_size = latents.shape[0]
        p = self.latent_patch_size
        c = self.latent_channel_size
        hp, wp, h_padded, w_padded = self._pad_to_patch_size(h, w)
        if h_padded != h or w_padded != w:
            latents = F.pad(latents, (0, w_padded - w, 0, h_padded - h))
        x = latents.reshape(batch_size, c, t, hp, p, wp, p)
        x = x.permute(0, 2, 3, 5, 4, 6, 1)
        return x.reshape(batch_size, t * hp * wp, p * p * c)

    def unpatchify(self, tokens: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        """``[B, t*hp*wp, p*p*C] -> [B, C, t, h, w]`` (crop h/w padding)."""
        batch_size = tokens.shape[0]
        p = self.latent_patch_size
        c = self.latent_channel_size
        hp, wp, h_padded, w_padded = self._pad_to_patch_size(h, w)
        x = tokens.reshape(batch_size, t, hp, wp, p, p, c)
        x = x.permute(0, 6, 1, 2, 4, 3, 5)
        x = x.reshape(batch_size, c, t, h_padded, w_padded)
        if h_padded != h or w_padded != w:
            x = x[:, :, :, :h, :w]
        return x

    # ------------------------------------------------------------------
    # Framework-faithful packed patchify / unpatchify (einsum ordering)
    # ------------------------------------------------------------------
    def _patchify_and_pack(
        self,
        tokens_vision: list[torch.Tensor],
        token_shapes: list[tuple[int, int, int]],
    ) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        p = self.latent_patch_size
        packed = []
        original_shapes: list[tuple[int, int, int]] = []
        for latent, (_t, _h, _w) in zip(tokens_vision, token_shapes):
            latent = latent.squeeze(0) if latent.dim() == 5 else latent  # [C, T, H, W]
            _, t_actual, h_actual, w_actual = latent.shape
            original_shapes.append((t_actual, h_actual, w_actual))
            h_padded = ((h_actual + p - 1) // p) * p
            w_padded = ((w_actual + p - 1) // p) * p
            if h_padded != h_actual or w_padded != w_actual:
                padded = torch.zeros((self.latent_channel, t_actual, h_padded, w_padded),
                                     device=latent.device,
                                     dtype=latent.dtype)
                padded[:, :, :h_actual, :w_actual] = latent
                latent = padded
            h_patches = h_padded // p
            w_patches = w_padded // p
            latent = latent.reshape(self.latent_channel, t_actual, h_patches, p, w_patches, p)
            latent = torch.einsum("cthpwq->thwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed.append(latent)
        return torch.cat(packed, dim=0), original_shapes

    def _unpatchify_and_unpack(
        self,
        packed_preds: torch.Tensor,
        token_shapes: list[tuple[int, int, int]],
        noisy_frame_indexes: list[torch.Tensor],
        original_shapes: list[tuple[int, int, int]] | None,
    ) -> list[torch.Tensor]:
        p = self.latent_patch_size
        outputs = []
        start_idx = 0
        for i, (t_c, _h_c, _w_c) in enumerate(token_shapes):
            if original_shapes is not None:
                _t_orig, h_orig, w_orig = original_shapes[i]
                h_padded = ((h_orig + p - 1) // p) * p
                w_padded = ((w_orig + p - 1) // p) * p
                h_patches = h_padded // p
                w_patches = w_padded // p
            else:
                h_orig, w_orig = _h_c * p, _w_c * p
                h_patches, w_patches = _h_c, _w_c

            nfi = noisy_frame_indexes[i]
            t_n = len(nfi)
            out = torch.zeros((self.latent_channel, t_c, h_orig, w_orig),
                              device=packed_preds.device,
                              dtype=packed_preds.dtype)
            num_patches = t_n * h_patches * w_patches
            if num_patches > 0:
                end_idx = start_idx + num_patches
                patches = packed_preds[start_idx:end_idx]
                patches = patches.reshape(t_n, h_patches, w_patches, p, p, self.latent_channel)
                latent = torch.einsum("thwpqc->cthpwq", patches)
                latent = latent.reshape(self.latent_channel, t_n, h_patches * p, w_patches * p)
                latent = latent[:, :, :h_orig, :w_orig]
                out[:, nfi] = latent
                start_idx = end_idx
            outputs.append(out.unsqueeze(0))  # [1, C, T, H, W]
        return outputs

    def _scatter_timestep_embeds(
        self,
        packed_tokens: torch.Tensor,
        packed_timestep_embeds: torch.Tensor,
        noisy_frame_indexes: list[torch.Tensor],
        token_shapes: list[tuple[int, ...]],
    ) -> torch.Tensor:
        """Add timestep embeds onto noisy patches (matches framework scatter_add)."""
        start_noisy_index = 0
        flat_idx = []
        for noisy_i, shape_i in zip(noisy_frame_indexes, token_shapes):
            spatial = math.prod(shape_i[1:])
            spatial_idx = torch.arange(spatial, device=packed_tokens.device)
            ni = (noisy_i * spatial).unsqueeze(-1).expand(-1, spatial)
            ni = ni.clone() + spatial_idx + start_noisy_index
            flat_idx.append(ni.flatten())
            start_noisy_index += math.prod(shape_i)
        flat = torch.cat(flat_idx, dim=0)
        flat = flat.unsqueeze(-1).expand(-1, packed_tokens.shape[1])
        return packed_tokens.scatter_add(0, flat, packed_timestep_embeds)

    def materialize_non_persistent_buffers(self, device: torch.device, dtype: torch.dtype | None = None) -> None:
        """Recompute non-persistent buffers lost by the meta-device FSDP load.

        Only ``rotary_emb.inv_freq`` is non-persistent (derived from
        ``rope_theta``, absent from the checkpoint). The FastVideo loader calls
        this after ``load_model_from_full_model_state_dict``.
        """
        del dtype  # inv_freq stays float32 regardless of compute dtype
        self.rotary_emb.reset_inv_freq(device)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(  # type: ignore[override]
        self,
        text_ids: torch.Tensor,
        text_indexes: torch.Tensor,
        position_ids: torch.Tensor,
        sequence_length: int,
        split_lens: list[int],
        attn_modes: list[str],
        vision_tokens: list[torch.Tensor],
        vision_token_shapes: list[tuple[int, int, int]],
        vision_sequence_indexes: torch.Tensor,
        vision_timesteps: torch.Tensor,
        vision_mse_loss_indexes: torch.Tensor,
        vision_noisy_frame_indexes: list[torch.Tensor],
        fps_vision: torch.Tensor | None = None,
        sound_tokens: list[torch.Tensor] | None = None,
        sound_token_shapes: list[tuple[int, int, int]] | None = None,
        sound_sequence_indexes: torch.Tensor | None = None,
        sound_timesteps: torch.Tensor | None = None,
        sound_mse_loss_indexes: torch.Tensor | None = None,
        sound_noisy_frame_indexes: list[torch.Tensor] | None = None,
        fps_sound: torch.Tensor | None = None,
        action_tokens: list[torch.Tensor] | None = None,
        action_token_shapes: list[tuple[int, ...]] | None = None,
        action_sequence_indexes: torch.Tensor | None = None,
        action_timesteps: torch.Tensor | None = None,
        action_mse_loss_indexes: torch.Tensor | None = None,
        action_noisy_frame_indexes: list[torch.Tensor] | None = None,
        action_domain_id: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Video-path forward returning ``{"last_hidden_state", "preds_vision"}``
        (plus ``"preds_sound"`` when sound tokens are provided for t2vs).

        Inputs mirror the framework ``PackedSequence`` fields for a single
        text(causal)+vision(full) sample. This is the surface the parity test
        and the converter exercise; a pipeline-facing wrapper composes these
        from a ``PackedSequence`` builder.
        """
        device = self.embed_tokens.weight.device

        # 1. Text embedding scattered into the packed sequence buffer
        text_embed = self.embed_tokens(text_ids)  # [N_text, hidden]
        target_dtype = text_embed.dtype
        packed_sequence = text_embed.new_zeros((sequence_length, self.hidden_size))
        packed_sequence[text_indexes] = text_embed

        # 2. Vision: patchify -> proj_in -> (+ additive 3d-rope pos emb) -> timestep scatter
        original_shapes: list[tuple[int, int, int]] | None = None
        if vision_tokens:
            packed_vision, original_shapes = self._patchify_and_pack(vision_tokens, vision_token_shapes)
            # Vision latents arrive in fp32 (noise/VAE); cast to the model's
            # compute dtype (bf16 at inference; no-op in the fp32 parity tests).
            packed_vision, _ = self.proj_in(packed_vision.to(target_dtype))

            if self.latent_pos_embed is not None:
                pos_emb = self.latent_pos_embed(vision_token_shapes, fps=fps_vision).to(target_dtype)
                packed_vision = packed_vision + pos_emb

            if vision_mse_loss_indexes.numel() > 0:
                timesteps = vision_timesteps * self.timestep_scale
                ts_embeds = self.time_embedder(timesteps).to(target_dtype)
                packed_vision = self._scatter_timestep_embeds(packed_vision, ts_embeds, vision_noisy_frame_indexes,
                                                              vision_token_shapes)

            packed_sequence[vision_sequence_indexes] = packed_vision

        # 2b. Sound (t2vs): pack [C, T] -> audio_proj_in -> + modality embed ->
        #     timestep scatter -> scatter into the (full) gen split. Mirrors the
        #     framework ``_encode_sound``; sound shares the vision "full" split.
        if sound_tokens:
            packed_sound = torch.cat(
                [s[:, :shp[0]].permute(1, 0) for s, shp in zip(sound_tokens, sound_token_shapes)],
                dim=0,
            )  # [total_sound_tokens, sound_dim]
            packed_sound, _ = self.audio_proj_in(packed_sound.to(target_dtype))
            packed_sound = packed_sound + self.audio_modality_embed
            if sound_mse_loss_indexes is not None and sound_mse_loss_indexes.numel() > 0:
                s_ts = sound_timesteps * self.timestep_scale
                s_embeds = self.time_embedder(s_ts).to(target_dtype)
                packed_sound = self._scatter_timestep_embeds(packed_sound, s_embeds, sound_noisy_frame_indexes,
                                                             sound_token_shapes)
            packed_sequence[sound_sequence_indexes] = packed_sound

        # 2c. Action (DomainAwareLinear): pack [T, D] per sample with a per-token
        #     domain id -> action_proj_in(domain) + action_modality_embed ->
        #     timestep scatter -> scatter into the gen split (framework
        #     ``_encode_action``). Action is domain-conditioned (per embodiment).
        if action_tokens:
            packed_action = torch.cat(
                [a[:shp[0]] for a, shp in zip(action_tokens, action_token_shapes)], dim=0,
            )  # [total_action_tokens, action_dim]
            per_token_domain = torch.cat(
                [d.reshape(1).expand(shp[0]) for d, shp in zip(action_domain_id, action_token_shapes)], dim=0,
            )  # [total_action_tokens]
            packed_action = self.action_proj_in(packed_action.to(target_dtype), per_token_domain)
            packed_action = packed_action + self.action_modality_embed.view(1, -1)
            if action_mse_loss_indexes is not None and action_mse_loss_indexes.numel() > 0:
                a_ts = action_timesteps * self.timestep_scale
                a_embeds = self.time_embedder(a_ts).to(target_dtype)
                packed_action = self._scatter_timestep_embeds(packed_action, a_embeds, action_noisy_frame_indexes,
                                                              action_token_shapes)
            packed_sequence[action_sequence_indexes] = packed_action

        # 3. RoPE for the full packed sequence (cos/sin per token); split und/gen
        cos, sin = self.rotary_emb(position_ids, device=device, dtype=target_dtype)  # [N, head_dim]
        und_idx, gen_idx = self._mode_indices(split_lens, attn_modes, device)
        cos_und, sin_und = cos[und_idx], sin[und_idx]
        cos_gen, sin_gen = cos[gen_idx], sin[gen_idx]

        # 4. Dual-pathway decoder over (und = text causal, gen = vision full)
        und_seq = packed_sequence[und_idx]
        gen_seq = packed_sequence[gen_idx]
        for layer in self.layers:
            und_seq, gen_seq = layer(und_seq, gen_seq, cos_und, sin_und, cos_gen, sin_gen)

        # 5. Final norms (und vs gen) and re-scatter into a joint buffer
        und_seq = self.norm(und_seq)
        gen_seq = self.norm_moe_gen(gen_seq)
        last_hidden_state = packed_sequence.new_zeros((sequence_length, self.hidden_size))
        last_hidden_state[und_idx] = und_seq
        last_hidden_state[gen_idx] = gen_seq

        # 6. Vision prediction: proj_out on noisy patches -> unpatchify
        output: dict[str, Any] = {}
        if vision_tokens and vision_mse_loss_indexes.numel() > 0:
            preds, _ = self.proj_out(last_hidden_state[vision_mse_loss_indexes])
            output["preds_vision"] = self._unpatchify_and_unpack(preds, vision_token_shapes,
                                                                 vision_noisy_frame_indexes, original_shapes)

        # 6b. Sound prediction (t2vs): audio_proj_out on noisy sound hidden
        #     states -> unpack to per-sample [C, T] (framework ``_decode_sound``).
        if sound_tokens and sound_mse_loss_indexes is not None and sound_mse_loss_indexes.numel() > 0:
            preds_sound, _ = self.audio_proj_out(last_hidden_state[sound_mse_loss_indexes])
            output["preds_sound"] = self._unpack_sound(preds_sound, sound_token_shapes, sound_noisy_frame_indexes)

        # 6c. Action prediction: action_proj_out(per-token domain) on noisy hidden
        #     states -> unpack to per-sample [T, D] (framework ``_decode_action``).
        if action_tokens and action_mse_loss_indexes is not None and action_mse_loss_indexes.numel() > 0:
            noisy_domain = torch.cat(
                [d.reshape(1).expand(len(nfi)) for d, nfi in zip(action_domain_id, action_noisy_frame_indexes)], dim=0,
            )
            preds_action = self.action_proj_out(last_hidden_state[action_mse_loss_indexes], noisy_domain)
            output["preds_action"] = self._unpack_action(preds_action, action_token_shapes, action_noisy_frame_indexes)

        output["last_hidden_state"] = last_hidden_state
        return output

    def _unpack_action(
        self,
        packed_preds: torch.Tensor,
        token_shapes: list[tuple[int, ...]],
        noisy_frame_indexes: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Scatter packed noisy action preds back to per-sample ``[T, D]`` (clean
        frames left zero). Mirrors framework ``unpack_action``."""
        outputs: list[torch.Tensor] = []
        start_idx = 0
        for shape, nfi in zip(token_shapes, noisy_frame_indexes):
            t = shape[0]
            out = torch.zeros((t, self.action_dim), device=packed_preds.device, dtype=packed_preds.dtype)
            t_n = len(nfi)
            if t_n > 0:
                out[nfi] = packed_preds[start_idx:start_idx + t_n]
                start_idx += t_n
            outputs.append(out)
        return outputs

    def _unpack_sound(
        self,
        packed_preds: torch.Tensor,
        token_shapes: list[tuple[int, int, int]],
        noisy_frame_indexes: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Scatter packed noisy sound preds back to per-sample ``[C, T]`` (clean
        frames left zero). Mirrors framework ``unpack_sound_latents``."""
        outputs: list[torch.Tensor] = []
        start_idx = 0
        for shape, nfi in zip(token_shapes, noisy_frame_indexes):
            t = shape[0]
            out = torch.zeros((self.sound_dim, t), device=packed_preds.device, dtype=packed_preds.dtype)
            t_n = len(nfi)
            if t_n > 0:
                out[:, nfi] = packed_preds[start_idx:start_idx + t_n].T
                start_idx += t_n
            outputs.append(out)
        return outputs

    @staticmethod
    def _mode_indices(split_lens: list[int], attn_modes: list[str],
                      device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Token indexes for causal (und) and full (gen) splits, in pack order."""
        und, gen = [], []
        start = 0
        for split_len, mode in zip(split_lens, attn_modes):
            rng = range(start, start + split_len)
            if mode == "causal":
                und.extend(rng)
            elif mode == "full":
                gen.extend(rng)
            start += split_len
        return (torch.tensor(und, dtype=torch.long, device=device),
                torch.tensor(gen, dtype=torch.long, device=device))
