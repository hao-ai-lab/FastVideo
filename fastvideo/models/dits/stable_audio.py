# SPDX-License-Identifier: Apache-2.0
"""Stable Audio Open 1.0 DiT — first-class FastVideo port.

Vendored from the official Stability-AI/stable-audio-tools repo
(`stable_audio_tools/models/dit.py` + `transformer.py` +
`blocks.py:FourierFeatures`) under Apache-2.0.  Stripped to the subset
the published model actually uses: continuous-transformer with rotary
position embeddings, cross-attention conditioning, prepend global
conditioning. No conformer / causal / sliding-window / differential /
flex-attention / memory-tokens / xpos branches.

Replaces the previous diffusers `StableAudioDiTModel` reuse so the
pipeline owns the numerical behaviour end-to-end (no
`from diffusers import ...` at runtime; see REVIEW item 30).
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


def _checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


# ---------------------------------------------------------------------------
# Fourier timestep features
# ---------------------------------------------------------------------------


class FourierFeatures(nn.Module):
    """Random-Fourier timestep encoding (upstream `blocks.FourierFeatures`)."""

    def __init__(self, in_features: int, out_features: int, std: float = 1.0) -> None:
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = 2 * math.pi * x @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


# ---------------------------------------------------------------------------
# Norms / scale
# ---------------------------------------------------------------------------


class LayerNorm(nn.Module):
    """Upstream `transformer.LayerNorm` — bias-free, optional fp32 cast."""

    def __init__(self, dim: int, bias: bool = False, fix_scale: bool = False, eps: float = 1e-5,
                 force_fp32: bool = False) -> None:
        super().__init__()
        if fix_scale:
            self.register_buffer("gamma", torch.ones(dim))
        else:
            self.gamma = nn.Parameter(torch.ones(dim))
        if bias:
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("beta", torch.zeros(dim))
        self.eps = eps
        self.force_fp32 = force_fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.force_fp32:
            return F.layer_norm(x, x.shape[-1:], weight=self.gamma, bias=self.beta, eps=self.eps)
        out = F.layer_norm(x.float(), x.shape[-1:], weight=self.gamma.float(),
                           bias=self.beta.float(), eps=self.eps)
        return out.to(x.dtype)


class LayerScale(nn.Module):
    """Upstream `transformer.LayerScale`."""

    def __init__(self, dim: int, init_val: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full([dim], init_val))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


# ---------------------------------------------------------------------------
# Rotary position embeddings (upstream variant — half-dim repeat-cat layout)
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, base_rescale_factor: float = 1.0,
                 interpolation_factor: float = 1.0) -> None:
        super().__init__()
        base *= base_rescale_factor**(dim / (dim - 2))
        inv_freq = 1.0 / (base**(torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.interpolation_factor = interpolation_factor
        self.register_buffer("scale", None)  # xpos disabled for SA

    def forward_from_seq_len(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        return self.forward(t)

    def forward(self, t: torch.Tensor):
        t = t.to(torch.float32) / self.interpolation_factor
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs, 1.0


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    out_dtype = t.dtype
    dtype = torch.float32
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs = freqs.to(dtype)[-seq_len:, :]
    t = t.to(dtype)
    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, "b n d -> b 1 n d")
    t_rot, t_unrot = t[..., :rot_dim], t[..., rot_dim:]
    t_rot = (t_rot * freqs.cos() * scale) + (_rotate_half(t_rot) * freqs.sin() * scale)
    return torch.cat((t_rot.to(out_dtype), t_unrot.to(out_dtype)), dim=-1)


# ---------------------------------------------------------------------------
# Feed-forward (SwiGLU)
# ---------------------------------------------------------------------------


class _GLU(nn.Module):

    def __init__(self, dim_in: int, dim_out: int, activation: nn.Module) -> None:
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x, gate = x.chunk(2, dim=-1)
        return x * self.act(gate)


class FeedForward(nn.Module):
    """Upstream `transformer.FeedForward` with `glu=True`, no_bias=False, no conv.

    Matches upstream's Sequential layout (`linear_in, Identity, linear_out,
    Identity`) so positional checkpoint keys (`ff.ff.0`, `ff.ff.2`) line up.
    """

    def __init__(self, dim: int, mult: int = 4, zero_init_output: bool = True) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        linear_in = _GLU(dim, inner_dim, nn.SiLU())
        linear_out = nn.Linear(inner_dim, dim, bias=True)
        if zero_init_output:
            nn.init.zeros_(linear_out.weight)
            nn.init.zeros_(linear_out.bias)
        self.ff = nn.Sequential(linear_in, nn.Identity(), linear_out, nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


# ---------------------------------------------------------------------------
# Attention (single non-causal flavour with optional cross-attention)
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Upstream `transformer.Attention` — non-causal, optional cross-attn,
    no qk_norm, no differential, no feat_scale."""

    def __init__(self, dim: int, dim_heads: int = 64, dim_context: int | None = None,
                 zero_init_output: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        dim_kv = dim_context if dim_context is not None else dim
        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads
        if dim_context is not None:
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)

    def _apply_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.num_heads != self.kv_heads:
            heads_per_kv_head = self.num_heads // self.kv_heads
            k = k.repeat_interleave(heads_per_kv_head, dim=1)
            v = v.repeat_interleave(heads_per_kv_head, dim=1)
        if flash_attn_func is not None:
            in_dtype = q.dtype
            q, k, v = (rearrange(t, "b h n d -> b n h d") for t in (q, k, v))
            if in_dtype not in (torch.float16, torch.bfloat16):
                q, k, v = (t.to(torch.float16) for t in (q, k, v))
            out = flash_attn_func(q, k, v, causal=False)
            return rearrange(out.to(in_dtype), "b n h d -> b h n d")
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None,
                rotary_pos_emb: tuple[torch.Tensor, float] | None = None) -> torch.Tensor:
        h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None
        kv_input = context if has_context else x
        if has_context:
            q = self.to_q(x)
            q = rearrange(q, "b n (h d) -> b h n d", h=h)
            k, v = self.to_kv(kv_input).chunk(2, dim=-1)
            k = rearrange(k, "b n (h d) -> b h n d", h=kv_h)
            v = rearrange(v, "b n (h d) -> b h n d", h=kv_h)
        else:
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q = rearrange(q, "b n (h d) -> b h n d", h=h)
            k = rearrange(k, "b n (h d) -> b h n d", h=h)
            v = rearrange(v, "b n (h d) -> b h n d", h=h)

        if rotary_pos_emb is not None:
            freqs, _ = rotary_pos_emb
            v_dtype = v.dtype
            if q.shape[-2] >= k.shape[-2]:
                ratio = q.shape[-2] / k.shape[-2]
                q_freqs, k_freqs = freqs, ratio * freqs
            else:
                ratio = k.shape[-2] / q.shape[-2]
                q_freqs, k_freqs = ratio * freqs, freqs
            q = _apply_rotary_pos_emb(q, q_freqs).to(v_dtype)
            k = _apply_rotary_pos_emb(k, k_freqs).to(v_dtype)

        out = self._apply_attn(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# ---------------------------------------------------------------------------
# Transformer block (rotary self-attn + cross-attn + SwiGLU FF, no global cond)
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):

    def __init__(self, dim: int, dim_heads: int = 64, cross_attend: bool = False,
                 dim_context: int | None = None, zero_init_branch_outputs: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.dim_heads = min(dim_heads, dim)
        self.cross_attend = cross_attend
        self.pre_norm = LayerNorm(dim)
        self.self_attn = Attention(dim, dim_heads=self.dim_heads,
                                   zero_init_output=zero_init_branch_outputs)
        self.self_attn_scale = nn.Identity()
        if cross_attend:
            self.cross_attend_norm = LayerNorm(dim)
            self.cross_attn = Attention(dim, dim_heads=self.dim_heads, dim_context=dim_context,
                                        zero_init_output=zero_init_branch_outputs)
            self.cross_attn_scale = nn.Identity()
        self.ff_norm = LayerNorm(dim)
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs)
        self.ff_scale = nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None,
                rotary_pos_emb: tuple[torch.Tensor, float] | None = None) -> torch.Tensor:
        x = x + self.self_attn_scale(self.self_attn(self.pre_norm(x), rotary_pos_emb=rotary_pos_emb))
        if context is not None and self.cross_attend:
            x = x + self.cross_attn_scale(
                self.cross_attn(self.cross_attend_norm(x), context=context))
        x = x + self.ff_scale(self.ff(self.ff_norm(x)))
        return x


# ---------------------------------------------------------------------------
# ContinuousTransformer (no conformer / sliding-window / memory tokens / abs-pos)
# ---------------------------------------------------------------------------


class ContinuousTransformer(nn.Module):

    def __init__(self, dim: int, depth: int, *, dim_heads: int = 64, dim_in: int | None = None,
                 dim_out: int | None = None, cross_attend: bool = False,
                 cond_token_dim: int | None = None, zero_init_branch_outputs: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()
        self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        self.layers = nn.ModuleList([
            TransformerBlock(dim, dim_heads=dim_heads, cross_attend=cross_attend,
                             dim_context=cond_token_dim,
                             zero_init_branch_outputs=zero_init_branch_outputs) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, prepend_embeds: torch.Tensor | None = None,
                context: torch.Tensor | None = None,
                use_checkpointing: bool = False) -> torch.Tensor:
        x = self.project_in(x)
        if prepend_embeds is not None:
            assert prepend_embeds.shape[-1] == x.shape[-1]
            x = torch.cat((prepend_embeds, x), dim=-2)
        rotary = self.rotary_pos_emb.forward_from_seq_len(x.shape[1])
        for layer in self.layers:
            if use_checkpointing:
                x = _checkpoint(layer, x, context=context, rotary_pos_emb=rotary)
            else:
                x = layer(x, context=context, rotary_pos_emb=rotary)
        return self.project_out(x)


# ---------------------------------------------------------------------------
# StableAudioDiT (= upstream DiffusionTransformer with SA defaults)
# ---------------------------------------------------------------------------


class StableAudioDiT(nn.Module):
    """Stable Audio Open 1.0 diffusion transformer.

    Public defaults match `stabilityai/stable-audio-open-1.0`'s
    `model_config.json["model"]["diffusion"]`:
        io_channels=64, embed_dim=1536, depth=24, num_heads=24,
        cond_token_dim=768, global_cond_dim=1536, project_cond_tokens=False
    """

    def __init__(self, io_channels: int = 64, embed_dim: int = 1536, depth: int = 24,
                 num_heads: int = 24, cond_token_dim: int = 768, global_cond_dim: int = 1536,
                 project_cond_tokens: bool = False, project_global_cond: bool = True) -> None:
        super().__init__()
        self.cond_token_dim = cond_token_dim
        timestep_features_dim = 256
        self.timestep_features = FourierFeatures(1, timestep_features_dim)
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )
        self.diffusion_objective = "v"

        cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
        self.to_cond_embed = nn.Sequential(
            nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim, bias=False),
        )

        global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
        self.to_global_embed = nn.Sequential(
            nn.Linear(global_cond_dim, global_embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(global_embed_dim, global_embed_dim, bias=False),
        )

        self.transformer = ContinuousTransformer(
            dim=embed_dim, depth=depth, dim_heads=embed_dim // num_heads, dim_in=io_channels,
            dim_out=io_channels, cross_attend=True, cond_token_dim=cond_embed_dim,
        )

        self.preprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

        self.io_channels = io_channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor, t: torch.Tensor, *, cross_attn_cond: torch.Tensor,
                global_embed: torch.Tensor) -> torch.Tensor:
        """Single-batch forward — CFG batching is done by the caller (no
        in-class CFG path; we never apply CFG dropout at inference)."""
        model_dtype = next(self.parameters()).dtype
        x = x.to(model_dtype)
        t = t.to(model_dtype)
        cross_attn_cond = cross_attn_cond.to(model_dtype)
        global_embed = global_embed.to(model_dtype)

        cross_attn_cond = self.to_cond_embed(cross_attn_cond)
        global_embed = self.to_global_embed(global_embed)
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None]))
        global_embed = global_embed + timestep_embed
        prepend_inputs = global_embed.unsqueeze(1)

        x = self.preprocess_conv(x) + x
        x = rearrange(x, "b c t -> b t c")
        out = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond)
        out = rearrange(out, "b t c -> b c t")[:, :, prepend_inputs.shape[1]:]
        return self.postprocess_conv(out) + out

    @classmethod
    def from_official_state_dict(cls, state_dict: dict[str, torch.Tensor],
                                 prefix: str = "model.model.") -> "StableAudioDiT":
        """Load weights from the official `model.safetensors` (as published
        by `stabilityai/stable-audio-open-1.0`). The diffusion model lives
        under `model.model.*` in that checkpoint (`ConditionedDiffusionModelWrapper.model`
        -> our `StableAudioDiT`).
        """
        cfg: dict[str, Any] = dict(io_channels=64, embed_dim=1536, depth=24, num_heads=24,
                                   cond_token_dim=768, global_cond_dim=1536,
                                   project_cond_tokens=False, project_global_cond=True)
        model = cls(**cfg)
        # Strip the prefix from official keys.
        own = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        missing, unexpected = model.load_state_dict(own, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"StableAudioDiT load mismatch — missing={missing[:5]} unexpected={unexpected[:5]}")
        return model
