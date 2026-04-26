# SPDX-License-Identifier: Apache-2.0
"""Stable Audio Open 1.0 DiT — first-class FastVideo port.

Vendored from the official Stability-AI/stable-audio-tools repo
(`stable_audio_tools/models/dit.py + transformer.py +
blocks.py:FourierFeatures`) under Apache-2.0.  Stripped to the subset
the published model uses: continuous-transformer with rotary position
embeddings, cross-attention conditioning, prepend global conditioning.

Layer reuse (matches Wan2.1 / LTX-2 conventions, REVIEW item 11 / 22c):
  * `nn.Linear`              → `fastvideo.layers.linear.ReplicatedLinear`
  * `nn.LayerNorm`           → `fastvideo.layers.layernorm.FP32LayerNorm`
  * raw flash_attn / SDPA    → `fastvideo.attention.LocalAttention`

Replaces the previous diffusers `StableAudioDiTModel` reuse so the
pipeline owns the numerical behaviour end-to-end (no
`from diffusers import ...` at runtime; see REVIEW item 30).
"""
from __future__ import annotations

import math
from typing import Any

import torch
from einops import rearrange
from torch import nn

from fastvideo.attention import LocalAttention
from fastvideo.layers.layernorm import FP32LayerNorm
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.platforms import AttentionBackendEnum

# Backends supported on a single GPU; mirrors the LTX-2 cross-attn choice.
_SUPPORTED_BACKENDS = (AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA)


# ---------------------------------------------------------------------------
# Fourier timestep features (no FastVideo equivalent — random-Fourier
# learned freqs, not the standard sinusoidal time embedding).
# ---------------------------------------------------------------------------


class FourierFeatures(nn.Module):

    def __init__(self, in_features: int, out_features: int, std: float = 1.0) -> None:
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = 2 * math.pi * x @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


# ---------------------------------------------------------------------------
# Rotary position embeddings — partial-rotary halves-swap layout (upstream
# variant). FastVideo's `_apply_rotary_emb` uses the interleaved-pair
# (`unbind(-1)`) convention; Stable Audio uses the halves-swap
# (`unbind(-2)` w/ `[-x2, x1]` cat) convention. Different math, kept
# local so the upstream weights load with bit-identical numerics.
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base**(torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("scale", None)

    def forward_from_seq_len(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs, 1.0


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    out_dtype = t.dtype
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs = freqs.to(torch.float32)[-seq_len:, :]
    t = t.to(torch.float32)
    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, "b n d -> b 1 n d")
    t_rot, t_unrot = t[..., :rot_dim], t[..., rot_dim:]
    t_rot = (t_rot * freqs.cos()) + (_rotate_half(t_rot) * freqs.sin())
    return torch.cat((t_rot.to(out_dtype), t_unrot.to(out_dtype)), dim=-1)


# ---------------------------------------------------------------------------
# Feed-forward (SwiGLU) — built on `ReplicatedLinear` (FastVideo's
# `fastvideo.layers.mlp.MLP` is non-gated, so we keep our own GLU).
# ---------------------------------------------------------------------------


class _GLU(nn.Module):

    def __init__(self, dim_in: int, dim_out: int, activation: nn.Module) -> None:
        super().__init__()
        self.act = activation
        self.proj = ReplicatedLinear(dim_in, dim_out * 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.proj(x)
        x, gate = x.chunk(2, dim=-1)
        return x * self.act(gate)


class FeedForward(nn.Module):
    """`linear_in (GLU) → Identity → linear_out → Identity` so positional
    checkpoint keys (`ff.0`, `ff.2`) line up with upstream's Sequential.
    """

    def __init__(self, dim: int, mult: int = 4, zero_init_output: bool = True) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        linear_in = _GLU(dim, inner_dim, nn.SiLU())
        linear_out = ReplicatedLinear(inner_dim, dim, bias=True)
        if zero_init_output:
            nn.init.zeros_(linear_out.weight)
            nn.init.zeros_(linear_out.bias)
        self.ff = nn.Sequential(linear_in, nn.Identity(), linear_out, nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mod in self.ff:
            if isinstance(mod, ReplicatedLinear):
                x, _ = mod(x)
            else:
                x = mod(x)
        return x


# ---------------------------------------------------------------------------
# Attention: ReplicatedLinear projections + LocalAttention compute.
# Cross-attention is GQA (24 query heads, 12 KV heads) — LocalAttention's
# SDPA backend handles GQA via `enable_gqa=True` and FlashAttn handles it
# natively.
# ---------------------------------------------------------------------------


class Attention(nn.Module):

    def __init__(self, dim: int, dim_heads: int = 64, dim_context: int | None = None,
                 zero_init_output: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        dim_kv = dim_context if dim_context is not None else dim
        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads
        if dim_context is not None:
            self.to_q = ReplicatedLinear(dim, dim, bias=False)
            self.to_kv = ReplicatedLinear(dim_kv, dim_kv * 2, bias=False)
        else:
            self.to_qkv = ReplicatedLinear(dim, dim * 3, bias=False)
        self.to_out = ReplicatedLinear(dim, dim, bias=False)
        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)

        self.attn = LocalAttention(num_heads=self.num_heads, head_size=dim_heads,
                                   num_kv_heads=self.kv_heads, causal=False,
                                   supported_attention_backends=_SUPPORTED_BACKENDS)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None,
                rotary_pos_emb: tuple[torch.Tensor, float] | None = None) -> torch.Tensor:
        h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None
        kv_input = context if has_context else x
        if has_context:
            q, _ = self.to_q(x)
            kv, _ = self.to_kv(kv_input)
            k, v = kv.chunk(2, dim=-1)
        else:
            qkv, _ = self.to_qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
        # LocalAttention expects [batch, seq_len, num_heads, head_dim].
        q = rearrange(q, "b n (h d) -> b n h d", h=h)
        k = rearrange(k, "b n (h d) -> b n h d", h=kv_h)
        v = rearrange(v, "b n (h d) -> b n h d", h=kv_h)

        if rotary_pos_emb is not None:
            freqs, _ = rotary_pos_emb
            v_dtype = v.dtype
            # The upstream rotary is partial (rot_dim < head_dim) and uses
            # the halves-swap convention — apply outside LocalAttention so
            # we can keep the bit-identical math.
            #   q,k arrive as [B, S, H, D]; the helper expects [B, H, S, D].
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            if q_t.shape[-2] >= k_t.shape[-2]:
                ratio = q_t.shape[-2] / k_t.shape[-2]
                q_freqs, k_freqs = freqs, ratio * freqs
            else:
                ratio = k_t.shape[-2] / q_t.shape[-2]
                q_freqs, k_freqs = ratio * freqs, freqs
            q = _apply_rotary_pos_emb(q_t, q_freqs).to(v_dtype).transpose(1, 2)
            k = _apply_rotary_pos_emb(k_t, k_freqs).to(v_dtype).transpose(1, 2)

        out = self.attn(q, k, v)
        out = rearrange(out, "b n h d -> b n (h d)")
        out, _ = self.to_out(out)
        return out


# ---------------------------------------------------------------------------
# Transformer block (rotary self-attn + cross-attn + SwiGLU FF)
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):

    def __init__(self, dim: int, dim_heads: int = 64, cross_attend: bool = False,
                 dim_context: int | None = None, zero_init_branch_outputs: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.dim_heads = min(dim_heads, dim)
        self.cross_attend = cross_attend
        self.pre_norm = FP32LayerNorm(dim, elementwise_affine=True)
        self.self_attn = Attention(dim, dim_heads=self.dim_heads,
                                   zero_init_output=zero_init_branch_outputs)
        if cross_attend:
            self.cross_attend_norm = FP32LayerNorm(dim, elementwise_affine=True)
            self.cross_attn = Attention(dim, dim_heads=self.dim_heads, dim_context=dim_context,
                                        zero_init_output=zero_init_branch_outputs)
        self.ff_norm = FP32LayerNorm(dim, elementwise_affine=True)
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None,
                rotary_pos_emb: tuple[torch.Tensor, float] | None = None) -> torch.Tensor:
        x = x + self.self_attn(self.pre_norm(x), rotary_pos_emb=rotary_pos_emb)
        if context is not None and self.cross_attend:
            x = x + self.cross_attn(self.cross_attend_norm(x), context=context)
        x = x + self.ff(self.ff_norm(x))
        return x


# ---------------------------------------------------------------------------
# ContinuousTransformer
# ---------------------------------------------------------------------------


class ContinuousTransformer(nn.Module):

    def __init__(self, dim: int, depth: int, *, dim_heads: int = 64, dim_in: int | None = None,
                 dim_out: int | None = None, cross_attend: bool = False,
                 cond_token_dim: int | None = None, zero_init_branch_outputs: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.project_in = (ReplicatedLinear(dim_in, dim, bias=False) if dim_in is not None
                           else nn.Identity())
        self.project_out = (ReplicatedLinear(dim, dim_out, bias=False) if dim_out is not None
                            else nn.Identity())
        self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        self.layers = nn.ModuleList([
            TransformerBlock(dim, dim_heads=dim_heads, cross_attend=cross_attend,
                             dim_context=cond_token_dim,
                             zero_init_branch_outputs=zero_init_branch_outputs) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, prepend_embeds: torch.Tensor | None = None,
                context: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(self.project_in, ReplicatedLinear):
            x, _ = self.project_in(x)
        if prepend_embeds is not None:
            assert prepend_embeds.shape[-1] == x.shape[-1]
            x = torch.cat((prepend_embeds, x), dim=-2)
        rotary = self.rotary_pos_emb.forward_from_seq_len(x.shape[1])
        for layer in self.layers:
            x = layer(x, context=context, rotary_pos_emb=rotary)
        if isinstance(self.project_out, ReplicatedLinear):
            x, _ = self.project_out(x)
        return x


# ---------------------------------------------------------------------------
# StableAudioDiT
# ---------------------------------------------------------------------------


class StableAudioDiT(nn.Module):
    """Stable Audio Open 1.0 diffusion transformer.

    Defaults match `stabilityai/stable-audio-open-1.0`'s
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
        # to_timestep_embed is a 2-layer MLP-like Sequential. Keep as
        # nn.Sequential of ReplicatedLinear to preserve checkpoint key layout.
        self.to_timestep_embed = nn.Sequential(
            ReplicatedLinear(timestep_features_dim, embed_dim, bias=True),
            nn.SiLU(),
            ReplicatedLinear(embed_dim, embed_dim, bias=True),
        )
        self.diffusion_objective = "v"

        cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
        self.to_cond_embed = nn.Sequential(
            ReplicatedLinear(cond_token_dim, cond_embed_dim, bias=False),
            nn.SiLU(),
            ReplicatedLinear(cond_embed_dim, cond_embed_dim, bias=False),
        )

        global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
        self.to_global_embed = nn.Sequential(
            ReplicatedLinear(global_cond_dim, global_embed_dim, bias=False),
            nn.SiLU(),
            ReplicatedLinear(global_embed_dim, global_embed_dim, bias=False),
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

    @staticmethod
    def _seq_apply(seq: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        for mod in seq:
            if isinstance(mod, ReplicatedLinear):
                x, _ = mod(x)
            else:
                x = mod(x)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor, *, cross_attn_cond: torch.Tensor,
                global_embed: torch.Tensor) -> torch.Tensor:
        """Single-batch forward — CFG batching is done by the caller."""
        model_dtype = next(self.parameters()).dtype
        x = x.to(model_dtype)
        t = t.to(model_dtype)
        cross_attn_cond = cross_attn_cond.to(model_dtype)
        global_embed = global_embed.to(model_dtype)

        cross_attn_cond = self._seq_apply(self.to_cond_embed, cross_attn_cond)
        global_embed = self._seq_apply(self.to_global_embed, global_embed)
        timestep_embed = self._seq_apply(self.to_timestep_embed, self.timestep_features(t[:, None]))
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
        """Load from the official `model.safetensors`. The diffusion model
        lives under `model.model.*` (`ConditionedDiffusionModelWrapper.model`
        → `DiTWrapper.model`).

        Two key remaps:
          * `<norm>.gamma`  →  `<norm>.weight`   (FP32LayerNorm uses
            nn.LayerNorm naming; upstream's bias-free LayerNorm uses
            x-transformers naming)
          * `<norm>.beta`   →  `<norm>.bias`
        """
        cfg: dict[str, Any] = dict(io_channels=64, embed_dim=1536, depth=24, num_heads=24,
                                   cond_token_dim=768, global_cond_dim=1536,
                                   project_cond_tokens=False, project_global_cond=True)
        model = cls(**cfg)
        own = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        # Remap LayerNorm key names: gamma/beta → weight/bias.
        remapped: dict[str, torch.Tensor] = {}
        for k, v in own.items():
            if k.endswith(".gamma"):
                remapped[k[:-len(".gamma")] + ".weight"] = v
            elif k.endswith(".beta"):
                remapped[k[:-len(".beta")] + ".bias"] = v
            else:
                remapped[k] = v
        missing, unexpected = model.load_state_dict(remapped, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"StableAudioDiT load mismatch — missing={missing[:5]} unexpected={unexpected[:5]}")
        return model
