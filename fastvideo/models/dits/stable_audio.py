# SPDX-License-Identifier: Apache-2.0
"""Stable Audio Open 1.0 DiT.

Continuous transformer with rotary self-attention, GQA cross-attention,
and prepend global conditioning. 24 layers, embed_dim=1536, head_dim=64.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from einops import rearrange
from torch import nn

from fastvideo.attention import LocalAttention
from fastvideo.configs.models.dits import StableAudioConfig
from fastvideo.layers.layernorm import FP32LayerNorm
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.loader.utils import get_param_names_mapping

# Single import-time snapshot — re-reading via `StableAudioConfig()` per
# `Attention.__init__` would rebuild the nested dataclass + regex map ~48
# times during a single DiT construction. Reused for the class-level
# attribute defaults below.
_DEFAULT_CONFIG = StableAudioConfig()
_SUPPORTED_BACKENDS = _DEFAULT_CONFIG.arch_config._supported_attention_backends


class FourierFeatures(nn.Module):
    """Random-Fourier learned-frequency timestep encoder."""

    def __init__(self, in_features: int, out_features: int, std: float = 1.0) -> None:
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = 2 * math.pi * x @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


# Partial-rotary with halves-swap (`unbind(-2)`, `[-x2, x1]`). Different
# from FastVideo's `_apply_rotary_emb` (interleaved pairs, `unbind(-1)`),
# so kept local.


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


# SwiGLU FF — local because `fastvideo.layers.mlp.MLP` is non-gated.


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
    # Sequential layout `(GLU, Identity, Linear, Identity)` keeps the
    # checkpoint keys at indices 0 and 2.

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


# Cross-attention is GQA (24 query heads, 12 KV heads); both backends
# (FlashAttn, SDPA with `enable_gqa=True`) handle it.


class Attention(nn.Module):

    def __init__(self, dim: int, dim_heads: int = 64, dim_context: int | None = None,
                 zero_init_output: bool = True, qk_norm: str | None = None) -> None:
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

        # `stable-audio-open-small` wraps Q/K in LayerNorm before attn
        # (`attn_kwargs.qk_norm = "ln"` in its `model_config.json`); the
        # 1.0 base does not. Names match upstream (`q_norm`/`k_norm`)
        # so the converted state dict loads strict.
        if qk_norm == "ln":
            self.q_norm = nn.LayerNorm(dim_heads)
            self.k_norm = nn.LayerNorm(dim_heads)
        elif qk_norm is None:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported qk_norm={qk_norm!r}; expected 'ln' or None.")

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
        q = self.q_norm(q)
        k = self.k_norm(k)

        if rotary_pos_emb is not None:
            freqs, _ = rotary_pos_emb
            v_dtype = v.dtype
            # Partial rotary (rot_dim < head_dim) with halves-swap, so
            # apply outside LocalAttention. q,k come in as [B, S, H, D];
            # transpose to [B, H, S, D] for the helper.
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


class TransformerBlock(nn.Module):

    def __init__(self, dim: int, dim_heads: int = 64, cross_attend: bool = False,
                 dim_context: int | None = None, zero_init_branch_outputs: bool = True,
                 qk_norm: str | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.dim_heads = min(dim_heads, dim)
        self.cross_attend = cross_attend
        self.pre_norm = FP32LayerNorm(dim, elementwise_affine=True)
        self.self_attn = Attention(dim, dim_heads=self.dim_heads,
                                   zero_init_output=zero_init_branch_outputs,
                                   qk_norm=qk_norm)
        if cross_attend:
            self.cross_attend_norm = FP32LayerNorm(dim, elementwise_affine=True)
            self.cross_attn = Attention(dim, dim_heads=self.dim_heads, dim_context=dim_context,
                                        zero_init_output=zero_init_branch_outputs,
                                        qk_norm=qk_norm)
        self.ff_norm = FP32LayerNorm(dim, elementwise_affine=True)
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None,
                rotary_pos_emb: tuple[torch.Tensor, float] | None = None) -> torch.Tensor:
        x = x + self.self_attn(self.pre_norm(x), rotary_pos_emb=rotary_pos_emb)
        if context is not None and self.cross_attend:
            x = x + self.cross_attn(self.cross_attend_norm(x), context=context)
        x = x + self.ff(self.ff_norm(x))
        return x


class ContinuousTransformer(nn.Module):

    def __init__(self, dim: int, depth: int, *, dim_heads: int = 64, dim_in: int | None = None,
                 dim_out: int | None = None, cross_attend: bool = False,
                 cond_token_dim: int | None = None, zero_init_branch_outputs: bool = True,
                 qk_norm: str | None = None) -> None:
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
                             zero_init_branch_outputs=zero_init_branch_outputs,
                             qk_norm=qk_norm) for _ in range(depth)
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


class StableAudioDiT(BaseDiT):
    """Stable Audio Open 1.0 diffusion transformer."""

    _fsdp_shard_conditions = _DEFAULT_CONFIG.arch_config._fsdp_shard_conditions
    _compile_conditions = _DEFAULT_CONFIG.arch_config._compile_conditions
    param_names_mapping = _DEFAULT_CONFIG.arch_config.param_names_mapping
    reverse_param_names_mapping: dict = {}

    def __init__(self, config: StableAudioConfig | None = None,
                 hf_config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = StableAudioConfig()
        super().__init__(config=config, hf_config=hf_config or {})
        arch = config.arch_config
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        io_channels = arch.io_channels
        embed_dim = arch.embed_dim
        depth = arch.depth
        num_heads = arch.num_attention_heads
        cond_token_dim = arch.cond_token_dim
        global_cond_dim = arch.global_cond_dim
        project_cond_tokens = arch.project_cond_tokens
        project_global_cond = arch.project_global_cond
        qk_norm = arch.qk_norm

        self.cond_token_dim = cond_token_dim
        timestep_features_dim = 256
        self.timestep_features = FourierFeatures(1, timestep_features_dim)
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
            qk_norm=qk_norm,
        )

        self.preprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

        self.io_channels = io_channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.__post_init__()

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
        """Forward over a single batch. CFG batching is the caller's job."""
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
        """Load from a raw `stable_audio_tools` monolithic state dict.
        Kept for tests / older checkpoints; production loads go through
        the standard `TransformerLoader` against the converted Diffusers
        repo.
        """
        model = cls()
        mapping_fn = get_param_names_mapping(model.config.arch_config.param_names_mapping)
        remapped: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if not k.startswith(prefix):
                continue
            new_key, _, _ = mapping_fn(k)
            remapped[new_key] = v
        missing, unexpected = model.load_state_dict(remapped, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"StableAudioDiT load mismatch — missing={missing[:5]} unexpected={unexpected[:5]}")
        return model


EntryClass = StableAudioDiT
