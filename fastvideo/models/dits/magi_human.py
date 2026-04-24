# SPDX-License-Identifier: Apache-2.0
"""daVinci-MagiHuman DiT (base variant).

Ported from https://github.com/GAIR-NLP/daVinci-MagiHuman
(inference/model/dit/dit_module.py, ~950 lines in the reference).

Architecture summary (verified against GAIR/daVinci-MagiHuman/base/ weights):

  - 40 transformer layers, hidden 5120, head_dim 128.
  - GQA with 40 query heads and 8 KV heads.
  - Multi-modality "sandwich": layers 0..3 and 36..39 use 3-way modality
    experts (video/audio/text) packed inside each linear as
    weight[..., out * 3, in]. Middle layers share a single expert.
  - Per-head attention gating: the QKV projection emits an extra
    num_heads_q channels that are sigmoid-gated onto the attention output.
  - Activation is GELU7 on layers 0..3 (non-gated, intermediate=4*hidden)
    and SwiGLU7 elsewhere (gated, intermediate=int(hidden*4*2/3)//4*4).
  - Position encoding is an element-wise Fourier embedding over 9-column
    coords (t,h,w + original TxHxW + reference TxHxW), not a standard
    1D/3D RoPE.
  - Forward takes a flat concatenated token stream (video first, then
    audio, then text) plus a modality map; the internal ModalityDispatcher
    permutes by modality before each linear so per-expert chunks line up.

Deviations from the "use fastvideo.layers primitives everywhere" guideline
in the add-model skill:

  - The packed-expert linears store weight as [out * num_experts, in].
    FastVideo's ReplicatedLinear does not model this layout; we use raw
    nn.Parameter with a small wrapper below. This is deliberate and scoped
    to this DiT: ReplicatedLinear still handles the adapter.* embedders
    and final_linear_{video,audio} (single-expert) here.
  - Self-attention is full-sequence and crosses modalities inside the flat
    concat stream; DistributedAttention assumes a clean spatial-sequence
    layout, so for the first port we use torch SDPA. Multi-GPU sequence
    parallelism is a follow-up.
  - torch.compile via magi_compiler is replaced with a plain nn.Module.

For the full history and shape-by-shape verification notes, see
.claude/skills/add-model/SKILL.md and the scaffold PR description.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from fastvideo.configs.models.dits.magi_human import (
    MagiHumanArchConfig,
    MagiHumanVideoConfig,
)
from fastvideo.models.dits.base import BaseDiT


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Modality(IntEnum):
    VIDEO = 0
    AUDIO = 1
    TEXT = 2


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


def swiglu7(x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    """Gated swish-GLU with OpenAI-OSS-style limits and +1 linear bias."""
    in_dtype = x.dtype
    x = x.to(torch.float32)
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return (out_glu * (x_linear + 1)).to(in_dtype)


def gelu7(x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    in_dtype = x.dtype
    x = x.to(torch.float32).clamp(max=limit)
    return (x * torch.sigmoid(alpha * x)).to(in_dtype)


# ---------------------------------------------------------------------------
# Modality dispatcher
# ---------------------------------------------------------------------------


class ModalityDispatcher:
    """Permute a flat token stream so same-modality tokens are contiguous.

    The DiT's multi-expert linears apply a different weight chunk per modality.
    Instead of carrying a branch inside each Linear, we pre-permute tokens so
    each chunk sees a contiguous slice, then un-permute before computing
    RoPE/attention across the full sequence.
    """

    def __init__(self, modality_mapping: torch.Tensor, num_modalities: int):
        self.modality_mapping = modality_mapping
        self.num_modalities = num_modalities
        self.permute_mapping = torch.argsort(modality_mapping)
        self.inv_permute_mapping = torch.argsort(self.permute_mapping)
        permuted = modality_mapping[self.permute_mapping]
        self.group_size = torch.bincount(permuted, minlength=num_modalities).to(torch.int32)
        self.group_size_cpu: list[int] = [int(x) for x in self.group_size.cpu().tolist()]

    def dispatch(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(x, self.group_size_cpu, dim=0))

    def undispatch(self, *chunks: torch.Tensor) -> torch.Tensor:
        return torch.cat(chunks, dim=0)

    @staticmethod
    def permute(x: torch.Tensor, permute_mapping: torch.Tensor) -> torch.Tensor:
        return x[permute_mapping]

    @staticmethod
    def inv_permute(x: torch.Tensor, inv_permute_mapping: torch.Tensor) -> torch.Tensor:
        return x[inv_permute_mapping]


# ---------------------------------------------------------------------------
# Norms, rotary embed
# ---------------------------------------------------------------------------


class MultiModalityRMSNorm(nn.Module):
    """RMSNorm with optional per-modality scale.

    When num_modality == 1, behaves identically to a standard RMSNorm with
    weight initialized to zero (effective weight is 1 + weight, hence the
    learnable +1 offset baked into the forward path). When num_modality > 1,
    the weight tensor packs per-modality scales along its flat axis and the
    dispatcher selects the right chunk per modality.
    """

    def __init__(self, dim: int, eps: float = 1e-6, num_modality: int = 1):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.num_modality = num_modality
        # Always stored in fp32; matches the reference initialization.
        self.weight = nn.Parameter(torch.zeros(dim * num_modality, dtype=torch.float32))

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        t = x.float()
        return t * torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(
        self,
        x: torch.Tensor,
        modality_dispatcher: Optional[ModalityDispatcher] = None,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        t = self._rms(x)
        if self.num_modality == 1:
            return (t * (self.weight + 1)).to(original_dtype)
        assert modality_dispatcher is not None, (
            "MultiModalityRMSNorm with num_modality>1 requires a dispatcher"
        )
        weight_chunks = self.weight.chunk(self.num_modality, dim=0)
        parts = modality_dispatcher.dispatch(t)
        for i in range(self.num_modality):
            parts[i] = parts[i] * (weight_chunks[i] + 1)
        return modality_dispatcher.undispatch(*parts).to(original_dtype)


def _freq_bands(num_bands: int, temperature: float = 10000.0) -> torch.Tensor:
    exp = torch.arange(0, num_bands, 1, dtype=torch.int64).float() / num_bands
    return 1.0 / (temperature ** exp)


class ElementWiseFourierEmbed(nn.Module):
    """Element-wise Fourier embedding over 9-column coords (t, h, w, T, H, W,
    ref_T, ref_H, ref_W). Produces a per-token positional embedding that
    acts as the RoPE angle input for attention.

    Weight: `bands` of shape `[dim // 8]` (fixed at init via freq_bands).
    """

    def __init__(
        self,
        dim: int,
        temperature: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dim = dim
        bands = _freq_bands(dim // 8, temperature=temperature).to(dtype)
        # `register_buffer` so state_dict keeps it, matching upstream naming.
        self.register_buffer("bands", bands)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [L, 9] = (t, h, w, T, H, W, ref_T, ref_H, ref_W)
        coords_xyz = coords[:, :3]
        sizes = coords[:, 3:6]
        refs = coords[:, 6:9]

        scales = (refs - 1) / (sizes - 1)
        scales[(refs == 1) & (sizes == 1)] = 1
        # Center H and W (leave time uncentered).
        centers = (sizes - 1) / 2
        centers[:, 0] = 0
        coords_xyz = coords_xyz - centers

        proj = coords_xyz.unsqueeze(-1) * scales.unsqueeze(-1) * self.bands  # [L, 3, B]
        sin_proj = proj.sin()
        cos_proj = proj.cos()
        return torch.cat((sin_proj, cos_proj), dim=1).flatten(1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply split-half RoPE. x: [..., seq, heads, head_dim].
    cos/sin: [..., seq, rot_dim/2].
    """
    rot_dim = cos.shape[-1] * 2
    cos = repeat(cos, "... d -> ... 1 (2 d)")
    sin = repeat(sin, "... d -> ... 1 (2 d)")
    x_rot = x[..., :rot_dim] * cos + _rotate_half(x[..., :rot_dim]) * sin
    if rot_dim < x.shape[-1]:
        return torch.cat([x_rot, x[..., rot_dim:]], dim=-1)
    return x_rot


# ---------------------------------------------------------------------------
# Packed-expert linear
# ---------------------------------------------------------------------------


class PackedExpertLinear(nn.Module):
    """Linear where the weight is packed per-modality along the output axis.

    Shapes:
        weight: [out_features * num_experts, in_features]
        bias:   [out_features * num_experts]  (optional)

    When `num_experts == 1`, behaves exactly like `nn.Linear`. When
    `num_experts > 1`, `forward` dispatches the input via the supplied
    `ModalityDispatcher`, applies the per-modality weight/bias chunk, and
    gathers the outputs in original order.

    Why not use `ReplicatedLinear`? Because the packed-expert layout is not
    what ReplicatedLinear (or any other fastvideo.layers.linear) is wired
    for. Using raw `nn.Parameter` keeps weight loading trivial (names map
    1:1 to the upstream checkpoint) and avoids quantization-path assumptions
    that don't match this layout.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int = 1,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.use_bias = bias
        self.weight = nn.Parameter(
            torch.empty(out_features * num_experts, in_features, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features * num_experts, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
        modality_dispatcher: Optional[ModalityDispatcher] = None,
    ) -> torch.Tensor:
        if self.num_experts == 1:
            return F.linear(x, self.weight, self.bias)
        assert modality_dispatcher is not None, (
            "PackedExpertLinear with num_experts>1 requires a dispatcher"
        )
        parts = modality_dispatcher.dispatch(x)
        w_chunks = self.weight.chunk(self.num_experts, dim=0)
        b_chunks = (
            self.bias.chunk(self.num_experts, dim=0)
            if self.bias is not None else [None] * self.num_experts
        )
        for i in range(self.num_experts):
            parts[i] = F.linear(parts[i], w_chunks[i], b_chunks[i])
        return modality_dispatcher.undispatch(*parts)


# ---------------------------------------------------------------------------
# Attention & MLP
# ---------------------------------------------------------------------------


@dataclass
class AttentionSubConfig:
    hidden_size: int
    num_heads_q: int
    num_heads_kv: int
    head_dim: int
    num_modality: int
    enable_attn_gating: bool


class MagiAttention(nn.Module):
    """Self-attention with GQA + optional per-head sigmoid gating."""

    def __init__(self, cfg: AttentionSubConfig):
        super().__init__()
        self.cfg = cfg
        self.gating_size = cfg.num_heads_q if cfg.enable_attn_gating else 0
        qkv_out = (
            cfg.num_heads_q * cfg.head_dim
            + 2 * cfg.num_heads_kv * cfg.head_dim
            + self.gating_size
        )
        self.pre_norm = MultiModalityRMSNorm(cfg.hidden_size, num_modality=cfg.num_modality)
        self.linear_qkv = PackedExpertLinear(
            cfg.hidden_size, qkv_out, num_experts=cfg.num_modality, bias=False,
        )
        self.linear_proj = PackedExpertLinear(
            cfg.num_heads_q * cfg.head_dim, cfg.hidden_size,
            num_experts=cfg.num_modality, bias=False,
        )
        self.q_norm = MultiModalityRMSNorm(cfg.head_dim, num_modality=cfg.num_modality)
        self.k_norm = MultiModalityRMSNorm(cfg.head_dim, num_modality=cfg.num_modality)

        self.q_size = cfg.num_heads_q * cfg.head_dim
        self.kv_size = cfg.num_heads_kv * cfg.head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        permute_mapping: torch.Tensor,
        inv_permute_mapping: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
    ) -> torch.Tensor:
        h = self.pre_norm(hidden_states, modality_dispatcher=modality_dispatcher).to(torch.bfloat16)
        qkv = self.linear_qkv(h, modality_dispatcher=modality_dispatcher).float()
        q, k, v, g = torch.split(
            qkv, [self.q_size, self.kv_size, self.kv_size, self.gating_size], dim=-1,
        )
        q = q.view(-1, self.cfg.num_heads_q, self.cfg.head_dim)
        k = k.view(-1, self.cfg.num_heads_kv, self.cfg.head_dim)
        v = v.view(-1, self.cfg.num_heads_kv, self.cfg.head_dim)
        g = g.view(-1, self.cfg.num_heads_q, 1) if self.gating_size else None

        q = self.q_norm(q, modality_dispatcher=modality_dispatcher)
        k = self.k_norm(k, modality_dispatcher=modality_dispatcher)

        # Un-permute before RoPE + attention so positional order reflects
        # the original (video, audio, text) concat — matches reference.
        q = ModalityDispatcher.inv_permute(q, inv_permute_mapping)
        k = ModalityDispatcher.inv_permute(k, inv_permute_mapping)
        v = ModalityDispatcher.inv_permute(v, inv_permute_mapping)
        if g is not None:
            g = ModalityDispatcher.inv_permute(g, inv_permute_mapping)

        # Element-wise Fourier embed packs sin/cos of 3 axes into a single
        # `rope` tensor. Match reference's split:
        #   sin_emb, cos_emb = rope.tensor_split(2, -1)
        # Note the reference calls apply_rotary_emb_torch with (cos_emb, sin_emb)
        # but passes sin_emb first in the split — we replicate that quirk
        # exactly so weight parity holds.
        sin_emb, cos_emb = rope.tensor_split(2, -1)
        q = apply_rotary_emb(q, cos_emb, sin_emb)
        k = apply_rotary_emb(k, cos_emb, sin_emb)

        # Single-GPU self-attention over the full concat stream. Repeat KV
        # heads to match Q heads (GQA).
        kv_repeat = self.cfg.num_heads_q // self.cfg.num_heads_kv
        if kv_repeat > 1:
            k = k.repeat_interleave(kv_repeat, dim=1)
            v = v.repeat_interleave(kv_repeat, dim=1)

        # [L, H, D] -> [1, H, L, D] for SDPA.
        q_t = q.to(torch.bfloat16).permute(1, 0, 2).unsqueeze(0)
        k_t = k.to(torch.bfloat16).permute(1, 0, 2).unsqueeze(0)
        v_t = v.to(torch.bfloat16).permute(1, 0, 2).unsqueeze(0)
        out = F.scaled_dot_product_attention(q_t, k_t, v_t)  # [1, H, L, D]
        out = out.squeeze(0).permute(1, 0, 2).contiguous()   # [L, H, D]

        # Re-permute to modality-grouped order so linear_proj's expert
        # chunks line up.
        out = ModalityDispatcher.permute(out, permute_mapping)
        if g is not None:
            g = ModalityDispatcher.permute(g, permute_mapping)
            out = out * torch.sigmoid(g.to(out.dtype))
        out = out.reshape(-1, self.cfg.num_heads_q * self.cfg.head_dim).to(torch.bfloat16)
        return self.linear_proj(out, modality_dispatcher=modality_dispatcher)


@dataclass
class MLPSubConfig:
    hidden_size: int
    intermediate_size: int
    activation: str              # "swiglu7" or "gelu7"
    num_modality: int
    gated: bool


class MagiMLP(nn.Module):
    def __init__(self, cfg: MLPSubConfig):
        super().__init__()
        self.cfg = cfg
        self.pre_norm = MultiModalityRMSNorm(cfg.hidden_size, num_modality=cfg.num_modality)
        up_out = cfg.intermediate_size * 2 if cfg.gated else cfg.intermediate_size
        self.up_gate_proj = PackedExpertLinear(
            cfg.hidden_size, up_out, num_experts=cfg.num_modality, bias=False,
        )
        self.down_proj = PackedExpertLinear(
            cfg.intermediate_size, cfg.hidden_size,
            num_experts=cfg.num_modality, bias=False,
        )
        self._act = swiglu7 if cfg.activation == "swiglu7" else gelu7

    def forward(
        self,
        x: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
    ) -> torch.Tensor:
        x = self.pre_norm(x, modality_dispatcher=modality_dispatcher).to(torch.bfloat16)
        x = self.up_gate_proj(x, modality_dispatcher=modality_dispatcher).float()
        x = self._act(x).to(torch.bfloat16)
        x = self.down_proj(x, modality_dispatcher=modality_dispatcher).float()
        return x


class MagiTransformerLayer(nn.Module):
    def __init__(self, arch: MagiHumanArchConfig, layer_idx: int):
        super().__init__()
        num_modality = 3 if layer_idx in arch.mm_layers else 1
        self.post_norm = layer_idx in arch.post_norm_layers

        self.attention = MagiAttention(AttentionSubConfig(
            hidden_size=arch.hidden_size,
            num_heads_q=arch.num_attention_heads,
            num_heads_kv=arch.num_heads_kv,
            head_dim=arch.head_dim,
            num_modality=num_modality,
            enable_attn_gating=arch.enable_attn_gating,
        ))

        is_gelu7 = layer_idx in arch.gelu7_layers
        if is_gelu7:
            intermediate = arch.hidden_size * 4
            gated = False
            activation = "gelu7"
        else:
            intermediate = (arch.hidden_size * 4 * 2 // 3) // 4 * 4
            gated = True
            activation = "swiglu7"

        self.mlp = MagiMLP(MLPSubConfig(
            hidden_size=arch.hidden_size,
            intermediate_size=intermediate,
            activation=activation,
            num_modality=num_modality,
            gated=gated,
        ))

        if self.post_norm:
            self.attn_post_norm = MultiModalityRMSNorm(arch.hidden_size, num_modality=num_modality)
            self.mlp_post_norm = MultiModalityRMSNorm(arch.hidden_size, num_modality=num_modality)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        permute_mapping: torch.Tensor,
        inv_permute_mapping: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
    ) -> torch.Tensor:
        attn_out = self.attention(
            hidden_states, rope, permute_mapping, inv_permute_mapping, modality_dispatcher,
        )
        if self.post_norm:
            attn_out = self.attn_post_norm(attn_out, modality_dispatcher=modality_dispatcher)
        hidden_states = hidden_states + attn_out

        mlp_out = self.mlp(hidden_states, modality_dispatcher=modality_dispatcher)
        if self.post_norm:
            mlp_out = self.mlp_post_norm(mlp_out, modality_dispatcher=modality_dispatcher)
        return hidden_states + mlp_out


# ---------------------------------------------------------------------------
# Adapter (per-modality embedders + Fourier RoPE producer)
# ---------------------------------------------------------------------------


class MagiAdapter(nn.Module):
    def __init__(self, arch: MagiHumanArchConfig):
        super().__init__()
        # Embedders stay in fp32 to match the reference dtype exactly.
        self.video_embedder = nn.Linear(
            arch.video_in_channels, arch.hidden_size, bias=True, dtype=torch.float32,
        )
        self.text_embedder = nn.Linear(
            arch.text_in_channels, arch.hidden_size, bias=True, dtype=torch.float32,
        )
        self.audio_embedder = nn.Linear(
            arch.audio_in_channels, arch.hidden_size, bias=True, dtype=torch.float32,
        )
        self.rope = ElementWiseFourierEmbed(arch.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        video_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rope = self.rope(coords_mapping)
        # Embedder dtypes may differ from x's dtype when FastVideo's FSDP
        # loader casts all weights to `pipeline_config.precision` (bf16).
        # Match the weight dtype per modality.
        v_w = self.video_embedder.weight
        a_w = self.audio_embedder.weight
        t_w = self.text_embedder.weight
        out = torch.zeros(
            x.shape[0], self.video_embedder.out_features,
            device=x.device, dtype=v_w.dtype,
        )
        out[text_mask] = self.text_embedder(
            x[text_mask, : self.text_embedder.in_features].to(t_w.dtype)
        ).to(out.dtype)
        out[audio_mask] = self.audio_embedder(
            x[audio_mask, : self.audio_embedder.in_features].to(a_w.dtype)
        ).to(out.dtype)
        out[video_mask] = self.video_embedder(
            x[video_mask, : self.video_embedder.in_features].to(v_w.dtype)
        ).to(out.dtype)
        return out, rope


# ---------------------------------------------------------------------------
# Top-level DiT
# ---------------------------------------------------------------------------


class _TransformerBlock(nn.Module):
    """Thin ModuleList wrapper to keep the 'block.layers.<i>' state_dict
    naming identical to the upstream checkpoint (which uses a magi_compile
    decorator producing `block.layers.<i>.*`)."""

    def __init__(self, arch: MagiHumanArchConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            MagiTransformerLayer(arch, i) for i in range(arch.num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        permute_mapping: torch.Tensor,
        inv_permute_mapping: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, rope, permute_mapping, inv_permute_mapping, modality_dispatcher)
        return x


_CFG = MagiHumanVideoConfig()


class MagiHumanDiT(BaseDiT):
    """Top-level DiT for daVinci-MagiHuman (base).

    Forward signature mirrors the reference `DiTModel.forward`: it takes a
    flat token stream, its per-token coords and modality mapping, and
    returns per-modality outputs packed into a max-channel-width tensor.

    This scaffold is single-GPU only; the `ulysses_scheduler().dispatch(...)`
    sequence-parallel wrapping in the reference has no equivalent here yet.
    """

    # BaseDiT requires these class attrs. Source them from the config so
    # they stay in sync with MagiHumanVideoConfig edits.
    _fsdp_shard_conditions = _CFG._fsdp_shard_conditions
    _compile_conditions = _CFG._compile_conditions
    _supported_attention_backends = _CFG._supported_attention_backends
    param_names_mapping = _CFG.param_names_mapping
    reverse_param_names_mapping = _CFG.reverse_param_names_mapping
    lora_param_names_mapping = _CFG.lora_param_names_mapping

    def __init__(self, config: MagiHumanVideoConfig, hf_config: dict | None = None, **kwargs):
        super().__init__(config=config, hf_config=hf_config or {})
        arch: MagiHumanArchConfig = getattr(config, "arch_config", config)
        self.arch = arch

        # BaseDiT contract instance vars.
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents

        self.adapter = MagiAdapter(arch)
        self.block = _TransformerBlock(arch)
        self.final_norm_video = MultiModalityRMSNorm(arch.hidden_size)
        self.final_norm_audio = MultiModalityRMSNorm(arch.hidden_size)
        self.final_linear_video = nn.Linear(
            arch.hidden_size, arch.video_in_channels, bias=False, dtype=torch.float32,
        )
        self.final_linear_audio = nn.Linear(
            arch.hidden_size, arch.audio_in_channels, bias=False, dtype=torch.float32,
        )

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        modality_mapping: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:                [L, max(V_ch, A_ch, T_ch)]
            coords_mapping:   [L, 9]
            modality_mapping: [L]  (int in {VIDEO, AUDIO, TEXT})
        Returns:
            out: [L, max(V_ch, A_ch)] with video channels in video slots and
                 audio channels in audio slots; text slots are zero.
        """
        dispatcher = ModalityDispatcher(modality_mapping, num_modalities=3)
        video_mask = modality_mapping == Modality.VIDEO
        audio_mask = modality_mapping == Modality.AUDIO
        text_mask = modality_mapping == Modality.TEXT

        x, rope = self.adapter(x, coords_mapping, video_mask, audio_mask, text_mask)
        x = x.to(torch.bfloat16)
        x = ModalityDispatcher.permute(x, dispatcher.permute_mapping)

        x = self.block(
            x, rope,
            permute_mapping=dispatcher.permute_mapping,
            inv_permute_mapping=dispatcher.inv_permute_mapping,
            modality_dispatcher=dispatcher,
        )
        x = ModalityDispatcher.inv_permute(x, dispatcher.inv_permute_mapping)

        x_video = x[video_mask].to(self.final_norm_video.weight.dtype)
        x_video = self.final_norm_video(x_video)
        x_video = self.final_linear_video(x_video)

        x_audio = x[audio_mask].to(self.final_norm_audio.weight.dtype)
        x_audio = self.final_norm_audio(x_audio)
        x_audio = self.final_linear_audio(x_audio)

        max_ch = max(self.arch.video_in_channels, self.arch.audio_in_channels)
        out = torch.zeros(x.shape[0], max_ch, device=x.device, dtype=x.dtype)
        out[video_mask, : self.arch.video_in_channels] = x_video.to(out.dtype)
        out[audio_mask, : self.arch.audio_in_channels] = x_audio.to(out.dtype)
        return out


EntryClass = MagiHumanDiT
