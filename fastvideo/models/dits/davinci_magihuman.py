# SPDX-License-Identifier: Apache-2.0
# Port of daVinci-MagiHuman (https://github.com/GAIR-NLP/daVinci-MagiHuman)
# Architecture: 15B single-stream Transformer with sandwich MoE, timestep-free.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from einops import repeat
from torch.nn.parameter import Parameter

from fastvideo.attention.layer import DistributedAttention
from fastvideo.configs.models import DiTConfig
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.quantization.base_config import QuantizationConfig
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.utils import set_weight_attrs

# ── Modality constants (must match checkpoint ordering) ───────────────────────
MODALITY_VIDEO = 0
MODALITY_AUDIO = 1
MODALITY_TEXT = 2
NUM_MODALITIES = 3

# ── Architecture defaults (keep in sync with DaVinciMagiHumanArchConfig) ──────
_MM_LAYERS = frozenset([0, 1, 2, 3, 36, 37, 38, 39])
_GELU7_LAYERS = frozenset([0, 1, 2, 3])


# ─── Custom activations ───────────────────────────────────────────────────────

def _swiglu7(
    x: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    """Clamped SwiGLU activation used in daVinci middle layers."""
    xf = x.to(torch.float32)
    gate, linear = xf[..., ::2], xf[..., 1::2]
    gate = gate.clamp(max=limit)
    linear = linear.clamp(-limit, limit)
    return (gate * torch.sigmoid(alpha * gate) * (linear + 1)).to(x.dtype)


def _gelu7(
    x: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    """Clamped GELU activation used in daVinci sandwich (first 4) layers."""
    xf = x.to(torch.float32)
    xf = xf.clamp(max=limit)
    return (xf * torch.sigmoid(alpha * xf)).to(x.dtype)


# ─── Positional encoding ──────────────────────────────────────────────────────

class ElementWiseFourierEmbed(nn.Module):
    """9-dim coordinate → Fourier positional embedding.

    coords columns: (t, h, w, T, H, W, ref_T, ref_H, ref_W)
    Output: [S, dim] where dim = head_dim.
    """

    def __init__(self, dim: int, temperature: float = 10000.0) -> None:
        super().__init__()
        # dim // 8: one freq-band set per (sin,cos) per 3 spatial axes
        num_bands = dim // 8
        exp = torch.arange(num_bands, dtype=torch.float32) / num_bands
        bands = 1.0 / (temperature ** exp)  # [B]
        self.register_buffer("bands", bands)
        self.dim = dim

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: [S, 9] → rope: [S, dim]"""
        coords_xyz = coords[:, :3]   # (t, h, w)
        sizes = coords[:, 3:6]       # (T, H, W)
        refs = coords[:, 6:9]        # (ref_T, ref_H, ref_W)

        scales = (refs - 1) / (sizes - 1)
        scales[(refs == 1) & (sizes == 1)] = 1.0

        centers = (sizes - 1) / 2
        centers[:, 0] = 0.0
        coords_xyz = coords_xyz - centers

        # [S, 3, B]
        proj = coords_xyz.unsqueeze(-1) * scales.unsqueeze(-1) * self.bands
        # sin and cos stacked → [S, 6*B] == [S, dim]
        return torch.cat((proj.sin(), proj.cos()), dim=1).flatten(1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embedding.

    x:   [B, S, H, D]
    cos: [S, D/2]
    sin: [S, D/2]
    """
    ro_dim = cos.shape[-1] * 2
    cos = repeat(cos, "s d -> 1 s 1 (2 d)")
    sin = repeat(sin, "s d -> 1 s 1 (2 d)")
    return torch.cat(
        [x[..., :ro_dim] * cos + _rotate_half(x[..., :ro_dim]) * sin,
         x[..., ro_dim:]],
        dim=-1,
    )


# ─── Modality dispatcher ──────────────────────────────────────────────────────

@dataclass
class ModalityDispatcher:
    """Computes permutation that groups tokens by modality.

    Tokens arrive in arbitrary interleaved order; this class sorts them
    so that all VIDEO tokens come first, then AUDIO, then TEXT.
    The same permute/inv_permute pair is reused for every layer.
    """

    permute_mapping: torch.Tensor
    inv_permute_mapping: torch.Tensor
    group_size_cpu: list[int]  # [n_video, n_audio, n_text]

    @classmethod
    def build(cls, modality_mapping: torch.Tensor) -> "ModalityDispatcher":
        perm = torch.argsort(modality_mapping)
        inv_perm = torch.argsort(perm)
        permuted = modality_mapping[perm]
        group_size = torch.bincount(permuted, minlength=NUM_MODALITIES)
        return cls(
            permute_mapping=perm,
            inv_permute_mapping=inv_perm,
            group_size_cpu=group_size.cpu().tolist(),
        )

    def permute(self, x: torch.Tensor) -> torch.Tensor:
        return x[self.permute_mapping]

    def inv_permute(self, x: torch.Tensor) -> torch.Tensor:
        return x[self.inv_permute_mapping]

    def split(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Split permuted tensor into per-modality chunks."""
        return list(torch.split(x, self.group_size_cpu, dim=0))

    @property
    def masks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """video_mask, audio_mask, text_mask — in ORIGINAL token order."""
        n = sum(self.group_size_cpu)
        inv = self.inv_permute_mapping
        video_end = self.group_size_cpu[0]
        audio_end = video_end + self.group_size_cpu[1]
        orig_video = torch.zeros(n, dtype=torch.bool, device=inv.device)
        orig_audio = torch.zeros(n, dtype=torch.bool, device=inv.device)
        orig_text = torch.zeros(n, dtype=torch.bool, device=inv.device)
        orig_video[inv[:video_end]] = True
        orig_audio[inv[video_end:audio_end]] = True
        orig_text[inv[audio_end:]] = True
        return orig_video, orig_audio, orig_text


# ─── MoE linear (sandwich layers) ────────────────────────────────────────────

class MoELinear(nn.Module):
    """Stacked per-modality linear layer for sandwich (mm) layers.

    Weight shape: [num_experts * out_features, in_features] — mirrors
    NativeMoELinear in the official code for checkpoint compatibility.
    Not LoRA-targetable (stacked weight is opaque to get_lora_layer).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int = NUM_MODALITIES,
        bias: bool = False,
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts

        self.weight = Parameter(
            torch.empty(num_experts * out_features, in_features,
                        dtype=params_dtype),
            requires_grad=False,
        )
        set_weight_attrs(self.weight, {
            "input_dim": 1,
            "output_dim": 0,
            "weight_loader": self.weight_loader,
        })

        if bias:
            self.bias = Parameter(
                torch.empty(num_experts * out_features, dtype=params_dtype),
                requires_grad=False,
            )
            set_weight_attrs(self.bias, {"weight_loader": self.weight_loader})
        else:
            self.register_parameter("bias", None)

    def weight_loader(
        self, param: Parameter, loaded_weight: torch.Tensor
    ) -> None:
        assert param.size() == loaded_weight.size(), (
            f"MoELinear weight shape mismatch: param={param.size()}, "
            f"loaded={loaded_weight.size()}")
        param.data.copy_(loaded_weight)

    def forward(
        self,
        x: torch.Tensor,
        token_counts: list[int],
    ) -> torch.Tensor:
        """Forward with per-expert dispatch.

        x:            [S, in_features] — already permuted by modality
        token_counts: [n_video, n_audio, n_text]
        """
        parts = torch.split(x, token_counts, dim=0)
        w_parts = self.weight.chunk(self.num_experts, dim=0)
        b_parts = (self.bias.chunk(self.num_experts, dim=0)
                   if self.bias is not None else [None] * self.num_experts)

        out_parts = []
        for xi, wi, bi in zip(parts, w_parts, b_parts):
            yi = torch.nn.functional.linear(xi.to(torch.bfloat16),
                                            wi.to(torch.bfloat16),
                                            bi)
            out_parts.append(yi)
        return torch.cat(out_parts, dim=0)


# ─── Per-modality RMS norm ────────────────────────────────────────────────────

class MultiModalityRMSNorm(nn.Module):
    """RMSNorm with optional per-modality weight (zeros init → scale=1 at start).

    For num_modality=1 (middle layers): single weight vector, standard RMSNorm.
    For num_modality=3 (sandwich layers): stacked weight [3*D], dispatch by token_counts.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        num_modality: int = 1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.num_modality = num_modality
        # zeros init: effective scale = weight + 1 = 1.0 initially
        self.weight = nn.Parameter(torch.zeros(dim * num_modality))

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(
        self,
        x: torch.Tensor,
        token_counts: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """
        x:            [S, D] — permuted token sequence
        token_counts: [n_video, n_audio, n_text] for mm layers, None for others
        """
        orig_dtype = x.dtype
        normed = self._rms(x)

        if self.num_modality == 1 or token_counts is None:
            return (normed * (self.weight + 1)).to(orig_dtype)

        # Per-modality scaling
        w_parts = self.weight.chunk(self.num_modality, dim=0)
        x_parts = torch.split(normed, token_counts, dim=0)
        return torch.cat(
            [xi * (wi + 1) for xi, wi in zip(x_parts, w_parts)],
            dim=0,
        ).to(orig_dtype)


# ─── Adapter (input projections + positional encoding) ───────────────────────

class DaVinciAdapter(nn.Module):
    """Projects per-modality tokens to hidden_size and computes rope."""

    def __init__(
        self,
        hidden_size: int,
        num_heads_q: int,
        video_in_channels: int,
        audio_in_channels: int,
        text_in_channels: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Renamed from official: video_embedder → video_proj, etc.
        self.video_proj = ReplicatedLinear(
            video_in_channels, hidden_size,
            bias=True, params_dtype=torch.float32,
            quant_config=quant_config, prefix=f"{prefix}.video_proj")
        self.text_proj = ReplicatedLinear(
            text_in_channels, hidden_size,
            bias=True, params_dtype=torch.float32,
            quant_config=quant_config, prefix=f"{prefix}.text_proj")
        self.audio_proj = ReplicatedLinear(
            audio_in_channels, hidden_size,
            bias=True, params_dtype=torch.float32,
            quant_config=quant_config, prefix=f"{prefix}.audio_proj")

        head_dim = hidden_size // num_heads_q
        self.rope = ElementWiseFourierEmbed(dim=head_dim)

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        video_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x:             [S, max(in_channels)] — packed mixed-modality tokens
        coords_mapping:[S, 9]
        *_mask:        [S] bool — which positions belong to each modality

        Returns:
            projected: [S, hidden_size]
            rope:      [S, head_dim]  (sin||cos)
        """
        rope = self.rope(coords_mapping)

        out = torch.zeros(
            x.shape[0], self.video_proj.output_size,
            device=x.device, dtype=x.dtype,
        )

        if video_mask.any():
            v_in = x[video_mask, :self.video_proj.input_size]
            v_out, _ = self.video_proj(v_in)
            out[video_mask] = v_out

        if audio_mask.any():
            a_in = x[audio_mask, :self.audio_proj.input_size]
            a_out, _ = self.audio_proj(a_in)
            out[audio_mask] = a_out

        if text_mask.any():
            t_in = x[text_mask, :self.text_proj.input_size]
            t_out, _ = self.text_proj(t_in)
            out[text_mask] = t_out

        return out, rope


# ─── Attention ────────────────────────────────────────────────────────────────

class DaVinciAttention(nn.Module):
    """Self-attention for one transformer layer.

    Sandwich (mm) layers use MoELinear for per-modality QKV + proj.
    Middle layers use ReplicatedLinear (LoRA-targetable).
    """

    _sdpa_diag_printed: bool = False  # class-level, fires once

    def __init__(
        self,
        hidden_size: int,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        is_mm_layer: bool,
        enable_attn_gating: bool = True,
        params_dtype: torch.dtype = torch.bfloat16,
        quant_config: Optional[QuantizationConfig] = None,
        supported_attention_backends: Optional[tuple] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.head_dim = head_dim
        self.enable_attn_gating = enable_attn_gating

        self.q_size = num_heads_q * head_dim
        self.kv_size = num_heads_kv * head_dim
        self.gating_size = num_heads_q if enable_attn_gating else 0
        qkv_out = self.q_size + self.kv_size * 2 + self.gating_size

        self.pre_norm = MultiModalityRMSNorm(
            hidden_size,
            num_modality=NUM_MODALITIES if is_mm_layer else 1,
        )
        self.q_norm = MultiModalityRMSNorm(
            head_dim,
            num_modality=NUM_MODALITIES if is_mm_layer else 1,
        )
        self.k_norm = MultiModalityRMSNorm(
            head_dim,
            num_modality=NUM_MODALITIES if is_mm_layer else 1,
        )

        if is_mm_layer:
            self.linear_qkv: nn.Module = MoELinear(
                hidden_size, qkv_out,
                num_experts=NUM_MODALITIES, bias=False,
                params_dtype=params_dtype,
            )
            self.linear_proj: nn.Module = MoELinear(
                self.q_size, hidden_size,
                num_experts=NUM_MODALITIES, bias=False,
                params_dtype=params_dtype,
            )
        else:
            self.linear_qkv = ReplicatedLinear(
                hidden_size, qkv_out,
                bias=False, params_dtype=params_dtype,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_qkv",
            )
            self.linear_proj = ReplicatedLinear(
                self.q_size, hidden_size,
                bias=False, params_dtype=params_dtype,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_proj",
            )

        self.attn = DistributedAttention(
            num_heads=num_heads_q,
            head_size=head_dim,
            num_kv_heads=num_heads_kv,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn",
        )

    def _project_qkv(
        self,
        x: torch.Tensor,
        token_counts: list[int],
        is_mm: bool,
    ) -> torch.Tensor:
        if is_mm:
            return self.linear_qkv(x.to(torch.bfloat16),
                                   token_counts=token_counts)
        else:
            out, _ = self.linear_qkv(x.to(torch.bfloat16))
            return out

    def _project_out(
        self,
        x: torch.Tensor,
        token_counts: list[int],
        is_mm: bool,
    ) -> torch.Tensor:
        if is_mm:
            return self.linear_proj(x, token_counts=token_counts)
        else:
            out, _ = self.linear_proj(x)
            return out

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        dispatcher: ModalityDispatcher,
        token_counts: list[int],
        is_mm: bool,
    ) -> torch.Tensor:
        """
        x:    [S, hidden_size] — permuted (video | audio | text)
        rope: [S, head_dim]   — in ORIGINAL token order
        """
        # Pre-norm
        x_norm = self.pre_norm(x.to(torch.bfloat16), token_counts if is_mm else None)

        # Project QKV
        qkv = self._project_qkv(x_norm, token_counts, is_mm).to(torch.float32)

        # Split QKV + optional gate
        q, k, v, g = torch.split(
            qkv,
            [self.q_size, self.kv_size, self.kv_size, self.gating_size],
            dim=-1,
        ) if self.enable_attn_gating else (
            *torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1),
            None,
        )

        # Reshape to [S, H, D]
        q = q.view(-1, self.num_heads_q, self.head_dim)
        k = k.view(-1, self.num_heads_kv, self.head_dim)
        v = v.view(-1, self.num_heads_kv, self.head_dim)
        if g is not None:
            g = g.view(-1, self.num_heads_q, 1)

        # Per-head Q/K norm: reshape to [S*H, head_dim] so each head vector is
        # normed independently; token_counts scaled by H to preserve modality
        # boundaries after the per-head reshape.
        S_seq = q.shape[0]
        tc_q = ([c * self.num_heads_q for c in token_counts]
                if (is_mm and token_counts) else None)
        tc_k = ([c * self.num_heads_kv for c in token_counts]
                if (is_mm and token_counts) else None)
        q = self.q_norm(
            q.reshape(S_seq * self.num_heads_q, self.head_dim), tc_q,
        ).reshape(S_seq, self.num_heads_q, self.head_dim)
        k = self.k_norm(
            k.reshape(S_seq * self.num_heads_kv, self.head_dim), tc_k,
        ).reshape(S_seq, self.num_heads_kv, self.head_dim)

        # Apply RoPE in ORIGINAL token order:
        # un-permute → apply → re-permute
        rope_orig = dispatcher.inv_permute(rope)  # [S, head_dim]
        sin_emb, cos_emb = rope_orig.tensor_split(2, -1)  # each [S, head_dim//2]

        q_orig = dispatcher.inv_permute(q)   # [S, H_q, D]
        k_orig = dispatcher.inv_permute(k)   # [S, H_kv, D]
        v_orig = dispatcher.inv_permute(v)   # [S, H_kv, D]

        # Add batch dim for DistributedAttention: [1, S, H, D]
        q_4d = q_orig.unsqueeze(0)
        k_4d = k_orig.unsqueeze(0)
        v_4d = v_orig.unsqueeze(0)

        q_4d = _apply_rope(q_4d, cos_emb, sin_emb)
        k_4d = _apply_rope(k_4d, cos_emb, sin_emb)

        # Cast to bf16 before SDPA: QKV was float32 for norm precision, but
        # flash-attn requires fp16/bf16. Without this, SDPA falls back to the
        # math kernel which materializes [H, S, S] — at S=130K that is ~2.7TB.
        q_4d = q_4d.to(torch.bfloat16)
        k_4d = k_4d.to(torch.bfloat16)
        v_4d = v_4d.to(torch.bfloat16)

        # Use F.scaled_dot_product_attention to bypass DistributedAttention
        # (which cats q/k/v requiring equal heads and asserts on GQA).
        # .contiguous() is required: permute() produces non-contiguous tensors
        # and flash-attn 2 requires contiguous inputs (otherwise triggers an
        # OOB vectorized_gather_kernel assert on the flash-attn CUDA stream).
        q_sd = q_4d.permute(0, 2, 1, 3).contiguous()  # [1, H_q, S, D]
        k_sd = k_4d.permute(0, 2, 1, 3).contiguous()  # [1, H_kv, S, D]
        v_sd = v_4d.permute(0, 2, 1, 3).contiguous()  # [1, H_kv, S, D]
        # Use enable_gqa for GQA — do NOT repeat_interleave k/v before SDPA.
        # repeat_interleave produces expanded tensors that trigger internal
        # device-side asserts inside PyTorch's flash attention kernel;
        # enable_gqa lets SDPA handle GQA natively without expansion.
        sdpa_kwargs = {}
        if self.num_heads_kv != self.num_heads_q:
            sdpa_kwargs["enable_gqa"] = True
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q_sd, k_sd, v_sd, **sdpa_kwargs)   # [1, H_q, S, D]
        attn_out = attn_out.permute(0, 2, 1, 3).squeeze(0)  # [S, H_q, D]

        # Re-permute back to modality-sorted order
        attn_out = dispatcher.permute(attn_out)      # [S, H_q, D]

        # Per-head gating
        if g is not None:
            attn_out = attn_out * torch.sigmoid(g)

        # Flatten heads and project
        attn_out = attn_out.view(-1, self.q_size).to(torch.bfloat16)
        return self._project_out(attn_out, token_counts, is_mm)


# ─── MLP ──────────────────────────────────────────────────────────────────────

class DaVinciMLP(nn.Module):
    """Feed-forward block.

    Sandwich layers: MoELinear (per-modality).
    Middle layers: ReplicatedLinear (shared, LoRA-targetable).
    Activation: swiglu7 for middle layers, gelu7 for sandwich layers.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        is_mm_layer: bool,
        use_gelu7: bool,
        params_dtype: torch.dtype = torch.bfloat16,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.use_gelu7 = use_gelu7
        gated = not use_gelu7  # swiglu7 is gated; gelu7 is not
        up_out = intermediate_size * 2 if gated else intermediate_size

        self.pre_norm = MultiModalityRMSNorm(
            hidden_size,
            num_modality=NUM_MODALITIES if is_mm_layer else 1,
        )

        if is_mm_layer:
            self.up_gate_proj: nn.Module = MoELinear(
                hidden_size, up_out,
                num_experts=NUM_MODALITIES, bias=False,
                params_dtype=params_dtype,
            )
            self.down_proj: nn.Module = MoELinear(
                intermediate_size, hidden_size,
                num_experts=NUM_MODALITIES, bias=False,
                params_dtype=params_dtype,
            )
        else:
            self.up_gate_proj = ReplicatedLinear(
                hidden_size, up_out,
                bias=False, params_dtype=params_dtype,
                quant_config=quant_config,
                prefix=f"{prefix}.up_gate_proj",
            )
            self.down_proj = ReplicatedLinear(
                intermediate_size, hidden_size,
                bias=False, params_dtype=params_dtype,
                quant_config=quant_config,
                prefix=f"{prefix}.down_proj",
            )

    def _up(
        self, x: torch.Tensor, token_counts: list[int], is_mm: bool
    ) -> torch.Tensor:
        if is_mm:
            return self.up_gate_proj(x, token_counts=token_counts)
        else:
            out, _ = self.up_gate_proj(x)
            return out

    def _down(
        self, x: torch.Tensor, token_counts: list[int], is_mm: bool
    ) -> torch.Tensor:
        if is_mm:
            return self.down_proj(x, token_counts=token_counts)
        else:
            out, _ = self.down_proj(x)
            return out

    def forward(
        self,
        x: torch.Tensor,
        token_counts: list[int],
        is_mm: bool,
    ) -> torch.Tensor:
        x_norm = self.pre_norm(x.to(torch.bfloat16),
                               token_counts if is_mm else None)
        up = self._up(x_norm, token_counts, is_mm).to(torch.float32)
        act = _gelu7(up) if self.use_gelu7 else _swiglu7(up)
        return self._down(act.to(torch.bfloat16), token_counts, is_mm)


# ─── Transformer layer ────────────────────────────────────────────────────────

class DaVinciTransformerLayer(nn.Module):
    """Single transformer layer (attention + MLP, pre-norm, optional post-norm).

    Layers in mm_layers (sandwich) are multi-modal: MoE weights per modality.
    Middle layers share weights across all modalities.
    Layers in gelu7_layers use GELU7 activation; others use SwiGLU7.
    """

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        enable_attn_gating: bool = True,
        mm_layers: frozenset = _MM_LAYERS,
        gelu7_layers: frozenset = _GELU7_LAYERS,
        post_norm_layers: frozenset = frozenset(),
        params_dtype: torch.dtype = torch.bfloat16,
        quant_config: Optional[QuantizationConfig] = None,
        supported_attention_backends: Optional[tuple] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.is_mm = layer_idx in mm_layers
        self.use_gelu7 = layer_idx in gelu7_layers
        self.has_post_norm = layer_idx in post_norm_layers

        # Intermediate size differs by activation type
        if self.use_gelu7:
            # GELU7 layers (0-3): no gating, simple 4× expansion
            intermediate_size = hidden_size * 4
        else:
            # SwiGLU7 layers (4-39): gated, 8/3× expansion rounded to mult-of-4
            intermediate_size = int(hidden_size * 4 * 2 / 3) // 4 * 4

        self.attention = DaVinciAttention(
            hidden_size=hidden_size,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            is_mm_layer=self.is_mm,
            enable_attn_gating=enable_attn_gating,
            params_dtype=params_dtype,
            quant_config=quant_config,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attention",
        )
        self.mlp = DaVinciMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            is_mm_layer=self.is_mm,
            use_gelu7=self.use_gelu7,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        if self.has_post_norm:
            self.attn_post_norm = MultiModalityRMSNorm(
                hidden_size,
                num_modality=NUM_MODALITIES if self.is_mm else 1,
            )
            self.mlp_post_norm = MultiModalityRMSNorm(
                hidden_size,
                num_modality=NUM_MODALITIES if self.is_mm else 1,
            )

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        dispatcher: ModalityDispatcher,
        token_counts: list[int],
    ) -> torch.Tensor:
        tc = token_counts if self.is_mm else None
        is_mm = self.is_mm

        attn_out = self.attention(x, rope, dispatcher, token_counts, is_mm)
        if self.has_post_norm:
            attn_out = self.attn_post_norm(attn_out, tc)
        x = x + attn_out

        mlp_out = self.mlp(x, token_counts, is_mm)
        if self.has_post_norm:
            mlp_out = self.mlp_post_norm(mlp_out, tc)
        x = x + mlp_out
        return x


# ─── Top-level DiT ────────────────────────────────────────────────────────────

class DaVinciMagiHuman(BaseDiT):
    """15B single-stream Transformer for audio-video generation.

    Architecture highlights:
    - 40-layer unified Transformer processing text, video, audio jointly.
    - Sandwich design: first/last 4 layers use MoE (per-modality projections),
      middle 32 layers share weights across modalities.
    - Timestep-free: denoising state inferred from input latents (no adaln).
    - Per-head scalar sigmoid attention gating for stability.
    - Custom 9-dim Fourier positional embedding (not standard RoPE).

    See: https://arxiv.org/abs/2603.21986
    """

    # ── Class-level attrs required by BaseDiT.__init_subclass__ ──────────────
    _fsdp_shard_conditions: list = ["layers"]
    _compile_conditions: list = ["layers"]
    param_names_mapping: dict = {
        # TransformerBlock wrapper removed; layers are direct children
        r"^block\.layers\.(\d+)\.(.*)$": r"layers.\1.\2",
        # Adapter embedder → proj rename
        r"^adapter\.video_embedder\.(.*)$": r"adapter.video_proj.\1",
        r"^adapter\.text_embedder\.(.*)$":  r"adapter.text_proj.\1",
        r"^adapter\.audio_embedder\.(.*)$": r"adapter.audio_proj.\1",
        # Final linear → proj rename
        r"^final_linear_video\.(.*)$": r"final_proj_video.\1",
        r"^final_linear_audio\.(.*)$": r"final_proj_audio.\1",
    }

    def __init__(
        self,
        config: DiTConfig,
        hf_config: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(config, hf_config, **kwargs)

        arch = config.arch_config
        hs = arch.hidden_size
        nq = arch.num_heads_q
        nkv = arch.num_heads_kv
        hd = arch.head_dim
        mm = frozenset(arch.mm_layers)
        g7 = frozenset(arch.gelu7_layers)
        pn = frozenset(getattr(arch, "post_norm_layers", []))
        supported_backends = getattr(arch, "_supported_attention_backends", None)

        self.hidden_size = hs
        self.num_attention_heads = nq
        self.num_channels_latents = arch.z_dim

        # If config carries an updated mapping, use it
        if arch.param_names_mapping:
            self.__class__.param_names_mapping = arch.param_names_mapping

        self.adapter = DaVinciAdapter(
            hidden_size=hs,
            num_heads_q=nq,
            video_in_channels=arch.video_in_channels,
            audio_in_channels=arch.audio_in_channels,
            text_in_channels=arch.text_in_channels,
            quant_config=config.quant_config,
            prefix="adapter",
        )

        self.layers = nn.ModuleList([
            DaVinciTransformerLayer(
                layer_idx=i,
                hidden_size=hs,
                num_heads_q=nq,
                num_heads_kv=nkv,
                head_dim=hd,
                enable_attn_gating=arch.enable_attn_gating,
                mm_layers=mm,
                gelu7_layers=g7,
                post_norm_layers=pn,
                params_dtype=torch.bfloat16,
                quant_config=config.quant_config,
                supported_attention_backends=supported_backends,
                prefix=f"layers.{i}",
            )
            for i in range(arch.num_layers)
        ])

        # Final norms + projections, one per output modality
        self.final_norm_video = MultiModalityRMSNorm(hs)
        self.final_norm_audio = MultiModalityRMSNorm(hs)

        # Renamed from official final_linear_{video,audio}
        self.final_proj_video = ReplicatedLinear(
            hs, arch.video_in_channels,
            bias=False, params_dtype=torch.float32,
            quant_config=config.quant_config,
            prefix="final_proj_video",
        )
        self.final_proj_audio = ReplicatedLinear(
            hs, arch.audio_in_channels,
            bias=False, params_dtype=torch.float32,
            quant_config=config.quant_config,
            prefix="final_proj_audio",
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor] | None = None,
        timestep: torch.LongTensor | None = None,
        # daVinci-specific inputs
        coords_mapping: torch.Tensor | None = None,
        modality_mapping: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Timestep-free forward pass.

        Args:
            hidden_states:    [S, max(video_in, audio_in)] packed mixed-modality
                              tokens (video + audio + text interleaved).
            encoder_hidden_states: unused (text is embedded via adapter).
            timestep:         unused — daVinci is timestep-free.
            coords_mapping:   [S, 9] positional coordinates per token.
            modality_mapping: [S] int — MODALITY_VIDEO/AUDIO/TEXT per token.

        Returns:
            [S, max(video_in, audio_in)] denoised token predictions.
        """
        assert coords_mapping is not None and modality_mapping is not None, (
            "DaVinciMagiHuman requires coords_mapping and modality_mapping")

        x = hidden_states  # [S, in_channels]

        # ── Build modality dispatcher ─────────────────────────────────────────
        dispatcher = ModalityDispatcher.build(modality_mapping)
        video_mask, audio_mask, text_mask = dispatcher.masks

        # ── Adapter: embed input tokens + compute positional encoding ─────────
        x, rope = self.adapter(x, coords_mapping, video_mask, audio_mask, text_mask)
        # x:    [S, hidden_size] in original (interleaved) order
        # rope: [S, head_dim]   in original order

        # ── Permute tokens by modality for MoE dispatch ───────────────────────
        x = dispatcher.permute(x).to(torch.bfloat16)
        rope_perm = dispatcher.permute(rope)  # [S, head_dim] in sorted order
        token_counts = dispatcher.group_size_cpu

        # ── Transformer layers ────────────────────────────────────────────────
        for layer in self.layers:
            x = layer(x, rope_perm, dispatcher, token_counts)

        # ── Restore original token order ──────────────────────────────────────
        x = dispatcher.inv_permute(x)

        # ── Per-modality output heads ──────────────────────────────────────────
        x_video = x[video_mask].to(self.final_norm_video.weight.dtype)
        x_video = self.final_norm_video(x_video)
        x_video, _ = self.final_proj_video(x_video)

        x_audio = x[audio_mask].to(self.final_norm_audio.weight.dtype)
        x_audio = self.final_norm_audio(x_audio)
        x_audio, _ = self.final_proj_audio(x_audio)

        arch = self.config.arch_config
        out_channels = max(arch.video_in_channels, arch.audio_in_channels)
        out = torch.zeros(
            x.shape[0], out_channels,
            device=x.device, dtype=x.dtype,
        )
        out[video_mask, :arch.video_in_channels] = x_video
        out[audio_mask, :arch.audio_in_channels] = x_audio
        return out
