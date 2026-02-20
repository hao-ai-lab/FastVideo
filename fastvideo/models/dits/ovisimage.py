# SPDX-License-Identifier: Apache-2.0
"""
Ovis-Image Transformer2D Model — Approach B (FastVideo-native implementation)

Architecture: FLUX-like MM-DiT with:
  - 6 double-stream (joint) transformer blocks
  - 27 single-stream transformer blocks
  - Qwen3 text encoder (2048-dim) projected to shared 3072-dim hidden space
  - FLUX-style 3D RoPE with axes_dims_rope=[16, 56, 56]
  - FastVideo DistributedAttention for SP support
  - ReplicatedLinear layers (FSDP-compatible, TP-ready)
  - CachableDiT base for TeaCache optimization

Weight attribute names match Diffusers OvisImageTransformer2DModel exactly,
so param_names_mapping = {} and weights load without any remapping.
"""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.attention import DistributedAttention
from fastvideo.configs.models.dits import OvisImageTransformer2DModelConfig
from fastvideo.configs.models.dits.base import DiTConfig
from fastvideo.distributed.communication_op import (
    sequence_model_parallel_all_gather, sequence_model_parallel_shard)
from fastvideo.forward_context import get_forward_context
from fastvideo.layers.layernorm import RMSNorm
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.models.dits.base import CachableDiT
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers: latent packing / unpacking and position IDs
# ---------------------------------------------------------------------------


def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack [B, C, H, W] -> [B, (H/2)*(W/2), C*4] for Ovis-Image transformer."""
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(B, (H // 2) * (W // 2), C * 4)


def _unpack_latents(latents: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Unpack [B, (H/2)*(W/2), C*4] -> [B, C, H, W]."""
    B, _, channels = latents.shape
    C = channels // 4
    latents = latents.view(B, H // 2, W // 2, C, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(B, C, H, W)


def _prepare_img_ids(H_half: int, W_half: int,
                     device: torch.device) -> torch.Tensor:
    """Image position IDs for RoPE: [H_half*W_half, 3] with (0, row, col)."""
    ids = torch.zeros(H_half, W_half, 3, device=device)
    ids[..., 1] = torch.arange(H_half, device=device)[:, None]
    ids[..., 2] = torch.arange(W_half, device=device)[None, :]
    return ids.reshape(H_half * W_half, 3)


def _prepare_txt_ids(seq_len: int, device: torch.device) -> torch.Tensor:
    """Text position IDs for RoPE: [seq_len, 3] with (0, i, i)."""
    ids = torch.zeros(seq_len, 3, device=device)
    ids[:, 1] = torch.arange(seq_len, device=device)
    ids[:, 2] = torch.arange(seq_len, device=device)
    return ids


# ---------------------------------------------------------------------------
# FLUX-style RoPE
# ---------------------------------------------------------------------------


class OvisImageRoPE(nn.Module):
    """
    FLUX-style 3D RoPE for Ovis-Image.

    Splits head_dim across three axes according to axes_dims_rope, computing
    separate frequency tables per axis and concatenating them.

    Position IDs: [..., 3] where components index (axis0, axis1, axis2).
    For 2D images: axis0=0, axis1=row, axis2=col.
    """

    def __init__(self, head_dim: int, axes_dims: list[int],
                 theta: float = 10000.0):
        super().__init__()
        assert sum(axes_dims) == head_dim, (
            f"sum(axes_dims)={sum(axes_dims)} != head_dim={head_dim}")
        self.head_dim = head_dim
        self.axes_dims = axes_dims
        self.theta = theta

    def _freqs_for_axis(self, dim: int,
                        device: torch.device) -> torch.Tensor:
        """Inverse frequency vector for one axis dimension."""
        half = dim // 2
        return 1.0 / (self.theta ** (
            torch.arange(0, half, device=device, dtype=torch.float32) / half))

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ids: position IDs [..., 3]
        Returns:
            (cos, sin) each [..., head_dim]
        """
        cos_parts, sin_parts = [], []
        for axis_idx, dim in enumerate(self.axes_dims):
            pos = ids[..., axis_idx].float()
            inv_freq = self._freqs_for_axis(dim, ids.device)
            freqs = torch.outer(pos.reshape(-1),
                                inv_freq).reshape(*pos.shape, -1)
            # Interleaved pairs [θ0,θ0,θ1,θ1,...] — matches Diffusers repeat_interleave_real=True
            emb = freqs.repeat_interleave(2, dim=-1)
            cos_parts.append(emb.cos())
            sin_parts.append(emb.sin())

        return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings — pair-wise complex rotation matching Diffusers
    apply_rotary_emb with use_real_unbind_dim=-1.

    q, k: [B, seq, n_heads, head_dim]
    cos, sin: [seq, head_dim]  (interleaved pairs encoding)
    """
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(2)

    # Unbind adjacent pairs: x -> (x_real, x_imag) each [..., head_dim//2]
    q_r, q_i = q.reshape(*q.shape[:-1], -1, 2).unbind(-1)
    k_r, k_i = k.reshape(*k.shape[:-1], -1, 2).unbind(-1)
    # Rotate: [-imag, real] — matches torch.stack([-x_imag, x_real]).flatten
    q_rot = torch.stack([-q_i, q_r], dim=-1).flatten(-2)
    k_rot = torch.stack([-k_i, k_r], dim=-1).flatten(-2)

    return (q.float() * cos + q_rot.float() * sin).to(q.dtype), (
        k.float() * cos + k_rot.float() * sin).to(k.dtype)


# ---------------------------------------------------------------------------
# Adaptive layer norms
# ---------------------------------------------------------------------------


class OvisAdaLayerNormZero(nn.Module):
    """
    Adaptive LayerNorm with zero-initialized modulation for double stream blocks.
    Produces 6 values: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, 6 * hidden_size, bias=True)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        emb = self.linear(F.silu(c))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            emb.chunk(6, dim=-1))
        # c is [B, hidden]; x is [B, seq, hidden] — unsqueeze for broadcast
        x_norm = self.norm(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        return x_norm, gate_msa.unsqueeze(1), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1), gate_mlp.unsqueeze(1)


class OvisAdaLayerNormZeroSingle(nn.Module):
    """
    Adaptive LayerNorm with zero-initialized modulation for single stream blocks.
    Produces 3 values: (shift, scale, gate).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, 3 * hidden_size, bias=True)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(F.silu(c))
        shift, scale, gate = emb.chunk(3, dim=-1)
        # c is [B, hidden]; x is [B, seq, hidden] — unsqueeze for broadcast
        x_norm = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x_norm, gate.unsqueeze(1)


class OvisAdaLayerNormContinuous(nn.Module):
    """Final adaptive LayerNorm before output projection."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, 2 * hidden_size, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        emb = self.linear(F.silu(c))
        scale, shift = emb.chunk(2, dim=-1)
        # c is [B, hidden]; x is [B, seq, hidden] — unsqueeze for broadcast
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# GEGLU feed-forward
# ---------------------------------------------------------------------------


class OvisGEGLUFeedForward(nn.Module):
    """
    GEGLU feed-forward used in double stream blocks.
    Attribute names match Diffusers ff.net structure for weight compatibility.
    """

    def __init__(self, hidden_size: int, ff_dim: int):
        super().__init__()
        # Diffusers stores as ff.net[0].proj (GEGLU: gate+up fused) and ff.net[2]
        self.net = nn.ModuleList([
            _GEGLUGateUp(hidden_size, ff_dim),  # index 0
            nn.Identity(),                        # index 1 (dropout placeholder)
            nn.Linear(ff_dim, hidden_size, bias=True),  # index 2
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net[0](x)   # GEGLU
        out = self.net[2](x)  # down projection
        return out


class _GEGLUGateUp(nn.Module):
    """SwiGLU gate+up projection matching Diffusers FeedForward(activation_fn='swiglu').

    Diffusers SwiGLU: proj -> [hidden, gate], return hidden * silu(gate).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features * 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, gate = self.proj(x).chunk(2, dim=-1)
        return hidden * F.silu(gate)


# ---------------------------------------------------------------------------
# Attention sub-modules (attr names match Diffusers for weight loading)
# ---------------------------------------------------------------------------


class _OvisDoubleAttn(nn.Module):
    """
    Joint attention for double stream blocks.
    Attribute names mirror Diffusers: to_q, to_k, to_v, to_out,
    add_q_proj, add_k_proj, add_v_proj, to_add_out, norm_q/k, norm_added_q/k.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int,
                 supported_attention_backends, prefix: str):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Image QKV + output
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=True)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=True)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=True)
        self.to_out = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=True)])

        # Text QKV + output
        self.add_q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.add_k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.add_v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.to_add_out = nn.Linear(hidden_size, hidden_size, bias=True)

        # QK-Norm
        self.norm_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, eps=1e-6)
        self.norm_added_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = RMSNorm(head_dim, eps=1e-6)

        # Distributed attention (SP-aware)
        self.attn_op = DistributedAttention(
            num_heads=num_heads,
            head_size=head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn_op")

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        img_cos: torch.Tensor,
        img_sin: torch.Tensor,
        txt_cos: torch.Tensor,
        txt_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, img_seq = img.shape[:2]
        txt_seq = txt.shape[1]

        # Image QKV
        img_q = self.to_q(img).view(B, img_seq, self.num_heads, self.head_dim)
        img_k = self.to_k(img).view(B, img_seq, self.num_heads, self.head_dim)
        img_v = self.to_v(img).view(B, img_seq, self.num_heads, self.head_dim)
        img_q = self.norm_q(img_q).to(img_v.dtype)
        img_k = self.norm_k(img_k).to(img_v.dtype)
        img_q, img_k = _apply_rope(img_q, img_k, img_cos, img_sin)

        # Text QKV
        txt_q = self.add_q_proj(txt).view(B, txt_seq, self.num_heads,
                                          self.head_dim)
        txt_k = self.add_k_proj(txt).view(B, txt_seq, self.num_heads,
                                          self.head_dim)
        txt_v = self.add_v_proj(txt).view(B, txt_seq, self.num_heads,
                                          self.head_dim)
        txt_q = self.norm_added_q(txt_q).to(txt_v.dtype)
        txt_k = self.norm_added_k(txt_k).to(txt_v.dtype)
        txt_q, txt_k = _apply_rope(txt_q, txt_k, txt_cos, txt_sin)

        # Joint attention via DistributedAttention
        img_attn, txt_attn = self.attn_op(img_q, img_k, img_v, txt_q, txt_k,
                                          txt_v)

        img_out = self.to_out[0](img_attn.reshape(B, img_seq, -1))
        txt_out = self.to_add_out(txt_attn.reshape(B, txt_seq, -1))
        return img_out, txt_out


class _OvisSingleAttn(nn.Module):
    """
    Single-stream attention (merged image+text).
    Attribute names match Diffusers for weight loading.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int,
                 supported_attention_backends, prefix: str):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(hidden_size, hidden_size, bias=True)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=True)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=True)
        self.norm_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, eps=1e-6)

        self.attn_op = DistributedAttention(
            num_heads=num_heads,
            head_size=head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn_op")

    def forward(self, x: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
        B, seq = x.shape[:2]
        q = self.to_q(x).view(B, seq, self.num_heads, self.head_dim)
        k = self.to_k(x).view(B, seq, self.num_heads, self.head_dim)
        v = self.to_v(x).view(B, seq, self.num_heads, self.head_dim)
        q = self.norm_q(q).to(v.dtype)
        k = self.norm_k(k).to(v.dtype)
        q, k = _apply_rope(q, k, cos, sin)
        # Single stream: pass img+txt jointly, no split
        attn_out, _ = self.attn_op(q, k, v, None, None, None)
        return attn_out.reshape(B, seq, -1)


# ---------------------------------------------------------------------------
# Double stream block
# ---------------------------------------------------------------------------


class OvisImageDoubleStreamBlock(nn.Module):
    """
    FLUX-style joint (double-stream) transformer block.

    Image and text each have their own adaptive LayerNorm and FFN, but share
    a joint cross-attention. Attribute names match Diffusers for weight compat.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int,
                 ff_dim: int, supported_attention_backends, prefix: str):
        super().__init__()

        # Image stream
        self.norm1 = OvisAdaLayerNormZero(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = OvisGEGLUFeedForward(hidden_size, ff_dim)

        # Text stream
        self.norm1_context = OvisAdaLayerNormZero(hidden_size)
        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False,
                                          eps=1e-6)
        self.ff_context = OvisGEGLUFeedForward(hidden_size, ff_dim)

        # Joint attention
        self.attn = _OvisDoubleAttn(hidden_size, num_heads, head_dim,
                                    supported_attention_backends,
                                    prefix=f"{prefix}.attn")

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        img_cos: torch.Tensor,
        img_sin: torch.Tensor,
        txt_cos: torch.Tensor,
        txt_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Adaptive norms
        img_n, img_gate_msa, img_shift_mlp, img_scale_mlp, img_gate_mlp = (
            self.norm1(img, vec))
        txt_n, txt_gate_msa, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = (
            self.norm1_context(txt, vec))

        # Joint attention
        img_attn, txt_attn = self.attn(img_n, txt_n, img_cos, img_sin, txt_cos,
                                       txt_sin)

        # Image: residual + norm + MLP
        img = img + img_gate_msa * img_attn
        img_ff_in = self.norm2(img) * (1 + img_scale_mlp) + img_shift_mlp
        img = img + img_gate_mlp * self.ff(img_ff_in)

        # Text: residual + norm + MLP
        txt = txt + txt_gate_msa * txt_attn
        txt_ff_in = self.norm2_context(txt) * (1 + txt_scale_mlp) + txt_shift_mlp
        txt = txt + txt_gate_mlp * self.ff_context(txt_ff_in)

        return img, txt


# ---------------------------------------------------------------------------
# Single stream block
# ---------------------------------------------------------------------------


class OvisImageSingleStreamBlock(nn.Module):
    """
    FLUX-style single-stream transformer block.

    Receives img and txt separately, concatenates txt-first internally (matching
    Diffusers), processes jointly, then splits and returns (txt, img).
    Attribute names match Diffusers for weight compat.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int,
                 mlp_ratio: float, supported_attention_backends, prefix: str):
        super().__init__()
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.norm = OvisAdaLayerNormZeroSingle(hidden_size)
        self.attn = _OvisSingleAttn(hidden_size, num_heads, head_dim,
                                    supported_attention_backends,
                                    prefix=f"{prefix}.attn")
        # proj_mlp outputs 2 × mlp_hidden_dim for SiLU gating (matching Diffusers)
        self.proj_mlp = nn.Linear(hidden_size, self.mlp_hidden_dim * 2, bias=True)
        self.proj_out = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size,
                                  bias=True)

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        temb: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        txt_seq = txt.shape[1]
        # Concat txt first then img (matching Diffusers convention)
        x = torch.cat([txt, img], dim=1)
        residual = x

        x_norm, gate = self.norm(x, temb)
        # SiLU-gated MLP (not GeLU): proj_mlp → [hidden, gate], silu(gate) * hidden
        mlp_out, mlp_gate = self.proj_mlp(x_norm).chunk(2, dim=-1)
        mlp_out = F.silu(mlp_gate) * mlp_out
        attn_out = self.attn(x_norm, cos, sin)
        combined = torch.cat([attn_out, mlp_out], dim=-1)
        x = residual + gate * self.proj_out(combined)

        txt_out = x[:, :txt_seq]
        img_out = x[:, txt_seq:]
        return txt_out, img_out


# ---------------------------------------------------------------------------
# Timestep + pooled text conditioning
# ---------------------------------------------------------------------------


def _timestep_embedding(t: torch.Tensor, dim: int,
                        max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) *
        torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class OvisTimestepEmbedder(nn.Module):
    """
    Timestep MLP matching Diffusers TimestepEmbedding weight names.

    Checkpoint keys: timestep_embedder.linear_1, timestep_embedder.linear_2
    """

    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(x)))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

_CFG = OvisImageTransformer2DModelConfig()


class OvisImageTransformer2DModel(CachableDiT):
    """
    Native FastVideo implementation of the Ovis-Image diffusion transformer.

    Architecture: FLUX-like MM-DiT
      - 6 double-stream (joint) blocks (transformer_blocks)
      - 27 single-stream blocks (single_transformer_blocks)
      - FLUX-style 3D RoPE (axes_dims_rope=[16, 56, 56])
      - FastVideo DistributedAttention (SP-compatible)
      - CachableDiT base (TeaCache-ready)

    Weight names match Diffusers OvisImageTransformer2DModel exactly
    => param_names_mapping = {} (no remapping, weights load directly).
    """

    # ---- Required CachableDiT class attributes ----
    _fsdp_shard_conditions = _CFG._fsdp_shard_conditions
    _compile_conditions: list = []
    _supported_attention_backends: tuple[
        AttentionBackendEnum, ...] = _CFG._supported_attention_backends
    param_names_mapping: dict = {}
    reverse_param_names_mapping: dict = {}
    lora_param_names_mapping: dict = {}

    def __init__(self, config: DiTConfig, hf_config: dict[str, Any],
                 **kwargs) -> None:
        super().__init__(config=config, hf_config=hf_config)

        arch = config.arch_config
        hidden_size: int = arch.hidden_size
        num_heads: int = arch.num_attention_heads
        head_dim: int = arch.attention_head_dim
        num_layers: int = arch.num_layers
        num_single_layers: int = arch.num_single_layers
        in_channels: int = arch.in_channels
        out_channels: int = (arch.out_channels
                             if arch.out_channels is not None else in_channels)
        joint_attention_dim: int = arch.joint_attention_dim

        # FastVideo required instance attributes
        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        self.num_channels_latents = arch.num_channels_latents  # 16 (VAE latent ch)
        self.out_channels = out_channels
        self.in_channels = in_channels

        ff_dim = hidden_size * 4  # standard MLP ratio

        # Input projections (weight names match Diffusers)
        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        # Norm applied to text encoder output before projection (matches Diffusers)
        self.context_embedder_norm = RMSNorm(joint_attention_dim, eps=1e-6)
        self.context_embedder = nn.Linear(joint_attention_dim, hidden_size,
                                          bias=True)

        # Timestep conditioning (purely from timestep, no pooled text)
        # Matches Diffusers: timestep_embedder.linear_1 / timestep_embedder.linear_2
        self._freq_dim = 256
        self.timestep_embedder = OvisTimestepEmbedder(self._freq_dim, hidden_size)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            OvisImageDoubleStreamBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                ff_dim=ff_dim,
                supported_attention_backends=self._supported_attention_backends,
                prefix=f"transformer_blocks.{i}",
            ) for i in range(num_layers)
        ])

        self.single_transformer_blocks = nn.ModuleList([
            OvisImageSingleStreamBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                mlp_ratio=4.0,
                supported_attention_backends=self._supported_attention_backends,
                prefix=f"single_transformer_blocks.{i}",
            ) for i in range(num_single_layers)
        ])

        # Output (weight names match Diffusers)
        self.norm_out = OvisAdaLayerNormContinuous(hidden_size)
        self.proj_out = nn.Linear(hidden_size, out_channels, bias=True)

        # RoPE
        self.rope = OvisImageRoPE(
            head_dim=head_dim,
            axes_dims=list(arch.axes_dims_rope),
        )

        self.__post_init__()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor]
        | None = None,
        guidance=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, C, T, H, W] or [B, C, H, W] latents
            encoder_hidden_states: [B, txt_seq, joint_attention_dim] or list
            timestep: [B] diffusion timestep
        Returns:
            Denoised latents in same shape as hidden_states
        """
        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = encoder_hidden_states[0]

        had_temporal = hidden_states.ndim == 5
        if had_temporal:
            hidden_states = hidden_states.squeeze(2)

        B, C, H, W = hidden_states.shape

        # Pack latents: [B, C, H, W] -> [B, img_seq, C*4]
        img_latents = _pack_latents(hidden_states)

        # Project to hidden_size
        img = self.x_embedder(img_latents)            # [B, img_seq, hidden_size]
        # Apply RMSNorm before projecting text (matches Diffusers context_embedder_norm)
        enc_norm = self.context_embedder_norm(encoder_hidden_states)
        txt = self.context_embedder(enc_norm)          # [B, txt_seq, hidden_size]

        txt_seq = txt.shape[1]
        img_seq = img.shape[1]

        # Timestep-only conditioning (Diffusers: timestep * 1000 then sinusoidal)
        # Input timestep is in [0, 1000]; sinusoidal embedding is computed at that scale
        t_emb = _timestep_embedding(timestep, self._freq_dim).to(img.dtype)
        temb = self.timestep_embedder(t_emb)           # [B, hidden_size]

        # RoPE position IDs — joint sequence: txt first, img second (matches Diffusers)
        img_ids = kwargs.get("img_ids")
        txt_ids = kwargs.get("txt_ids")
        if img_ids is None:
            img_ids = _prepare_img_ids(H // 2, W // 2, hidden_states.device)
        if txt_ids is None:
            txt_ids = _prepare_txt_ids(txt_seq, hidden_states.device)

        # Joint RoPE (txt first, then img)
        joint_ids = torch.cat([txt_ids, img_ids], dim=0)
        joint_cos, joint_sin = self.rope(joint_ids)
        joint_cos = joint_cos.to(img.dtype)
        joint_sin = joint_sin.to(img.dtype)
        txt_cos = joint_cos[:txt_seq]
        txt_sin = joint_sin[:txt_seq]
        img_cos = joint_cos[txt_seq:]
        img_sin = joint_sin[txt_seq:]

        # TeaCache early exit check
        forward_context = get_forward_context()
        forward_batch = getattr(forward_context, "forward_batch", None)
        enable_teacache = (forward_batch is not None
                           and getattr(forward_batch, "enable_teacache", False))
        if enable_teacache:
            original_img = img.clone()

        # Sequence Parallelism: shard image sequence across SP ranks
        img, _ = sequence_model_parallel_shard(img, dim=1)

        # Double-stream blocks: temb as conditioning
        for block in self.transformer_blocks:
            img, txt = block(img, txt, temb, img_cos, img_sin, txt_cos, txt_sin)

        # Single-stream blocks: blocks handle txt/img concat internally (txt first)
        for block in self.single_transformer_blocks:
            txt, img = block(img, txt, temb, joint_cos, joint_sin)

        # Gather SP shards
        img = sequence_model_parallel_all_gather(img, dim=1)

        if enable_teacache:
            self.maybe_cache_states(img, original_img)

        # Output (img stream only)
        img = self.norm_out(img, temb)
        img = self.proj_out(img)

        output = _unpack_latents(img, H, W)
        if had_temporal:
            output = output.unsqueeze(2)

        return output

    # ------------------------------------------------------------------
    # TeaCache interface (CachableDiT)
    # ------------------------------------------------------------------

    def maybe_cache_states(self, hidden_states: torch.Tensor,
                           original_hidden_states: torch.Tensor) -> None:
        """Cache residual between current and original hidden states."""
        self.previous_resiual = hidden_states - original_hidden_states

    def should_skip_forward_for_cached_states(self, **kwargs) -> bool:
        """TeaCache skip decision — not yet calibrated for Ovis-Image."""
        forward_context = get_forward_context()
        forward_batch = getattr(forward_context, "forward_batch", None)
        if forward_batch is None:
            return False
        return False  # Always compute for now; calibrate coefficients later
