# Copyright 2026 Linum Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native single-device Pyramid-JiT image denoiser, adapted from Linum-AI/pjit-21k.

The three transformer stacks operate on image tokens, caption tokens, and their
concatenation. Their padding masks are local to each stack. We pass these to
FastVideo's attention implementations directly instead of using a single
sequence-parallel attention context for the entire denoising step.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from fastvideo.attention.backends.sdpa import SDPAMetadata
from fastvideo.attention.selector import effective_attention_backend, get_attn_backend
from fastvideo.configs.models.dits.pjit import PJiTConfig
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.models.dits.base import BaseDiT
from fastvideo.platforms import AttentionBackendEnum


class _RMSNorm(nn.Module):
    """Linum rounds the normalized input before multiplying by its fp32 gain."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def _rotary_frequencies(config):
    # Construct in fp64 even when initializing on meta. Storing the rounded
    # real/imaginary pairs in fp32 keeps the exact rotation used by Linum and
    # allows MPS to consume the table without moving complex128 onto Metal.
    head_pairs = config.head_dim // 2
    spatial_pairs = head_pairs // 3
    widths = [head_pairs - 2 * spatial_pairs, spatial_pairs, spatial_pairs]
    axes = []
    for pairs, scale in zip(widths, (1.0, config.rope_spatial_scale, config.rope_spatial_scale), strict=True):
        positions = torch.arange(config.rope_max_positions, dtype=torch.float64, device="cpu") * scale
        powers = torch.arange(pairs, dtype=torch.float64, device="cpu") / pairs
        angles = torch.outer(positions, 1.0 / torch.pow(10000.0, powers))
        axes.append(torch.stack([angles.cos(), angles.sin()], -1).float())
    return torch.cat(axes, dim=1)


def _grid_rotations(freqs, frames, height, width):
    if max(frames, height, width) > freqs.size(0):
        raise ValueError(f"PyramidJiT patch grid {(frames, height, width)} exceeds RoPE table length {freqs.size(0)}")
    half = freqs.size(1)
    spatial = half // 3
    temporal, rows, cols = freqs.split([half - 2 * spatial, spatial, spatial], dim=1)
    return torch.cat([
        temporal[:frames, None, None].expand(-1, height, width, -1, -1),
        rows[None, :height, None].expand(frames, -1, width, -1, -1),
        cols[None, None, :width].expand(frames, height, -1, -1, -1),
    ], dim=3).flatten(0, 2)


def _apply_rotations(x, rope):
    cos, sin = rope[None, :, None].unbind(-1)
    even, odd = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2).to(x.dtype)


class _Attention(nn.Module):

    def __init__(self, config, backend):
        super().__init__()
        dim, heads = config.dim, config.num_heads
        self.q = ReplicatedLinear(dim, dim)
        self.k = ReplicatedLinear(dim, dim)
        self.v = ReplicatedLinear(dim, dim)
        self.o = ReplicatedLinear(dim, dim)
        self.norm_q = _RMSNorm(dim, config.eps)
        self.norm_k = _RMSNorm(dim, config.eps)
        self.gate_proj = ReplicatedLinear(dim, heads)
        self.heads = heads
        self.use_flash = backend.get_name() == "FLASH_ATTN"
        self.impl = backend.get_impl_cls()(num_heads=heads, head_size=config.head_dim,
                                           softmax_scale=config.head_dim ** -0.5, causal=False, num_kv_heads=heads)

    def forward(self, x, mask, rope, metadata):
        padding = ~mask.unsqueeze(-1)
        q = self.norm_q(self.q(x)[0].masked_fill(padding, 0)).unflatten(-1, (self.heads, -1))
        k = self.norm_k(self.k(x)[0].masked_fill(padding, 0)).unflatten(-1, (self.heads, -1))
        v = self.v(x)[0].masked_fill(padding, 0).unflatten(-1, (self.heads, -1))
        q, k = _apply_rotations(q, rope), _apply_rotations(k, rope)
        dtype = q.dtype
        if self.use_flash:
            # The upstream packed flash kernel receives bf16 explicitly. In
            # SDPA the surrounding autocast context controls compute precision.
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
        attended = self.impl.forward(q, k, v, metadata).to(dtype)
        gate = self.gate_proj(x)[0].masked_fill(padding, -1e4).sigmoid().unsqueeze(-1)
        return self.o((attended * gate).flatten(2))[0].masked_fill(padding, 0)


class _SwiGLU(nn.Module):

    def __init__(self, dim, ffn_dim):
        super().__init__()
        width = math.ceil(int(2 * ffn_dim / 3) / 256) * 256
        self.w13 = ReplicatedLinear(dim, 2 * width, bias=False)
        self.w2 = ReplicatedLinear(width, dim, bias=False)

    def forward(self, x):
        gate, value = self.w13(x)[0].chunk(2, -1)
        return self.w2(F.silu(gate) * value)[0]


class _Block(nn.Module):

    def __init__(self, config, ffn_dim, backend):
        super().__init__()
        self.norm1 = _RMSNorm(config.dim, config.eps)
        self.attn = _Attention(config, backend)
        self.post_norm1 = _RMSNorm(config.dim, config.eps)
        self.norm2 = _RMSNorm(config.dim, config.eps)
        self.ffn = _SwiGLU(config.dim, ffn_dim)
        self.post_norm2 = _RMSNorm(config.dim, config.eps)
        self.adaln_up = ReplicatedLinear(config.adaln_rank, 4 * config.dim)

    def forward(self, x, c, mask, rope, metadata):
        sa, ga, sf, gf = self.adaln_up(c)[0].unsqueeze(1).chunk(4, dim=-1)
        attn = self.attn(self.norm1(x) * (1 + sa), mask, rope, metadata)
        x = x + ga.tanh() * (self.post_norm1(attn) * mask.unsqueeze(-1))
        ffn = self.ffn(self.norm2(x) * (1 + sf))
        return x + gf.tanh() * (self.post_norm2(ffn) * mask.unsqueeze(-1))


class _PatchEmbed(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.proj1 = nn.Conv3d(config.in_channels, config.bottleneck_dim, config.patch, stride=config.patch, bias=False)
        self.proj2 = nn.Conv3d(config.bottleneck_dim, config.dim, 1)

    def forward(self, x):
        return self.proj2(self.proj1(x))


class _Head(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.dim)
        self.proj = ReplicatedLinear(config.dim, math.prod(config.patch) * config.out_channels)
        self.modulation = nn.Parameter(torch.zeros(1, 2, config.dim))

    def forward(self, x, e):
        shift, scale = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        return self.proj(self.norm(x) * (1 + scale) + shift)[0]


_DEFAULTS = PJiTConfig()


class PyramidJiT(BaseDiT):
    _supported_attention_backends = _DEFAULTS._supported_attention_backends
    param_names_mapping = _DEFAULTS.param_names_mapping
    reverse_param_names_mapping = _DEFAULTS.reverse_param_names_mapping
    lora_param_names_mapping = _DEFAULTS.lora_param_names_mapping

    def __init__(self, config: PJiTConfig, hf_config: dict | None = None, **kwargs):
        super().__init__(config, hf_config or {}, **kwargs)
        if config.quant_config is not None:
            raise NotImplementedError("PyramidJiT expects unquantized fp32 master weights")
        self.hidden_size = config.dim
        self.num_attention_heads = config.num_heads
        self.num_channels_latents = config.out_channels
        self.text_len = config.text_len
        self.patch = tuple(config.patch)
        self.patch_embed = _PatchEmbed(config)
        self.time_embed = nn.Sequential(nn.Linear(config.freq_dim, config.dim), nn.SiLU(), nn.Linear(config.dim, config.dim))
        self.image_adaln = nn.Sequential(nn.SiLU(), nn.Linear(config.dim, config.adaln_rank))
        self.text_adaln = nn.Sequential(nn.SiLU(), nn.Linear(config.dim, config.adaln_rank))
        self.text_proj = nn.Sequential(nn.Linear(config.text_dim, config.dim, bias=False), nn.GELU(approximate="tanh"),
                                       nn.Linear(config.dim, config.dim, bias=False))
        backend = get_attn_backend(config.head_dim, torch.bfloat16, self._supported_attention_backends,
                                   default_backend=AttentionBackendEnum.TORCH_SDPA,
                                   requested=effective_attention_backend(config))
        self.blocks = nn.ModuleList(_Block(config, config.ffn_dim, backend) for _ in range(config.num_layers))
        self.image_refiner = nn.ModuleList(_Block(config, config.refiner_ffn_dim, backend)
                                           for _ in range(config.refiner_blocks))
        self.text_refiner = nn.ModuleList(_Block(config, config.refiner_ffn_dim, backend)
                                          for _ in range(config.refiner_blocks))
        self.head = _Head(config)
        self.register_buffer("rotations", _rotary_frequencies(config), persistent=False)

    def _time_embedding(self, t, dtype):
        # The release builds these short sinusoids in fp64. MPS has no fp64
        # arithmetic, so do that small operation on CPU on this platform.
        device = t.device
        positions = (t.cpu() if device.type == "mps" else t).double()
        half = self.config.freq_dim // 2
        powers = torch.arange(half, dtype=torch.float64, device=positions.device) / half
        angles = torch.outer(positions, torch.pow(10000., -powers))
        return self.time_embed(torch.cat([angles.cos(), angles.sin()], -1).to(dtype).to(device))

    @staticmethod
    def _run_blocks(blocks, x, c, mask, rope):
        # The shared backend takes a key-padding mask. Projection and residual
        # masks in each block also zero padded queries, as in Linum's implementation.
        metadata = SDPAMetadata(current_timestep=0, attn_mask=mask)
        for block in blocks:
            x = block(x, c, mask, rope, metadata)
        return x

    def forward(self, x_t, text, text_lens, t):
        """Map noisy [B,3,1,H,W] pixels and padded caption embeddings to clean pixels."""
        frames, height, width = x_t.shape[2:]
        pt, ph, pw = self.patch
        x_t = F.pad(x_t, (0, -width % pw, 0, -height % ph, 0, -frames % pt))
        image = self.patch_embed(x_t)
        grid = image.shape[2:]
        image = image.flatten(2).transpose(1, 2).contiguous()
        image_count = image.size(1)
        image_mask = torch.ones(image.shape[:2], dtype=torch.bool, device=image.device)
        text_mask = torch.arange(self.text_len, device=image.device)[None, :] < text_lens[:, None]
        image_rope = _grid_rotations(self.rotations, *grid)
        text_rope = self.rotations[0:1].expand(self.text_len, -1, -1)
        e = self._time_embedding(t, image.dtype)
        c_image = self.image_adaln(e)
        image = self._run_blocks(self.image_refiner, image, c_image, image_mask, image_rope)
        text = self._run_blocks(self.text_refiner, self.text_proj(text), self.text_adaln(e), text_mask, text_rope)
        mask = torch.cat([image_mask, text_mask], dim=1)
        tokens = torch.cat([image, text], dim=1).masked_fill(~mask.unsqueeze(-1), 0)
        tokens = self._run_blocks(self.blocks, tokens, c_image, mask, torch.cat([image_rope, text_rope], dim=0))
        patches = self.head(tokens[:, :image_count], e)
        fp, hp, wp = grid
        pixels = patches.reshape(x_t.size(0), fp, hp, wp, pt, ph, pw, self.config.out_channels)
        pixels = pixels.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(x_t.size(0), self.config.out_channels,
                                                                   fp * pt, hp * ph, wp * pw)
        return pixels[:, :, :frames, :height, :width]


EntryClass = PyramidJiT
