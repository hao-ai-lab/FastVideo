# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code=no-untyped-call
"""MiniMax-H3 video VAE for the Apple Silicon MLX runtime.

Faithful MLX port of ``fastvideo/models/vaes/minimax_h3_video.py`` (itself
parity-validated against the official diffusers implementation):

- **Encoder**: causal 3D CNN — causal temporal padding, reflect spatial
  padding, per-frame GroupNorm, residual blocks, spatial/temporal downsampling,
  then a 1x1x1 ``quant_conv`` producing mean/logvar channels.
- **Decoder**: 1x1x1 ``post_quant_conv``, then a 36-layer ViT — per-head Q/K
  RMSNorm, three-axis rotary embedding (theta 100, rotary width 48 of 64 head
  dims), register tokens plus a final zero class token, SwiGLU feed-forward,
  residual scale vectors, FP32 norm accumulation, final LayerNorm, output
  projection, and channel-major unpatchification (temporal patch 4, spatial
  patch 16x16) — with the exact clip chunking (``clip_length=17``,
  ``token_drop=3``), frame pre-padding, overlap blending, tail trimming, and
  optional spatial tiling of the released model.

Released weights are FP32; the loader streams shards so peak memory stays
bounded, and can optionally store bf16/fp16 after callers have measured the
dtype drift against the FP32 acceptance gate.

Production code here never imports PyTorch. Torch parity references live in
the tests under ``tests/local_tests/minimax_h3/``.
"""

from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import mlx.core as mx

from fastvideo.logger import init_logger

logger = init_logger(__name__)


def _ct(x, *order):
    """Contiguous transpose: some MLX builds silently mis-compute ops on
    strided views, so every layout change materializes."""
    return mx.contiguous(x.transpose(*order))


PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class MiniMaxH3VideoVAEConfigView:
    """Architecture constants (defaults mirror the released vae/config.json)."""

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 24
    block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024)
    layers_per_block: int = 2
    spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1)
    temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1)
    norm_num_groups: int = 32
    norm_eps: float = 1e-6
    decoder_num_layers: int = 36
    decoder_num_attention_heads: int = 32
    decoder_attention_head_dim: int = 64
    decoder_num_register_tokens: int = 4
    decoder_ffn_mult: int = 4
    decoder_rope_theta: float = 100.0
    decoder_rope_dim_ratio: float = 0.75
    decoder_norm_eps: float = 1e-5
    clip_length: int = 17
    token_drop: int = 3
    latents_mean: tuple[float, ...] | None = None
    latents_std: tuple[float, ...] | None = None

    @classmethod
    def from_vae_dir(cls, vae_dir: str | Path) -> MiniMaxH3VideoVAEConfigView:
        config = json.loads((Path(vae_dir) / "config.json").read_text())
        latents_mean = tuple(float(v) for v in config.get("latents_mean", ()))
        latents_std = tuple(float(v) for v in config.get("latents_std", ()))
        return cls(
            in_channels=int(config["in_channels"]),
            out_channels=int(config["out_channels"]),
            latent_channels=int(config["latent_channels"]),
            block_out_channels=tuple(int(v) for v in config["block_out_channels"]),
            layers_per_block=int(config["layers_per_block"]),
            spatial_downsample_factors=tuple(int(v) for v in config["spatial_downsample_factors"]),
            temporal_downsample_factors=tuple(int(v) for v in config["temporal_downsample_factors"]),
            norm_num_groups=int(config["norm_num_groups"]),
            norm_eps=float(config["norm_eps"]),
            decoder_num_layers=int(config["decoder_num_layers"]),
            decoder_num_attention_heads=int(config["decoder_num_attention_heads"]),
            decoder_attention_head_dim=int(config["decoder_attention_head_dim"]),
            decoder_num_register_tokens=int(config["decoder_num_register_tokens"]),
            decoder_ffn_mult=int(config.get("decoder_ffn_mult", 4)),
            decoder_rope_theta=float(config["decoder_rope_theta"]),
            decoder_rope_dim_ratio=float(config["decoder_rope_dim_ratio"]),
            decoder_norm_eps=float(config["decoder_norm_eps"]),
            clip_length=int(config["clip_length"]),
            token_drop=int(config["token_drop"]),
            latents_mean=latents_mean or None,
            latents_std=latents_std or None,
        )

    @property
    def spatial_compression_ratio(self) -> int:
        return math.prod(self.spatial_downsample_factors)

    @property
    def temporal_compression_ratio(self) -> int:
        return math.prod(self.temporal_downsample_factors)

    @property
    def tokens_chunk_size(self) -> int:
        """Latent frames decoded per clip chunk (ceil(clip_length / ratio))."""
        return math.ceil(self.clip_length / self.temporal_compression_ratio)

    @property
    def token_overlap(self) -> int:
        return (-self.token_drop) % self.tokens_chunk_size

    @property
    def frame_pre_padding(self) -> int:
        return (-self.clip_length) % self.temporal_compression_ratio

    @property
    def frame_overlap(self) -> int:
        return max(self.token_overlap * self.temporal_compression_ratio - self.frame_pre_padding, 0)


# ---------------------------------------------------------------------------
# MLX primitives (channels-last conv3d: input (N,D,H,W,C), weight (O,kT,kH,kW,C))
# ---------------------------------------------------------------------------


def _reflect_pad_axis(x, axis: int, left: int, right: int):
    """Torch-style reflect padding (edge value excluded), version-portable."""
    if left == 0 and right == 0:
        return x
    size = x.shape[axis]
    pieces = []
    if left > 0:
        pieces.append(mx.take(x, mx.array(list(range(left, 0, -1))), axis=axis))
    pieces.append(x)
    if right > 0:
        indices = list(range(size - 2, size - 2 - right, -1))
        pieces.append(mx.take(x, mx.array(indices), axis=axis))
    return mx.contiguous(mx.concatenate(pieces, axis=axis))


def _causal_conv3d(x, weight, bias=None, *, stride=(1, 1, 1), spatial_padding: int = 0, temporal_padding: int = 0):
    """Reflect spatial padding + zero causal temporal padding, then conv3d.

    Accepts and returns torch-style (N, C, T, H, W); MLX conv3d runs
    channels-last internally.
    """
    x = _ct(x, 0, 2, 3, 4, 1)
    if spatial_padding > 0:
        x = _reflect_pad_axis(x, 2, spatial_padding, spatial_padding)
        x = _reflect_pad_axis(x, 3, spatial_padding, spatial_padding)
    if temporal_padding > 0:
        x = mx.pad(x, ((0, 0), (temporal_padding, 0), (0, 0), (0, 0), (0, 0)))
    y = mx.conv3d(mx.contiguous(x), weight, stride=stride)
    if bias is not None:
        y = y + bias[None, None, None, None, :]
    return _ct(y, 0, 4, 1, 2, 3)


def _group_norm_per_frame(x, weight, bias, *, num_groups: int, eps: float):
    """GroupNorm over channels for every (batch, frame) independently.

    x: (N, C, T, H, W) -> same layout. Mirrors MiniMaxH3VideoGroupNorm.
    """
    n, c, t, h, w = x.shape
    assert c % num_groups == 0
    grouped = _ct(x, 0, 2, 1, 3, 4).reshape(n * t, num_groups, c // num_groups, h, w)
    mean = grouped.mean(axis=(2, 3, 4), keepdims=True)
    var = ((grouped - mean)**2).mean(axis=(2, 3, 4), keepdims=True)
    grouped = (grouped - mean) / mx.sqrt(var + eps)
    grouped = grouped.reshape(n, t, c, h, w)
    grouped = _ct(grouped, 0, 2, 1, 3, 4)
    return grouped * weight[None, :, None, None, None] + bias[None, :, None, None, None]


def _rms_norm_no_affine(x, eps: float):
    return x / mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)


def _silu(x):
    return x * (mx.sigmoid(x))


def _linear(x, weight, bias=None):
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


def _rotary_cos_sin(position_ids, *, rotary_dim: int, theta: float):
    """(S, 3) positions -> cos/sin each (S, 1, rotary_dim), fp32.

    inv_freq has rotary_dim//6 entries per axis; the three axes are
    concatenated and doubled (matches MiniMaxH3VideoRotaryPosEmbed). The
    singleton middle axis broadcasts over attention heads.
    """
    freqs_per_axis = rotary_dim // 6
    inv_freq = 1.0 / (theta**(mx.arange(0, freqs_per_axis, dtype=mx.float32) * (6.0 / rotary_dim)))
    angles = 2.0 * math.pi * position_ids[:, :, None].astype(mx.float32) * inv_freq[None, None, :]
    # (S, 3, F) -> (S, 3*F) -> doubled -> (S, 6*F == rotary_dim)
    angles = angles.reshape(position_ids.shape[0], 3 * freqs_per_axis)
    angles = mx.concatenate([angles, angles], axis=-1)[:, None, None, :]
    return mx.cos(angles), mx.sin(angles)


def _apply_rotary(query, key, cos, sin):
    """Half-split rotation on the rotary prefix of each head (fp32)."""
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    k_rot, k_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    q_rot = q_rot.astype(mx.float32)
    k_rot = k_rot.astype(mx.float32)
    q_first, q_second = mx.split(q_rot, 2, axis=-1)
    k_first, k_second = mx.split(k_rot, 2, axis=-1)
    q_out = q_rot * cos + mx.concatenate([-q_second, q_first], axis=-1) * sin
    k_out = k_rot * cos + mx.concatenate([-k_second, k_first], axis=-1) * sin
    return (mx.concatenate([q_out.astype(query.dtype), q_pass],
                           axis=-1), mx.concatenate([k_out.astype(key.dtype), k_pass], axis=-1))


def _attention(block: dict[str, Any], x, cos, sin, *, num_heads: int, head_dim: int, eps: float):
    """x: (S, B, D) sequence-major. Returns (S, B, D)."""
    seq_len, batch, hidden = x.shape
    q = _linear(x, block["attn.to_q.weight"], block["attn.to_q.bias"])
    k = _linear(x, block["attn.to_k.weight"], block["attn.to_k.bias"])
    v = _linear(x, block["attn.to_v.weight"], block["attn.to_v.bias"])

    # Per-head Q/K RMSNorm without affine parameters, accumulated in fp32.
    q = q.reshape(seq_len, batch, num_heads, head_dim)
    k = k.reshape(seq_len, batch, num_heads, head_dim)
    v = v.reshape(seq_len, batch, num_heads, head_dim)
    q = _rms_norm_no_affine(q.astype(mx.float32), eps).astype(x.dtype)
    k = _rms_norm_no_affine(k.astype(mx.float32), eps).astype(x.dtype)
    if cos is not None:
        q, k = _apply_rotary(q, k, cos, sin)
    # -> (B, H, S, Dh); contiguous() matters: some MLX builds silently
    # mis-compute fused SDPA on strided views.
    q = _ct(q, 1, 2, 0, 3)
    k = _ct(k, 1, 2, 0, 3)
    v = _ct(v, 1, 2, 0, 3)
    attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5)
    attended = _ct(attended, 0, 2, 1, 3).reshape(seq_len, batch, hidden)
    return _linear(attended, block["attn.to_out.0.weight"], block["attn.to_out.0.bias"])


def _feed_forward(block: dict[str, Any], x):
    hidden = _linear(x, block["ff.net.0.proj.weight"], block["ff.net.0.proj.bias"])
    value, gate = mx.split(hidden, 2, axis=-1)
    return _linear(value * _silu(gate), block["ff.net.2.weight"], block["ff.net.2.bias"])


def _transformer_block(block: dict[str, Any], x, scale1, scale2, cos, sin, *, num_heads: int, head_dim: int,
                       norm_eps: float, qk_norm_eps: float):
    """x: (S, B, D) sequence-major like the torch reference."""
    normed = _rms_norm_affine(x.astype(mx.float32), block["norm1.weight"], norm_eps).astype(x.dtype)
    x = x + _attention(block, normed, cos, sin, num_heads=num_heads, head_dim=head_dim, eps=qk_norm_eps) * scale1
    normed = _rms_norm_affine(x.astype(mx.float32), block["norm2.weight"], norm_eps).astype(x.dtype)
    return x + _feed_forward(block, normed) * scale2


def _rms_norm_affine(x_fp32, weight, eps: float):
    return x_fp32 / mx.sqrt(mx.mean(x_fp32 * x_fp32, axis=-1, keepdims=True) + eps) * weight


def _resnet_block(weights: dict[str, Any], prefix: str, x, *, num_groups: int, eps: float):
    residual = x
    h = _silu(
        _group_norm_per_frame(x,
                              weights[f"{prefix}.norm1.weight"],
                              weights[f"{prefix}.norm1.bias"],
                              num_groups=num_groups,
                              eps=eps))
    h = _causal_conv3d(h,
                       weights[f"{prefix}.conv1.weight"],
                       weights[f"{prefix}.conv1.bias"],
                       spatial_padding=1,
                       temporal_padding=2)
    h = _silu(
        _group_norm_per_frame(h,
                              weights[f"{prefix}.norm2.weight"],
                              weights[f"{prefix}.norm2.bias"],
                              num_groups=num_groups,
                              eps=eps))
    h = _causal_conv3d(h,
                       weights[f"{prefix}.conv2.weight"],
                       weights[f"{prefix}.conv2.bias"],
                       spatial_padding=1,
                       temporal_padding=2)
    shortcut_key = f"{prefix}.conv_shortcut.weight"
    if shortcut_key in weights:
        residual = _causal_conv3d(x, weights[shortcut_key], weights[f"{prefix}.conv_shortcut.bias"])
    return residual + h


# ---------------------------------------------------------------------------
# The MLX H3 video VAE
# ---------------------------------------------------------------------------


class MLXMiniMaxH3VideoVAE:
    """Encoder + decoder for the released MiniMax-H3 video VAE."""

    def __init__(self, weights: dict[str, Any], config: MiniMaxH3VideoVAEConfigView, *, has_encoder: bool = True):
        self.weights = weights
        self.config = config
        self.has_encoder = has_encoder
        self.latent_channels = config.latent_channels
        self.spatial_compression_ratio = config.spatial_compression_ratio
        self.temporal_compression_ratio = config.temporal_compression_ratio
        self.num_heads = config.decoder_num_attention_heads
        self.head_dim = config.decoder_attention_head_dim
        self.dim = self.num_heads * self.head_dim
        self.rotary_dim = int(self.head_dim * config.decoder_rope_dim_ratio)
        self._blocks: list[dict[str, Any]] | None = None
        arch = config
        mean = arch.latents_mean if arch.latents_mean is not None else [0.0] * arch.latent_channels
        std = arch.latents_std if arch.latents_std is not None else [1.0] * arch.latent_channels
        self._latents_mean = np.asarray(mean, dtype=np.float32).reshape(1, -1, 1, 1, 1)
        self._latents_std = np.asarray(std, dtype=np.float32).reshape(1, -1, 1, 1, 1)
        self._pixel_mean = np.asarray(PIXEL_MEAN, dtype=np.float32).reshape(1, -1, 1, 1, 1)
        self._pixel_std = np.asarray(PIXEL_STD, dtype=np.float32).reshape(1, -1, 1, 1, 1)

    # -- normalization -------------------------------------------------

    def normalize_latents(self, latents):
        return (latents - mx.array(self._latents_mean)) / mx.array(self._latents_std)

    def denormalize_latents(self, latents):
        return latents * mx.array(self._latents_std) + mx.array(self._latents_mean)

    def normalize_pixels(self, pixels):
        return (pixels - mx.array(self._pixel_mean)) / mx.array(self._pixel_std)

    def denormalize_pixels(self, sample):
        return sample * mx.array(self._pixel_std) + mx.array(self._pixel_mean)

    # -- encoder ---------------------------------------------------------

    def _encode_clip(self, x):
        """(N, 3, T, H, W) pixels -> (N, 2C, T', H', W') moments."""
        w = self.weights
        cfg = self.config
        x = _causal_conv3d(x,
                           w["encoder.conv_in.weight"],
                           w["encoder.conv_in.bias"],
                           spatial_padding=1,
                           temporal_padding=2)
        for index in range(len(cfg.block_out_channels)):
            td = cfg.temporal_downsample_factors[index]
            sd = cfg.spatial_downsample_factors[index]
            for layer in range(cfg.layers_per_block):
                x = _resnet_block(w,
                                  f"encoder.down_blocks.{index}.resnets.{layer}",
                                  x,
                                  num_groups=cfg.norm_num_groups,
                                  eps=cfg.norm_eps)
            if td * sd > 1:
                if sd == 2:
                    x = _reflect_pad_axis(x, 3, 0, 1)
                    x = _reflect_pad_axis(x, 4, 0, 1)
                x = _causal_conv3d(
                    x,
                    w[f"encoder.down_blocks.{index}.downsamplers.0.conv.weight"],
                    w[f"encoder.down_blocks.{index}.downsamplers.0.conv.bias"],
                    stride=(td, sd, sd),
                    temporal_padding=2,
                )
        x = _silu(
            _group_norm_per_frame(x,
                                  w["encoder.norm_out.weight"],
                                  w["encoder.norm_out.bias"],
                                  num_groups=cfg.norm_num_groups,
                                  eps=cfg.norm_eps))
        x = _causal_conv3d(x,
                           w["encoder.conv_out.weight"],
                           w["encoder.conv_out.bias"],
                           spatial_padding=1,
                           temporal_padding=2)
        x = _causal_conv3d(x, w["quant_conv.weight"], w["quant_conv.bias"])
        return x

    def encode(self, pixels):
        """Normalized pixels (1, 3, T, H, W) -> (mean, logvar) each (1, C, T', H', W').

        Pads the clip to ``clip_length`` and drops ``token_drop`` trailing
        moment frames exactly like the reference ``_encode``.
        """
        if not self.has_encoder:
            raise RuntimeError("This MLX H3 video VAE was loaded without encoder weights.")
        cfg = self.config
        num_frames = pixels.shape[2]
        if num_frames % cfg.clip_length != 0:
            pad_frames = (-num_frames) % cfg.clip_length
            tail = mx.repeat(pixels[:, :, -1:], pad_frames, axis=2)
            pixels = mx.concatenate([pixels, tail], axis=2)
        moment_chunks: list[Any] = []
        for index in range(pixels.shape[2] // cfg.clip_length):
            clip = pixels[:, :, index * cfg.clip_length:(index + 1) * cfg.clip_length]
            moment_chunks.append(self._encode_clip(clip))
        moments = mx.concatenate(moment_chunks, axis=2)
        if cfg.token_drop > 0:
            moments = moments[:, :, :-cfg.token_drop]
        mean, logvar = mx.split(moments, 2, axis=1)
        logvar = mx.clip(logvar, -30.0, 20.0)
        return mean, logvar

    def encode_keyframe(self, pixels):
        """Single-frame conditioning encode without chunk padding."""
        if pixels.shape[2] != 1:
            raise ValueError(f"encode_keyframe expects exactly one frame, got {pixels.shape}.")
        moments = self._encode_clip(pixels)
        mean, logvar = mx.split(moments, 2, axis=1)
        logvar = mx.clip(logvar, -30.0, 20.0)
        return mean, logvar

    @staticmethod
    def sample_posterior(mean, logvar, noise):
        """Reparameterization with an explicit noise array (deterministic parity)."""
        return mean + mx.exp(0.5 * logvar) * noise

    # -- decoder ---------------------------------------------------------

    def _decoder_blocks(self) -> list[dict[str, Any]]:
        """Split the flat weight dict into per-block dicts once."""
        if self._blocks is None:
            blocks: list[dict[str, Any]] = [{} for _ in range(self.config.decoder_num_layers)]
            prefix = "decoder.transformer_blocks."
            for key, value in self.weights.items():
                if key.startswith(prefix):
                    index, sub = key[len(prefix):].split(".", 1)
                    blocks[int(index)][sub] = value
            self._blocks = blocks
        return self._blocks

    def _decode_clip(self, z):
        """(1, C, T_tokens, H', W') latents -> (1, 3, T_tokens*4, H, W) pixels."""
        w = self.weights
        cfg = self.config
        z = _causal_conv3d(z, w["post_quant_conv.weight"], w["post_quant_conv.bias"])
        batch, channels, num_frames, height, width = z.shape
        # (B, C, T, H, W) -> (B*T*H*W, C): sequence-major tokens.
        tokens = _ct(z, 0, 2, 3, 4, 1).reshape(-1, channels)
        tokens = _linear(tokens, w["decoder.proj_in.weight"], w["decoder.proj_in.bias"])
        num_patches = tokens.shape[0]

        num_reg = self.config.decoder_num_register_tokens
        register = mx.broadcast_to(w["decoder.register_tokens"], (batch, num_reg, self.dim)).reshape(-1, self.dim)
        cls_token = mx.zeros((batch, self.dim), dtype=tokens.dtype)
        hidden = mx.concatenate([tokens, register, cls_token], axis=0)[:, None, :]  # (S, B, D)

        grids = [2.0 * (mx.arange(0.5, size, dtype=mx.float32) / size) - 1.0 for size in (num_frames, height, width)]
        mesh_t, mesh_h, mesh_w = mx.meshgrid(grids[0], grids[1], grids[2], indexing="ij")
        position_ids = mx.stack([mesh_t.reshape(-1), mesh_h.reshape(-1), mesh_w.reshape(-1)], axis=-1)
        suffix = mx.zeros((batch * (cfg.decoder_num_register_tokens + 1), 3), dtype=position_ids.dtype)
        position_ids = mx.concatenate([position_ids, suffix], axis=0)  # (S, 3)
        cos, sin = _rotary_cos_sin(position_ids, rotary_dim=self.rotary_dim, theta=cfg.decoder_rope_theta)

        for block in self._decoder_blocks():
            hidden = _transformer_block(
                block,
                hidden,
                block["scale1"],
                block["scale2"],
                cos,
                sin,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                norm_eps=cfg.decoder_norm_eps,
                qk_norm_eps=cfg.decoder_norm_eps,
            )
            mx.eval(hidden)  # per-block sync (36 blocks build a huge lazy graph)
        hidden = _layer_norm(hidden, w["decoder.norm_out.weight"], w["decoder.norm_out.bias"], eps=cfg.decoder_norm_eps)
        output = _linear(hidden, w["decoder.proj_out.weight"], w["decoder.proj_out.bias"])[:num_patches]

        patch_t = cfg.temporal_compression_ratio
        patch_h = patch_w = cfg.spatial_compression_ratio
        output = output.reshape(num_patches, cfg.out_channels, patch_t, patch_h, patch_w)
        output = output.reshape(batch, num_frames, height, width, cfg.out_channels, patch_t, patch_h, patch_w)
        output = _ct(output, 0, 4, 1, 5, 2, 6, 3, 7)
        return output.reshape(batch, cfg.out_channels, num_frames * patch_t, height * patch_h, width * patch_w)

    def _split_tiles(self, length: int, tile_size: int, min_overlap: int):
        if tile_size >= length:
            return [0], [length], []
        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
            num_tiles += 1
        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for index in range(remaining // self.spatial_compression_ratio):
            overlaps[index % (num_tiles - 1)] += self.spatial_compression_ratio
        starts = [0]
        for index in range(num_tiles - 1):
            starts.append(starts[-1] + tile_size - overlaps[index])
        return starts, [tile_size] * num_tiles, overlaps

    @staticmethod
    def _blend(a, b, blend_extent: int, dim: int):
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        positions = mx.arange(blend_extent, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = (1 - positions / blend_extent).reshape(shape)
        weight_b = (positions / blend_extent).reshape(shape)
        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b
        if blend_extent == b.shape[dim]:
            return blended
        rest = [slice(None)] * b.ndim
        rest[dim] = slice(blend_extent, None)
        return mx.concatenate([blended, b[tuple(rest)]], axis=dim)

    def _stitch_tiles(self, tiles, height_overlaps, width_overlaps):
        result_rows = []
        for row_index, row in enumerate(tiles):
            result_row = []
            for column_index, tile in enumerate(row):
                if row_index > 0:
                    tile = self._blend(tiles[row_index - 1][column_index], tile, height_overlaps[row_index - 1], -2)
                if column_index > 0:
                    tile = self._blend(row[column_index - 1], tile, width_overlaps[column_index - 1], -1)
                if row_index < len(tiles) - 1:
                    tile = tile[..., :-height_overlaps[row_index], :]
                if column_index < len(row) - 1:
                    tile = tile[..., :, :-width_overlaps[column_index]]
                result_row.append(tile)
            result_rows.append(mx.concatenate(result_row, axis=-1))
        return mx.concatenate(result_rows, axis=-2)

    def decode_clip_tiled(self,
                          z,
                          tile_sample_min_height: int,
                          tile_sample_min_width: int,
                          tile_sample_min_overlap_height: int = 64,
                          tile_sample_min_overlap_width: int = 64):
        """One clip through the decoder with spatial tiling (memory bounded)."""
        height = z.shape[-2] * self.spatial_compression_ratio
        width = z.shape[-1] * self.spatial_compression_ratio
        y_starts, y_lengths, y_overlaps = self._split_tiles(height, tile_sample_min_height,
                                                            tile_sample_min_overlap_height)
        x_starts, x_lengths, x_overlaps = self._split_tiles(width, tile_sample_min_width, tile_sample_min_overlap_width)
        ratio = self.spatial_compression_ratio
        rows = []
        for y_start, y_length in zip(y_starts, y_lengths, strict=False):
            row = []
            for x_start, x_length in zip(x_starts, x_lengths, strict=False):
                tile = z[..., y_start // ratio:y_start // ratio + y_length // ratio,
                         x_start // ratio:x_start // ratio + x_length // ratio]
                row.append(self._decode_clip(tile))
            rows.append(row)
        return self._stitch_tiles(rows, y_overlaps, x_overlaps)

    def decode(self,
               z,
               *,
               tiled: bool = True,
               tile_sample_min_height: int = 256,
               tile_sample_min_width: int = 256,
               tile_sample_min_overlap_height: int = 64,
               tile_sample_min_overlap_width: int = 64):
        """Chunked decode of normalized latents (1, C, T_lat, H', W') -> (1, 3, T, H, W)."""
        cfg = self.config
        tokens_chunk_size = cfg.tokens_chunk_size
        token_drop = cfg.token_drop
        temporal_ratio = self.temporal_compression_ratio
        chunk_num_frames = tokens_chunk_size * temporal_ratio
        num_tokens = z.shape[2] + token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
        if pad_tokens > 0:
            tail = mx.repeat(z[:, :, -1:], pad_tokens, axis=2)
            z = mx.concatenate([z, tail], axis=2)

        def decode_one(chunk):
            if tiled:
                return self.decode_clip_tiled(chunk, tile_sample_min_height, tile_sample_min_width,
                                              tile_sample_min_overlap_height, tile_sample_min_overlap_width)
            return self._decode_clip(chunk)

        decoded_chunks = []
        overlap = None
        for index in range(num_chunks):
            start = index * tokens_chunk_size
            clip = decode_one(z[:, :, start:start + tokens_chunk_size + cfg.token_overlap])
            for overlap_index in range(int(token_drop > 0) + 1):
                frame_start = overlap_index * chunk_num_frames
                chunk = clip[:, :, frame_start:frame_start + chunk_num_frames]
                chunk = chunk[:, :, cfg.frame_pre_padding:]
                if overlap_index == 0:
                    if overlap is not None:
                        chunk = self._blend(overlap, chunk, cfg.frame_overlap, dim=-3)
                    mx.eval(chunk)  # materialize per clip; keeps the lazy graph bounded
                    decoded_chunks.append(chunk)
                else:
                    overlap = chunk
                    mx.eval(overlap)
        if overlap is not None:
            decoded_chunks.append(overlap)
        decoded = mx.concatenate(decoded_chunks, axis=2)

        if pad_tokens > 0:
            intra_tail = cfg.clip_length % temporal_ratio
            num_tokens_before_pad = z.shape[2] - pad_tokens
            pad_frames = sum(intra_tail if intra_tail and (num_tokens_before_pad + offset) %
                             tokens_chunk_size == 0 else temporal_ratio for offset in range(pad_tokens))
            decoded = decoded[:, :, :-pad_frames]
        return decoded


def _layer_norm(x, weight, bias, eps: float):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean)**2).mean(axis=-1, keepdims=True)
    return (x - mean) / mx.sqrt(var + eps) * weight + bias


# ---------------------------------------------------------------------------
# Streaming safetensors loading
# ---------------------------------------------------------------------------


def _shards(component_dir: Path) -> list[Path]:
    component_dir = Path(component_dir)
    index = component_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return sorted({component_dir / shard for shard in set(weight_map.values())})
    single = component_dir / "diffusion_pytorch_model.safetensors"
    if single.exists():
        return [single]
    shards = sorted(component_dir.glob("*.safetensors"))
    if shards:
        return shards
    raise FileNotFoundError(f"No safetensors found under {component_dir}")


_DTYPE_MAP = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}


def mlx_h3_video_vae_from_dir(vae_dir: str | Path,
                              *,
                              include_encoder: bool = True,
                              storage_dtype: str = "fp32",
                              config: MiniMaxH3VideoVAEConfigView | None = None) -> MLXMiniMaxH3VideoVAE:
    """Load the released H3 video VAE, streaming one shard at a time.

    ``storage_dtype="fp32"`` keeps the released numerics. bf16/fp16 halve
    residency; measure drift against the FP32 acceptance gate before shipping
    a reduced-dtype configuration.
    """
    import mlx.core as mx

    vae_dir = Path(vae_dir)
    config = config or MiniMaxH3VideoVAEConfigView.from_vae_dir(vae_dir)
    cast_dtype = _DTYPE_MAP[storage_dtype]

    wanted_prefixes: tuple[str, ...] = ("post_quant_conv.", "decoder.")
    if include_encoder:
        wanted_prefixes = ("quant_conv.", "encoder.", "post_quant_conv.", "decoder.")

    weights: dict[str, Any] = {}
    for shard in _shards(vae_dir):
        arrays = mx.load(str(shard))
        for key, source in arrays.items():
            if not key.startswith(wanted_prefixes):
                continue
            array = source.astype(cast_dtype)
            if array.ndim == 5:  # conv3d (O, I, kT, kH, kW) -> (O, kT, kH, kW, I)
                array = _ct(array, 0, 2, 3, 4, 1)
            mx.eval(array)
            del source
            weights[key] = array
        del arrays
        gc.collect()
        mx.clear_cache()

    required = [
        "post_quant_conv.weight", "post_quant_conv.bias", "decoder.proj_in.weight", "decoder.norm_out.weight",
        "decoder.proj_out.weight", "decoder.register_tokens"
    ]
    missing = [key for key in required if key not in weights]
    if missing:
        raise KeyError(f"H3 video VAE at {vae_dir} is missing required tensors: {missing}")
    if include_encoder:
        encoder_required = [
            "quant_conv.weight", "encoder.conv_in.weight", "encoder.conv_out.weight", "encoder.norm_out.weight"
        ]
        missing = [key for key in encoder_required if key not in weights]
        if missing:
            raise KeyError(f"H3 video VAE encoder at {vae_dir} is missing required tensors: {missing}")
    vae = MLXMiniMaxH3VideoVAE(weights, config, has_encoder=include_encoder)
    expected_blocks = config.decoder_num_layers
    found_blocks = {int(key.split(".")[2]) for key in weights if key.startswith("decoder.transformer_blocks.")}
    if found_blocks != set(range(expected_blocks)):
        raise KeyError(f"H3 video VAE decoder blocks incomplete: found {sorted(found_blocks)} "
                       f"of {expected_blocks}.")
    return vae
