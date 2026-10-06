# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 SR causal video VAE (KVAE): spatial x16, temporal x4 (causal, ``1 + 4k`` frames).

Clips are encoded / decoded in temporal segments (16 pixel frames, the first one 17; 4 latent frames, the first one 5).
Every causal conv carries its last input frames to the next segment, so memory is bounded by one segment whatever the
clip length, and the segmentation is part of the numerics: it fixes which frames the first-frame special cases see.

Conventions: ``normalize_data`` maps ``[0, 255]`` pixels to ``x / 128 - 1``; ``encode`` returns ``(latent,
split_list)`` where ``latent`` is the raw (unscaled) posterior mean and ``split_list`` the pixel-frame segment sizes;
``decode`` returns an object whose ``.sample`` is in the normalized pixel range. State-dict keys are the checkpoint's
own ``encoder.*`` / ``decoder.*`` keys, so the loader loads strictly without renaming.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.configs.models.vaes.kandinsky6_sr import Kandinsky6SRVAEArchConfig, Kandinsky6SRVAEConfig
from fastvideo.models.vaes.wanvae import WanRMS_norm

# Pixel frames per encode segment after the first (which has one more: the causal first frame).
_SEGMENT_FRAMES = 16
# Inputs above this many elements are convolved in temporal chunks: at 4x SR decode tiles (~1536x832) the
# full-resolution 256-channel convs of the last decoder level see ~5.6e9 elements per segment, beyond 32-bit
# indexing, and chunking also keeps the transient conv workspace to a fraction of the activation.
_MAX_CONV_NUMEL = 2 * 10**9


@dataclass
class DecoderOutput:
    sample: torch.Tensor


class _SegmentCache:
    """Causal state carried from one temporal segment to the next during a single encode / decode call."""

    def __init__(self) -> None:
        self.first = True
        self.padding: dict[nn.Module, torch.Tensor] = {}


def _silu(x: torch.Tensor) -> torch.Tensor:
    # The output norms use this form and the resblocks use F.silu; the two are not bitwise equal in bf16.
    return x * torch.sigmoid(x)


class _K6RMSNorm(WanRMS_norm):
    """``WanRMS_norm`` with an explicit float32 upcast around the normalize, matching the diffusers reference's
    ``Kandinsky6SRRMSNorm`` (which upcasts unconditionally, not only for fp16/bf16/fp8 inputs): at bf16/fp16
    activations, normalizing without upcasting measurably drifts from the reference over many resnet blocks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = F.normalize(x.float(), dim=1).to(x.dtype)
        return normed * self.scale * self.gamma + self.bias


class _ChunkedConv3d(nn.Conv3d):
    """``nn.Conv3d`` that convolves inputs larger than ``_MAX_CONV_NUMEL`` in overlapping temporal chunks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_parts = math.ceil(x.numel() / _MAX_CONV_NUMEL)
        if num_parts <= 1:
            return super().forward(x)
        k, frames = self.kernel_size[0], x.size(2)
        if self.padding[0] != 0:
            raise ValueError(f"temporal chunking needs an unpadded time axis, got kernel {self.kernel_size}, "
                             f"padding {self.padding}")
        if self.stride[0] != 1:
            # A strided conv can't reuse the stride-1 path's "prefix each chunk with the previous
            # chunk's tail" trick below (a strided conv's output doesn't align with input chunk
            # boundaries), so slide one kernel-width window at a time instead -- matches the diffusers
            # reference's Kandinsky6SRSafeConv3d, only raising if even a single window would overflow.
            stride = self.stride[0]
            window_numel = x.shape[0] * x.shape[1] * k * x.shape[3] * x.shape[4]
            if window_numel >= _MAX_CONV_NUMEL:
                raise ValueError(f"frames are too big for Conv3d even one stride-{stride} window at a time "
                                 f"({window_numel} elements per window)")
            # Bare super() doesn't resolve inside a list comprehension's own implicit scope, hence the loop.
            window_outputs = []
            for i in range(0, frames - k + 1, stride):
                window_outputs.append(super().forward(x[:, :, i:i + k]))
            return torch.cat(window_outputs, dim=2)
        step = math.ceil(frames / num_parts)
        last = frames - step * (math.ceil(frames / step) - 1)
        if k > 1 and min(step, last) < k:
            # A chunk would be shorter than the kernel: convolve one output frame at a time instead.
            windows = [(i, i + k) for i in range(frames - k + 1)]
        else:
            # Each input chunk [start, end) is prefixed with the previous chunk's last k - 1 frames.
            windows = [(max(start - k + 1, 0), min(start + step, frames)) for start in range(0, frames, step)]
        out: torch.Tensor | None = None
        for lo, hi in windows:
            y = super().forward(x[:, :, lo:hi])
            if out is None:
                out = y.new_empty((*y.shape[:2], frames - k + 1, *y.shape[3:]))
            out[:, :, lo:lo + y.size(2)] = y
        assert out is not None
        return out


class CausalConv3d(nn.Module):
    """Conv3d that is causal in time: the first segment is left-padded with copies of its first frame, later
    segments with the tail of the previous one; height / width are zero padded."""

    def __init__(self, chan_in: int, chan_out: int, kernel_size: int | tuple[int, int, int],
                 stride: tuple[int, int, int] = (1, 1, 1)) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, ) * 3
        time_kernel, height_kernel, width_kernel = kernel_size
        self.height_pad = height_kernel // 2
        self.width_pad = width_kernel // 2
        self.time_pad = time_kernel - 1
        self.time_kernel = time_kernel
        self.time_stride = stride[0]
        self.conv = _ChunkedConv3d(chan_in, chan_out, kernel_size, stride=stride)

    def forward(self, x: torch.Tensor, cache: _SegmentCache) -> torch.Tensor:
        s, k = self.time_stride, self.time_kernel
        batch, _, frames, height, width = x.shape
        x = F.pad(x, (self.width_pad, self.width_pad, self.height_pad, self.height_pad, 0, 0))
        if cache.first:
            first_frame = x[:, :, :1]
            padding = first_frame.expand(-1, -1, self.time_pad, -1, -1)
        else:
            padding = cache.padding[self]

        # conv(cat([padding, x])) without materializing the concatenation of the whole segment: only the outputs
        # whose window overlaps ``padding`` are computed on a small concatenated head.
        out_frames = frames if s == 1 else (frames + 1) // 2
        output = x.new_empty((batch, self.conv.out_channels, out_frames, height, width))
        head_out = math.ceil(padding.size(2) / s)
        head_in = head_out * s - padding.size(2)
        if head_out > 0:
            output[:, :, :head_out] = self.conv(torch.cat([padding, x[:, :, :head_in + k - s]], dim=2))
        if head_out < out_frames:
            output[:, :, head_out:] = self.conv(x[:, :, head_in:])

        # Keep exactly the input frames the next segment's first output window needs.
        pad_offset = head_in + s * math.trunc((frames - head_in - k) / s) + s
        if pad_offset < 0:  # segment shorter than the kernel
            cache.padding[self] = torch.cat([padding[:, :, pad_offset:], x], dim=2)
        else:
            cache.padding[self] = x[:, :, pad_offset:].clone()
        return output


class ResnetBlock3D(nn.Module):
    """norm -> SiLU -> causal conv, twice, plus a 1x1 shortcut when the width changes. Decoder blocks use the
    latent-conditioned ``SpatialNorm3D`` (``zq_channels`` set)."""

    def __init__(self, in_channels: int, out_channels: int, zq_channels: int | None = None) -> None:
        super().__init__()
        if zq_channels is None:
            self.norm1: nn.Module = _K6RMSNorm(in_channels, images=False)
            self.norm2: nn.Module = _K6RMSNorm(out_channels, images=False)
        else:
            self.norm1 = SpatialNorm3D(in_channels, zq_channels)
            self.norm2 = SpatialNorm3D(out_channels, zq_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)
        if in_channels != out_channels:
            self.nin_shortcut = _ChunkedConv3d(in_channels, out_channels, kernel_size=1)

    def _norm(self, norm: nn.Module, h: torch.Tensor, zq: torch.Tensor | None, cache: _SegmentCache) -> torch.Tensor:
        return norm(h) if zq is None else norm(h, zq, cache)

    def forward(self, x: torch.Tensor, cache: _SegmentCache, zq: torch.Tensor | None = None) -> torch.Tensor:
        h = F.silu(self._norm(self.norm1, x, zq, cache), inplace=True)
        h = self.conv1(h, cache)
        h = F.silu(self._norm(self.norm2, h, zq, cache), inplace=True)
        h = self.conv2(h, cache)
        if hasattr(self, "nin_shortcut"):
            x = self.nin_shortcut(x)
        return x + h


def _chunked_interpolate_nearest(x: torch.Tensor, size: tuple[int, int, int], channels: int = 32) -> torch.Tensor:
    """``F.interpolate(x, size, mode="nearest")`` in ``channels``-wide chunks along dim 1, to bound peak memory at
    large SR-decode tile sizes (matches the diffusers reference's chunked ``zq`` interpolation)."""
    if x.shape[1] <= channels:
        return F.interpolate(x, size=size, mode="nearest")
    return torch.cat([F.interpolate(chunk, size=size, mode="nearest") for chunk in torch.split(x, channels, dim=1)],
                     dim=1)


class SpatialNorm3D(nn.Module):
    """RMS norm modulated by the latent: ``norm(f) * conv_y(zq) + conv_b(zq)`` with ``zq`` nearest-resized to ``f``."""

    def __init__(self, f_channels: int, zq_channels: int) -> None:
        super().__init__()
        self.norm_layer = _K6RMSNorm(f_channels, images=False)
        self.conv_y = _ChunkedConv3d(zq_channels, f_channels, kernel_size=1)
        self.conv_b = _ChunkedConv3d(zq_channels, f_channels, kernel_size=1)

    def forward(self, f: torch.Tensor, zq: torch.Tensor, cache: _SegmentCache) -> torch.Tensor:
        frames, height, width = f.shape[-3:]
        if cache.first:
            # The causal first latent frame maps to exactly one pixel frame; the rest cover the remaining frames.
            zq_first = _chunked_interpolate_nearest(zq[:, :, :1], (1, height, width))
            if zq.size(2) > 1:
                zq_rest = _chunked_interpolate_nearest(zq[:, :, 1:], (frames - 1, height, width))
                zq = torch.cat([zq_first, zq_rest], dim=2)
            else:
                zq = zq_first
        else:
            zq = _chunked_interpolate_nearest(zq, (frames, height, width))
        norm_f = self.norm_layer(f)
        norm_f.mul_(self.conv_y(zq))
        norm_f.add_(self.conv_b(zq))
        return norm_f


class PXSDownsample(nn.Module):
    """x2 spatial (strided conv + channel-averaged pixel-unshuffle) and optional x2 causal temporal (two causal convs
    + average pooling) downsample; doubles the channels."""

    def __init__(self, in_channels: int, compress_time: bool) -> None:
        super().__init__()
        out_channels = 2 * in_channels
        self.spatial_conv = _ChunkedConv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                           padding=(0, 1, 1))
        self.compress_time = compress_time
        if compress_time:
            self.temporal_conv = nn.Sequential(
                CausalConv3d(out_channels, out_channels, kernel_size=(2, 1, 1)),
                CausalConv3d(out_channels, out_channels, kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            )
        self.linear = _ChunkedConv3d(out_channels, out_channels, kernel_size=1)

    def _spatial(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        pxs = F.pixel_unshuffle(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), 2)
        pxs = pxs.view(b * t, 2 * c, 2, h // 2, w // 2).mean(dim=2)
        pxs = pxs.reshape(b, t, 2 * c, h // 2, w // 2).permute(0, 2, 1, 3, 4)
        return self.spatial_conv(x) + pxs

    def _temporal(self, x: torch.Tensor, cache: _SegmentCache) -> torch.Tensor:
        b, c, t, h, w = x.shape
        pooled = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, t)
        if cache.first:  # the causal first frame is kept as is
            first, rest = pooled[..., :1], pooled[..., 1:]
            pooled = torch.cat([first, F.avg_pool1d(rest, kernel_size=2, stride=2)], dim=-1) if t > 1 else first
        else:
            pooled = F.avg_pool1d(pooled, kernel_size=2, stride=2)
        pooled = pooled.reshape(b, h, w, c, -1).permute(0, 3, 4, 1, 2)
        conv = self.temporal_conv[1](self.temporal_conv[0](x, cache), cache)
        return conv + pooled

    def forward(self, x: torch.Tensor, cache: _SegmentCache) -> torch.Tensor:
        out = self._spatial(x)
        if self.compress_time:
            out = self._temporal(out, cache)
        return self.linear(out)


class PXSUpsample(nn.Module):
    """Optional x2 causal temporal (frame repeat + causal conv residual) then x2 spatial (nearest + conv residual)
    upsample."""

    def __init__(self, in_channels: int, compress_time: bool) -> None:
        super().__init__()
        self.spatial_conv = _ChunkedConv3d(in_channels, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.compress_time = compress_time
        if compress_time:
            self.temporal_conv = CausalConv3d(in_channels, in_channels, kernel_size=(3, 1, 1))
        self.linear = _ChunkedConv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cache: _SegmentCache) -> torch.Tensor:
        if self.compress_time:
            x = x.repeat_interleave(2, dim=2)
            if cache.first:  # the causal first frame is not duplicated
                x = x[:, :, 1:]
            x = self.temporal_conv(x, cache) + x
        frames, height, width = x.shape[-3:]
        # Plain F.interpolate silently corrupts most elements once the tensor exceeds ~2**31 elements (it appears
        # to do 32-bit index arithmetic internally): chunk along channels like SpatialNorm3D's zq resize does.
        x = _chunked_interpolate_nearest(x, (frames, height * 2, width * 2))
        x.add_(self.spatial_conv(x))
        return self.linear(x)


def _level_module(**children: nn.Module) -> nn.Module:
    module = nn.Module()
    for name, child in children.items():
        setattr(module, name, child)
    return module


class Encoder3D(nn.Module):

    def __init__(self, *, ch: int, ch_mult: list[float], num_res_blocks: int, in_channels: int, z_channels: int,
                 temporal_compress_times: int, temporal_compress_start_level: int) -> None:
        super().__init__()
        self.num_res_blocks = num_res_blocks
        time_levels = range(temporal_compress_start_level,
                            temporal_compress_start_level + int(math.log2(temporal_compress_times)))
        self.conv_in = CausalConv3d(in_channels, round(ch * ch_mult[0]), kernel_size=3)
        self.down = nn.ModuleList()
        block_in = round(ch * ch_mult[0])
        for level, mult in enumerate(ch_mult):
            block_out = round(ch * mult)
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock3D(block_in, block_out))
                block_in = block_out
            down = _level_module(block=blocks)
            if level != len(ch_mult) - 1:
                down.downsample = PXSDownsample(block_in, compress_time=level in time_levels)
                block_in *= 2
            self.down.append(down)
        self.mid = _level_module(block_1=ResnetBlock3D(block_in, block_in), block_2=ResnetBlock3D(block_in, block_in))
        self.norm_out = _K6RMSNorm(block_in, images=False)
        self.conv_out = CausalConv3d(block_in, 2 * z_channels, kernel_size=3)

    def forward(self, x: torch.Tensor, cache: _SegmentCache) -> torch.Tensor:
        h = self.conv_in(x, cache)
        for down in self.down:
            for block in down.block:
                h = block(h, cache)
            if hasattr(down, "downsample"):
                h = down.downsample(h, cache)
        h = self.mid.block_2(self.mid.block_1(h, cache), cache)
        return self.conv_out(_silu(self.norm_out(h)), cache)


class Decoder3D(nn.Module):

    def __init__(self, *, ch: int, ch_mult: list[float], num_res_blocks: int, out_ch: int, z_channels: int,
                 temporal_compress_times: int, temporal_compress_start_level: int) -> None:
        super().__init__()
        num_levels = len(ch_mult)
        # Mirror of the encoder's temporally compressing levels.
        time_levels = range(num_levels - temporal_compress_start_level - int(math.log2(temporal_compress_times)),
                            num_levels - temporal_compress_start_level)
        block_in = round(ch * ch_mult[-1])
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3)
        self.mid = _level_module(block_1=ResnetBlock3D(block_in, block_in, z_channels),
                                 block_2=ResnetBlock3D(block_in, block_in, z_channels))
        up_levels: list[nn.Module] = []
        for level in reversed(range(num_levels)):
            block_out = round(ch * ch_mult[level])
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock3D(block_in, block_out, z_channels))
                block_in = block_out
            up = _level_module(block=blocks)
            if level != 0:
                up.upsample = PXSUpsample(block_in, compress_time=level in time_levels)
            up_levels.insert(0, up)
        self.up = nn.ModuleList(up_levels)
        self.norm_out = SpatialNorm3D(block_in, z_channels)
        self.conv_out = CausalConv3d(block_in, out_ch, kernel_size=3)

    def forward(self, z: torch.Tensor, cache: _SegmentCache) -> torch.Tensor:
        h = self.conv_in(z, cache)
        h = self.mid.block_2(self.mid.block_1(h, cache, z), cache, z)
        for up in reversed(self.up):
            for block in up.block:
                h = block(h, cache, z)
            if hasattr(up, "upsample"):
                h = up.upsample(h, cache)
        return self.conv_out(_silu(self.norm_out(h, z, cache)), cache)


# Code-path-selecting knobs of the released checkpoint; anything else would need layers this module does not have.
_REQUIRED_ENCODER_KNOBS = {"norm_type": "rms_norm", "padding_mode": "zeros", "downsample_version": 2,
                           "fix_pxs": True, "double_z": True}
_REQUIRED_DECODER_KNOBS = {"norm_type": "rms_norm", "padding_mode": "zeros"}
_ARCH_KEYS = ("ch", "ch_mult", "num_res_blocks", "z_channels", "temporal_compress_times",
              "temporal_compress_start_level")


def _arch_kwargs(conf: dict[str, Any], required: dict[str, Any], extra: str, which: str) -> dict[str, Any]:
    for key, value in required.items():
        if conf.get(key) != value:
            raise ValueError(f"Kandinsky6SRVAE supports only {which}_config[{key!r}] == {value!r}, "
                             f"got {conf.get(key)!r}")
    missing = [key for key in (*_ARCH_KEYS, extra) if key not in conf]
    if missing:
        raise ValueError(f"Kandinsky6SRVAE {which}_config is missing {missing}")
    return {key: conf[key] for key in (*_ARCH_KEYS, extra)}


class Kandinsky6SRVAE(nn.Module):

    def __init__(self, config: Kandinsky6SRVAEConfig) -> None:
        super().__init__()
        self.config = config
        arch = config.arch_config
        assert isinstance(arch, Kandinsky6SRVAEArchConfig)
        if not arch.encoder_config or not arch.decoder_config:
            raise ValueError("Kandinsky6SRVAE needs `encoder_config` and `decoder_config` (KVAE architecture) in the "
                             "component config.json.")
        enc = _arch_kwargs(dict(arch.encoder_config), _REQUIRED_ENCODER_KNOBS, "in_channels", "encoder")
        dec = _arch_kwargs(dict(arch.decoder_config), _REQUIRED_DECODER_KNOBS, "out_ch", "decoder")
        self.temporal_compression = enc["temporal_compress_times"]
        self.encoder = Encoder3D(**enc)
        self.decoder = Decoder3D(**dec)

    @property
    def scaling_factor(self) -> float:
        return float(self.config.arch_config.scaling_factor)

    @property
    def spatial_factor(self) -> int:
        return int(self.config.arch_config.spatial_factor)

    @property
    def temporal_factor(self) -> int:
        return int(self.config.arch_config.temporal_factor)

    @staticmethod
    def normalize_data(data: torch.Tensor) -> torch.Tensor:
        return data / 128 - 1.0

    @staticmethod
    def denormalize_data(data: torch.Tensor) -> torch.Tensor:
        return (data + 1) * 128

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        """``[B, C, T, H, W]`` normalized pixels -> ``(raw latent mean, pixel-frame segment sizes)``."""
        split_list = [_SEGMENT_FRAMES + 1]
        remaining = x.size(2) - split_list[0]
        while remaining > 0:
            split_list.append(_SEGMENT_FRAMES)
            remaining -= _SEGMENT_FRAMES
        split_list[-1] += remaining
        cache = _SegmentCache()
        latents = []
        for segment in torch.split(x, split_list, dim=2):
            moments = self.encoder(segment, cache)
            cache.first = False
            latents.append(moments.chunk(2, dim=1)[0])  # deterministic posterior: the mean
        return torch.cat(latents, dim=2), split_list

    def decode(self, z: torch.Tensor) -> DecoderOutput:
        """``[B, C, T', h, w]`` raw latent -> ``.sample`` ``[B, 3, T, H, W]`` in the normalized pixel range."""
        segment = _SEGMENT_FRAMES // self.temporal_compression
        num_frames = z.size(2)
        if num_frames == 1:
            split_list = [1]
        else:
            split_list = [segment] * ((num_frames - 1) // segment)
            if (num_frames - 1) % segment:
                split_list.append((num_frames - 1) % segment)
            split_list[0] += 1
        cache = _SegmentCache()
        samples = []
        for chunk in torch.split(z, split_list, dim=2):
            samples.append(self.decoder(chunk, cache))
            cache.first = False
        return DecoderOutput(sample=torch.cat(samples, dim=2))


EntryClass = Kandinsky6SRVAE
