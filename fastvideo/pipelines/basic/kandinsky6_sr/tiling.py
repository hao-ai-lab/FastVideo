# SPDX-License-Identifier: Apache-2.0
"""Tile planning and stitching for Kandinsky6 video super-resolution.

The SR DiT was trained on fixed base resolutions, so a source frame is cut into overlapping tiles whose upscaled size
is one of those bases, each tile is super-resolved independently and the decoded tiles are blended back with a Hanning
window.  Tile positions are aligned to the VAE spatial factor so the same grid addresses both the latent and the
decoded pixels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional

#: Trained base resolutions ``visual_size -> [(H, W), ...]`` of the SR DiT.
RESOLUTIONS: dict[int, tuple[tuple[int, int], ...]] = {
    512: ((512, 512), (512, 768), (768, 512)),
}


@dataclass(frozen=True)
class TileGrid:
    """Tile size and per-axis tile start positions, in pixels of the (pre-upscaled) source frame."""

    tile_h: int
    tile_w: int
    tops: tuple[int, ...]
    lefts: tuple[int, ...]

    @property
    def num_tiles(self) -> int:
        return len(self.tops) * len(self.lefts)

    def to_latent(self, spatial_factor: int) -> TileGrid:
        values = (self.tile_h, self.tile_w, *self.tops, *self.lefts)
        if any(value % spatial_factor for value in values):
            raise ValueError(f"Tile grid {self} is not aligned to the VAE spatial factor {spatial_factor}.")
        return TileGrid(self.tile_h // spatial_factor, self.tile_w // spatial_factor,
                        tuple(top // spatial_factor for top in self.tops),
                        tuple(left // spatial_factor for left in self.lefts))

    def extract(self, video: torch.Tensor) -> list[torch.Tensor]:
        """``[..., H, W]`` -> row-major list of ``[..., tile_h, tile_w]`` views."""
        return [video[..., top:top + self.tile_h, left:left + self.tile_w] for top in self.tops for left in self.lefts]


def _axis_positions(length: int, tile: int, min_overlap: float, snap: int) -> tuple[int, ...]:
    """Fewest evenly spaced starts covering ``[0, length - tile]`` with at least ``min_overlap`` overlap.

    Positions are multiples of ``snap`` when both ``length`` and ``tile`` are; otherwise the same layout is computed at
    pixel precision and :meth:`TileGrid.to_latent` rejects the unaligned grid.
    """
    if tile >= length:
        return (0, )
    unit = snap if length % snap == 0 and tile % snap == 0 else 1
    span_units = (length - tile) // unit
    max_stride_units = max(1, math.floor(tile * (1.0 - min_overlap) / unit))
    count = math.ceil(span_units / max_stride_units) + 1
    return tuple(round(i * span_units / (count - 1)) * unit for i in range(count))


def plan_tiles(height: int, width: int, visual_size: int, scale: int, min_overlap: float,
               spatial_factor: int) -> TileGrid:
    """Tile grid of a ``height x width`` source frame for an integer tiling ``scale``.

    The base resolution is the trained one closest in aspect ratio; each tile is ``base / scale`` source pixels, so its
    upscaled size is exactly the base resolution the DiT was trained on.
    """
    if visual_size not in RESOLUTIONS:
        raise ValueError(f"Unsupported SR visual_size={visual_size}; known sizes: {sorted(RESOLUTIONS)}")
    if not 0.0 <= min_overlap < 1.0:
        raise ValueError(f"sr_tile_min_overlap must be in [0, 1), got {min_overlap}")
    ratio = width / height
    base_h, base_w = min(RESOLUTIONS[visual_size], key=lambda hw: abs(hw[1] / hw[0] - ratio))
    if base_h % scale or base_w % scale:
        raise ValueError(f"Tiling scale {scale} does not divide the base resolution {base_h}x{base_w}.")
    tile_h, tile_w = base_h // scale, base_w // scale
    return TileGrid(tile_h, tile_w, _axis_positions(height, tile_h, min_overlap, spatial_factor),
                    _axis_positions(width, tile_w, min_overlap, spatial_factor))


def stitch_tiles(tiles: list[torch.Tensor],
                 grid: TileGrid,
                 height: int,
                 width: int,
                 scale: int,
                 frame_chunk: int = 16) -> torch.Tensor:
    """Blend row-major float ``[0, 255]`` tiles ``[C, T, tile_h * scale, tile_w * scale]`` into a single uint8
    ``[C, T, H * scale, W * scale]`` video, quantizing once at the very end (not per tile): pre-quantizing each
    tile to uint8 before blending would blend already-rounded values instead of the continuous pixel values,
    compounding rounding error at every overlap -- matches the diffusers reference's ``decode_latents``/
    ``__call__``, which accumulates ``video_acc += tile * window`` in float and rounds only the final result.

    Each tile is weighted by a 2D Hanning window without zero endpoints (so border pixels covered by a single tile keep
    a non-zero weight) and the accumulation is normalised per pixel, which also handles uneven overlaps.  Frames are
    blended in chunks to bound host memory; frames are independent, so chunking does not change the result.
    """
    channels, frames, hr_h, hr_w = tiles[0].shape
    window_h = torch.hann_window(hr_h + 2)[1:-1]
    window_w = torch.hann_window(hr_w + 2)[1:-1]
    window = (window_h[:, None] * window_w[None, :])[None, None]
    out = torch.empty((channels, frames, height * scale, width * scale), dtype=torch.uint8)
    for t0 in range(0, frames, max(1, frame_chunk)):
        t1 = min(frames, t0 + frame_chunk)
        acc = torch.zeros((channels, t1 - t0, height * scale, width * scale))
        weight = torch.zeros((1, 1, height * scale, width * scale))
        tile_iter = iter(tiles)
        for top in grid.tops:
            for left in grid.lefts:
                y, x = top * scale, left * scale
                acc[:, :, y:y + hr_h, x:x + hr_w] += next(tile_iter)[:, t0:t1] * window
                weight[:, :, y:y + hr_h, x:x + hr_w] += window
        out[:, t0:t1] = (acc / weight.clamp(min=1e-6)).clamp(0, 255).to(torch.uint8)
    return out


def resolve_scale(scale: float) -> tuple[int, float]:
    """Requested total upscale -> ``(tiling_scale, pixel_pre_upscale)``.

    The latent upscaler only has x2 and x4 entries; the fractional x2.25 is an x1.125 pixel pre-upscale followed by x2
    tiling.
    """
    if scale == 2.25:
        return 2, 1.125
    if scale in (2.0, 4.0):
        return int(scale), 1.0
    raise ValueError(f"sr_resolution_scale must be 2, 4 or 2.25, got {scale}")


def pre_upscale(video: torch.Tensor, factor: float, spatial_multiple: int, frame_chunk: int = 16) -> torch.Tensor:
    """Bilinear pixel upscale of a ``[T, C, H, W]`` uint8 video, target size rounded to ``spatial_multiple``."""
    height, width = video.shape[-2:]
    size = (max(spatial_multiple,
                round(height * factor / spatial_multiple) * spatial_multiple),
            max(spatial_multiple,
                round(width * factor / spatial_multiple) * spatial_multiple))
    chunks = []
    for t0 in range(0, video.shape[0], frame_chunk):
        resized = functional.interpolate(video[t0:t0 + frame_chunk].float(),
                                         size=size,
                                         mode="bilinear",
                                         align_corners=False)
        chunks.append(resized.round_().clamp_(0, 255).to(torch.uint8))
    return torch.cat(chunks, dim=0)
