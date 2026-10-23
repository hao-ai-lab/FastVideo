# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 SR tile planning, stitching and scale handling (CPU)."""
from __future__ import annotations

import pytest
import torch

from fastvideo.pipelines.basic.kandinsky6_sr.tiling import (TileGrid, plan_tiles, pre_upscale, resolve_scale,
                                                           stitch_tiles)


@pytest.mark.parametrize("size", [(512, 768), (768, 512), (512, 512), (576, 864), (720, 1280), (1088, 1920),
                                  (208, 384), (240, 432)])
@pytest.mark.parametrize("scale", [2, 4])
def test_tile_grids_cover_the_frame_overlap_and_stay_latent_aligned(size, scale):
    height, width = size
    grid = plan_tiles(height, width, 512, scale, 0.2, 16)
    assert grid.tops[0] == 0 and grid.lefts[0] == 0
    assert grid.tops[-1] + grid.tile_h == max(height, grid.tile_h)
    assert grid.lefts[-1] + grid.tile_w == max(width, grid.tile_w)
    assert (grid.tile_h * scale, grid.tile_w * scale) in ((512, 512), (512, 768), (768, 512))
    latent = grid.to_latent(16)
    assert latent.tile_h * 16 == grid.tile_h and latent.num_tiles == grid.num_tiles
    for a, b in zip(grid.tops, grid.tops[1:]):
        assert grid.tile_h - (b - a) >= 0.2 * grid.tile_h - 16


def test_unaligned_grid_is_rejected_for_the_latent_tiles():
    with pytest.raises(ValueError, match="not aligned"):
        TileGrid(256, 384, (0, 100), (0, 128)).to_latent(16)


def test_extract_returns_row_major_tiles():
    video = torch.arange(4 * 6).reshape(1, 4, 6)
    tiles = TileGrid(2, 3, (0, 2), (0, 3)).extract(video)
    assert [tile[0, 0, 0].item() for tile in tiles] == [0, 3, 12, 15]


def test_scale_that_does_not_divide_the_base_resolution_is_rejected():
    with pytest.raises(ValueError, match="does not divide"):
        plan_tiles(512, 768, 512, 3, 0.2, 16)


def test_stitching_constant_tiles_reproduces_the_constant():
    grid = TileGrid(32, 48, (0, 16, 32), (0, 24, 48))  # uneven overlaps, as at the frame edges
    tiles = [torch.full((3, 4, 64, 96), 173, dtype=torch.uint8) for _ in range(grid.num_tiles)]
    out = stitch_tiles(tiles, grid, 64, 96, scale=2, frame_chunk=3)
    assert out.shape == (3, 4, 128, 192) and out.dtype == torch.uint8
    assert bool(((out.int() - 173).abs() <= 1).all())  # normalised blend, truncated to uint8


def test_scale_requests():
    assert resolve_scale(2) == (2, 1.0) and resolve_scale(4.0) == (4, 1.0) and resolve_scale(2.25) == (2, 1.125)
    for bad in (1, 3, 8, 2.5, 0):
        with pytest.raises(ValueError, match="2, 4 or 2.25"):
            resolve_scale(bad)


def test_pre_upscale_targets_are_vae_aligned_and_reach_the_2_25_total():
    video = torch.randint(0, 256, (2, 3, 512, 768), dtype=torch.uint8)
    up = pre_upscale(video, 1.125, 16, frame_chunk=1)
    assert up.shape[-2:] == (576, 864) and up.dtype == torch.uint8
    assert torch.equal(up, pre_upscale(video, 1.125, 16))  # frames are independent: chunking changes nothing
    odd = pre_upscale(torch.zeros(1, 3, 100, 130, dtype=torch.uint8), 1.125, 16)
    assert odd.shape[-2] % 16 == 0 and odd.shape[-1] % 16 == 0
