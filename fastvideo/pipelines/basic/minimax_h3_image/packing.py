# SPDX-License-Identifier: Apache-2.0
"""Canvas, rotary layout, and patch packing for MiniMax-H3-Image.

Spatial coordinates are built with ``numpy.linspace(..., endpoint=False)`` in
float64. ``torch.linspace`` does not reproduce that grid.
"""

from __future__ import annotations

import math

import numpy as np
import torch

MINIMAX_H3_IMAGE_TAG = 0
MINIMAX_H3_IMAGE_TEXT_TAG = 1
MINIMAX_H3_IMAGE_CANVAS_MULTIPLE = 32
MINIMAX_H3_IMAGE_MIN_GENERATION_AREA = 2048 * 2048
MINIMAX_H3_IMAGE_MAX_SEQUENCE_LENGTH = 4096
MINIMAX_H3_IMAGE_MAX_REFERENCES = 16
MINIMAX_H3_IMAGE_REFERENCE_NOISE_AUG = 0.999
MINIMAX_H3_IMAGE_NUM_STEPS = 6
MINIMAX_H3_IMAGE_SAMPLE_SIGMAS = (
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0,
)
_ROPE_SPATIAL_SCALE = 32

REFERENCE_MIN_SHORT_EDGE = 1024
REFERENCE_MAX_SHORT_EDGE = 2048
VISION_MIN_SHORT_EDGE = 1024
VISION_MAX_SHORT_EDGE = 1024

TEXT_TOKEN_TAGS_KEY = "minimax_h3_image_text_token_tags"
REFERENCE_IMAGES_KEY = "minimax_h3_image_reference_images"
VISION_IMAGES_KEY = "minimax_h3_image_vision_images"
OUTPUT_HEIGHT_KEY = "minimax_h3_image_output_height"
OUTPUT_WIDTH_KEY = "minimax_h3_image_output_width"


def edit_label(index: int) -> str:
    """Label for reference ``index`` (1-based). Image edits omit the angle brackets used by video Ref2VA."""
    return f"Picture {index}: "


def get_image_canvas_size(height: int, width: int, min_generation_area: int, multiple: int) -> tuple[int, int]:
    """Canvas for a requested ``height x width``. Smaller images scale up, then both sides snap to ``multiple``."""
    if height <= 0 or width <= 0:
        raise ValueError(f"Image size must be positive, got {height}x{width}.")
    if min_generation_area <= 0 or multiple <= 0:
        raise ValueError("min_generation_area and canvas multiple must be positive.")
    scale = max(1.0, math.sqrt(min_generation_area / (width * height)))
    return (
        max(multiple,
            round(height * scale / multiple) * multiple),
        max(multiple,
            round(width * scale / multiple) * multiple),
    )


def bound_short_edge(
    width: int,
    height: int,
    min_short_edge: int,
    max_short_edge: int,
    multiple: int,
) -> tuple[int, int]:
    """Resize target for one reference consumer. Sides stay put when the short edge is already in range."""
    short_edge = min(width, height)
    if min_short_edge <= short_edge <= max_short_edge:
        return width, height
    scale = (min_short_edge if short_edge < min_short_edge else max_short_edge) / short_edge
    return (
        max(multiple,
            round(width * scale / multiple) * multiple),
        max(multiple,
            round(height * scale / multiple) * multiple),
    )


def patchify_video_latents(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """Pack ``(batch, channels, frames, height, width)`` into frame-major rows."""
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"Latents of shape {tuple(latents.shape)} are not divisible by the patch {patch_size}.")
    latents = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(np.ascontiguousarray(grid)).to(torch.float64)


def _frame_position_grid(
    latent_height: int,
    latent_width: int,
    patch_h: int,
    patch_w: int,
) -> torch.Tensor:
    sqrt_area = float(np.sqrt(latent_height * latent_width))
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1)


def build_image_packed_sequence(
    text_token_tags: torch.Tensor,
    reference_grids: list[tuple[int, int]],
    latent_height: int,
    latent_width: int,
    patch_size: tuple[int, int, int],
    image_tag: int = MINIMAX_H3_IMAGE_TAG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build ``[text | reference images | generated image]`` positions, tags, and index tensors."""
    _, patch_h, patch_w = patch_size
    num_text_tokens = int(text_token_tags.shape[0])
    num_reference_rows = sum((height // patch_h) * (width // patch_w) for height, width in reference_grids)
    num_target_rows = (latent_height // patch_h) * (latent_width // patch_w)
    sequence_length = num_text_tokens + num_reference_rows + num_target_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)
    for height, width in reference_grids:
        row_count = (height // patch_h) * (width // patch_w)
        frame_grid = _frame_position_grid(height, width, patch_h, patch_w)
        position_ids[cursor:cursor + row_count, 0] = rotary_time
        position_ids[cursor:cursor + row_count, 1:] = frame_grid
        cursor += row_count
        rotary_time += 1.0
    frame_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)
    position_ids[cursor:, 0] = rotary_time
    position_ids[cursor:, 1:] = frame_grid

    text_indices = torch.arange(num_text_tokens)
    image_indices = torch.arange(num_text_tokens, sequence_length)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[image_indices] = image_tag
    return position_ids, token_tags, image_indices, text_indices, num_reference_rows


def rounded_sample_sigmas(sigmas: tuple[float, ...] = MINIMAX_H3_IMAGE_SAMPLE_SIGMAS) -> torch.Tensor:
    """Round each sigma through the float32 timestep ``1 - sigma`` and back, matching the release sampler."""
    timesteps = torch.tensor([1.0 - sigma for sigma in sigmas[:-1]], dtype=torch.float32)
    return torch.cat([1.0 - timesteps, torch.zeros(1)])
