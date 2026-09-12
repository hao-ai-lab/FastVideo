# SPDX-License-Identifier: Apache-2.0
"""Deterministic math and RNG helpers for Helios pyramid sampling."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Calculate the dynamic flow shift for a pyramid stage."""
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    return image_seq_len * slope + intercept


def get_num_latent_chunks(
    num_frames: int,
    num_latent_frames_per_chunk: int,
    temporal_scale_factor: int,
) -> int:
    pixel_frames_per_chunk = (num_latent_frames_per_chunk - 1) * temporal_scale_factor + 1
    return max(1, math.ceil(num_frames / pixel_frames_per_chunk))


def get_generated_pixel_frames(num_latent_frames: int, temporal_scale_factor: int) -> int:
    return (num_latent_frames - 1) // temporal_scale_factor * temporal_scale_factor + 1


def build_helios_frame_indices(
    history_sizes: Sequence[int],
    num_latent_frames_per_chunk: int,
    keep_first_frame: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(history_sizes) != 3:
        raise ValueError(f"Helios requires three history sizes, got {list(history_sizes)}")
    if not keep_first_frame:
        raise ValueError("The initial FastVideo Helios T2V port requires keep_first_frame=True")

    history_long, history_mid, history_one = history_sizes
    indices = torch.arange(
        1 + sum(history_sizes) + num_latent_frames_per_chunk,
        device=device,
    )
    prefix, long_indices, mid_indices, one_indices, current_indices = indices.split([
        1,
        history_long,
        history_mid,
        history_one,
        num_latent_frames_per_chunk,
    ])
    short_indices = torch.cat([prefix, one_indices])
    return (
        current_indices.unsqueeze(0),
        short_indices.unsqueeze(0),
        mid_indices.unsqueeze(0),
        long_indices.unsqueeze(0),
    )


def downsample_to_pyramid_base(latents: torch.Tensor, num_stages: int) -> torch.Tensor:
    batch_size, channels, num_frames, height, width = latents.shape
    flattened = latents.permute(0, 2, 1, 3, 4).reshape(
        batch_size * num_frames,
        channels,
        height,
        width,
    )
    for _ in range(num_stages - 1):
        height //= 2
        width //= 2
        flattened = F.interpolate(flattened, size=(height, width), mode="bilinear") * 2
    return flattened.reshape(
        batch_size,
        num_frames,
        channels,
        height,
        width,
    ).permute(0, 2, 1, 3, 4)


def _randn_tensor(
    shape: tuple[int, ...],
    generator: torch.Generator | list[torch.Generator],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if isinstance(generator, list):
        if len(generator) != shape[0]:
            raise ValueError(f"Expected one generator per batch item ({shape[0]}), got {len(generator)}")
        samples = [
            torch.randn(
                (1, *shape[1:]),
                generator=item,
                device=item.device,
                dtype=dtype,
            ).to(device) for item in generator
        ]
        return torch.cat(samples, dim=0)
    return torch.randn(
        shape,
        generator=generator,
        device=generator.device,
        dtype=dtype,
    ).to(device)


def sample_block_noise(
    scheduler,
    shape: tuple[int, int, int, int, int],
    patch_size: tuple[int, int, int],
    device: torch.device,
    generator: torch.Generator | list[torch.Generator],
) -> torch.Tensor:
    if isinstance(generator, list):
        generator = generator[0]
    batch_size, channels, num_frames, height, width = shape
    _, patch_height, patch_width = patch_size
    if height % patch_height or width % patch_width:
        raise ValueError(f"Noise shape {(height, width)} must be divisible by patch {(patch_height, patch_width)}")

    block_size = patch_height * patch_width
    gamma = scheduler.config.gamma
    covariance = (torch.eye(block_size, device=device) * (1 + gamma) -
                  torch.ones(block_size, block_size, device=device) * gamma)
    covariance += torch.eye(block_size, device=device) * 1e-8
    cholesky = torch.linalg.cholesky(covariance.float())

    block_count = batch_size * channels * num_frames * (height // patch_height) * (width // patch_width)
    standard_noise = torch.randn(
        block_count,
        block_size,
        generator=generator,
        device=generator.device,
    ).to(device)
    noise = standard_noise @ cholesky.T
    noise = noise.view(
        batch_size,
        channels,
        num_frames,
        height // patch_height,
        width // patch_width,
        patch_height,
        patch_width,
    )
    return noise.permute(0, 1, 2, 3, 5, 4, 6).reshape(shape)
