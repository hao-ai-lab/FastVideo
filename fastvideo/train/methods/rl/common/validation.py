# SPDX-License-Identifier: Apache-2.0
"""Shared validation helpers for RL training methods."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


@dataclass(slots=True)
class RLValidationConfig:
    every_steps: int = 0
    num_steps: int = 40 # Reference DiffusionNFT sampling num steps for best visual quality
    num_prompts: int = 16
    batch_size: int = 16
    log_samples: bool = True
    seed: int = 42
    data_path: str | None = None
    sampling: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> RLValidationConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(f"method.validation must be a mapping, got {type(raw).__name__}")
        data_path = raw.get("data_path", None)
        sampling = raw.get("sampling", None)
        if sampling is not None and not isinstance(sampling, dict):
            raise ValueError(f"method.validation.sampling must be a mapping, got {type(sampling).__name__}")
        return cls(
            every_steps=max(0, int(raw.get("every_steps", 0) or 0)),
            num_steps=max(1, int(raw.get("num_steps", 40) or 40)),
            num_prompts=max(1, int(raw.get("num_prompts", 16) or 16)),
            batch_size=max(1, int(raw.get("batch_size", 16) or 16)),
            log_samples=bool(raw.get("log_samples", True)),
            seed=int(raw.get("seed", 42) or 42),
            data_path=(None if data_path in (None, "") else str(data_path)),
            sampling=(dict(sampling) if sampling is not None else None),
        )


def validation_shard_indices(
    num_prompts: int,
    *,
    rank: int,
    world_size: int,
) -> list[tuple[int, bool]]:
    """Return fixed validation prompt indices for one distributed rank."""
    num_prompts = max(1, int(num_prompts))
    world_size = max(1, int(world_size))
    per_rank = int(math.ceil(num_prompts / world_size))
    padded_total = per_rank * world_size
    return [((idx % num_prompts), idx < num_prompts) for idx in range(rank, padded_total, world_size)]


def validation_caption(
    prompt: str,
    rewards: dict[str, float],
) -> str:
    reward_parts = [f"{key}: {float(rewards[key]):.4f}" for key in sorted(rewards)]
    return f"{' | '.join(reward_parts)} | {prompt[:1000]}"


def media_to_video_array(media: torch.Tensor) -> Any:
    """Convert decoded media to a tracker video array.

    Accepts ``[C, T, H, W]`` tensors. ``[C, H, W]`` tensors are treated as
    ``T=1`` media. Output follows the existing tracker convention used
    elsewhere in FastVideo: ``[T, C, H, W]`` uint8.
    """
    if media.ndim == 3:
        media = media.unsqueeze(1)
    if media.ndim != 4:
        raise ValueError("media must have shape [C, T, H, W] or [C, H, W], "
                         f"got {tuple(media.shape)}")
    video = (media.detach().float().clamp(0, 1) * 255).round().to(torch.uint8)
    return video.permute(1, 0, 2, 3).contiguous().cpu().numpy()
