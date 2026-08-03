# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Literal

import torch


@dataclass(frozen=True)
class MiniMaxH3Reference:
    """A deferred FL2VA/Ref2VA input; media decoding belongs to a pipeline stage."""

    source: Any
    media_type: Literal["image", "video", "audio"] = "image"
    anchor: Literal["first", "last"] = "first"
    fps: float | None = None
    sample_rate: int | None = None


@dataclass(frozen=True)
class MiniMaxH3Layout:
    """One packed joint sequence and the geometry needed to interpret it."""

    sequence_length: int
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_condition_video_rows: int
    num_condition_audio_rows: int
    num_video_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int


@dataclass
class MiniMaxH3State:
    """Typed shared state carried between future MiniMax-H3 pipeline stages."""

    layout: MiniMaxH3Layout | None = None
    text_token_tags: torch.Tensor | None = None
    video_latents: torch.Tensor | None = None
    audio_latents: torch.Tensor | None = None
    condition_video_latents: list[torch.Tensor] = field(default_factory=list)
    condition_audio_latents: list[torch.Tensor] = field(default_factory=list)
    references: list[MiniMaxH3Reference] = field(default_factory=list)
