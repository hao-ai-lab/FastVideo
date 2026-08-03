# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

MINIMAX_H3_STATE_KEY = "minimax_h3"


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
    """Typed shared state carried between MiniMax-H3 pipeline stages."""

    height: int | None = None
    width: int | None = None
    num_frames: int | None = None
    num_latent_frames: int | None = None
    latent_height: int | None = None
    latent_width: int | None = None
    num_audio_latents: int | None = None
    keyframes: list[Any] = field(default_factory=list)
    keyframe_anchors: tuple[Literal["first", "last"], ...] = ()
    layout: MiniMaxH3Layout | None = None
    prompt_embeds: torch.Tensor | None = None
    text_token_tags: torch.Tensor | None = None
    video_latents: torch.Tensor | None = None
    audio_latents: torch.Tensor | None = None
    condition_video_latents: torch.Tensor | None = None
    condition_audio_latents: torch.Tensor | None = None
    video_timesteps: torch.Tensor | None = None
    audio_timesteps: torch.Tensor | None = None
    row_timestep_plan: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    references: list[MiniMaxH3Reference] = field(default_factory=list)


def get_minimax_h3_state(batch: "ForwardBatch") -> MiniMaxH3State:
    """Return the one family-local state object for a request."""
    state = batch.extra.get(MINIMAX_H3_STATE_KEY)
    if state is None:
        state = MiniMaxH3State()
        batch.extra[MINIMAX_H3_STATE_KEY] = state
    if not isinstance(state, MiniMaxH3State):
        raise TypeError(f"batch.extra[{MINIMAX_H3_STATE_KEY!r}] must be MiniMaxH3State, got {type(state).__name__}.")
    return state
