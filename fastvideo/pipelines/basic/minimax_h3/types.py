# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

MINIMAX_H3_STATE_KEY = "minimax_h3"


@dataclass(frozen=True)
class MiniMaxH3Reference:
    """One deferred Ref2VA medium; decoding belongs to the preparation stage."""

    source: Any
    media_type: Literal["image", "video", "audio"] = "image"
    soundtrack: Any | None = None
    fps: float | None = None
    sample_rate: int | None = None

    def __post_init__(self) -> None:
        if self.source is None:
            raise ValueError("A MiniMax-H3 reference requires a media source.")
        if self.media_type not in ("image", "video", "audio"):
            raise ValueError(f"Unsupported MiniMax-H3 reference type: {self.media_type!r}.")
        if self.soundtrack is not None and self.media_type != "video":
            raise ValueError("Only a video reference may carry a separate soundtrack.")
        if self.fps is not None and (self.media_type != "video" or isinstance(self.fps, bool)
                                     or not isinstance(self.fps, Real) or self.fps <= 0):
            raise ValueError("Reference `fps` must be positive and is only valid for video.")
        if self.sample_rate is not None and (self.media_type == "image" or isinstance(self.sample_rate, bool)
                                             or not isinstance(self.sample_rate, Integral) or self.sample_rate <= 0):
            raise ValueError("Reference `sample_rate` must be positive and is only valid for audio-bearing media.")


@dataclass
class MiniMaxH3PreparedReference:
    """Decoded and normalized Ref2VA media plus its resolved latent geometry."""

    media_type: Literal["image", "video", "audio"]
    has_audio: bool = False
    image: Any | None = None
    frames: Any | None = None
    waveform: torch.Tensor | None = None
    block_timestamps: list[float] = field(default_factory=list)
    num_latent_frames: int = 1
    latent_height: int = 0
    latent_width: int = 0
    num_audio_latents: int = 0


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
    prepared_references: list[MiniMaxH3PreparedReference] = field(default_factory=list)


def get_minimax_h3_state(batch: "ForwardBatch") -> MiniMaxH3State:
    """Return the one family-local state object for a request."""
    state = batch.extra.get(MINIMAX_H3_STATE_KEY)
    if state is None:
        state = MiniMaxH3State()
        batch.extra[MINIMAX_H3_STATE_KEY] = state
    if not isinstance(state, MiniMaxH3State):
        raise TypeError(f"batch.extra[{MINIMAX_H3_STATE_KEY!r}] must be MiniMaxH3State, got {type(state).__name__}.")
    return state
