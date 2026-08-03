# SPDX-License-Identifier: Apache-2.0
"""Dual-scheduler timestep planning for MiniMax H3."""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    build_row_timesteps,
)
from fastvideo.pipelines.basic.minimax_h3.types import get_minimax_h3_state
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult


class MiniMaxH3TimestepPreparationStage(PipelineStage):
    """Create the video/audio schedules and per-row timestep lookup for each interval."""

    def __init__(self, scheduler: Any, audio_scheduler: Any) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.audio_scheduler = audio_scheduler

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        state = get_minimax_h3_state(batch)
        result = VerificationResult()
        result.add_check("layout", state.layout, V.not_none)
        result.add_check("video_latents", state.video_latents, V.with_dims(2))
        result.add_check("audio_latents", state.audio_latents, V.with_dims(2))
        result.add_check("num_inference_steps", batch.num_inference_steps, V.positive_int)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        state = get_minimax_h3_state(batch)
        result = VerificationResult()
        result.add_check("video_timesteps", state.video_timesteps, V.with_dims(1))
        result.add_check("audio_timesteps", state.audio_timesteps, V.with_dims(1))
        result.add_check("row_timestep_plan", state.row_timestep_plan, V.list_not_empty)
        result.add_check("batch.timesteps", batch.timesteps, V.with_dims(1))
        return result

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        del fastvideo_args
        state = get_minimax_h3_state(batch)
        if state.layout is None or state.video_latents is None:
            raise ValueError("MiniMax-H3 layout and latents must be prepared before timesteps.")
        device = state.video_latents.device
        self.scheduler.set_timesteps(batch.num_inference_steps, device=device)
        self.audio_scheduler.set_timesteps(batch.num_inference_steps, device=device)
        state.video_timesteps = self.scheduler.timesteps
        state.audio_timesteps = self.audio_scheduler.timesteps
        if state.video_timesteps is None or state.audio_timesteps is None:
            raise ValueError("MiniMax-H3 schedulers did not produce timesteps.")
        if len(state.video_timesteps) != len(state.audio_timesteps):
            raise ValueError("MiniMax-H3 video and audio schedules must have the same number of intervals.")

        state.row_timestep_plan = []
        for video_timestep, audio_timestep in zip(state.video_timesteps, state.audio_timesteps, strict=False):
            video_value = float(video_timestep.item())
            audio_value = float(audio_timestep.item())
            unique, inverse = build_row_timesteps(
                state.layout,
                video_timestep=video_value,
                audio_timestep=audio_value,
                condition_video_timestep=max(video_value, MINIMAX_H3_KEYFRAME_NOISE_AUG),
                condition_audio_timestep=1.0,
            )
            state.row_timestep_plan.append((unique.to(device), inverse.to(device)))
        batch.timesteps = state.video_timesteps
        return batch


__all__ = ["MiniMaxH3TimestepPreparationStage"]
