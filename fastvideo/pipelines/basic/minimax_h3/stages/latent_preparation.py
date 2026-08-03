# SPDX-License-Identifier: Apache-2.0
"""Packed joint audio/video latent preparation for MiniMax H3."""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    patchify_video_latents,
    randn_tensor,
)
from fastvideo.pipelines.basic.minimax_h3.types import get_minimax_h3_state
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult


def _first_parameter(module: Any) -> torch.Tensor | None:
    parameters = getattr(module, "parameters", None)
    return None if parameters is None else next(parameters(), None)


def _module_device(module: Any) -> torch.device:
    parameter = _first_parameter(module)
    return torch.device("cpu") if parameter is None else parameter.device


def _arch_value(module: Any, name: str) -> Any:
    config = getattr(module, "config", None)
    arch = getattr(config, "arch_config", config)
    value = getattr(arch, name, None)
    if value is None:
        value = getattr(module, name, None)
    if value is None:
        raise ValueError(f"MiniMax-H3 component {type(module).__name__} does not expose `{name}`.")
    return value


class MiniMaxH3LatentPreparationStage(PipelineStage):
    """Draw target video then target audio noise and prepend condition rows."""

    def __init__(self, transformer: Any, vae: Any, audio_vae: Any) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.audio_vae = audio_vae

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        state = get_minimax_h3_state(batch)
        result = VerificationResult()
        result.add_check("layout", state.layout, V.not_none)
        result.add_check("num_latent_frames", state.num_latent_frames, V.positive_int)
        result.add_check("latent_height", state.latent_height, V.positive_int)
        result.add_check("latent_width", state.latent_width, V.positive_int)
        result.add_check("num_audio_latents", state.num_audio_latents, V.positive_int)
        result.add_check("latents", batch.latents, V.none_or_tensor)
        result.add_check("audio_latents", batch.audio_latents, V.none_or_tensor)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        state = get_minimax_h3_state(batch)
        result = VerificationResult()
        result.add_check("video_latents", state.video_latents, V.with_dims(2))
        result.add_check("audio_latents", state.audio_latents, V.with_dims(2))
        result.add_check("batch.latents", batch.latents, lambda value: value is None)
        result.add_check("batch.audio_latents", batch.audio_latents, lambda value: value is None)
        return result

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        del fastvideo_args
        state = get_minimax_h3_state(batch)
        if state.layout is None:
            raise ValueError("MiniMax-H3 layout must be prepared before target latents.")
        geometry = (state.num_latent_frames, state.latent_height, state.latent_width, state.num_audio_latents)
        if any(value is None for value in geometry):
            raise ValueError("MiniMax-H3 input geometry is incomplete.")
        num_latent_frames, latent_height, latent_width, num_audio_latents = (int(value) for value in geometry
                                                                             if value is not None)

        patch_size = tuple(int(value) for value in _arch_value(self.transformer, "patch_size"))
        video_channels = int(_arch_value(self.vae, "latent_channels"))
        audio_channels = int(_arch_value(self.audio_vae, "latent_channels"))
        device = _module_device(self.transformer)

        video_noise = batch.latents
        expected_video_shape = (1, video_channels, num_latent_frames, latent_height, latent_width)
        if video_noise is None:
            video_noise = randn_tensor(
                expected_video_shape,
                generator=batch.generator,
                device=device,
                dtype=torch.float32,
            )
        elif tuple(video_noise.shape) != expected_video_shape:
            raise ValueError(
                f"MiniMax-H3 injected video latents must have shape {expected_video_shape}, got {tuple(video_noise.shape)}."
            )
        video_rows = patchify_video_latents(video_noise.to(device=device, dtype=torch.float32), patch_size)

        audio_noise = batch.audio_latents
        expected_audio_shape = (MINIMAX_H3_AUDIO_CHANNELS, audio_channels, num_audio_latents)
        if audio_noise is None:
            audio_rows = randn_tensor(
                (num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS, audio_channels),
                generator=batch.generator,
                device=device,
                dtype=torch.float32,
            )
        else:
            if tuple(audio_noise.shape) != expected_audio_shape:
                raise ValueError(f"MiniMax-H3 injected audio latents must have shape {expected_audio_shape}, "
                                 f"got {tuple(audio_noise.shape)}.")
            audio_rows = audio_noise.to(device=device, dtype=torch.float32).permute(0, 2, 1).reshape(-1, audio_channels)

        if state.condition_video_latents is not None:
            video_rows = torch.cat([state.condition_video_latents.to(device), video_rows])
        if state.condition_audio_latents is not None:
            audio_rows = torch.cat([state.condition_audio_latents.to(device), audio_rows])

        if video_rows.shape[0] != state.layout.video_indices.numel():
            raise ValueError("MiniMax-H3 packed video row count does not match its layout.")
        if audio_rows.shape[0] != state.layout.audio_indices.numel():
            raise ValueError("MiniMax-H3 packed audio row count does not match its layout.")
        state.video_latents = video_rows
        state.audio_latents = audio_rows
        batch.latents = None
        batch.audio_latents = None
        return batch


__all__ = ["MiniMaxH3LatentPreparationStage"]
