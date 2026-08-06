# SPDX-License-Identifier: Apache-2.0
"""Request validation and latent initialization for MAGI-2 inference."""

from __future__ import annotations

import random

import numpy as np
import torch

import fastvideo.envs as envs
from fastvideo.fastvideo_args import FastVideoArgs, WorkloadType
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult


class Magi2InputValidationStage(PipelineStage):
    """Validate the published ten-second 1080p T2V and I2V request profile."""

    def __init__(
        self,
        output_frames: int,
        output_height: int,
        output_width: int,
        output_fps: int,
    ) -> None:
        """Store the fixed output geometry supported by the release checkpoint."""
        super().__init__()
        self.output_frames = output_frames
        self.output_height = output_height
        self.output_width = output_width
        self.output_fps = output_fps

    def verify_input(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        """Return the default record; ``forward`` emits model-specific errors."""
        del batch, fastvideo_args
        return VerificationResult()

    def verify_output(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        """Return the default record after request normalization and seeding."""
        del batch, fastvideo_args
        return VerificationResult()

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Require release geometry and reset every request's random-number stream."""
        if fastvideo_args.workload_type not in (WorkloadType.T2V, WorkloadType.I2V):
            raise ValueError("MAGI-2 supports only T2V and I2V workloads")
        if not isinstance(batch.prompt, str) or not batch.prompt.strip():
            raise ValueError("MAGI-2 requires one non-empty prompt string")
        expected_geometry = (
            self.output_frames,
            self.output_height,
            self.output_width,
            self.output_fps,
        )
        received_geometry = (
            batch.num_frames,
            batch.height,
            batch.width,
            batch.fps,
        )
        if received_geometry != expected_geometry:
            raise ValueError(
                "MAGI-2 Preview supports 249 frames at 1088x1920 and 25 fps; "
                f"received frames/height/width/fps={received_geometry}"
            )
        if batch.num_inference_steps <= 0 or batch.num_inference_steps_sr <= 0:
            raise ValueError("MAGI-2 denoising step counts must be positive")
        seed = 42 if batch.seed is None else int(batch.seed)
        batch.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if fastvideo_args.deterministic or envs.MAGI2_DETERMINISTIC:
            random.seed(seed)
            np.random.seed(seed)
        return batch


class Magi2LatentPreparationStage(PipelineStage):
    """Draw video noise before audio noise with the official release shapes."""

    def __init__(
        self,
        video_channels: int,
        video_length: int,
        video_height: int,
        video_width: int,
        audio_length: int,
        audio_channels: int,
    ) -> None:
        """Store the preview video and audio latent geometry."""
        super().__init__()
        self.video_shape = (
            1,
            video_channels,
            video_length,
            video_height,
            video_width,
        )
        self.audio_shape = (1, audio_length, audio_channels)

    def verify_input(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        """Return the default record for a validated and seeded request."""
        del batch, fastvideo_args
        return VerificationResult()

    def verify_output(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        """Return the default record for joint video and audio noise tensors."""
        del batch, fastvideo_args
        return VerificationResult()

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Draw FP32 video and audio noise in the release model's RNG order."""
        del fastvideo_args
        device = torch.device("cuda", torch.cuda.current_device())
        batch.latents = torch.randn(
            self.video_shape,
            device=device,
            dtype=torch.float32,
        )
        batch.audio_latents = torch.randn(
            self.audio_shape,
            device=device,
            dtype=torch.float32,
        )
        return batch


__all__ = ["Magi2InputValidationStage", "Magi2LatentPreparationStage"]
