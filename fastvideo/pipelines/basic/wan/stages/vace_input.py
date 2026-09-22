# SPDX-License-Identifier: Apache-2.0
"""Wan-VACE input preparation: mask video, reference images, and reference-only pixels."""

from __future__ import annotations

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.vision_utils import load_video, normalize, numpy_to_pt, pil_to_numpy, resize
from fastvideo.pipelines.basic.wan.stages.vace_conditioning import preprocess_vace_reference_images
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult


class WanVACEInputStage(PipelineStage):
    """Prepare VACE-specific inputs after shared validation and text encoding."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.mask_path is not None:
            mask_images, _ = load_video(batch.mask_path, return_fps=True)
            if batch.num_frames is not None and len(mask_images) > batch.num_frames:
                mask_images = mask_images[:batch.num_frames]
            mask_numpy = normalize(pil_to_numpy([
                resize(img, batch.height, batch.width, resize_mode="default", resample="lanczos")
                for img in mask_images
            ]))
            batch.mask_video = numpy_to_pt(mask_numpy).permute(1, 0, 2, 3).unsqueeze(0)

        if batch.references:
            batch.vace_reference_images = preprocess_vace_reference_images(
                batch.references,
                (batch.height, batch.width),
                self.device,
                torch.float32,
            )
            batch.vace_num_reference_frames = len(batch.vace_reference_images)

        # Official Wan2.1 reference-only path: no src video/mask, synthesize zero pixels.
        if batch.video_path is None and batch.references and batch.video_latent is None:
            batch.video_latent = torch.zeros(
                1,
                3,
                batch.num_frames,
                batch.height,
                batch.width,
                dtype=torch.float32,
            )
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        return result
