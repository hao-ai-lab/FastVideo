# SPDX-License-Identifier: Apache-2.0
"""Trim the decoded video to the draft length and drop the per-request H3 state."""

from __future__ import annotations

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.open_h3_regenerate_2k.draft_video import DRAFT_FRAMES
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult


class Regenerate2KFinalizeStage(PipelineStage):
    """The released pipeline writes a silent video: the audio stream only conditions the DiT and is never decoded."""

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("output", batch.output, V.with_dims(5))
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("output", batch.output, V.with_dims(5))
        return result

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        # The VAE decodes 124 frames (37 latent frames); the draft has 121.
        batch.output = batch.output[:, :, :DRAFT_FRAMES]
        batch.prompt_embeds = []
        batch.latents = None
        batch.audio_latents = None
        for key in tuple(batch.extra):
            if key.startswith("minimax_h3_"):
                batch.extra.pop(key)
        return batch


__all__ = ["Regenerate2KFinalizeStage"]
