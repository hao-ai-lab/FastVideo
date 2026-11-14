# SPDX-License-Identifier: Apache-2.0
"""Latent preparation for Wan-VACE (reference-frame temporal padding)."""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.latent_preparation import LatentPreparationStage


class WanVACELatentPreparationStage(LatentPreparationStage):

    def latent_num_frames(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> int:
        # Each reference image occupies one extra leading latent frame.
        return super().latent_num_frames(batch, fastvideo_args) + batch.vace_num_reference_frames
