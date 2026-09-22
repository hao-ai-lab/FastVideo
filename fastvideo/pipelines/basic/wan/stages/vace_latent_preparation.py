# SPDX-License-Identifier: Apache-2.0
"""Latent preparation for Wan-VACE (reference-frame temporal padding)."""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.latent_preparation import LatentPreparationStage


class WanVACELatentPreparationStage(LatentPreparationStage):

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        num_ref = batch.vace_num_reference_frames
        temporal_ratio = fastvideo_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        original_num_frames = batch.num_frames
        if num_ref > 0 and original_num_frames is not None:
            batch.num_frames = int(original_num_frames) + num_ref * temporal_ratio
        super().forward(batch, fastvideo_args)
        batch.num_frames = original_num_frames
        return batch
