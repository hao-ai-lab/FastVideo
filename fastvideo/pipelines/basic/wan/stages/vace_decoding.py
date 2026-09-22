# SPDX-License-Identifier: Apache-2.0
"""Decode VACE outputs after stripping reference-frame latents."""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.decoding import DecodingStage


class WanVACEDecodingStage(DecodingStage):

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        num_ref = batch.vace_num_reference_frames
        if num_ref > 0 and batch.latents is not None:
            batch.latents = batch.latents[:, :, num_ref:]
        return super().forward(batch, fastvideo_args)
