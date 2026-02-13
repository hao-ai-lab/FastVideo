# SPDX-License-Identifier: Apache-2.0
"""
Stable Audio decoding stage: decode latents to audio via Oobleck VAE.
"""
from __future__ import annotations

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.utils import PRECISION_TO_TYPE


class StableAudioDecodingStage(PipelineStage):
    """Decode Stable Audio latents to waveform using pretransform (Oobleck VAE)."""

    def __init__(self, pretransform) -> None:
        super().__init__()
        self.pretransform = pretransform

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        latents = batch.latents
        if latents is None:
            raise ValueError("Latents must be provided before decoding.")

        device = get_local_torch_device()
        self.pretransform = self.pretransform.to(device)
        latents = latents.to(device)

        vae_dtype = PRECISION_TO_TYPE.get(
            getattr(fastvideo_args.pipeline_config, "vae_precision", "fp32"),
            torch.float32,
        )
        latents = latents.to(vae_dtype)

        audio = self.pretransform.decode(latents)
        batch.output = audio
        return batch
