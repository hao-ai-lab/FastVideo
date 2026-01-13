# SPDX-License-Identifier: Apache-2.0
"""
Audio decoding stage for LTX-2 pipelines.
"""

from __future__ import annotations

import os

import torch

from fastvideo.models.dits.ltx2 import DEFAULT_LTX2_VOCODER_OUTPUT_SAMPLE_RATE

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)


class LTX2AudioDecodingStage(PipelineStage):
    """Decode LTX-2 audio latents into a waveform."""

    def __init__(self, audio_decoder, vocoder) -> None:
        super().__init__()
        self.audio_decoder = audio_decoder
        self.vocoder = vocoder

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        audio_latents = batch.extra.get("ltx2_audio_latents")
        if audio_latents is None:
            return batch

        device = get_local_torch_device()
        self.audio_decoder = self.audio_decoder.to(device)
        self.vocoder = self.vocoder.to(device)
        audio_latents = audio_latents.to(device)

        disable_autocast = os.getenv("LTX2_DISABLE_AUDIO_AUTOCAST", "1") == "1"
        with torch.no_grad(), torch.autocast(
                device_type="cuda",
                dtype=audio_latents.dtype,
                enabled=not disable_autocast,
        ):
            decoded_spec = self.audio_decoder(audio_latents)
            audio_wave = self.vocoder(decoded_spec).squeeze(0).float()

        # Move to CPU for pickling across process boundary
        batch.extra["audio"] = audio_wave.cpu()
        batch.extra[
            "audio_sample_rate"] = DEFAULT_LTX2_VOCODER_OUTPUT_SAMPLE_RATE
        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("audio_latents", batch.extra.get("ltx2_audio_latents"),
                         V.none_or_tensor)
        return result
