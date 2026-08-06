# SPDX-License-Identifier: Apache-2.0
"""Stable Audio decoding for the MAGI-2 joint video-audio pipeline."""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import resample

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.dits.magi2_runtime import psm
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult

MAGI2_AUDIO_TIME_STRETCH = 441.0 / 512.0


def resample_magi2_audio(
    sample_major_audio: np.ndarray,
    time_stretching: float = MAGI2_AUDIO_TIME_STRETCH,
) -> np.ndarray:
    """Resample sample-major stereo audio with MAGI-2's FFT interpolation."""
    output_length = int(sample_major_audio.shape[0] * time_stretching)
    return resample(sample_major_audio, output_length)


@torch.inference_mode()
def decode_magi2_audio(audio_vae, sample_major_latent: torch.Tensor) -> np.ndarray:
    """Decode a ``[latent samples, channels]`` tensor into stereo samples."""
    waveform = audio_vae.decode(sample_major_latent.T)
    sample_major_audio = waveform.squeeze(0).T.cpu().numpy()
    return resample_magi2_audio(sample_major_audio)


class Magi2AudioDecodingStage(PipelineStage):
    """Decode preview audio latents on the context-parallel leader rank."""

    def __init__(self, audio_vae) -> None:
        """Store the published Stable Audio decoder."""
        super().__init__()
        self.audio_vae = audio_vae

    def verify_input(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        """Accept the audio latent that the joint denoising stage produced."""
        del batch, fastvideo_args
        return VerificationResult()

    def verify_output(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        """Return the default stage verification record."""
        del batch, fastvideo_args
        return VerificationResult()

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Decode and store sample-major audio on the output-producing rank."""
        if batch.audio_latents is None:
            raise ValueError("MAGI-2 audio decoding requires audio_latents")
        if not psm.is_group_first_rank("cp"):
            return batch
        device = torch.device("cuda", torch.cuda.current_device())
        self.audio_vae.to(device=device, dtype=torch.float32)
        sample_major_latent = batch.audio_latents.squeeze(0).to(device)
        batch.extra["audio"] = decode_magi2_audio(
            self.audio_vae,
            sample_major_latent,
        )
        batch.extra["audio_sample_rate"] = int(self.audio_vae.sampling_rate)
        if fastvideo_args.vae_cpu_offload:
            self.audio_vae.to("cpu")
            torch.cuda.empty_cache()
        return batch


__all__ = [
    "MAGI2_AUDIO_TIME_STRETCH",
    "Magi2AudioDecodingStage",
    "decode_magi2_audio",
    "resample_magi2_audio",
]
