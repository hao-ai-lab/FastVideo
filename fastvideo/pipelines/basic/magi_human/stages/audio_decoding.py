# SPDX-License-Identifier: Apache-2.0
"""Audio decoding stage for daVinci-MagiHuman.

Takes the denoised audio latent that `MagiHumanDenoisingStage` leaves
on `batch.audio_latents` and decodes it to a waveform using the
Stable Audio Open 1.0 VAE. Mirrors the upstream post-process path
(see `MagiEvaluator.post_process` in
daVinci-MagiHuman/inference/pipeline/video_generate.py:503):

    latent_audio.squeeze(0)                 # (L, C_latent)
    audio = self.audio_vae.decode(latent_audio.T)   # (1, audio_ch, samples)
    audio = audio.squeeze(0).T.cpu().numpy()        # (samples, audio_ch)
    audio = resample_audio_sinc(audio, _UPSTREAM_AUDIO_TIME_STRETCH)

The stage stores the resampled waveform on `batch.extra["audio"]`
(shape `[samples, audio_channels]`) and the sample rate on
`batch.extra["audio_sample_rate"]`. FastVideo's `VideoGenerator._mux_audio`
then reads those, writes a temp wav, and muxes it into the output mp4
via PyAV — same plumbing LTX-2 and Stable Audio use.
"""
from __future__ import annotations

import numpy as np
import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult

# 441/512 is daVinci-MagiHuman's audio time-stretch ratio that aligns
# the 44.1 kHz Stable-Audio output with the 25-fps video frame rate.
# See daVinci-MagiHuman/inference/pipeline/video_generate.py:516.
_UPSTREAM_AUDIO_TIME_STRETCH = 441.0 / 512.0

# Stable Audio Open 1.0 native sample rate (per stabilityai/stable-audio-open-1.0
# model card and fastvideo/configs/models/vaes/oobleck.py::OobleckVAEArchConfig.sampling_rate).
_SA_AUDIO_OPEN_SAMPLE_RATE = 44100


def _resample_sinc(audio: np.ndarray, time_stretching: float) -> np.ndarray:
    """Sinc-interpolate the audio to `new_length = int(L * time_stretching)`.

    Mirrors upstream `video_process.resample_audio_sinc` (which delegates to
    `torchaudio.functional.resample` under the hood); to avoid the hard
    dependency and mixed-library gotchas, we do the same via
    `torch.nn.functional` on a tensor.
    """
    if time_stretching == 1.0:
        return audio
    t = torch.from_numpy(audio.astype(np.float32))
    new_length = int(t.shape[0] * time_stretching)
    # t: [L, C] -> [1, C, L] for interpolate
    t_bct = t.T.unsqueeze(0)
    t_bct = torch.nn.functional.interpolate(
        t_bct,
        size=new_length,
        mode="linear",
        align_corners=True,
    )
    t = t_bct.squeeze(0).T  # [new_length, C]
    return t.numpy()


class MagiHumanAudioDecodingStage(PipelineStage):
    """Decode `batch.audio_latents` to a waveform using Stable Audio's VAE.

    The VAE is loaded lazily by `SAAudioVAEModel.sa_audio_vae_model` — the
    first call triggers a snapshot_download (requires HF token + accepted
    terms on stabilityai/stable-audio-open-1.0).
    """

    def __init__(
        self,
        audio_vae,
        time_stretching: float = _UPSTREAM_AUDIO_TIME_STRETCH,
    ) -> None:
        super().__init__()
        self.audio_vae = audio_vae
        self.time_stretching = time_stretching

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        latent_audio = getattr(batch, "audio_latents", None)
        if latent_audio is None:
            # Nothing to decode — silent output.
            return batch

        # Upstream shape: `[B, L, C_latent]` from the DiT; AutoencoderOobleck
        # expects `[B, C_latent, L]`. MagiEvaluator.post_process does
        # `latent_audio.squeeze(0); audio_vae.decode(latent_audio.T)`
        # (which yields `[C_latent, L]`, implicit batch=1). We keep the
        # batch dim and transpose L<->C.
        latent_bcl = latent_audio.permute(0, 2, 1).contiguous()

        # Decode: [B, C_latent, L] -> [B, audio_channels, samples]
        audio_out = self.audio_vae.decode(latent_bcl)

        audio_np = audio_out.squeeze(0).T.float().cpu().numpy()
        audio_np = _resample_sinc(audio_np, self.time_stretching)

        # Conform to FastVideo convention: VideoGenerator._mux_audio
        # reads these two keys and muxes via PyAV.
        if batch.extra is None:
            batch.extra = {}
        batch.extra["audio"] = audio_np
        batch.extra["audio_sample_rate"] = int(getattr(self.audio_vae, "sampling_rate", _SA_AUDIO_OPEN_SAMPLE_RATE))
        return batch
