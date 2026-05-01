# SPDX-License-Identifier: Apache-2.0
"""Stable Audio decoding: latent -> waveform via OobleckVAE.

Slices the output to `[audio_start_in_s, audio_end_in_s]` and stashes
the result on `batch.extra["audio"]` + `["audio_sample_rate"]` for
`VideoGenerator._mux_audio` to pick up.
"""
from __future__ import annotations

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult


class StableAudioDecodingStage(PipelineStage):
    """Decode latent → audio waveform + slice to [start, end]."""

    def __init__(self, vae) -> None:
        super().__init__()
        self.vae = vae

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        pc = fastvideo_args.pipeline_config
        latents = batch.latents

        decoded = self.vae.decode(latents)
        if hasattr(decoded, "sample"):  # tolerate tensor or dataclass
            decoded = decoded.sample

        sr = int(getattr(self.vae, "sampling_rate", pc.sampling_rate))
        start_in_s = float(batch.extra.get("audio_start_in_s", pc.audio_start_in_s))
        end_in_s = float(batch.extra.get("audio_end_in_s", pc.audio_end_in_s))
        decoded = decoded[:, :, int(start_in_s * sr):int(end_in_s * sr)]

        if batch.extra is None:
            batch.extra = {}
        # `_mux_audio` / `_write_pcm_wav` want `[samples, channels]`.
        batch.extra["audio"] = decoded.squeeze(0).T.detach().float().cpu().numpy()
        batch.extra["audio_sample_rate"] = sr
        batch.extra["audio_only"] = True
        # Raw tensor for parity tests.
        batch.extra["decoded_audio"] = decoded.detach().cpu()

        # `VideoGenerator.generate_video` is video-shaped (asserts
        # `output_batch.output is not None`); fill with a placeholder of
        # the expected `[B, 3, num_frames, H, W]` shape — the real audio
        # is on `batch.extra` above. Pure-audio workload support tracked
        # in REVIEW item 28.
        b = decoded.shape[0]
        n_frames = int(getattr(batch, "num_frames", 1) or 1)
        h = int(getattr(batch, "height", 1) or 1)
        w = int(getattr(batch, "width", 1) or 1)
        batch.output = torch.zeros((b, 3, n_frames, h, w), dtype=torch.uint8)
        return batch
