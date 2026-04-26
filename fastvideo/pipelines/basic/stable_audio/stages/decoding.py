# SPDX-License-Identifier: Apache-2.0
"""Stable Audio decoding: latent → waveform via FastVideo's first-class
`OobleckVAE`. Slices the output to the requested [start, end] sample
range and stashes it on `batch.extra["audio"]` + `batch.extra["audio_sample_rate"]`
for the standard `VideoGenerator._mux_audio` glue.
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

        # Our OobleckVAE wrapper exposes .decode(latent) -> waveform
        # tensor of shape [B, audio_channels, samples]. The
        # SAAudioVAEModel wrapper unwraps the diffusers-style
        # OobleckDecoderOutput and returns the raw tensor.
        decoded = self.vae.decode(latents)
        if hasattr(decoded, "sample"):  # tolerate either tensor or dataclass
            decoded = decoded.sample

        # Slice to [start, end] by sample.
        sr = int(getattr(self.vae, "sampling_rate", pc.sampling_rate))
        start_in_s = float(batch.extra.get("audio_start_in_s", pc.audio_start_in_s))
        end_in_s = float(batch.extra.get("audio_end_in_s", pc.audio_end_in_s))
        start_idx = int(start_in_s * sr)
        end_idx = int(end_in_s * sr)
        decoded = decoded[:, :, start_idx:end_idx]

        # Conform to FastVideo audio-mux convention.
        if batch.extra is None:
            batch.extra = {}
        # Shape that VideoGenerator._mux_audio expects: 1D or 2D
        # `[samples, channels]`. Our decode produces [B, C, samples];
        # squeeze batch + transpose.
        audio_np = decoded.squeeze(0).T.detach().float().cpu().numpy()
        batch.extra["audio"] = audio_np
        batch.extra["audio_sample_rate"] = sr
        # Also store raw decoded tensor so the parity test can compare
        # against diffusers' .audios output.
        batch.extra["decoded_audio"] = decoded.detach().cpu()

        # `VideoGenerator.generate_video` is video-shaped: it pre-allocates
        # `[B, 3, num_frames, H, W]` and asserts `output_batch.output is not
        # None`. Audio-only workloads have no native slot, so fill the
        # video output with zeros of the expected shape — the real audio
        # lives on `batch.extra` above. Pure-audio workload support is
        # tracked under REVIEW item 28 (`WorkloadType` lacks audio
        # variants).
        b = decoded.shape[0]
        n_frames = int(getattr(batch, "num_frames", 1) or 1)
        h = int(getattr(batch, "height", 1) or 1)
        w = int(getattr(batch, "width", 1) or 1)
        batch.output = torch.zeros((b, 3, n_frames, h, w), dtype=torch.uint8)
        return batch
