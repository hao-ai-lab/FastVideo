# SPDX-License-Identifier: Apache-2.0
"""Post-refiner latent capture and Turbo VAE decoding for MAGI-2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

import fastvideo.envs as envs
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.dits.magi2_runtime import psm
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult


class Magi2LatentSavingStage(PipelineStage):
    """Write each post-refiner video latent on the output-producing rank."""

    def __init__(self) -> None:
        """Initialize the per-process sample counter used in latent filenames."""
        super().__init__()
        self.sample_index = 0

    def verify_input(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        """Return the default record for the post-refiner latent."""
        del batch, fastvideo_args
        return VerificationResult()

    def verify_output(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        """Return the default record after the optional filesystem write."""
        del batch, fastvideo_args
        return VerificationResult()

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Save ``latent_N.pt`` when ``MAGI2_SAVE_LATENT_PATH`` is configured."""
        del fastvideo_args
        latent_directory = envs.MAGI2_SAVE_LATENT_PATH
        if not latent_directory or not psm.is_group_first_rank("cp"):
            return batch
        if batch.latents is None:
            raise ValueError("MAGI-2 latent saving requires a post-refiner latent")
        output_directory = Path(latent_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / f"latent_{self.sample_index}.pt"
        torch.save(batch.latents.detach().cpu(), output_path)
        self.sample_index += 1
        return batch


class Magi2VideoDecodingStage(PipelineStage):
    """Decode the 1080p latent with Turbo VAE on one rank per video."""

    performance_component_metric = "vae_decode_time_s"

    def __init__(self, turbo_vae: Any | None) -> None:
        """Store the distilled Turbo VAE decoder loaded from ``ckpt/turbo_vae``."""
        super().__init__()
        self.turbo_vae = turbo_vae

    def verify_input(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        """Return the default record; ``forward`` validates the video latent."""
        del batch, fastvideo_args
        return VerificationResult()

    def verify_output(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        """Return the default record because non-leader ranks intentionally skip decode."""
        del batch, fastvideo_args
        return VerificationResult()

    @torch.inference_mode()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Decode and store a CPU float video tensor in ``[B,C,T,H,W]`` layout."""
        if not psm.is_group_first_rank("cp"):
            batch.output = None
            return batch
        if batch.latents is None:
            raise ValueError("MAGI-2 Turbo VAE decoding requires a video latent")
        if self.turbo_vae is None:
            raise RuntimeError("MAGI-2 requires the distilled Turbo VAE decoder")
        device = torch.device("cuda", torch.cuda.current_device())
        self.turbo_vae.to(device=device, dtype=torch.bfloat16)
        decoder_input = batch.latents.squeeze(0).to(torch.bfloat16)
        if decoder_input.dim() == 4:
            decoder_input = decoder_input.unsqueeze(0)
        decoded_video = self.turbo_vae.decode(decoder_input).float()
        batch.output = decoded_video.mul(0.5).add(0.5).clamp(0, 1).cpu()
        if fastvideo_args.vae_cpu_offload:
            self.turbo_vae.to("cpu")
            torch.cuda.empty_cache()
        return batch


__all__ = ["Magi2LatentSavingStage", "Magi2VideoDecodingStage"]
