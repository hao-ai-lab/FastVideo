# SPDX-License-Identifier: Apache-2.0
"""Video-VAE keyframe conditioning for MiniMax H3 FL2VA."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLMiniMaxH3
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    keyframe_condition_noise,
    patchify_video_latents,
)
from fastvideo.pipelines.basic.minimax_h3.stages._module_lifecycle import (
    maybe_offload_module,
    module_device,
    move_module_to_local_device,
)
from fastvideo.pipelines.basic.minimax_h3.stages._vae_conditioning import arch_value, latent_stats, sample_posterior
from fastvideo.pipelines.basic.minimax_h3.types import get_minimax_h3_state
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult


class MiniMaxH3KeyframeEncodingStage(PipelineStage):
    """Encode first/last keyframes and draw their request-seeded condition noise."""

    def __init__(self, vae: AutoencoderKLMiniMaxH3, transformer: Any, scheduler: Any) -> None:
        super().__init__()
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        state = get_minimax_h3_state(batch)
        result = VerificationResult()
        result.add_check("keyframes", state.keyframes, V.is_list)
        result.add_check("latent_height", state.latent_height, V.positive_int)
        result.add_check("latent_width", state.latent_width, V.positive_int)
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        state = get_minimax_h3_state(batch)
        result = VerificationResult()
        result.add_check("condition_video_latents", state.condition_video_latents, V.none_or_tensor)
        return result

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        state = get_minimax_h3_state(batch)
        if not state.keyframes:
            state.condition_video_latents = None
            return batch
        if state.latent_height is None or state.latent_width is None:
            raise ValueError("MiniMax-H3 input geometry must be prepared before keyframe encoding.")

        patch_size = tuple(int(value) for value in arch_value(self.transformer, "patch_size"))
        latent_channels = int(arch_value(self.vae, "latent_channels"))
        self.vae, vae_device, _ = move_module_to_local_device(self.vae)
        try:
            latents_mean, latents_std = latent_stats(self.vae, (1, -1, 1, 1, 1))
            clean_rows = []
            for image in state.keyframes:
                pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None, :, None]
                pixels = pixels.to(device=vae_device, dtype=torch.float32).div_(255.0)
                pixels = self.vae.normalize_pixels(pixels)
                posterior = self.vae.encode_keyframe(pixels).latent_dist
                latents = sample_posterior(posterior, MINIMAX_H3_KEYFRAME_ENCODE_SEED)
                latents = latents.to(torch.float16).float().cpu()
                clean_rows.append(patchify_video_latents((latents - latents_mean) / latents_std, patch_size))
        finally:
            self.vae = maybe_offload_module(
                self.vae,
                enabled=bool(getattr(fastvideo_args, "vae_cpu_offload", False)),
            )

        transformer_device = module_device(self.transformer)
        condition_noise = keyframe_condition_noise(
            ((1, state.latent_height, state.latent_width), ) * len(state.keyframes),
            patch_size,
            latent_channels,
            generator=batch.generator,
            device=transformer_device,
        )
        state.condition_video_latents = self.scheduler.scale_noise(
            torch.cat(clean_rows).to(transformer_device),
            MINIMAX_H3_KEYFRAME_NOISE_AUG,
            condition_noise,
        )
        return batch


__all__ = ["MiniMaxH3KeyframeEncodingStage"]
