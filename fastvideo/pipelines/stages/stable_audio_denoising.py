# SPDX-License-Identifier: Apache-2.0
"""
Stable Audio denoising stage using k-diffusion v-prediction sampling.
"""
from __future__ import annotations

from typing import Any

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage

logger = init_logger(__name__)


class StableAudioDenoisingStage(PipelineStage):
    """Run k-diffusion v-prediction sampling for Stable Audio."""

    def __init__(self, transformer) -> None:
        super().__init__()
        self.transformer = transformer

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        latents = batch.latents
        if latents is None:
            raise ValueError("Latents must be provided before denoising.")

        conditioning = batch.extra.get("stable_audio_conditioning")
        if conditioning is None:
            raise ValueError(
                "Conditioning must be set by StableAudioConditioningStage.")

        cfg_scale = batch.guidance_scale or 6.0
        steps = batch.num_inference_steps or 250

        cross_attn = conditioning.get("prompt", (None, None))
        cross_attn_cond, cross_attn_mask = cross_attn[0], cross_attn[1]
        global_cond = batch.extra.get("stable_audio_global_cond")

        cond_inputs = {
            "cross_attn_cond": cross_attn_cond,
            "cross_attn_cond_mask": cross_attn_mask,
            "global_embed": global_cond,
        }

        device = latents.device

        def _to_device(x: Any) -> Any:
            if x is None:
                return x
            if torch.is_tensor(x):
                return x.to(device)
            return x

        cond_inputs = {k: _to_device(v) for k, v in cond_inputs.items()}

        from fastvideo.models.stable_audio.sampling import sample_stable_audio

        sampled = sample_stable_audio(
            self.transformer.model,
            batch.latents,
            steps=steps,
            device=device,
            cfg_scale=cfg_scale,
            **cond_inputs,
        )

        batch.latents = sampled
        logger.info("[StableAudio] Denoising done: steps=%d", steps)
        return batch
