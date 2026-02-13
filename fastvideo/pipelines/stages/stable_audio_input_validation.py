# SPDX-License-Identifier: Apache-2.0
"""
Stable Audio input validation: audio-specific checks.
"""
from __future__ import annotations

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.input_validation import InputValidationStage


class StableAudioInputValidationStage(InputValidationStage):
    """Input validation for Stable Audio: uses duration instead of height/width."""

    def _generate_seeds(self, batch: ForwardBatch,
                        fastvideo_args: FastVideoArgs):
        seed = batch.seed
        num_videos_per_prompt = batch.num_videos_per_prompt
        assert seed is not None
        batch.seeds = [seed + i for i in range(num_videos_per_prompt)]
        batch.generator = [
            torch.Generator("cpu").manual_seed(s) for s in batch.seeds
        ]

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        self._generate_seeds(batch, fastvideo_args)

        if batch.prompt is None and batch.prompt_embeds is None:
            raise ValueError(
                "Either `prompt` or `prompt_embeds` must be provided")

        if batch.num_inference_steps <= 0:
            raise ValueError(
                f"num_inference_steps must be positive, got {batch.num_inference_steps}"
            )

        if batch.do_classifier_free_guidance and batch.guidance_scale <= 0:
            raise ValueError(
                f"guidance_scale must be positive when using CFG, got {batch.guidance_scale}"
            )

        duration = batch.duration_seconds or 10.0
        batch.extra["stable_audio_duration"] = duration
        return batch
