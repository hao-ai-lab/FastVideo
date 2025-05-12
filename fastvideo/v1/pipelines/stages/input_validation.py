# SPDX-License-Identifier: Apache-2.0
"""
Input validation stage for diffusion pipelines.
"""

import math

import torch

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage

logger = init_logger(__name__)


class InputValidationStage(PipelineStage):
    """
    Stage for validating and preparing inputs for diffusion pipelines.
    
    This stage validates that all required inputs are present and properly formatted
    before proceeding with the diffusion process.
    """

    def _generate_seeds(self, batch: ForwardBatch,
                        fastvideo_args: FastVideoArgs):
        """Generate seeds for the inference"""
        seed = batch.seed
        num_videos_per_prompt = batch.num_videos_per_prompt

        assert seed is not None
        seeds = [seed + i for i in range(num_videos_per_prompt)]
        batch.seeds = seeds
        # Peiyuan: using GPU seed will cause A100 and H100 to generate different results...
        batch.generator = [
            torch.Generator("cpu").manual_seed(seed) for seed in seeds
        ]

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Validate and prepare inputs.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The validated batch information.
        """
        self._generate_seeds(batch, fastvideo_args)

        # Ensure prompt is properly formatted
        if batch.prompt is None and batch.prompt_embeds is None:
            raise ValueError(
                "Either `prompt` or `prompt_embeds` must be provided")

        # Ensure negative prompt is properly formatted if using classifier-free guidance
        if (batch.do_classifier_free_guidance and batch.negative_prompt is None
                and batch.negative_prompt_embeds is None):
            raise ValueError(
                "For classifier-free guidance, either `negative_prompt` or "
                "`negative_prompt_embeds` must be provided")

        # Validate height and width
        if batch.height is None or batch.width is None:
            raise ValueError(
                "Height and width must be provided. Please set `height` and `width`."
            )
        if batch.height % 8 != 0 or batch.width % 8 != 0:
            raise ValueError(
                f"Height and width must be divisible by 8 but are {batch.height} and {batch.width}."
            )

        # Validate number of inference steps
        if batch.num_inference_steps <= 0:
            raise ValueError(
                f"Number of inference steps must be positive, but got {batch.num_inference_steps}"
            )

        # Validate guidance scale if using classifier-free guidance
        if batch.do_classifier_free_guidance and batch.guidance_scale <= 0:
            raise ValueError(
                f"Guidance scale must be positive, but got {batch.guidance_scale}"
            )

        # Adjust number of frames based on number of GPUs
        temporal_scale_factor = fastvideo_args.vae_config.arch_config.temporal_compression_ratio
        orig_latent_num_frames = (batch.num_frames -
                                  1) // temporal_scale_factor + 1
        if orig_latent_num_frames % fastvideo_args.num_gpus != 0:
            if batch.num_frames_round_down:
                new_latent_num_frames = math.floor(
                    orig_latent_num_frames /
                    fastvideo_args.num_gpus) * fastvideo_args.num_gpus
            else:
                new_latent_num_frames = math.ceil(
                    orig_latent_num_frames /
                    fastvideo_args.num_gpus) * fastvideo_args.num_gpus
            new_num_frames = (new_latent_num_frames -
                              1) * temporal_scale_factor + 1
            logger.info(
                "Adjusting number of frames from %s to %s based on number of GPUs (%s)",
                batch.num_frames, new_num_frames, fastvideo_args.num_gpus)
            batch.num_frames = new_num_frames

        return batch
