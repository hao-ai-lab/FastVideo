# SPDX-License-Identifier: Apache-2.0
"""
Input validation stage for diffusion pipelines.
"""

import torch

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.vision_utils import load_image
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage
from fastvideo.v1.pipelines.stages.validators import (StageValidators,
                                                      VerificationResult)

logger = init_logger(__name__)

# Alias for convenience
V = StageValidators


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

        # for i2v, get image from image_path
        if batch.image_path is not None:
            image = load_image(batch.image_path)
            batch.pil_image = image

        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify input validation stage inputs."""
        result = VerificationResult()
        result.add_check("seed", batch.seed, [V.not_none, V.positive_int])
        result.add_check("num_videos_per_prompt", batch.num_videos_per_prompt,
                         V.positive_int)
        result.add_check(
            "prompt_or_embeds", None, lambda _: V.string_or_list_strings(
                batch.prompt) or V.list_not_empty(batch.prompt_embeds))
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("num_inference_steps", batch.num_inference_steps,
                         V.positive_int)
        result.add_check(
            "guidance_scale", batch.guidance_scale, lambda x: not batch.
            do_classifier_free_guidance or V.positive_float(x))
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify input validation stage outputs."""
        result = VerificationResult()
        result.add_check("seeds", batch.seeds, V.list_not_empty)
        result.add_check("generator", batch.generator,
                         V.generator_or_list_generators)
        return result
