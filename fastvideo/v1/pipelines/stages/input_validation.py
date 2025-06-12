# SPDX-License-Identifier: Apache-2.0
"""
Input validation stage for diffusion pipelines.
"""

from typing import Optional

import torch
from fastvideo.v1.models.vision_utils import (load_image, pil_to_numpy,
                                              numpy_to_pt, normalize,
                                              get_default_height_width,
                                              resize)
from fastvideo.v1.distributed import get_torch_device
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage
import PIL.Image



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
        
        # for i2v, get image from image_path 
        if batch.image_path is not None:
            image = load_image(batch.image_path)
            batch.pil_image = image

            # image = self.preprocess_image(
            #     image,
            #     vae_scale_factor=self.vae.spatial_compression_ratio,
            #     height=batch.height,
            #     width=batch.width).to(get_torch_device(), dtype=torch.float32)
            # image = image.unsqueeze(2)
            # batch.preprocessed_image = image

        return batch

    def preprocess_image(
            self,
            image: PIL.Image.Image,
            vae_scale_factor: int,
            height: Optional[int] = None,
            width: Optional[int] = None,
            resize_mode: str = "default",  # "default", "fill", "crop"
    ) -> torch.Tensor:
        image = [image]

        height, width = get_default_height_width(image[0], vae_scale_factor,
                                                 height, width)
        image = [
            resize(i, height, width, resize_mode=resize_mode) for i in image
        ]
        image = pil_to_numpy(image)  # to np
        image = numpy_to_pt(image)  # to pt

        do_normalize = True
        if image.min() < 0:
            do_normalize = False
        if do_normalize:
            image = normalize(image)

        return image