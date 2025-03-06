"""
Input validation stage for diffusion pipelines.
"""

from typing import Optional, Union, List
import torch

from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.inference_args import InferenceArgs
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class InputValidationStage(PipelineStage):
    """
    Stage for validating and preparing inputs for diffusion pipelines.
    
    This stage validates that all required inputs are present and properly formatted
    before proceeding with the diffusion process.
    """
    
    def _call_implementation(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """
        Validate and prepare inputs.
        
        Args:
            batch: The current batch information.
            inference_args: The inference arguments.
            
        Returns:
            The validated batch information.
        """
        # Ensure prompt is properly formatted
        if batch.prompt is None and batch.prompt_embeds is None:
            raise ValueError("Either `prompt` or `prompt_embeds` must be provided")
            
        # Ensure negative prompt is properly formatted if using classifier-free guidance
        if batch.do_classifier_free_guidance:
            if batch.negative_prompt is None and batch.negative_prompt_embeds is None:
                raise ValueError(
                    "For classifier-free guidance, either `negative_prompt` or "
                    "`negative_prompt_embeds` must be provided"
                )
        
        # Validate height and width
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
            
        # Set device if not already set
        if batch.device is None:
            batch.device = self.device
            
        # Set data type if not already set
        if batch.data_type is None:
            batch.data_type = inference_args.precision
            
        return batch 