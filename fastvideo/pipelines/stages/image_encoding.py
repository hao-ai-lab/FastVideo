# SPDX-License-Identifier: Apache-2.0
"""
Image encoding stages for I2V diffusion pipelines.

This module contains implementations of image encoding stages for diffusion pipelines.
"""

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)


class ImageEncodingStage(PipelineStage):
    """
    Stage for encoding image prompts into embeddings for diffusion models.
    
    This stage handles the encoding of image prompts into the embedding space
    expected by the diffusion model.
    """

    def __init__(self, image_encoder, image_processor) -> None:
        """
        Initialize the prompt encoding stage.
        
        Args:
            enable_logging: Whether to enable logging for this stage.
            is_secondary: Whether this is a secondary image encoder.
        """
        super().__init__()
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Encode the prompt into image encoder hidden states.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with encoded prompt embeddings.
        """
        self.image_encoder = self.image_encoder.to(get_local_torch_device())

        image = batch.pil_image

        image_inputs = self.image_processor(
            images=image, return_tensors="pt").to(get_local_torch_device())
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = self.image_encoder(**image_inputs)
            image_embeds = outputs.last_hidden_state

        batch.image_embeds.append(image_embeds)

        if fastvideo_args.image_encoder_cpu_offload:
            self.image_encoder.to('cpu')

        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify image encoding stage inputs."""
        result = VerificationResult()
        result.add_check("pil_image", batch.pil_image, V.not_none)
        result.add_check("image_embeds", batch.image_embeds, V.is_list)
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify image encoding stage outputs."""
        result = VerificationResult()
        result.add_check("image_embeds", batch.image_embeds,
                         V.list_of_tensors_dims(3))
        return result
