# SPDX-License-Identifier: Apache-2.0
"""
Conditioning stage for diffusion pipelines.
"""

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult


class ConditioningStage(PipelineStage):
    """
    Stage for applying conditioning to the diffusion process.
    
    This stage handles the application of conditioning, such as classifier-free guidance,
    to the diffusion process.
    """

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Apply conditioning to the diffusion process.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with applied conditioning.
        """
        # Forward is a no-op: CFG is applied via two separate
        # transformer forward passes inside DenoisingStage (e.g.
        # denoising.py:364-394, :706, :930). The class is kept because
        # verify_input / verify_output still validate CFG fields and
        # disable CFG when prompt_embeds is empty.
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify conditioning stage inputs."""
        result = VerificationResult()
        if not batch.prompt_embeds:
            # No text encoder/prompt embeddings: skip checks and effectively disable CFG.
            batch.do_classifier_free_guidance = False
            return result
        result.add_check("do_classifier_free_guidance", batch.do_classifier_free_guidance, V.bool_value)
        result.add_check("guidance_scale", batch.guidance_scale, V.positive_float)
        # Matrix-Game allow empty prompt
        # embeddings when CFG isn't enabled.
        if batch.do_classifier_free_guidance or batch.prompt_embeds:
            result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
            result.add_check("negative_prompt_embeds", batch.negative_prompt_embeds,
                             lambda x: not batch.do_classifier_free_guidance or V.list_not_empty(x))
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify conditioning stage outputs."""
        result = VerificationResult()
        if batch.prompt_embeds is None or not batch.prompt_embeds:
            batch.do_classifier_free_guidance = False
            return result
        if batch.do_classifier_free_guidance or batch.prompt_embeds:
            result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        return result
