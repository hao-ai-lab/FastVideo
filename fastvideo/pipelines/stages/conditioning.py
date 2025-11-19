# SPDX-License-Identifier: Apache-2.0
"""
Conditioning stage for diffusion pipelines.
"""

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)


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
        # TODO!!
        if not batch.do_classifier_free_guidance:
            return batch
        else:
            return batch

        logger.info("batch.negative_prompt_embeds: %s",
                    batch.negative_prompt_embeds)
        logger.info("do_classifier_free_guidance: %s",
                    batch.do_classifier_free_guidance)
        logger.info("cfg_scale: %s", batch.guidance_scale)

        # Ensure negative prompt embeddings are available
        assert batch.negative_prompt_embeds is not None, (
            "Negative prompt embeddings are required for classifier-free guidance"
        )

        # Concatenate primary embeddings and masks
        batch.prompt_embeds = torch.cat(
            [batch.negative_prompt_embeds, batch.prompt_embeds])
        if batch.attention_mask is not None:
            batch.attention_mask = torch.cat(
                [batch.negative_attention_mask, batch.attention_mask])

        # Concatenate secondary embeddings and masks if present
        if batch.prompt_embeds_2 is not None:
            batch.prompt_embeds_2 = torch.cat(
                [batch.negative_prompt_embeds_2, batch.prompt_embeds_2])
        if batch.attention_mask_2 is not None:
            batch.attention_mask_2 = torch.cat(
                [batch.negative_attention_mask_2, batch.attention_mask_2])

        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify conditioning stage inputs."""
        result = VerificationResult()
        if not batch.prompt_embeds:
            # No text encoder/prompt embeddings: skip checks and effectively disable CFG.
            batch.do_classifier_free_guidance = False
            return result
        result.add_check("do_classifier_free_guidance",
                         batch.do_classifier_free_guidance, V.bool_value)
        result.add_check("guidance_scale", batch.guidance_scale,
                         V.positive_float)
        # Matrix-Game allow empty prompt
        # embeddings when CFG isn't enabled.
        if batch.do_classifier_free_guidance or batch.prompt_embeds:
            result.add_check("prompt_embeds", batch.prompt_embeds,
                             V.list_not_empty)
            result.add_check(
                "negative_prompt_embeds", batch.negative_prompt_embeds,
                lambda x: not batch.do_classifier_free_guidance
                or V.list_not_empty(x))
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify conditioning stage outputs."""
        result = VerificationResult()
        if batch.prompt_embeds is None or not batch.prompt_embeds:
            batch.do_classifier_free_guidance = False
            return result
        if batch.do_classifier_free_guidance or batch.prompt_embeds:
            result.add_check("prompt_embeds", batch.prompt_embeds,
                             V.list_not_empty)
        return result
