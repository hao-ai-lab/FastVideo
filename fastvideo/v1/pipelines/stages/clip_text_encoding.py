# SPDX-License-Identifier: Apache-2.0
"""
Prompt encoding stages for diffusion pipelines.

This module contains implementations of prompt encoding stages for diffusion pipelines.
"""

import torch

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage

logger = init_logger(__name__)


class CLIPTextEncodingStage(PipelineStage):
    """
    Stage for encoding text prompts into embeddings for diffusion models.
    
    This stage handles the encoding of text prompts into the embedding space
    expected by the diffusion model.
    """

    def __init__(self, text_encoder, tokenizer) -> None:
        """
        Initialize the prompt encoding stage.
        
        Args:
            enable_logging: Whether to enable logging for this stage.
            is_secondary: Whether this is a secondary text encoder.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Encode the prompt into text encoder hidden states.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with encoded prompt embeddings.
        """
        if fastvideo_args.use_cpu_offload:
            self.text_encoder = self.text_encoder.to(batch.device)

        text_inputs = self.tokenizer(
            batch.prompt,
            truncation=True,
            # better way to handle this?
            max_length=77,
            return_tensors="pt",
        )
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = self.text_encoder(input_ids=text_inputs["input_ids"].to(
                batch.device), )
        prompt_embeds = outputs["pooler_output"]

        batch.prompt_embeds.append(prompt_embeds)

        if batch.do_classifier_free_guidance:
            negative_text_inputs = self.tokenizer(
                batch.negative_prompt,
                truncation=True,
                # better way to handle this?
                max_length=77,
                return_tensors="pt",
            )
            with set_forward_context(current_timestep=0, attn_metadata=None):
                negative_outputs = self.text_encoder(
                    input_ids=negative_text_inputs["input_ids"].to(
                        batch.device), )
            negative_prompt_embeds = negative_outputs["pooler_output"]

            assert batch.negative_prompt_embeds is not None
            batch.negative_prompt_embeds.append(negative_prompt_embeds)

        if fastvideo_args.use_cpu_offload:
            self.text_encoder.to('cpu')
            torch.cuda.empty_cache()

        return batch
