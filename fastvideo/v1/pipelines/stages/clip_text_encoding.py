# SPDX-License-Identifier: Apache-2.0
"""
Prompt encoding stages for diffusion pipelines.

This module contains implementations of prompt encoding stages for diffusion pipelines.
"""

from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.logger import init_logger

from ..pipeline_batch_info import ForwardBatch
from .base import PipelineStage

logger = init_logger(__name__)


class CLIPTextEncodingStage(PipelineStage):
    """
    Stage for encoding text prompts into embeddings for diffusion models.
    
    This stage handles the encoding of text prompts into the embedding space
    expected by the diffusion model.
    """

    def __init__(self, text_encoder, tokenizer):
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
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """
        Encode the prompt into text encoder hidden states.
        
        Args:
            batch: The current batch information.
            inference_args: The inference arguments.
            
        Returns:
            The batch with encoded prompt embeddings.
        """

        text_inputs = self.tokenizer(
            batch.prompt,
            truncation=True,
            # better way to handle this?
            max_length=77,
            return_tensors="pt",
        )
        with set_forward_context():
            outputs = self.text_encoder(input_ids=text_inputs["input_ids"].to(
                batch.device), )
        prompt_embeds = outputs["pooler_output"]

        batch.prompt_embeds.append(prompt_embeds)

        return batch
