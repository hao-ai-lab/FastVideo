# SPDX-License-Identifier: Apache-2.0
"""
Prompt encoding stages for diffusion pipelines.

This module contains implementations of prompt encoding stages for diffusion pipelines.
"""
import torch

from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage

logger = init_logger(__name__)

class T5EncodingStage(PipelineStage):
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
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

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

        text = batch.prompt
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(batch.device)
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=mask,
            )
        assert torch.isnan(outputs).sum() == 0
        prompt_embeds = outputs[:, :seq_lens, :].to(batch.device)
        batch.prompt_embeds.append(prompt_embeds)

        if batch.do_classifier_free_guidance:
            negative_text = batch.negative_prompt
            negative_text_inputs = self.tokenizer(
                negative_text,
                padding="max_length",
                truncation=True,
                max_length=512,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(batch.device)
            text_input_ids, mask = negative_text_inputs.input_ids, negative_text_inputs.attention_mask
            seq_lens = mask.gt(0).sum(dim=1).long()
            with set_forward_context(current_timestep=0, attn_metadata=None):
                negative_outputs = self.text_encoder(
                    input_ids=text_input_ids,
                    attention_mask=mask,
                )
            assert torch.isnan(negative_outputs).sum() == 0
            neg_prompt_embeds = negative_outputs[:, :seq_lens, :].to(batch.device)
            batch.negative_prompt_embeds.append(neg_prompt_embeds)

        return batch
