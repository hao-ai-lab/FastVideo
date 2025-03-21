"""
Prompt encoding stages for diffusion pipelines.

This module contains implementations of prompt encoding stages for diffusion pipelines.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import torch

from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.pipelines.stages import PipelineStage
from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


class PromptEncodingStage(PipelineStage):
    """
    Stage for encoding text prompts into embeddings for diffusion models.
    
    This stage handles the encoding of text prompts into the embedding space
    expected by the diffusion model.
    """
    
    def __init__(self, enable_logging: bool = False, is_secondary: bool = False):
        """
        Initialize the prompt encoding stage.
        
        Args:
            enable_logging: Whether to enable logging for this stage.
            is_secondary: Whether this is a secondary text encoder.
        """
        super().__init__(enable_logging=enable_logging)
        self.is_secondary = is_secondary
        
    def _call_implementation(
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
        if self.is_secondary:
            assert self.text_encoder_2 is not None, "Secondary text encoder is not set"
            text_encoder = self.text_encoder_2
        else:
            text_encoder = self.text_encoder
        
        prompt: Union[str, List[str]] = batch.prompt
        device: torch.device = batch.device
        num_videos_per_prompt: int = batch.num_videos_per_prompt
        data_type: str = batch.data_type
        
        # Get the right prompt embeds and attention masks based on whether this is primary or secondary
        if self.is_secondary:
            prompt_embeds = batch.prompt_embeds_2
            attention_mask = batch.attention_mask_2
        else:
            prompt_embeds = batch.prompt_embeds
            attention_mask = batch.attention_mask

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            # TODO(will-refactor): check this
            # if isinstance(self, TextualInversionLoaderMixin):
            #     prompt = self.maybe_convert_prompt(prompt, text_encoder.tokenizer)

            text_inputs = text_encoder.text2tokens(prompt)
            prompt_outputs = text_encoder.encode(text_inputs, device=device)
            prompt_embeds = prompt_outputs.hidden_state
            # TODO(will): support clip_skip
            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(bs_embed * num_videos_per_prompt, seq_len)

        if text_encoder is not None:
            # TODO(will-refactor): use text_encoder.dtype
            prompt_embeds_dtype = torch.float16
        elif self.transformer is not None:
            prompt_embeds_dtype = self.transformer.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        # TODO(will-refactor): check this. Not needed in v0 due to encoder
        # outputs already in the right shape
        if prompt_embeds.ndim == 1:
            prompt_embeds = prompt_embeds.unsqueeze(0)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # Set the appropriate attributes based on whether this is primary or secondary
        if self.is_secondary:
            batch.prompt_embeds_2 = prompt_embeds
            batch.attention_mask_2 = attention_mask
        else:
            batch.prompt_embeds = prompt_embeds
            batch.attention_mask = attention_mask


        return batch