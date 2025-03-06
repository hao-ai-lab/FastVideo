"""
Prompt encoding stages for diffusion pipelines.

This module contains implementations of prompt encoding stages for diffusion pipelines.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import torch

from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.inference_args import InferenceArgs
from fastvideo.pipelines.stages.base import PromptEncodingStage
from fastvideo.models.hunyuan.text_encoder import TextEncoder
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class PromptEncodingStage(PromptEncodingStage):
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
            # Process text inputs
            text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)
            prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type, device=device)
            prompt_embeds = prompt_outputs.hidden_state

            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(bs_embed * num_videos_per_prompt, seq_len)

        # Determine the correct dtype for prompt embeddings
        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype
        elif hasattr(self, 'transformer') and self.transformer is not None:
            prompt_embeds_dtype = self.transformer.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        # Duplicate embeddings for each generation per prompt
        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # Set the appropriate attributes based on whether this is primary or secondary
        if self.is_secondary:
            batch.prompt_embeds_2 = prompt_embeds
            batch.attention_mask_2 = attention_mask
        else:
            batch.prompt_embeds = prompt_embeds
            batch.attention_mask = attention_mask

        # If this is the primary encoder and we need negative prompts
        if not self.is_secondary and batch.do_classifier_free_guidance and batch.negative_prompt_embeds is None:
            # Process negative prompt
            negative_prompt = batch.negative_prompt if batch.negative_prompt is not None else ""
            negative_text_inputs = text_encoder.text2tokens(negative_prompt, data_type=data_type)
            negative_prompt_outputs = text_encoder.encode(negative_text_inputs, data_type=data_type, device=device)
            batch.negative_prompt_embeds = negative_prompt_outputs.hidden_state
            batch.negative_attention_mask = negative_prompt_outputs.attention_mask
            
            # Format negative embeddings
            if batch.negative_prompt_embeds.ndim == 2:
                bs_embed, _ = batch.negative_prompt_embeds.shape
                batch.negative_prompt_embeds = batch.negative_prompt_embeds.repeat(1, num_videos_per_prompt)
                batch.negative_prompt_embeds = batch.negative_prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
            else:
                bs_embed, seq_len, _ = batch.negative_prompt_embeds.shape
                batch.negative_prompt_embeds = batch.negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
                batch.negative_prompt_embeds = batch.negative_prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        return batch


class StandardPromptEncodingStage(PromptEncodingStage):
    """
    Standard prompt encoding stage for diffusion pipelines.
    
    This stage encodes the prompt(s) using a text encoder.
    """
    
    needs_text_encoder = True
    needs_tokenizer = True
    
    def __init__(self, max_length: int = 77):
        """
        Initialize the prompt encoding stage.
        
        Args:
            max_length: The maximum length of the prompt.
        """
        super().__init__()
        self.max_length = max_length
    
    def __call__(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """
        Encode the prompt(s).
        
        Args:
            batch: The current batch information.
            inference_args: The inference arguments.
            
        Returns:
            The updated batch information after prompt encoding.
        """
        # Skip if prompt is already processed
        if batch.is_prompt_processed:
            return batch
        
        # Get the prompt(s)
        prompt = batch.prompt
        if prompt is None:
            prompt = [""] * batch.batch_size
        elif isinstance(prompt, str):
            prompt = [prompt]
        
        # Get the negative prompt(s)
        negative_prompt = batch.negative_prompt
        if negative_prompt is None:
            negative_prompt = [""] * len(prompt)
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        
        # Ensure the number of negative prompts matches the number of prompts
        if len(negative_prompt) != len(prompt):
            negative_prompt = negative_prompt * len(prompt)
        
        # Tokenize the prompt(s)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        # Tokenize the negative prompt(s)
        negative_text_inputs = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_text_input_ids = negative_text_inputs.input_ids.to(self.device)
        
        # Encode the prompt(s)
        prompt_embeds = self.text_encoder(text_input_ids)[0]
        
        # Encode the negative prompt(s)
        negative_prompt_embeds = self.text_encoder(negative_text_input_ids)[0]
        
        # Update the batch
        batch.prompt_embeds = prompt_embeds
        batch.negative_prompt_embeds = negative_prompt_embeds
        batch.is_prompt_processed = True
        
        return batch


class DualEncoderPromptEncodingStage(PromptEncodingStage):
    """
    Dual encoder prompt encoding stage for diffusion pipelines.
    
    This stage encodes the prompt(s) using two text encoders.
    """
    
    needs_text_encoder = True
    needs_tokenizer = True
    needs_text_encoder_2 = True
    needs_tokenizer_2 = True
    
    def __init__(self, max_length: int = 77, max_length_2: int = 77):
        """
        Initialize the dual encoder prompt encoding stage.
        
        Args:
            max_length: The maximum length of the prompt for the first encoder.
            max_length_2: The maximum length of the prompt for the second encoder.
        """
        super().__init__()
        self.max_length = max_length
        self.max_length_2 = max_length_2
    
    def __call__(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """
        Encode the prompt(s) using two text encoders.
        
        Args:
            batch: The current batch information.
            inference_args: The inference arguments.
            
        Returns:
            The updated batch information after prompt encoding.
        """
        # Skip if prompt is already processed
        if batch.is_prompt_processed:
            return batch
        
        # Get the prompt(s)
        prompt = batch.prompt
        if prompt is None:
            prompt = [""] * batch.batch_size
        elif isinstance(prompt, str):
            prompt = [prompt]
        
        # Get the negative prompt(s)
        negative_prompt = batch.negative_prompt
        if negative_prompt is None:
            negative_prompt = [""] * len(prompt)
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        
        # Ensure the number of negative prompts matches the number of prompts
        if len(negative_prompt) != len(prompt):
            negative_prompt = negative_prompt * len(prompt)
        
        # Tokenize the prompt(s) for the first encoder
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        # Tokenize the negative prompt(s) for the first encoder
        negative_text_inputs = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_text_input_ids = negative_text_inputs.input_ids.to(self.device)
        
        # Encode the prompt(s) with the first encoder
        prompt_embeds = self.text_encoder(text_input_ids)[0]
        
        # Encode the negative prompt(s) with the first encoder
        negative_prompt_embeds = self.text_encoder(negative_text_input_ids)[0]
        
        # Tokenize the prompt(s) for the second encoder
        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.max_length_2,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2.input_ids.to(self.device)
        
        # Tokenize the negative prompt(s) for the second encoder
        negative_text_inputs_2 = self.tokenizer_2(
            negative_prompt,
            padding="max_length",
            max_length=self.max_length_2,
            truncation=True,
            return_tensors="pt",
        )
        negative_text_input_ids_2 = negative_text_inputs_2.input_ids.to(self.device)
        
        # Encode the prompt(s) with the second encoder
        prompt_embeds_2 = self.text_encoder_2(text_input_ids_2)[0]
        
        # Encode the negative prompt(s) with the second encoder
        negative_prompt_embeds_2 = self.text_encoder_2(negative_text_input_ids_2)[0]
        
        # Update the batch
        batch.prompt_embeds = prompt_embeds
        batch.negative_prompt_embeds = negative_prompt_embeds
        batch.prompt_embeds_2 = prompt_embeds_2
        batch.negative_prompt_embeds_2 = negative_prompt_embeds_2
        batch.is_prompt_processed = True
        
        return batch 