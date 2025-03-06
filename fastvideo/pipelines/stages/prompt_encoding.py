"""
Prompt encoding stages for diffusion pipelines.

This module contains implementations of prompt encoding stages for diffusion pipelines.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import torch

from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.inference_args import InferenceArgs
from fastvideo.pipelines.stages.base import PromptEncodingStage


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