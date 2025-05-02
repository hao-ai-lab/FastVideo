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


class TextEncodingStage(PipelineStage):
    """
    Stage for encoding text prompts into embeddings for diffusion models.
    
    This stage handles the encoding of text prompts into the embedding space
    expected by the diffusion model.
    """

    def __init__(self, text_encoders, tokenizers) -> None:
        """
        Initialize the prompt encoding stage.
        
        Args:
            enable_logging: Whether to enable logging for this stage.
            is_secondary: Whether this is a secondary text encoder.
        """
        super().__init__()
        self.tokenizers = tokenizers
        self.text_encoders = text_encoders

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
        assert len(self.tokenizers) == len(self.text_encoders)
        assert len(self.text_encoders) == len(
            fastvideo_args.text_encoder_configs)

        for tokenizer, text_encoder, encoder_config, preprocess_func, postprocess_func in zip(
                self.tokenizers, self.text_encoders,
                fastvideo_args.text_encoder_configs,
                fastvideo_args.preprocess_text_funcs,
                fastvideo_args.postprocess_text_funcs):
            if fastvideo_args.use_cpu_offload:
                text_encoder = text_encoder.to(fastvideo_args.device)

            assert isinstance(batch.prompt, str)
            text = preprocess_func(batch.prompt)
            text_inputs = tokenizer(text, **encoder_config.tokenizer_kwargs).to(
                fastvideo_args.device)
            input_ids = text_inputs["input_ids"]
            attention_mask = text_inputs["attention_mask"]
            with set_forward_context(current_timestep=0, attn_metadata=None):
                outputs = text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            prompt_embeds = postprocess_func(outputs)

            batch.prompt_embeds.append(prompt_embeds)

            if batch.do_classifier_free_guidance:
                assert isinstance(batch.negative_prompt, str)
                negative_text = preprocess_func(batch.negative_prompt)
                negative_text_inputs = tokenizer(
                    negative_text,
                    **encoder_config.tokenizer_kwargs).to(fastvideo_args.device)
                negative_input_ids = negative_text_inputs["input_ids"]
                negative_attention_mask = negative_text_inputs["attention_mask"]
                with set_forward_context(current_timestep=0,
                                         attn_metadata=None):
                    negative_outputs = text_encoder(
                        input_ids=negative_input_ids,
                        attention_mask=negative_attention_mask,
                        output_hidden_states=True,
                    )
                negative_prompt_embeds = postprocess_func(negative_outputs)

                assert batch.negative_prompt_embeds is not None
                batch.negative_prompt_embeds.append(negative_prompt_embeds)

            if fastvideo_args.use_cpu_offload:
                text_encoder.to('cpu')
                torch.cuda.empty_cache()

        return batch
