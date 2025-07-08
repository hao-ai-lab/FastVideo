# SPDX-License-Identifier: Apache-2.0
"""
Prompt encoding stages for diffusion pipelines.

This module contains implementations of prompt encoding stages for diffusion pipelines.
"""

import weakref

from fastvideo.v1.distributed import get_local_torch_device
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage
from fastvideo.v1.pipelines.stages.validators import StageValidators as V
from fastvideo.v1.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)


class TextEncodingStage(PipelineStage):
    """
    Stage for encoding text prompts into embeddings for diffusion models.
    
    This stage handles the encoding of text prompts into the embedding space
    expected by the diffusion model.
    """

    def __init__(self, text_encoders, tokenizers, pipeline=None) -> None:
        """
        Initialize the prompt encoding stage.
        
        Args:
            enable_logging: Whether to enable logging for this stage.
            is_secondary: Whether this is a secondary text encoder.
        """
        super().__init__()
        self.tokenizers = tokenizers
        self.text_encoders = text_encoders
        self.pipeline = weakref.ref(pipeline) if pipeline else None

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

        # if not fastvideo_args.model_loaded["text_encoder"]:
        #     text_encoder_loader = TextEncoderLoader()
        #     self.text_encoder = text_encoder_loader.load(
        #         fastvideo_args.model_paths["text_encoder"], fastvideo_args)
        #     pipeline = self.pipeline() if self.pipeline else None
        #     if pipeline:
        #         pipeline.add_module("text_encoder", self.text_encoder)
        #     fastvideo_args.model_loaded["text_encoder"] = True

        # if not fastvideo_args.model_loaded["tokenizer"]:
        #     tokenizer_loader = TokenizerLoader()
        #     self.tokenizer = tokenizer_loader.load(
        #         fastvideo_args.model_paths["tokenizer"], fastvideo_args)
        #     pipeline = self.pipeline() if self.pipeline else None
        #     if pipeline:
        #         pipeline.add_module("tokenizer", self.tokenizer)
        #     fastvideo_args.model_loaded["tokenizer"] = True

        assert len(self.tokenizers) == len(self.text_encoders)
        assert len(self.text_encoders) == len(
            fastvideo_args.pipeline_config.text_encoder_configs)

        for tokenizer, text_encoder, encoder_config, preprocess_func, postprocess_func in zip(
                self.tokenizers,
                self.text_encoders,
                fastvideo_args.pipeline_config.text_encoder_configs,
                fastvideo_args.pipeline_config.preprocess_text_funcs,
                fastvideo_args.pipeline_config.postprocess_text_funcs,
                strict=True):

            assert isinstance(batch.prompt, str | list)
            if isinstance(batch.prompt, str):
                batch.prompt = [batch.prompt]
            texts = []
            for prompt_str in batch.prompt:
                texts.append(preprocess_func(prompt_str))
            text_inputs = tokenizer(texts,
                                    **encoder_config.tokenizer_kwargs).to(
                                        get_local_torch_device())
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
            if batch.prompt_attention_mask is not None:
                batch.prompt_attention_mask.append(attention_mask)

            if batch.do_classifier_free_guidance:
                assert isinstance(batch.negative_prompt, str)
                negative_text = preprocess_func(batch.negative_prompt)
                negative_text_inputs = tokenizer(
                    negative_text, **encoder_config.tokenizer_kwargs).to(
                        get_local_torch_device())
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
                if batch.negative_attention_mask is not None:
                    batch.negative_attention_mask.append(
                        negative_attention_mask)

            # deallocate text encoder and tokenizer if on mps
            # if torch.backends.mps.is_available():
            #     logger.info(
            #         "Memory before deallocating text encoder and tokenizer: %s",
            #         torch.mps.current_allocated_memory())
            #     del text_encoder
            #     pipeline = self.pipeline() if self.pipeline else None
            #     if pipeline is not None and "text_encoder" in pipeline.modules:
            #         del pipeline.modules["text_encoder"]
            #     del tokenizer
            #     if pipeline is not None and "tokenizer" in pipeline.modules:
            #         del pipeline.modules["tokenizer"]
            #     gc.collect()
            #     torch.mps.empty_cache()
            #     logger.info(
            #         "Memory after deallocating text encoder and tokenizer: %s",
            #         torch.mps.current_allocated_memory())
            #     fastvideo_args.model_loaded["text_encoder"] = False
            #     fastvideo_args.model_loaded["tokenizer"] = False

        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify text encoding stage inputs."""
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, V.string_or_list_strings)
        result.add_check(
            "negative_prompt", batch.negative_prompt, lambda x: not batch.
            do_classifier_free_guidance or V.string_not_empty(x))
        result.add_check("do_classifier_free_guidance",
                         batch.do_classifier_free_guidance, V.bool_value)
        result.add_check("prompt_embeds", batch.prompt_embeds, V.is_list)
        result.add_check("negative_prompt_embeds", batch.negative_prompt_embeds,
                         V.none_or_list)
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify text encoding stage outputs."""
        result = VerificationResult()
        result.add_check("prompt_embeds", batch.prompt_embeds,
                         V.list_of_tensors_min_dims(2))
        result.add_check(
            "negative_prompt_embeds", batch.negative_prompt_embeds,
            lambda x: not batch.do_classifier_free_guidance or V.
            list_of_tensors_with_min_dims(x, 2))
        return result
