# SPDX-License-Identifier: Apache-2.0
"""
Prompt encoding stages for diffusion pipelines.

This module contains implementations of prompt encoding stages for diffusion pipelines.
"""

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

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

    @torch.no_grad()
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
            fastvideo_args.pipeline_config.text_encoder_configs)

        # Encode positive prompt with all available encoders
        assert batch.prompt is not None
        prompt_text: str | list[str] = batch.prompt
        all_indices: list[int] = list(range(len(self.text_encoders)))
        prompt_embeds_list, prompt_masks_list = self.encode_text(
            prompt_text,
            fastvideo_args,
            encoder_index=all_indices,
            return_attention_mask=True,
        )

        for pe in prompt_embeds_list:
            batch.prompt_embeds.append(pe)
        if batch.prompt_attention_mask is not None:
            for am in prompt_masks_list:
                batch.prompt_attention_mask.append(am)

        # Encode negative prompt if CFG is enabled
        if batch.do_classifier_free_guidance:
            assert isinstance(batch.negative_prompt, str)
            neg_embeds_list, neg_masks_list = self.encode_text(
                batch.negative_prompt,
                fastvideo_args,
                encoder_index=all_indices,
                return_attention_mask=True,
            )

            assert batch.negative_prompt_embeds is not None
            for ne in neg_embeds_list:
                batch.negative_prompt_embeds.append(ne)
            if batch.negative_attention_mask is not None:
                for nm in neg_masks_list:
                    batch.negative_attention_mask.append(nm)

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

    @torch.no_grad()
    def encode_text(
        self,
        text: str | list[str],
        fastvideo_args: FastVideoArgs,
        encoder_index: int | list[int] | None = None,
        return_attention_mask: bool = False,
        return_type: str = "list",  # one of: "list", "dict", "stack"
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        max_length: int | None = None,
        truncation: bool | None = None,
        padding: bool | str | None = None,
    ):
        """
        Encode plain text using selected text encoder(s) and return embeddings.

        Args:
            text: A single string or a list of strings to encode.
            fastvideo_args: The inference arguments providing pipeline config,
                including tokenizer and encoder settings, preprocess and postprocess
                functions.
            encoder_index: Encoder selector by index. Accepts an int or list of ints.
            return_attention_mask: If True, also return attention masks for each
                selected encoder.
            return_type: "list" (default) returns a list aligned with selection;
                "dict" returns a dict keyed by encoder index as a string; "stack" stacks along a
                new first dimension (requires matching shapes).
            device: Optional device override for inputs; defaults to local torch device.
            dtype: Optional dtype to cast returned embeddings to.
            max_length: Optional per-call tokenizer override.
            truncation: Optional per-call tokenizer override.
            padding: Optional per-call tokenizer override.

        Returns:
            Depending on return_type and return_attention_mask:
            - list: List[Tensor] or (List[Tensor], List[Tensor])
            - dict: Dict[str, Tensor] or (Dict[str, Tensor], Dict[str, Tensor])
            - stack: Tensor of shape [num_encoders, ...] or a tuple with stacked
              attention masks
        """

        assert len(self.tokenizers) == len(self.text_encoders)
        assert len(self.text_encoders) == len(
            fastvideo_args.pipeline_config.text_encoder_configs)

        # Resolve selection into indices
        encoder_cfgs = fastvideo_args.pipeline_config.text_encoder_configs
        if encoder_index is None:
            indices: list[int] = [0]
        elif isinstance(encoder_index, int):
            indices = [encoder_index]
        else:
            indices = list(encoder_index)
        # validate range
        num_encoders = len(self.text_encoders)
        for idx in indices:
            if idx < 0 or idx >= num_encoders:
                raise IndexError(
                    f"encoder index {idx} out of range [0, {num_encoders-1}]")

        # Validate indices are within range
        num_encoders = len(self.text_encoders)

        # Normalize input to list[str]
        assert isinstance(text, str | list)
        if isinstance(text, str):
            texts: list[str] = [text]
        else:
            texts = text

        embeds_list: list[torch.Tensor] = []
        attn_masks_list: list[torch.Tensor] = []

        preprocess_funcs = fastvideo_args.pipeline_config.preprocess_text_funcs
        postprocess_funcs = fastvideo_args.pipeline_config.postprocess_text_funcs
        encoder_cfgs = fastvideo_args.pipeline_config.text_encoder_configs

        if return_type not in ("list", "dict", "stack"):
            raise ValueError(
                f"Invalid return_type '{return_type}'. Expected one of: 'list', 'dict', 'stack'"
            )

        target_device = device if device is not None else get_local_torch_device(
        )

        for i in indices:
            tokenizer = self.tokenizers[i]
            text_encoder = self.text_encoders[i]
            encoder_config = encoder_cfgs[i]
            preprocess_func = preprocess_funcs[i]
            postprocess_func = postprocess_funcs[i]

            processed_texts: list[str] = []
            for prompt_str in texts:
                processed_texts.append(preprocess_func(prompt_str))

            tok_kwargs = dict(encoder_config.tokenizer_kwargs)
            if max_length is not None:
                tok_kwargs["max_length"] = max_length
            if truncation is not None:
                tok_kwargs["truncation"] = truncation
            if padding is not None:
                tok_kwargs["padding"] = padding

            text_inputs = tokenizer(processed_texts,
                                    **tok_kwargs).to(target_device)

            input_ids = text_inputs["input_ids"]
            attention_mask = text_inputs["attention_mask"]

            with set_forward_context(current_timestep=0, attn_metadata=None):
                outputs = text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            prompt_embeds = postprocess_func(outputs)
            if dtype is not None:
                prompt_embeds = prompt_embeds.to(dtype=dtype)
            embeds_list.append(prompt_embeds)
            if return_attention_mask:
                attn_masks_list.append(attention_mask)

        # Shape results according to return_type
        if return_type == "list":
            if return_attention_mask:
                return embeds_list, attn_masks_list
            return embeds_list

        if return_type == "dict":
            key_strs = [str(i) for i in indices]
            embeds_dict = {
                k: v
                for k, v in zip(key_strs, embeds_list, strict=False)
            }
            if return_attention_mask:
                attn_dict = {
                    k: v
                    for k, v in zip(key_strs, attn_masks_list, strict=False)
                }
                return embeds_dict, attn_dict
            return embeds_dict

        # return_type == "stack"
        # Validate shapes are compatible
        base_shape = list(embeds_list[0].shape)
        for t in embeds_list[1:]:
            if list(t.shape) != base_shape:
                raise ValueError(
                    f"Cannot stack embeddings with differing shapes: {[list(t.shape) for t in embeds_list]}"
                )
        stacked_embeds = torch.stack(embeds_list, dim=0)
        if return_attention_mask:
            base_mask_shape = list(attn_masks_list[0].shape)
            for m in attn_masks_list[1:]:
                if list(m.shape) != base_mask_shape:
                    raise ValueError(
                        f"Cannot stack attention masks with differing shapes: {[list(m.shape) for m in attn_masks_list]}"
                    )
            stacked_masks = torch.stack(attn_masks_list, dim=0)
            return stacked_embeds, stacked_masks
        return stacked_embeds

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
