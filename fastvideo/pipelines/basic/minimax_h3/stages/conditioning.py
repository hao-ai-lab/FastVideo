# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL conditioning stage for MiniMax H3."""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConditioner
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_IMAGE_PAD_TOKEN,
    MINIMAX_H3_TEXT_ENCODER_LAYER,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    MINIMAX_H3_VISION_END_TOKEN,
    MINIMAX_H3_VISION_START_TOKEN,
)
from fastvideo.pipelines.basic.minimax_h3.stages._module_lifecycle import (
    maybe_offload_module,
    move_module_to_local_device,
)
from fastvideo.pipelines.basic.minimax_h3.types import get_minimax_h3_state
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult


def _token_ids(tokenized: Any) -> list[int]:
    input_ids = tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized.input_ids
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("MiniMax H3 tokenization must produce exactly one sequence.")
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def _module_dtype(module: Any) -> torch.dtype:
    dtype = getattr(module, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    parameter = next(module.parameters(), None)
    return torch.float32 if parameter is None else parameter.dtype


class MiniMaxH3ConditioningStage(PipelineStage):
    """Encode the verbatim prompt and optional keyframes with Qwen3-VL."""

    def __init__(self, conditioner: MiniMaxH3Qwen3VLConditioner, tokenizer: Any, processor: Any) -> None:
        super().__init__()
        self.conditioner = conditioner
        self.tokenizer = tokenizer
        self.processor = processor

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        state = get_minimax_h3_state(batch)
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, lambda value: isinstance(value, str))
        result.add_check("keyframes", state.keyframes, V.is_list)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        state = get_minimax_h3_state(batch)
        result = VerificationResult()
        result.add_check("prompt_embeds", state.prompt_embeds, V.with_dims(3))
        result.add_check("text_token_tags", state.text_token_tags, V.with_dims(1))
        result.add_check("batch.prompt_embeds", batch.prompt_embeds, V.list_of_tensors_dims(3))
        return result

    def _encode_prompt(
        self,
        prompt: str | list[str] | None,
        images: list[Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(prompt, str):
            raise ValueError(f"MiniMax H3 requires `prompt` to be a single string, got {type(prompt)}.")

        hidden_state_index = MINIMAX_H3_TEXT_ENCODER_LAYER
        num_hidden_layers = getattr(self.conditioner, "num_hidden_layers", None)
        if num_hidden_layers is None:
            num_hidden_layers = getattr(self.conditioner.config, "num_hidden_layers", None)
        if num_hidden_layers is None or num_hidden_layers <= hidden_state_index:
            raise ValueError(f"MiniMax H3 requires more than {hidden_state_index} Qwen3-VL decoder layers to read "
                             f"`hidden_states[{hidden_state_index}]`, got {num_hidden_layers}.")

        token_ids: list[int] = []
        token_tags: list[int] = []
        pixel_values = None
        image_grid_thw = None

        if images:
            vision_inputs = self.processor.image_processor(images=images, return_tensors="pt")
            pixel_values = vision_inputs["pixel_values"]
            image_grid_thw = vision_inputs["image_grid_thw"]
            merge_area = int(self.processor.image_processor.merge_size)**2
            vision_start_id = int(self.tokenizer.convert_tokens_to_ids(MINIMAX_H3_VISION_START_TOKEN))
            image_pad_id = int(self.tokenizer.convert_tokens_to_ids(MINIMAX_H3_IMAGE_PAD_TOKEN))
            vision_end_id = int(self.tokenizer.convert_tokens_to_ids(MINIMAX_H3_VISION_END_TOKEN))

            for index in range(len(images)):
                num_image_tokens = int(image_grid_thw[index].prod().item()) // merge_area
                label_ids = _token_ids(self.tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False))
                vision_ids = [vision_start_id] + [image_pad_id] * num_image_tokens + [vision_end_id]
                token_ids.extend(label_ids)
                token_ids.extend(vision_ids)
                token_tags.extend([MINIMAX_H3_TEXT_TAG] * len(label_ids))
                token_tags.extend([MINIMAX_H3_VIDEO_TAG] * len(vision_ids))

        prompt_ids = _token_ids(self.tokenizer(prompt, add_special_tokens=False))
        token_ids.extend(prompt_ids)
        token_tags.extend([MINIMAX_H3_TEXT_TAG] * len(prompt_ids))

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        mm_token_type_ids = torch.as_tensor(self.processor.create_mm_token_type_ids([token_ids]),
                                            dtype=torch.long,
                                            device=device)
        dtype = _module_dtype(self.conditioner)
        outputs = self.conditioner(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=None if pixel_values is None else pixel_values.to(device=device, dtype=dtype),
            image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(device=device),
            use_cache=False,
            output_hidden_states=True,
        )
        if outputs.hidden_states is None or len(outputs.hidden_states) <= hidden_state_index:
            raise ValueError(f"Qwen3-VL did not return `hidden_states[{hidden_state_index}]`.")
        prompt_embeds = outputs.hidden_states[hidden_state_index].to(device=device, dtype=dtype)
        return prompt_embeds, torch.tensor(token_tags, dtype=torch.long)

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        state = get_minimax_h3_state(batch)
        self.conditioner, device, moved_for_forward = move_module_to_local_device(self.conditioner)
        try:
            prompt_embeds, text_token_tags = self._encode_prompt(batch.prompt, state.keyframes, device)
        finally:
            self.conditioner = maybe_offload_module(
                self.conditioner,
                enabled=moved_for_forward and bool(getattr(fastvideo_args, "text_encoder_cpu_offload", False)),
            )
        state.prompt_embeds = prompt_embeds
        state.text_token_tags = text_token_tags
        batch.prompt_embeds = [prompt_embeds]
        return batch


__all__ = ["MiniMaxH3ConditioningStage"]
