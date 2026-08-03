# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL conditioning stage for MiniMax H3."""

from __future__ import annotations

from typing import Any

import numpy as np
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
from fastvideo.pipelines.basic.minimax_h3.packing_ref2va import (
    build_ref2va_presentation,
    sample_reference_video_frames,
)
from fastvideo.pipelines.basic.minimax_h3.stages._module_lifecycle import (
    maybe_offload_module,
    move_module_to_local_device,
)
from fastvideo.pipelines.basic.minimax_h3.types import MiniMaxH3PreparedReference, get_minimax_h3_state
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


def _create_mm_token_type_ids(processor: Any, token_ids: list[int]) -> list[list[int]]:
    """Build Qwen3-VL modality IDs across old and new Transformers releases."""
    create_ids = getattr(processor, "create_mm_token_type_ids", None)
    if callable(create_ids):
        return create_ids([token_ids])

    modality_ids = [0] * len(token_ids)
    for modality, modality_type in (("image", 1), ("video", 2), ("audio", 3)):
        special_ids = getattr(processor, f"{modality}_token_ids", None)
        if special_ids is None:
            special_id = getattr(processor, f"{modality}_token_id", None)
            special_ids = [] if special_id is None else [special_id]
        special_ids = {int(special_id) for special_id in special_ids if special_id is not None}
        for index, token_id in enumerate(token_ids):
            if token_id in special_ids:
                modality_ids[index] = modality_type
    return [modality_ids]


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

        return self._encode_tokens(
            token_ids,
            token_tags,
            device,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    def _encode_tokens(
        self,
        token_ids: list[int],
        token_tags: list[int],
        device: torch.device,
        **vision_inputs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_state_index = MINIMAX_H3_TEXT_ENCODER_LAYER
        num_hidden_layers = getattr(self.conditioner, "num_hidden_layers", None)
        if num_hidden_layers is None:
            config = getattr(self.conditioner, "config", None)
            arch = getattr(config, "arch_config", config)
            num_hidden_layers = getattr(arch, "num_hidden_layers", None)
        if num_hidden_layers is None or num_hidden_layers <= hidden_state_index:
            raise ValueError(f"MiniMax H3 requires more than {hidden_state_index} Qwen3-VL decoder layers to read "
                             f"`hidden_states[{hidden_state_index}]`, got {num_hidden_layers}.")

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        mm_token_type_ids = torch.as_tensor(_create_mm_token_type_ids(self.processor, token_ids),
                                            dtype=torch.long,
                                            device=device)
        dtype = _module_dtype(self.conditioner)
        outputs = self.conditioner(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            output_hidden_states=True,
            **{
                name:
                None if value is None else value.to(
                    device=device,
                    dtype=dtype if name in {"pixel_values", "pixel_values_videos"} else None,
                )
                for name, value in vision_inputs.items()
            },
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


class MiniMaxH3Ref2VAConditioningStage(MiniMaxH3ConditioningStage):
    """Encode the ordered Ref2VA presentation with Qwen3-VL image and video inputs."""

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        state = get_minimax_h3_state(batch)
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, lambda value: isinstance(value, str))
        result.add_check("prepared_references", state.prepared_references, V.list_not_empty)
        return result

    def _encode_references(
        self,
        prompt: str | list[str] | None,
        references: list[MiniMaxH3PreparedReference],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(prompt, str):
            raise ValueError(f"MiniMax H3 requires `prompt` to be a single string, got {type(prompt)}.")

        merge_area = int(self.processor.image_processor.merge_size)**2
        pixel_values = None
        image_grid_thw = None
        image_token_counts: list[int] = []
        images = [reference.image for reference in references if reference.media_type == "image"]
        if any(image is None for image in images):
            raise ValueError("MiniMax-H3 reference images must be prepared before conditioning.")
        if images:
            vision = self.processor.image_processor(images=images, return_tensors="pt")
            pixel_values = vision["pixel_values"]
            image_grid_thw = vision["image_grid_thw"]
            image_token_counts = [int(grid.prod().item()) // merge_area for grid in image_grid_thw]

        pixel_values_videos = None
        video_grid_thw = None
        video_block_token_counts: list[int] = []
        videos = [reference for reference in references if reference.media_type == "video"]
        if videos:
            if any(reference.frames is None for reference in videos):
                raise ValueError("MiniMax-H3 reference videos must be prepared before conditioning.")
            sampled = [sample_reference_video_frames(reference.frames) for reference in videos]
            for reference, (_, timestamps) in zip(videos, sampled, strict=True):
                reference.block_timestamps = timestamps
            vision = self.processor.video_processor(
                videos=[np.stack(frames) for frames, _ in sampled],
                do_sample_frames=False,
                return_tensors="pt",
            )
            pixel_values_videos = vision["pixel_values_videos"]
            video_grid_thw = vision["video_grid_thw"]
            video_block_token_counts = [int(grid[1]) * int(grid[2]) // merge_area for grid in video_grid_thw]
            for reference, grid in zip(videos, video_grid_thw, strict=True):
                if int(grid[0]) != len(reference.block_timestamps):
                    raise ValueError(f"Qwen3-VL produced {int(grid[0])} blocks for a reference video, but "
                                     f"MiniMax-H3 labels {len(reference.block_timestamps)}.")

        token_ids, token_tags = build_ref2va_presentation(
            self.tokenizer,
            prompt,
            references,
            image_token_counts,
            video_block_token_counts,
        )
        return self._encode_tokens(
            token_ids,
            token_tags,
            device,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        state = get_minimax_h3_state(batch)
        self.conditioner, device, moved_for_forward = move_module_to_local_device(self.conditioner)
        try:
            prompt_embeds, text_token_tags = self._encode_references(
                batch.prompt,
                state.prepared_references,
                device,
            )
        finally:
            self.conditioner = maybe_offload_module(
                self.conditioner,
                enabled=moved_for_forward and bool(getattr(fastvideo_args, "text_encoder_cpu_offload", False)),
            )
        state.prompt_embeds = prompt_embeds
        state.text_token_tags = text_token_tags
        batch.prompt_embeds = [prompt_embeds]
        return batch


__all__ = ["MiniMaxH3ConditioningStage", "MiniMaxH3Ref2VAConditioningStage"]
