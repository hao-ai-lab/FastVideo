# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL conditioning for MiniMax H3 FL2VA and Ref2VA."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.distributed.tensor import DTensor

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConditioner
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_IMAGE_PAD_TOKEN,
    MINIMAX_H3_TEXT_ENCODER_LAYER,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_PAD_TOKEN,
    MINIMAX_H3_VIDEO_TAG,
    MINIMAX_H3_VISION_END_TOKEN,
    MINIMAX_H3_VISION_START_TOKEN,
)
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference, sample_reference_video_frames
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import MINIMAX_H3_KEYFRAMES_KEY
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

MINIMAX_H3_TEXT_TOKEN_TAGS_KEY = "minimax_h3_text_token_tags"


def _token_ids(tokenized: Any) -> list[int]:
    input_ids = tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized.input_ids
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("MiniMax H3 tokenization must produce exactly one sequence.")
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


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
        resolved_ids = {int(special_id) for special_id in special_ids if special_id is not None}
        for index, token_id in enumerate(token_ids):
            if token_id in resolved_ids:
                modality_ids[index] = modality_type
    return [modality_ids]


def build_ref2va_presentation(
    tokenizer: Any,
    prompt: str,
    references: list[MiniMaxH3PreparedReference],
    image_token_counts: list[int],
    video_block_token_counts: list[int],
) -> tuple[list[int], list[int]]:
    """Tokenize ordered reference labels, vision blocks, then the prompt."""

    def text(value: str) -> tuple[list[int], list[int]]:
        ids = _token_ids(tokenizer(value, add_special_tokens=False))
        return ids, [MINIMAX_H3_TEXT_TAG] * len(ids)

    def vision(pad_token: str, count: int) -> tuple[list[int], list[int]]:
        ids = ([int(tokenizer.convert_tokens_to_ids(MINIMAX_H3_VISION_START_TOKEN))] +
               [int(tokenizer.convert_tokens_to_ids(pad_token))] * count +
               [int(tokenizer.convert_tokens_to_ids(MINIMAX_H3_VISION_END_TOKEN))])
        return ids, [MINIMAX_H3_VIDEO_TAG] * len(ids)

    token_ids: list[int] = []
    token_tags: list[int] = []

    def emit(segment: tuple[list[int], list[int]]) -> None:
        token_ids.extend(segment[0])
        token_tags.extend(segment[1])

    counts = {"image": 0, "video": 0, "audio": 0}
    for reference in references:
        if reference.has_audio:
            counts["audio"] += 1
            emit(text(f"<Audio {counts['audio']}>: "))
        if reference.media_type == "image":
            counts["image"] += 1
            if counts["image"] > len(image_token_counts):
                raise ValueError("Missing Qwen token count for a reference image.")
            emit(text(f"<Picture {counts['image']}>: "))
            emit(vision(MINIMAX_H3_IMAGE_PAD_TOKEN, image_token_counts[counts["image"] - 1]))
        elif reference.media_type == "video":
            counts["video"] += 1
            if counts["video"] > len(video_block_token_counts):
                raise ValueError("Missing Qwen token count for a reference video.")
            emit(text(f"<Video {counts['video']}>: "))
            for timestamp in reference.block_timestamps:
                emit(text(f"<{timestamp:.1f} seconds>"))
                emit(vision(MINIMAX_H3_VIDEO_PAD_TOKEN, video_block_token_counts[counts["video"] - 1]))
        elif reference.media_type != "audio":
            raise ValueError(f"Unsupported prepared reference type: {reference.media_type!r}.")
    if counts["image"] != len(image_token_counts) or counts["video"] != len(video_block_token_counts):
        raise ValueError("Qwen vision token counts do not match the ordered references.")
    emit(text(prompt))
    return token_ids, token_tags


class MiniMaxH3ConditioningStage(PipelineStage):
    """Encode the prompt and ordered visual presentation with Qwen3-VL."""

    performance_component_metric = "text_encoder_time_s"

    def __init__(
        self,
        conditioner: MiniMaxH3Qwen3VLConditioner,
        tokenizer: Any,
        processor: Any,
        *,
        ref2va: bool = False,
    ) -> None:
        super().__init__()
        self.conditioner = conditioner
        self.tokenizer = tokenizer
        self.processor = processor
        self.ref2va = ref2va

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, lambda value: isinstance(value, str))
        if self.ref2va:
            result.add_check("references", batch.references, V.list_not_empty)
        else:
            result.add_check("keyframes", batch.extra.get(MINIMAX_H3_KEYFRAMES_KEY), V.is_list)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_of_tensors_dims(3))
        result.add_check("text_token_tags", batch.extra.get(MINIMAX_H3_TEXT_TOKEN_TAGS_KEY), V.with_dims(1))
        return result

    def _encode_tokens(
        self,
        token_ids: list[int],
        token_tags: list[int],
        device: torch.device,
        **vision_inputs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_state_index = MINIMAX_H3_TEXT_ENCODER_LAYER
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        mm_token_type_ids = torch.as_tensor(
            _create_mm_token_type_ids(self.processor, token_ids),
            dtype=torch.long,
            device=device,
        )
        dtype = self.conditioner.dtype
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
        return (
            outputs.hidden_states[hidden_state_index].to(device=device, dtype=dtype),
            torch.tensor(token_tags, dtype=torch.long),
        )

    def _encode_fl2va(self, batch: ForwardBatch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch.prompt, str):
            raise ValueError("MiniMax H3 requires `prompt` to be a single string.")
        images = batch.extra.get(MINIMAX_H3_KEYFRAMES_KEY, [])
        if not isinstance(images, list):
            raise TypeError("MiniMax-H3 keyframes must be a list.")

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

        prompt_ids = _token_ids(self.tokenizer(batch.prompt, add_special_tokens=False))
        token_ids.extend(prompt_ids)
        token_tags.extend([MINIMAX_H3_TEXT_TAG] * len(prompt_ids))
        return self._encode_tokens(
            token_ids,
            token_tags,
            device,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    def _encode_ref2va(self, batch: ForwardBatch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch.prompt, str):
            raise ValueError("MiniMax H3 requires `prompt` to be a single string.")
        references = list(batch.references or [])
        if not references or not all(isinstance(item, MiniMaxH3PreparedReference) for item in references):
            raise TypeError("MiniMax-H3 Ref2VA conditioning requires prepared references.")

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
            batch.prompt,
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
        device = get_local_torch_device()
        first_param = next(self.conditioner.parameters(), None)
        moved_for_forward = (fastvideo_args.text_encoder_cpu_offload and first_param is not None
                             and not isinstance(first_param, DTensor))
        if moved_for_forward:
            self.conditioner.to(device)
        try:
            if self.ref2va:
                prompt_embeds, text_token_tags = self._encode_ref2va(batch, device)
            else:
                prompt_embeds, text_token_tags = self._encode_fl2va(batch, device)
        finally:
            if moved_for_forward:
                self.conditioner.to("cpu")
        batch.prompt_embeds = [prompt_embeds]
        batch.extra[MINIMAX_H3_TEXT_TOKEN_TAGS_KEY] = text_token_tags
        return batch


__all__ = [
    "MINIMAX_H3_TEXT_TOKEN_TAGS_KEY",
    "MiniMaxH3ConditioningStage",
    "build_ref2va_presentation",
]
