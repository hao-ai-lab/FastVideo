# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL conditioning for MiniMax H3 FL2VA and Ref2VA."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from torch.distributed.tensor import DTensor

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConditioner
from fastvideo.pipelines.basic.minimax_h3.memo import ContentMemo, image_key
from fastvideo.pipelines.basic.minimax_h3.prompt_embeds_cache import PromptEmbedsCache
from fastvideo.profiler import nvtx_range
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_IMAGE_PAD_TOKEN,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_PAD_TOKEN,
    MINIMAX_H3_VIDEO_TAG,
    MINIMAX_H3_VISION_END_TOKEN,
    MINIMAX_H3_VISION_START_TOKEN,
)
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference, sample_reference_video_frames
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import (
    MINIMAX_H3_KEYFRAME_ANCHORS_KEY,
    MINIMAX_H3_KEYFRAMES_KEY,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

MINIMAX_H3_TEXT_TOKEN_TAGS_KEY = "minimax_h3_text_token_tags"

logger = init_logger(__name__)

_CACHE_FORMAT = 1


def prompt_embeds_cache_key(batch: Any, ref2va: bool, model_path: Any, device: Any) -> str | None:
    """Stable digest of everything the conditioning stage consumes.

    Returns ``None`` for presentations this cache cannot safely key: a video
    reference gets its block timestamps assigned while presenting, so its
    output is not a pure function of the prepared object.
    """
    parts = [
        "h3-prompt-embeds",
        str(_CACHE_FORMAT),
        "ref2va" if ref2va else "fl2va",
        str(model_path),
        str(device),
        str(batch.prompt),
    ]
    anchors = batch.extra.get(MINIMAX_H3_KEYFRAME_ANCHORS_KEY, ()) or ()
    keyframes = batch.extra.get(MINIMAX_H3_KEYFRAMES_KEY, []) or []
    if len(anchors) != len(keyframes):
        return None
    try:
        for anchor, image in zip(anchors, keyframes, strict=True):
            parts.append(f"kf:{anchor}:{image_key(image)}")
    except Exception:
        return None
    if ref2va:
        for reference in batch.references or []:
            if getattr(reference, "frames", None) is not None or reference.media_type == "video":
                return None
            if reference.image is None:
                return None
            try:
                parts.append(f"ref:{reference.media_type}:{bool(reference.has_audio)}:{image_key(reference.image)}")
            except Exception:
                return None
    return hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=16).hexdigest()


def _token_ids(tokenized: Any) -> list[int]:
    input_ids = tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized.input_ids
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("MiniMax H3 tokenization must produce exactly one sequence.")
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


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
        manage_cpu_offload: bool = True,
    ) -> None:
        super().__init__()
        self.conditioner = conditioner
        self.tokenizer = tokenizer
        self.processor = processor
        self.ref2va = ref2va
        self.manage_cpu_offload = manage_cpu_offload
        self._ref2va_memo = ContentMemo()
        self._prompt_embeds_cache = PromptEmbedsCache()

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
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        dtype = self.conditioner.dtype
        prompt_embeds = self.conditioner(
            input_ids,
            **{
                name:
                None if value is None else value.to(
                    device=device,
                    dtype=dtype if name in {"pixel_values", "pixel_values_videos"} else None,
                )
                for name, value in vision_inputs.items()
            },
        )
        if prompt_embeds.ndim != 2 or prompt_embeds.shape[0] != len(token_ids):
            raise ValueError(f"MiniMax-H3 slim text encoder returned unexpected shape={tuple(prompt_embeds.shape)}")
        return (
            prompt_embeds.unsqueeze(0).to(device=device, dtype=dtype),
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
        # Video references get their block timestamps assigned while presenting,
        # so only presentations without a video reference are memoized. The key
        # hashes pixels, so a disabled memo must not pay for building it.
        has_video = any(reference.media_type == "video" for reference in references)
        images_ready = all(reference.image is not None for reference in references if reference.media_type == "image")
        if self._ref2va_memo.enabled and not has_video and images_ready:
            key = (str(device), batch.prompt,
                   tuple((reference.media_type, bool(reference.has_audio),
                          image_key(reference.image) if reference.media_type == "image" else None)
                         for reference in references))
            return self._ref2va_memo.get_or_compute(key, lambda: self._present_ref2va(batch, references, device))
        return self._present_ref2va(batch, references, device)

    def _present_ref2va(self, batch: ForwardBatch, references: list[MiniMaxH3PreparedReference],
                        device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
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
        """Encode one H3 prompt presentation and attach its packed text features."""
        device = get_local_torch_device()
        # Stages built without __init__ (tests, subclasses) have no cache; treat that as off.
        cache = getattr(self, "_prompt_embeds_cache", None)
        cache_key: str | None = None
        if cache is not None and cache.enabled:
            cache_key = prompt_embeds_cache_key(batch, self.ref2va, getattr(fastvideo_args, "model_path", None), device)
            cached = cache.get(cache_key)
            if cached is not None:
                cached_embeds, cached_tags = cached
                batch.prompt_embeds = [cached_embeds.to(device=device)]
                batch.extra[MINIMAX_H3_TEXT_TOKEN_TAGS_KEY] = cached_tags
                # Never touch self.conditioner here: the deferred text encoder stays
                # unmaterialized, so an identical request skips its multi-GiB reload.
                logger.info("MiniMax-H3 conditioning cache hit (%s); Qwen3-VL stays unmaterialized", cache_key)
                return batch
        first_param = next(self.conditioner.parameters(), None)
        moved_for_forward = (getattr(self, "manage_cpu_offload", True) and fastvideo_args.text_encoder_cpu_offload
                             and first_param is not None and not isinstance(first_param, DTensor)
                             and getattr(self.conditioner, "_h3_encoder_layerwise_device", None) is None)
        if moved_for_forward:
            self.conditioner.to(device)
        try:
            # Keep both H3 prompt-presentation modes under one text-encoding
            # range so Nsight Systems exposes their complete conditioning cost.
            with nvtx_range("minimax_h3.text_encoding"):
                if self.ref2va:
                    prompt_embeds, text_token_tags = self._encode_ref2va(batch, device)
                else:
                    prompt_embeds, text_token_tags = self._encode_fl2va(batch, device)
        finally:
            if moved_for_forward:
                self.conditioner.to("cpu")
        batch.prompt_embeds = [prompt_embeds]
        batch.extra[MINIMAX_H3_TEXT_TOKEN_TAGS_KEY] = text_token_tags
        if cache is not None:
            cache.put(cache_key, prompt_embeds, text_token_tags)
        return batch


__all__ = [
    "MINIMAX_H3_TEXT_TOKEN_TAGS_KEY",
    "MiniMaxH3ConditioningStage",
    "build_ref2va_presentation",
]
