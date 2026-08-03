# SPDX-License-Identifier: Apache-2.0
"""Reference preparation and packed-layout primitives for MiniMax H3 Ref2VA."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from PIL import Image

from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_IMAGE_PAD_TOKEN,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_PAD_TOKEN,
    MINIMAX_H3_VIDEO_TAG,
    MINIMAX_H3_VISION_END_TOKEN,
    MINIMAX_H3_VISION_START_TOKEN,
    _ROPE_FRAME_RESCALE,
    _ROPE_FRAMES_PER_LATENT,
    _spatial_position_grid,
    _temporal_position_grid,
    resolve_canvas_size,
)
from fastvideo.pipelines.basic.minimax_h3.types import MiniMaxH3Layout, MiniMaxH3PreparedReference

MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048
MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS = 2.0
MINIMAX_H3_QWEN_TEMPORAL_PATCH = 2
MINIMAX_H3_MAX_REFERENCE_IMAGES = 9
MINIMAX_H3_MAX_REFERENCE_VIDEOS = 3
MINIMAX_H3_MAX_REFERENCE_AUDIOS = 3
MINIMAX_H3_MAX_REFERENCES = 12


def _token_ids(tokenized: Any) -> list[int]:
    input_ids = tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized.input_ids
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("MiniMax-H3 tokenization must produce exactly one sequence.")
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def _reference_temporal_span(num_latent_frames: int) -> float:
    # This call site uses sequential float64 addition; the target layout keeps
    # NumPy's pairwise sum. They differ at the final ULP for longer clips.
    return sum(_ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
               for index in range(num_latent_frames))


def _frame_position_grid(
    latent_height: int,
    latent_width: int,
    patch_h: int,
    patch_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def _fill_audio_positions(
    position_ids: torch.Tensor,
    rows: slice,
    num_audio_latents: int,
    rotary_time: float,
    width_grid: torch.Tensor,
) -> None:
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[rows, 2] = torch.cat([
        torch.full((num_audio_latents, ), float(width_grid[0]), dtype=torch.float64),
        torch.full((num_audio_latents, ), float(width_grid[-1]), dtype=torch.float64),
    ])


def _num_video_rows(reference: MiniMaxH3PreparedReference, patch_size: tuple[int, int, int]) -> int:
    patch_t, patch_h, patch_w = patch_size
    geometry = (reference.num_latent_frames, reference.latent_height, reference.latent_width)
    if any(value <= 0 for value in geometry):
        raise ValueError(f"Incomplete visual reference geometry: {geometry}.")
    if any(value % patch for value, patch in zip(geometry, patch_size, strict=True)):
        raise ValueError(f"Visual reference geometry {geometry} is not divisible by patch {patch_size}.")
    return (reference.num_latent_frames // patch_t) * (reference.latent_height // patch_h) * (reference.latent_width //
                                                                                              patch_w)


def build_ref2va_packed_sequence(
    text_token_tags: torch.Tensor,
    references: list[MiniMaxH3PreparedReference],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
) -> MiniMaxH3Layout:
    """Build `[text | ordered references | target audio | target video]`."""
    if patch_size != (1, 2, 2):
        raise ValueError(f"MiniMax-H3 Ref2VA requires patch_size=(1, 2, 2), got {patch_size}.")
    if text_token_tags.ndim != 1:
        raise ValueError(f"text_token_tags must be one-dimensional, got {tuple(text_token_tags.shape)}.")
    valid_text_tags = (text_token_tags == MINIMAX_H3_TEXT_TAG) | (text_token_tags == MINIMAX_H3_VIDEO_TAG)
    if not bool(valid_text_tags.all()):
        raise ValueError("text_token_tags may contain only text and vision tags.")
    if not references:
        raise ValueError("Ref2VA requires at least one prepared reference.")

    patch_t, patch_h, patch_w = patch_size
    target_geometry = (num_latent_frames, latent_height, latent_width)
    if any(value <= 0 for value in target_geometry) or num_audio_latents <= 0:
        raise ValueError("Ref2VA target latent geometry must be positive.")
    if any(value % patch for value, patch in zip(target_geometry, patch_size, strict=True)):
        raise ValueError(f"Target geometry {target_geometry} is not divisible by patch {patch_size}.")

    visual_rows = {
        id(reference): _num_video_rows(reference, patch_size)
        for reference in references if reference.media_type != "audio"
    }
    for reference in references:
        if reference.media_type == "audio" and not reference.has_audio:
            raise ValueError("An audio reference must carry decoded waveform latents.")
        if reference.has_audio and reference.num_audio_latents <= 0:
            raise ValueError("An audio-bearing reference has no resolved audio latents.")

    num_text_tokens = int(text_token_tags.shape[0])
    num_target_video_rows = (num_latent_frames // patch_t) * (latent_height // patch_h) * (latent_width // patch_w)
    num_target_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    num_reference_video_rows = sum(visual_rows.values())
    num_reference_audio_rows = sum(reference.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS for reference in references
                                   if reference.has_audio)
    sequence_length = (num_text_tokens + num_reference_video_rows + num_reference_audio_rows + num_target_audio_rows +
                       num_target_video_rows)

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
    target_frame_grid, target_width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    video_indices: list[torch.Tensor] = []
    audio_indices: list[torch.Tensor] = []
    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)
    for reference in references:
        if reference.media_type == "image":
            count = visual_rows[id(reference)]
            rows = slice(cursor, cursor + count)
            cursor = rows.stop
            video_indices.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_position_grid(
                reference.latent_height,
                reference.latent_width,
                patch_h,
                patch_w,
            )
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            rotary_time += 1.0
        elif reference.media_type == "audio":
            count = reference.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
            rows = slice(cursor, cursor + count)
            cursor = rows.stop
            audio_indices.append(torch.arange(rows.start, rows.stop))
            _fill_audio_positions(position_ids, rows, reference.num_audio_latents, rotary_time, target_width_grid)
            rotary_time += float(reference.num_audio_latents)
        elif reference.media_type == "video":
            audio_count = reference.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS if reference.has_audio else 0
            video_count = visual_rows[id(reference)]
            audio_rows = slice(cursor, cursor + audio_count)
            video_rows = slice(audio_rows.stop, audio_rows.stop + video_count)
            cursor = video_rows.stop
            if audio_count:
                audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
            video_indices.append(torch.arange(video_rows.start, video_rows.stop))

            frame_grid, width_grid = _frame_position_grid(
                reference.latent_height,
                reference.latent_width,
                patch_h,
                patch_w,
            )
            if audio_count:
                _fill_audio_positions(
                    position_ids,
                    audio_rows,
                    reference.num_audio_latents,
                    rotary_time,
                    width_grid,
                )
            frame_time = _temporal_position_grid(reference.num_latent_frames, rotary_time)
            rows_per_frame = frame_grid.shape[0]
            position_ids[video_rows, 0] = frame_time.repeat_interleave(rows_per_frame)
            position_ids[video_rows, 1:] = frame_grid.repeat(reference.num_latent_frames // patch_t, 1)
            rotary_time += max(
                float(reference.num_audio_latents if reference.has_audio else 0),
                _reference_temporal_span(reference.num_latent_frames),
            )
        else:
            raise ValueError(f"Unsupported prepared reference type: {reference.media_type!r}.")

    audio_start = cursor
    video_start = audio_start + num_target_audio_rows
    _fill_audio_positions(position_ids, slice(audio_start, video_start), num_audio_latents, rotary_time,
                          target_width_grid)
    frame_time = _temporal_position_grid(num_latent_frames, rotary_time)
    position_ids[video_start:, 0] = frame_time.repeat_interleave(target_frame_grid.shape[0])
    position_ids[video_start:, 1:] = target_frame_grid.repeat(num_latent_frames // patch_t, 1)

    video_indices.append(torch.arange(video_start, sequence_length))
    audio_indices.append(torch.arange(audio_start, video_start))
    packed_video_indices = torch.cat(video_indices)
    packed_audio_indices = torch.cat(audio_indices)
    text_indices = torch.arange(num_text_tokens)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[packed_audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[packed_video_indices] = MINIMAX_H3_VIDEO_TAG

    return MiniMaxH3Layout(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=packed_video_indices,
        audio_indices=packed_audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_reference_video_rows,
        num_condition_audio_rows=num_reference_audio_rows,
        num_video_latent_frames=num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=num_audio_latents,
    )


def resolve_reference_image_size(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"A reference image must have a positive size, got {width}x{height}.")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")
    scale = MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    return (
        max(multiple,
            round(height * scale / multiple) * multiple),
        max(multiple,
            round(width * scale / multiple) * multiple),
    )


def reference_media_to_uint8(media: Any) -> np.ndarray:
    if isinstance(media, list):
        if not media:
            raise ValueError("A reference video must contain at least one frame.")
        return np.stack([reference_media_to_uint8(item) for item in media])
    if isinstance(media, Image.Image):
        return np.asarray(media.convert("RGB"))
    if isinstance(media, torch.Tensor):
        media = media.movedim(-3, -1).detach().cpu().numpy()
    media = np.asarray(media)
    if media.dtype != np.uint8:
        media = (media * 255.0).round().clip(0, 255).astype(np.uint8)
    return media


def prepare_reference_image(image: Image.Image, height: int, width: int) -> Image.Image:
    return image if image.size == (width, height) else image.resize((width, height), Image.Resampling.LANCZOS)


def resample_reference_frames(frames: np.ndarray, fps: float) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] == 0:
        raise ValueError(f"A reference video must be non-empty RGB frames, got {tuple(frames.shape)}.")
    if fps <= 0:
        raise ValueError(f"A reference video must have a positive frame rate, got {fps}.")
    if fps == MINIMAX_H3_FPS:
        return frames
    scale = MINIMAX_H3_FPS / fps
    slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
    repeats = np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5))
    return np.repeat(frames, repeats, axis=0)


def prepare_reference_frames(frames: np.ndarray, num_frames: int) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] == 0:
        raise ValueError(f"A reference video must be non-empty RGB frames, got {tuple(frames.shape)}.")
    frames = frames[:num_frames]
    height, width = resolve_canvas_size(frames.shape[2], frames.shape[1])
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)) for frame in frames])


def sample_reference_video_frames(frames: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    if frames.ndim != 4 or frames.shape[0] == 0:
        raise ValueError("A prepared reference video must contain frames.")
    stride = MINIMAX_H3_FPS / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS
    indices: list[int] = []
    cursor = 0.0
    while round(cursor) < frames.shape[0]:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride
    timestamps = [index / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS for index in range(len(indices))]
    timestamps += [timestamps[-1]] * (-len(timestamps) % MINIMAX_H3_QWEN_TEMPORAL_PATCH)
    block_timestamps = [(timestamps[index] + timestamps[index + MINIMAX_H3_QWEN_TEMPORAL_PATCH - 1]) / 2
                        for index in range(0, len(timestamps), MINIMAX_H3_QWEN_TEMPORAL_PATCH)]
    return [frames[index] for index in indices], block_timestamps


def prepare_reference_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int,
    max_duration: float,
) -> torch.Tensor:
    waveform = torch.as_tensor(waveform).detach().cpu()
    if waveform.ndim != 2 or waveform.shape[0] not in (1, MINIMAX_H3_AUDIO_CHANNELS):
        raise ValueError(
            f"A reference soundtrack must be mono or stereo [channels, samples], got {tuple(waveform.shape)}.")
    if waveform.shape[-1] == 0:
        raise ValueError("A reference soundtrack must contain samples.")
    if sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("Reference audio sample rates must be positive.")
    waveform = waveform.to(torch.float32)[:, :int(max_duration * sample_rate)]
    if waveform.shape[0] == 1:
        waveform = waveform.expand(MINIMAX_H3_AUDIO_CHANNELS, -1).contiguous()
    if sample_rate == target_sample_rate:
        return waveform
    try:
        import torchaudio
    except ImportError as error:
        raise ImportError("Resampling MiniMax-H3 reference audio requires torchaudio.") from error
    return torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)


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


def trim_reference_num_frames(num_frames: int) -> int:
    if num_frames < 1:
        raise ValueError(f"A reference video must have at least one frame, got {num_frames}.")
    return (
        max(1,
            (num_frames - MINIMAX_H3_LATENTS_PER_CHUNK) // MINIMAX_H3_FRAMES_PER_CHUNK) * MINIMAX_H3_FRAMES_PER_CHUNK +
        MINIMAX_H3_LATENTS_PER_CHUNK)


__all__ = [
    "MINIMAX_H3_MAX_REFERENCES",
    "MINIMAX_H3_MAX_REFERENCE_AUDIOS",
    "MINIMAX_H3_MAX_REFERENCE_IMAGES",
    "MINIMAX_H3_MAX_REFERENCE_VIDEOS",
    "build_ref2va_packed_sequence",
    "build_ref2va_presentation",
    "prepare_reference_frames",
    "prepare_reference_image",
    "prepare_reference_waveform",
    "reference_media_to_uint8",
    "resample_reference_frames",
    "resolve_reference_image_size",
    "sample_reference_video_frames",
    "trim_reference_num_frames",
]
