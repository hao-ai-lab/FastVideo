# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MiniMax and HuggingFace Teams.

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

if TYPE_CHECKING:
    from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference

MINIMAX_H3_VIDEO_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
MINIMAX_H3_AUDIO_TAG = 2

MINIMAX_H3_FPS = 24
MINIMAX_H3_SHORT_EDGE = 768
MINIMAX_H3_MAX_PIXELS = 768 * 1344
MINIMAX_H3_CANVAS_MULTIPLE = 32
MINIMAX_H3_MIN_ASPECT_RATIO = 1 / 4
MINIMAX_H3_MAX_ASPECT_RATIO = 4
MINIMAX_H3_MIN_DURATION = 5.0
MINIMAX_H3_MAX_DURATION = 15.0
MINIMAX_H3_FRAMES_PER_CHUNK = 17
MINIMAX_H3_LATENTS_PER_CHUNK = 5
MINIMAX_H3_TEXT_ENCODER_LAYER = 50
MINIMAX_H3_VISION_START_TOKEN = "<|vision_start|>"
MINIMAX_H3_IMAGE_PAD_TOKEN = "<|image_pad|>"
MINIMAX_H3_VIDEO_PAD_TOKEN = "<|video_pad|>"
MINIMAX_H3_VISION_END_TOKEN = "<|vision_end|>"
MINIMAX_H3_AUDIO_LATENTS_PER_SECOND = 40
MINIMAX_H3_AUDIO_CHANNELS = 2
MINIMAX_H3_KEYFRAME_NOISE_AUG = 0.999
MINIMAX_H3_KEYFRAME_ENCODE_SEED = 42

_PATCH_SIZE_CACHE: dict[int, tuple[int, int, int]] = {}


def h3_dit_patch_size(fastvideo_args: Any) -> tuple[int, int, int]:
    """Read DiT patch size from pipeline config, not live transformer weights."""
    dit_config = getattr(getattr(fastvideo_args, "pipeline_config", None), "dit_config", None)
    cached = _PATCH_SIZE_CACHE.get(id(dit_config)) if dit_config is not None else None
    if cached is not None:
        return cached
    patch_size = getattr(dit_config, "patch_size", None)
    if patch_size is None:
        raise ValueError("MiniMax-H3 requires pipeline_config.dit_config.patch_size.")
    axes = tuple(int(axis) for axis in patch_size)
    if len(axes) != 3 or min(axes) <= 0:
        raise ValueError(f"MiniMax-H3 patch_size must be three positive ints, got {patch_size!r}.")
    values = (axes[0], axes[1], axes[2])
    if dit_config is not None:
        _PATCH_SIZE_CACHE[id(dit_config)] = values
    return values


def h3_latent_channels(model_config: Any, name: str) -> int:
    """Read VAE latent width from arch config, not a live VAE proxy."""
    arch = getattr(model_config, "arch_config", None)
    value = getattr(arch, "latent_channels", None)
    if value is None:
        raise ValueError(f"MiniMax-H3 requires {name}.arch_config.latent_channels")
    return int(value)


MINIMAX_H3_ROPE_FRAME_RESCALE = 5.0 / 3.0
MINIMAX_H3_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


@dataclass(frozen=True)
class MiniMaxH3PackedLayout:
    """One packed joint sequence and the geometry needed to interpret it."""

    sequence_length: int
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_condition_video_rows: int
    num_condition_audio_rows: int
    num_video_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int
    # Per-reference conditioning spans in packed order (Ref2VA only; empty
    # elsewhere): (kind, row_count, latent_shape) covering exactly the rows
    # between the text tokens and the target audio. kind is "audio" / "image"
    # / "video" / "continuation"; latent_shape is the reference's raw latent (frames, height,
    # width) for visual kinds and (0, 0, 0) for audio rows. A video reference
    # with sound contributes two entries — its audio rows precede its video
    # rows. Retained so VSA-H3 can 3-D-tile (and sparsify) reference-video
    # regions instead of exempting them wholesale.
    reference_segments: tuple[tuple[str, int, tuple[int, int, int]], ...] = ()
    # Capacity rows trail all real modalities and have no output indices.
    pad_tokens: int = 0

    @property
    def real_sequence_length(self) -> int:
        return self.sequence_length - self.pad_tokens

    @property
    def valid(self) -> torch.Tensor:
        return torch.arange(self.sequence_length, device=self.token_tags.device) < self.real_sequence_length

    def __post_init__(self) -> None:
        _validate_packed_rows(self)


@dataclass(frozen=True)
class MiniMaxH3PackedBatchLayout:
    """Several independent H3 documents concatenated into one DiT stream.

    H3 already calls the text/audio/video rows belonging to one sample a
    "packed" sequence.  This second-level layout is deliberately named a
    batch layout: every child remains an independent document and attention
    backends must use :attr:`document_lengths` as block-diagonal boundaries.
    Position IDs are copied from the children, so every document restarts its
    own RoPE coordinate system.
    """

    documents: tuple[MiniMaxH3PackedLayout, ...]
    sequence_length: int
    document_offsets: tuple[int, ...]
    document_lengths: tuple[int, ...]
    text_lengths: tuple[int, ...]
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    pad_tokens: int = 0

    @property
    def real_sequence_length(self) -> int:
        return self.sequence_length - self.pad_tokens

    @property
    def valid(self) -> torch.Tensor:
        return torch.arange(self.sequence_length, device=self.token_tags.device) < self.real_sequence_length

    def __post_init__(self) -> None:
        _validate_packed_rows(self)
        if any(document.pad_tokens for document in self.documents):
            raise ValueError("Padding must trail the entire packed sequence, not individual documents")
        if len(self.documents) < 2:
            raise ValueError("A packed H3 batch must contain at least two documents")
        if len(self.document_offsets) != len(self.documents) + 1 or self.document_offsets[0] != 0:
            raise ValueError("document_offsets must contain one leading zero and one end offset per document")
        if self.document_offsets[-1] != self.real_sequence_length:
            raise ValueError("The final H3 document offset must equal the real sequence length (before padding)")
        if tuple(end - start for start, end in zip(self.document_offsets[:-1], self.document_offsets[1:],
                                                   strict=True)) != self.document_lengths:
            raise ValueError("H3 document offsets and lengths disagree")
        if sum(self.text_lengths) != int(self.text_indices.numel()):
            raise ValueError("H3 packed-batch text lengths do not cover text_indices")
        if self.position_ids.shape != (self.sequence_length, 3):
            raise ValueError("H3 packed-batch position_ids must cover the complete sequence")
        if self.token_tags.shape != (self.sequence_length, ):
            raise ValueError("H3 packed-batch token_tags must cover the complete sequence")


@dataclass(frozen=True)
class MiniMaxH3ContinuationLayout:
    """Latent geometry of a continuation clip (``reference.MiniMaxH3PreparedContinuation``)."""

    num_latent_frames: int
    # 0 for a clip without a soundtrack.
    num_audio_latents: int


def _validate_packed_rows(layout: MiniMaxH3PackedLayout | MiniMaxH3PackedBatchLayout) -> None:
    if type(layout.pad_tokens) is not int or layout.pad_tokens < 0:
        raise ValueError("pad_tokens must be a non-negative integer")
    real_tokens = sum(indices.numel() for indices in (layout.video_indices, layout.audio_indices, layout.text_indices))
    if real_tokens != layout.real_sequence_length or real_tokens <= 0:
        raise ValueError("H3 real token counts must be exact, with capacity padding appended")
    if layout.position_ids.shape != (layout.sequence_length,
                                     3) or layout.token_tags.shape != (layout.sequence_length, ):
        raise ValueError("H3 positions and token tags must cover real tokens and trailing padding")


def pad_packed_sequence(
    layout: MiniMaxH3PackedLayout | MiniMaxH3PackedBatchLayout,
    pad_tokens: int,
) -> MiniMaxH3PackedLayout | MiniMaxH3PackedBatchLayout:
    """Append inert rows without changing any real position or modality index.

    Padding uses an existing AdaLN tag/time, not a new learned modality. Its
    validity comes from the layout/attention boundaries, never from token tags.
    """
    if type(pad_tokens) is not int or pad_tokens < 0:
        raise ValueError("pad_tokens must be a non-negative integer")
    if layout.pad_tokens:
        raise ValueError("The H3 sequence already has capacity padding")
    if not pad_tokens:
        return layout
    return replace(
        layout,
        sequence_length=layout.sequence_length + pad_tokens,
        pad_tokens=pad_tokens,
        position_ids=torch.nn.functional.pad(layout.position_ids, (0, 0, 0, pad_tokens)),
        token_tags=torch.nn.functional.pad(layout.token_tags, (0, pad_tokens)),
    )


def build_packed_batch_sequence(documents: tuple[MiniMaxH3PackedLayout, ...], ) -> MiniMaxH3PackedBatchLayout:
    """Concatenate independent T2VA document layouts without changing them."""
    if len(documents) < 2:
        raise ValueError("Cross-sample sequence packing requires at least two H3 documents")
    if any(document.num_condition_video_rows or document.num_condition_audio_rows or document.reference_segments
           for document in documents):
        raise ValueError("Cross-sample sequence packing currently supports T2VA documents only")

    offsets = [0]
    video_indices: list[torch.Tensor] = []
    audio_indices: list[torch.Tensor] = []
    text_indices: list[torch.Tensor] = []
    for document in documents:
        offset = offsets[-1]
        video_indices.append(document.video_indices + offset)
        audio_indices.append(document.audio_indices + offset)
        text_indices.append(document.text_indices + offset)
        offsets.append(offset + document.sequence_length)

    sequence_length = offsets[-1]
    return MiniMaxH3PackedBatchLayout(
        documents=documents,
        sequence_length=sequence_length,
        document_offsets=tuple(offsets),
        document_lengths=tuple(document.sequence_length for document in documents),
        text_lengths=tuple(int(document.text_indices.numel()) for document in documents),
        position_ids=torch.cat([document.position_ids for document in documents], dim=0),
        token_tags=torch.cat([document.token_tags for document in documents], dim=0),
        video_indices=torch.cat(video_indices),
        audio_indices=torch.cat(audio_indices),
        text_indices=torch.cat(text_indices),
    )


def resolve_canvas_size(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}.")
    ratio = aspect_width / aspect_height
    if not MINIMAX_H3_MIN_ASPECT_RATIO <= ratio <= MINIMAX_H3_MAX_ASPECT_RATIO:
        raise ValueError(f"MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got {aspect_width}:{aspect_height}.")

    if ratio >= 1:
        width, height = MINIMAX_H3_SHORT_EDGE * ratio, float(MINIMAX_H3_SHORT_EDGE)
    else:
        width, height = float(MINIMAX_H3_SHORT_EDGE), MINIMAX_H3_SHORT_EDGE / ratio
    area = width * height
    if area > MINIMAX_H3_MAX_PIXELS:
        scale = (MINIMAX_H3_MAX_PIXELS / area)**0.5
        width, height = width * scale, height * scale
    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    return max(multiple, round(height / multiple) * multiple), max(multiple, round(width / multiple) * multiple)


def align_num_frames(num_frames: int) -> int:
    if num_frames < 1:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    while num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


# Duration limits are seconds; requests are validated against the aligned buckets
# they map to. The 15-second target (360 frames) pads to 362 on the causal-VAE
# grid, so 362 is the largest accepted bucket.
MINIMAX_H3_MIN_ALIGNED_FRAMES = align_num_frames(int(MINIMAX_H3_MIN_DURATION * MINIMAX_H3_FPS))
MINIMAX_H3_MAX_ALIGNED_FRAMES = align_num_frames(int(MINIMAX_H3_MAX_DURATION * MINIMAX_H3_FPS))


def video_latent_num_frames(num_frames: int) -> int:
    if num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        raise ValueError(f"`num_frames` must be of the form 17 * n + 5, got {num_frames}.")
    return (num_frames - MINIMAX_H3_LATENTS_PER_CHUNK) // MINIMAX_H3_FRAMES_PER_CHUNK * MINIMAX_H3_LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int) -> int:
    return int(round(num_frames / MINIMAX_H3_FPS * MINIMAX_H3_AUDIO_LATENTS_PER_SECOND))


def prepare_keyframe_image(image: Image.Image, height: int, width: int, stretch: bool) -> Image.Image:
    if image.size == (width, height):
        return image
    if stretch:
        return image.resize((width, height), Image.Resampling.LANCZOS)
    scale = max(width / image.size[0], height / image.size[1])
    resized_size = (max(width, round(image.size[0] * scale)), max(height, round(image.size[1] * scale)))
    left = max(0, (resized_size[0] - width) // 2)
    top = max(0, (resized_size[1] - height) // 2)
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    return resized.crop((left, top, left + width, top + height))


def patchify_video_latents(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"Latents of shape {tuple(latents.shape)} are not divisible by the patch {patch_size}.")
    latents = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    # Channel-major patch features are a checkpoint contract; do not use einops's common patch-major order.
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()


def unpatchify_video_tokens(
    rows: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    channels: int,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    patch_t, patch_h, patch_w = patch_size
    rows = rows.reshape(
        -1,
        num_latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(-1, channels, num_latent_frames, latent_height, latent_width).contiguous()


def unpack_audio_tokens(rows: torch.Tensor, num_audio_latents: int) -> torch.Tensor:
    rows = rows.reshape(MINIMAX_H3_AUDIO_CHANNELS, num_audio_latents, rows.shape[-1])
    return rows.permute(0, 2, 1).contiguous()


def spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # NumPy endpoint=False and float64 are part of the reference RoPE coordinate arithmetic.
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [
            MINIMAX_H3_ROPE_FRAME_RESCALE *
            MINIMAX_H3_ROPE_FRAMES_PER_LATENT[index % len(MINIMAX_H3_ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _temporal_position_span(num_latent_frames: int) -> float:
    # Preserve NumPy's pairwise summation: sequential torch summation diverges in the final ulp for longer clips.
    spans = np.ones(num_latent_frames, dtype=np.float64) * MINIMAX_H3_ROPE_FRAME_RESCALE
    for index, frames in enumerate(MINIMAX_H3_ROPE_FRAMES_PER_LATENT):
        spans[index::len(MINIMAX_H3_ROPE_FRAMES_PER_LATENT)] *= frames
    return float(spans.sum())


def build_packed_sequence(
        text_token_tags: torch.Tensor,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
        patch_size: tuple[int, int, int],
        keyframe_anchors: tuple[str, ...] = (),
) -> MiniMaxH3PackedLayout:
    if text_token_tags.ndim != 1:
        raise ValueError(f"text_token_tags must be one-dimensional, got {tuple(text_token_tags.shape)}.")
    valid_text_tags = (text_token_tags == MINIMAX_H3_TEXT_TAG) | (text_token_tags == MINIMAX_H3_VIDEO_TAG)
    if not bool(valid_text_tags.all()):
        raise ValueError("text_token_tags may contain only text and vision tags; semantic padding is not allowed.")

    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text_tokens = text_token_tags.shape[0]
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows

    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            anchor_time = (float(num_text_tokens) + _temporal_position_span(num_latent_frames) -
                           MINIMAX_H3_ROPE_FRAME_RESCALE)
        else:
            raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")
        rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    _fill_audio_positions(position_ids, slice(audio_start, video_start), num_audio_latents, float(num_text_tokens),
                          width_grid)

    video_positions = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_positions[:, :, 0] = temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
    video_positions[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_positions.reshape(-1, 3)

    video_indices = torch.cat([torch.arange(condition_start, audio_start), torch.arange(video_start, sequence_length)])
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text_tokens)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(device=token_tags.device, dtype=torch.long)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG
    return MiniMaxH3PackedLayout(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_condition_rows,
        num_condition_audio_rows=0,
        num_video_latent_frames=num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=num_audio_latents,
    )


def _reference_temporal_span(num_latent_frames: int) -> float:
    """Advance the Ref2VA clock using its required sequential summation order."""
    return sum(MINIMAX_H3_ROPE_FRAME_RESCALE *
               MINIMAX_H3_ROPE_FRAMES_PER_LATENT[index % len(MINIMAX_H3_ROPE_FRAMES_PER_LATENT)]
               for index in range(num_latent_frames))


def _frame_position_grid(
    latent_height: int,
    latent_width: int,
    patch_h: int,
    patch_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = spatial_position_grid(latent_width, patch_w, sqrt_area)
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
    continuation: MiniMaxH3ContinuationLayout | None = None,
) -> MiniMaxH3PackedLayout:
    """Build `[text | ordered references | continuation clip | target audio | target video]`.

    A continuation clip follows the references and advances the rotary clock as a video reference with its
    soundtrack would; its video rows then take the coordinates of the target's first latent frames and its audio rows
    those of the target's first audio latents, so the clip conditions the start of the target in place (the
    reference context of ComfyUI-H3-Continuum). VSA-H3 keeps its rows dense.
    """
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

    visual_row_counts = [
        0 if reference.media_type == "audio" else _num_video_rows(reference, patch_size) for reference in references
    ]
    for reference in references:
        if reference.media_type == "audio" and not reference.has_audio:
            raise ValueError("An audio reference must carry decoded waveform latents.")
        if reference.has_audio and reference.num_audio_latents <= 0:
            raise ValueError("An audio-bearing reference has no resolved audio latents.")

    num_text_tokens = int(text_token_tags.shape[0])
    num_target_video_rows = (num_latent_frames // patch_t) * (latent_height // patch_h) * (latent_width // patch_w)
    num_target_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    num_reference_video_rows = sum(visual_row_counts)
    num_reference_audio_rows = sum(reference.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS for reference in references
                                   if reference.has_audio)
    rows_per_target_frame = (latent_height // patch_h) * (latent_width // patch_w)
    if continuation is not None:
        if not 0 < continuation.num_latent_frames < num_latent_frames or continuation.num_audio_latents < 0:
            raise ValueError(f"Continuation clip {continuation} does not fit a target of {num_latent_frames} "
                             "latent frames.")
        num_reference_video_rows += continuation.num_latent_frames * rows_per_target_frame
        num_reference_audio_rows += continuation.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    sequence_length = (num_text_tokens + num_reference_video_rows + num_reference_audio_rows + num_target_audio_rows +
                       num_target_video_rows)

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
    target_frame_grid, target_width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    video_indices: list[torch.Tensor] = []
    audio_indices: list[torch.Tensor] = []
    reference_segments: list[tuple[str, int, tuple[int, int, int]]] = []
    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)
    for reference, visual_row_count in zip(references, visual_row_counts, strict=True):
        visual_latent_shape = (reference.num_latent_frames, reference.latent_height, reference.latent_width)
        if reference.media_type == "image":
            reference_segments.append(("image", visual_row_count, visual_latent_shape))
            rows = slice(cursor, cursor + visual_row_count)
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
            reference_segments.append(("audio", count, (0, 0, 0)))
            rows = slice(cursor, cursor + count)
            cursor = rows.stop
            audio_indices.append(torch.arange(rows.start, rows.stop))
            _fill_audio_positions(position_ids, rows, reference.num_audio_latents, rotary_time, target_width_grid)
            rotary_time += float(reference.num_audio_latents)
        elif reference.media_type == "video":
            audio_count = reference.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS if reference.has_audio else 0
            if audio_count:
                reference_segments.append(("audio", audio_count, (0, 0, 0)))
            reference_segments.append(("video", visual_row_count, visual_latent_shape))
            audio_rows = slice(cursor, cursor + audio_count)
            video_rows = slice(audio_rows.stop, audio_rows.stop + visual_row_count)
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
            frame_time = temporal_position_grid(reference.num_latent_frames, rotary_time)
            rows_per_frame = frame_grid.shape[0]
            position_ids[video_rows, 0] = frame_time.repeat_interleave(rows_per_frame)
            position_ids[video_rows, 1:] = frame_grid.repeat(reference.num_latent_frames // patch_t, 1)
            rotary_time += max(
                float(reference.num_audio_latents if reference.has_audio else 0),
                _reference_temporal_span(reference.num_latent_frames),
            )
        else:
            raise ValueError(f"Unsupported prepared reference type: {reference.media_type!r}.")

    if continuation is not None:
        audio_count = continuation.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
        visual_row_count = continuation.num_latent_frames * rows_per_target_frame
        if audio_count:
            reference_segments.append(("audio", audio_count, (0, 0, 0)))
        # Not "video": _h3_vsa_ref2va_segments keeps every other span dense.
        reference_segments.append(
            ("continuation", visual_row_count, (continuation.num_latent_frames, latent_height, latent_width)))
        audio_rows = slice(cursor, cursor + audio_count)
        video_rows = slice(audio_rows.stop, audio_rows.stop + visual_row_count)
        cursor = video_rows.stop
        if audio_count:
            audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
        video_indices.append(torch.arange(video_rows.start, video_rows.stop))
        rotary_time += max(float(continuation.num_audio_latents),
                           _reference_temporal_span(continuation.num_latent_frames))
        frame_time = temporal_position_grid(continuation.num_latent_frames, rotary_time)
        position_ids[video_rows, 0] = frame_time.repeat_interleave(rows_per_target_frame)
        position_ids[video_rows, 1:] = target_frame_grid.repeat(continuation.num_latent_frames // patch_t, 1)
        if audio_count:
            _fill_audio_positions(position_ids, audio_rows, continuation.num_audio_latents, rotary_time,
                                  target_width_grid)

    audio_start = cursor
    video_start = audio_start + num_target_audio_rows
    _fill_audio_positions(position_ids, slice(audio_start, video_start), num_audio_latents, rotary_time,
                          target_width_grid)
    frame_time = temporal_position_grid(num_latent_frames, rotary_time)
    position_ids[video_start:, 0] = frame_time.repeat_interleave(target_frame_grid.shape[0])
    position_ids[video_start:, 1:] = target_frame_grid.repeat(num_latent_frames // patch_t, 1)

    video_indices.append(torch.arange(video_start, sequence_length))
    audio_indices.append(torch.arange(audio_start, video_start))
    packed_video_indices = torch.cat(video_indices)
    packed_audio_indices = torch.cat(audio_indices)
    text_indices = torch.arange(num_text_tokens)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(device=token_tags.device, dtype=torch.long)
    token_tags[packed_audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[packed_video_indices] = MINIMAX_H3_VIDEO_TAG

    return MiniMaxH3PackedLayout(
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
        reference_segments=tuple(reference_segments),
    )


def build_row_timesteps(
    layout: MiniMaxH3PackedLayout | MiniMaxH3PackedBatchLayout,
    video_timestep: float | tuple[float, ...],
    audio_timestep: float | tuple[float, ...],
    condition_video_timestep: float | tuple[float, ...],
    condition_audio_timestep: float | tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(layout, MiniMaxH3PackedBatchLayout):
        count = len(layout.documents)

        def _per_document(value: float | tuple[float, ...], name: str) -> tuple[float, ...]:
            if isinstance(value, tuple):
                if len(value) != count:
                    raise ValueError(f"{name} has {len(value)} values for {count} packed H3 documents")
                return tuple(float(item) for item in value)
            return (float(value), ) * count

        video_values = _per_document(video_timestep, "video_timestep")
        audio_values = _per_document(audio_timestep, "audio_timestep")
        condition_video_values = _per_document(condition_video_timestep, "condition_video_timestep")
        condition_audio_values = _per_document(condition_audio_timestep, "condition_audio_timestep")
        row_timesteps = []
        for index, document in enumerate(layout.documents):
            unique, inverse = build_row_timesteps(
                document,
                video_values[index],
                audio_values[index],
                condition_video_values[index],
                condition_audio_values[index],
            )
            row_timesteps.append(unique[inverse])
        unique, inverse = torch.unique(torch.cat(row_timesteps), sorted=True, return_inverse=True)
        # Reuse an existing time table entry; padding must not change the real
        # document clocks, their unique time values, or their inverse indices.
        return unique, torch.nn.functional.pad(inverse, (0, layout.pad_tokens))

    if any(
            isinstance(value, tuple)
            for value in (video_timestep, audio_timestep, condition_video_timestep, condition_audio_timestep)):
        raise ValueError("Single-document H3 timestep inputs must be scalars")
    row_timesteps = torch.full((layout.real_sequence_length, ), video_timestep, dtype=torch.float32)
    row_timesteps[layout.video_indices[:layout.num_condition_video_rows]] = condition_video_timestep
    row_timesteps[layout.audio_indices[layout.num_condition_audio_rows:]] = audio_timestep
    row_timesteps[layout.audio_indices[:layout.num_condition_audio_rows]] = condition_audio_timestep
    unique, inverse = torch.unique(row_timesteps, sorted=True, return_inverse=True)
    return unique, torch.nn.functional.pad(inverse, (0, layout.pad_tokens))


def keyframe_condition_noise(
    condition_latent_shapes: tuple[tuple[int, int, int], ...],
    patch_size: tuple[int, int, int],
    latent_channels: int,
    generator: torch.Generator | list[torch.Generator] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    rows = []
    for num_latent_frames, latent_height, latent_width in condition_latent_shapes:
        noise = randn_tensor(
            (1, latent_channels, num_latent_frames, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        rows.append(patchify_video_latents(noise, patch_size))
    return torch.cat(rows)


def reference_condition_generator(noise_seed: int, *, device: torch.device) -> torch.Generator:
    """Create the per-record generator shared by Ref2VA train and validation."""
    if isinstance(noise_seed, bool) or not isinstance(noise_seed, Integral) or int(noise_seed) < 0:
        raise ValueError(f"Ref2VA reference noise seed must be a non-negative integer, got {noise_seed!r}")
    return torch.Generator(device=device).manual_seed(int(noise_seed))
