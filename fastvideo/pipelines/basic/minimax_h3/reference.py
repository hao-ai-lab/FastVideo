# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 reference inputs and media preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
import os
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageOps
import torch

from fastvideo.models.vision_utils import load_image
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    resolve_canvas_size,
)

MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048
MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS = 2.0
MINIMAX_H3_QWEN_TEMPORAL_PATCH = 2
MINIMAX_H3_MAX_REFERENCE_IMAGES = 9
MINIMAX_H3_MAX_REFERENCE_VIDEOS = 3
MINIMAX_H3_MAX_REFERENCE_AUDIOS = 3
MINIMAX_H3_MAX_REFERENCES = 12


@dataclass(frozen=True)
class MiniMaxH3Reference:
    """One ordered Ref2VA input; decoding is deferred until request preparation."""

    source: Any
    media_type: Literal["image", "video", "audio"] = "image"
    soundtrack: Any | None = None
    fps: float | None = None
    sample_rate: int | None = None

    def __post_init__(self) -> None:
        if self.source is None:
            raise ValueError("A MiniMax-H3 reference requires a media source.")
        if self.media_type not in ("image", "video", "audio"):
            raise ValueError(f"Unsupported MiniMax-H3 reference type: {self.media_type!r}.")
        if self.soundtrack is not None and self.media_type != "video":
            raise ValueError("Only a video reference may carry a separate soundtrack.")
        if self.fps is not None and (self.media_type != "video" or isinstance(self.fps, bool)
                                     or not isinstance(self.fps, Real) or self.fps <= 0):
            raise ValueError("Reference `fps` must be positive and is only valid for video.")
        if self.sample_rate is not None and (self.media_type == "image" or isinstance(self.sample_rate, bool)
                                             or not isinstance(self.sample_rate, Integral) or self.sample_rate <= 0):
            raise ValueError("Reference `sample_rate` must be positive and is only valid for audio-bearing media.")


@dataclass
class MiniMaxH3PreparedReference:
    """Decoded Ref2VA media and the latent geometry resolved by condition encoding."""

    media_type: Literal["image", "video", "audio"]
    has_audio: bool = False
    image: Image.Image | None = None
    frames: np.ndarray | None = None
    waveform: torch.Tensor | None = None
    block_timestamps: list[float] = field(default_factory=list)
    num_latent_frames: int = 1
    latent_height: int = 0
    latent_width: int = 0
    num_audio_latents: int = 0


def validate_references(references: list[Any]) -> list[MiniMaxH3Reference]:
    """Validate the ordered Ref2VA media list and its per-modality limits."""
    if not references:
        raise ValueError("MiniMax-H3 Ref2VA requires at least one reference.")
    if not all(isinstance(reference, MiniMaxH3Reference) for reference in references):
        raise TypeError("Every Ref2VA entry must be a MiniMaxH3Reference.")

    typed = list(references)
    counts = {
        media_type: sum(reference.media_type == media_type for reference in typed)
        for media_type in ("image", "video", "audio")
    }
    limits = {
        "image": MINIMAX_H3_MAX_REFERENCE_IMAGES,
        "video": MINIMAX_H3_MAX_REFERENCE_VIDEOS,
        "audio": MINIMAX_H3_MAX_REFERENCE_AUDIOS,
    }
    for media_type, limit in limits.items():
        if counts[media_type] > limit:
            raise ValueError(f"MiniMax-H3 accepts at most {limit} {media_type} references.")
    if len(typed) > MINIMAX_H3_MAX_REFERENCES:
        raise ValueError(f"MiniMax-H3 accepts at most {MINIMAX_H3_MAX_REFERENCES} references.")
    if counts["audio"] == len(typed):
        raise ValueError("Audio references must be paired with at least one image or video reference.")
    return typed


def _import_av() -> Any:
    try:
        import av
    except ImportError as error:
        raise ImportError("Decoding MiniMax-H3 video or audio references requires PyAV.") from error
    return av


def _decode_audio_stream(av_module: Any, container: Any, stream: Any) -> tuple[torch.Tensor, int]:
    sample_rate = int(stream.codec_context.sample_rate)
    resampler = av_module.audio.resampler.AudioResampler(format="fltp", layout=stream.layout, rate=sample_rate)
    chunks: list[torch.Tensor] = []
    for frame in container.decode(stream):
        chunks.extend(torch.from_numpy(item.to_ndarray()) for item in resampler.resample(frame))
    chunks.extend(torch.from_numpy(item.to_ndarray()) for item in resampler.resample(None))
    if not chunks:
        raise ValueError("The reference audio stream contains no samples.")
    return torch.cat(chunks, dim=-1).to(torch.float32), sample_rate


def decode_reference_audio(source: str | os.PathLike[str]) -> tuple[torch.Tensor, int]:
    """Decode the first audio stream without imposing H3's target sample rate."""
    av_module = _import_av()
    with av_module.open(str(source)) as container:
        if not container.streams.audio:
            raise ValueError(f"Reference media {source!s} contains no audio stream.")
        return _decode_audio_stream(av_module, container, container.streams.audio[0])


def decode_reference_video(source: str | os.PathLike[str]) -> tuple[np.ndarray, float, tuple[torch.Tensor, int] | None]:
    """Decode RGB frames, display rotation, frame rate, and an optional soundtrack."""
    av_module = _import_av()
    with av_module.open(str(source)) as container:
        if not container.streams.video:
            raise ValueError(f"Reference media {source!s} contains no video stream.")
        stream = container.streams.video[0]
        frames = []
        rotation = 0.0
        for frame in container.decode(stream):
            rotation = float(getattr(frame, "rotation", 0.0) or 0.0)
            frames.append(frame.to_ndarray(format="rgb24"))
        if not frames:
            raise ValueError(f"Reference media {source!s} contains no video frames.")
        rate = stream.average_rate or getattr(stream, "guessed_rate", None)
        if rate is None:
            raise ValueError(f"Reference media {source!s} does not expose a frame rate.")

        soundtrack = None
        if container.streams.audio:
            container.seek(0)
            soundtrack = _decode_audio_stream(av_module, container, container.streams.audio[0])

    result = np.stack(frames)
    turns = round(rotation / 90.0) % 4
    if turns:
        result = np.ascontiguousarray(np.rot90(result, k=-turns, axes=(1, 2)))
    return result, float(rate), soundtrack


def resolve_reference_image_size(width: int, height: int) -> tuple[int, int]:
    """Resolve the released 2048-short-edge reference-image canvas."""
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
    """Normalize PIL, tensor, NumPy, or frame-list media to channel-last uint8."""
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
    """Resize an RGB reference image to its resolved condition canvas."""
    return image if image.size == (width, height) else image.resize((width, height), Image.Resampling.LANCZOS)


def resample_reference_frames(frames: np.ndarray, fps: float) -> np.ndarray:
    """Nearest-resample reference frames onto H3's fixed 24-fps timeline."""
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
    """Trim a reference video to the request and resize every RGB frame."""
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] == 0:
        raise ValueError(f"A reference video must be non-empty RGB frames, got {tuple(frames.shape)}.")
    frames = frames[:num_frames]
    height, width = resolve_canvas_size(frames.shape[2], frames.shape[1])
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)) for frame in frames])


def sample_reference_video_frames(frames: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    """Sample the released 2-fps Qwen presentation and its block timestamps."""
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
    """Normalize reference audio to stereo float32 at the audio-VAE rate."""
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
    except (ImportError, OSError):
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(sample_rate, target_sample_rate)
        resampled = resample_poly(
            waveform.numpy(),
            up=target_sample_rate // divisor,
            down=sample_rate // divisor,
            axis=-1,
        )
        return torch.from_numpy(np.asarray(resampled, dtype=np.float32))
    return torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)


def trim_reference_num_frames(num_frames: int) -> int:
    """Trim video references to the causal VAE's complete chunk geometry."""
    if num_frames < 1:
        raise ValueError(f"A reference video must have at least one frame, got {num_frames}.")
    return (
        max(1,
            (num_frames - MINIMAX_H3_LATENTS_PER_CHUNK) // MINIMAX_H3_FRAMES_PER_CHUNK) * MINIMAX_H3_FRAMES_PER_CHUNK +
        MINIMAX_H3_LATENTS_PER_CHUNK)


def _resolve_audio_source(source: Any) -> tuple[torch.Tensor, int | None]:
    if isinstance(source, str | os.PathLike):
        return decode_reference_audio(source)
    return torch.as_tensor(source), None


def prepare_reference(
    reference: MiniMaxH3Reference,
    num_frames: int,
    target_sample_rate: int,
) -> MiniMaxH3PreparedReference:
    """Decode and normalize one deferred Ref2VA medium."""
    prepared = MiniMaxH3PreparedReference(media_type=reference.media_type)
    if reference.media_type == "image":
        source = reference.source
        image = load_image(str(source)) if isinstance(source, str | os.PathLike) else source
        if not isinstance(image, Image.Image):
            pixels = reference_media_to_uint8(image)
            if pixels.ndim != 3 or pixels.shape[-1] != 3:
                raise ValueError(f"An image reference must be RGB, got {tuple(pixels.shape)}.")
            image = Image.fromarray(pixels)
        image = ImageOps.exif_transpose(image).convert("RGB")
        height, width = resolve_reference_image_size(*image.size)
        prepared.image = prepare_reference_image(image, height, width)
        return prepared

    if reference.media_type == "video":
        decoded_soundtrack = None
        decoded_sample_rate = None
        if isinstance(reference.source, str | os.PathLike):
            frames, decoded_fps, soundtrack = decode_reference_video(reference.source)
            if soundtrack is not None:
                decoded_soundtrack, decoded_sample_rate = soundtrack
        else:
            frames = reference_media_to_uint8(reference.source)
            decoded_fps = float(MINIMAX_H3_FPS)
        fps = float(reference.fps if reference.fps is not None else decoded_fps)
        prepared.frames = prepare_reference_frames(resample_reference_frames(frames, fps), num_frames)

        if reference.soundtrack is not None:
            waveform, source_rate = _resolve_audio_source(reference.soundtrack)
            decoded_soundtrack = waveform
            decoded_sample_rate = source_rate
        if decoded_soundtrack is not None:
            sample_rate = reference.sample_rate or decoded_sample_rate or target_sample_rate
            prepared.waveform = prepare_reference_waveform(
                decoded_soundtrack,
                int(sample_rate),
                target_sample_rate,
                max_duration=num_frames / MINIMAX_H3_FPS,
            )
            prepared.has_audio = True
        elif reference.sample_rate is not None:
            raise ValueError("A silent video reference cannot specify `sample_rate`.")
        return prepared

    waveform, decoded_sample_rate = _resolve_audio_source(reference.source)
    sample_rate = reference.sample_rate or decoded_sample_rate or target_sample_rate
    prepared.waveform = prepare_reference_waveform(
        waveform,
        int(sample_rate),
        target_sample_rate,
        max_duration=num_frames / MINIMAX_H3_FPS,
    )
    prepared.has_audio = True
    return prepared


__all__ = [
    "MINIMAX_H3_MAX_REFERENCES",
    "MINIMAX_H3_MAX_REFERENCE_AUDIOS",
    "MINIMAX_H3_MAX_REFERENCE_IMAGES",
    "MINIMAX_H3_MAX_REFERENCE_VIDEOS",
    "MINIMAX_H3_QWEN_TEMPORAL_PATCH",
    "MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS",
    "MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE",
    "MiniMaxH3PreparedReference",
    "MiniMaxH3Reference",
    "decode_reference_audio",
    "decode_reference_video",
    "prepare_reference",
    "prepare_reference_frames",
    "prepare_reference_image",
    "prepare_reference_waveform",
    "reference_media_to_uint8",
    "resample_reference_frames",
    "resolve_reference_image_size",
    "sample_reference_video_frames",
    "trim_reference_num_frames",
    "validate_references",
]
