# SPDX-License-Identifier: Apache-2.0
"""Ordered Ref2VA reference inputs and their media preparation, without PyTorch.

The CUDA path keeps the same logic in
``fastvideo.pipelines.basic.minimax_h3.reference``, but that module is built on
``torch.Tensor`` (waveforms) and imports ``fastvideo.models.vision_utils`` for
image loading, both of which pull the CUDA stack into an MLX-only install. The
Apple Silicon runtime therefore carries its own port with NumPy waveforms.

The stages here mirror the CUDA module step for step, because the values they
produce feed a layout the two runtimes share: decode and normalize each medium,
then resolve the reference canvas the video VAE and the Qwen presentation read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import math
from numbers import Integral, Real
import os
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageOps

from fastvideo.mlx_runtime.minimax_h3 import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_FPS,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    MINIMAX_H3_MAX_DURATION,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
)

MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048
MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS = 2.0
MINIMAX_H3_QWEN_TEMPORAL_PATCH = 2
MINIMAX_H3_MAX_REFERENCE_IMAGES = 9
MINIMAX_H3_MAX_REFERENCE_VIDEOS = 3
MINIMAX_H3_MAX_REFERENCE_AUDIOS = 3
MINIMAX_H3_MAX_REFERENCES = 12

# Canvas constants the shared packing contract fixes for every reference medium.
MINIMAX_H3_SHORT_EDGE = 768
MINIMAX_H3_MAX_PIXELS = 768 * 1344
MINIMAX_H3_CANVAS_MULTIPLE = 32
MINIMAX_H3_MIN_ASPECT_RATIO = 1 / 4
MINIMAX_H3_MAX_ASPECT_RATIO = 4

MINIMAX_H3_VISION_START_TOKEN = "<|vision_start|>"
MINIMAX_H3_IMAGE_PAD_TOKEN = "<|image_pad|>"
MINIMAX_H3_VIDEO_PAD_TOKEN = "<|video_pad|>"
MINIMAX_H3_VISION_END_TOKEN = "<|vision_end|>"


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
    """Decoded Ref2VA media and the latent geometry resolved by condition encoding.

    ``waveform`` is ``(channels, samples)`` float32, not a tensor: the MLX side
    never needs a device-resident waveform, the audio VAE converts on encode.
    """

    media_type: Literal["image", "video", "audio"]
    has_audio: bool = False
    image: Image.Image | None = None
    frames: np.ndarray | None = None
    waveform: np.ndarray | None = None
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


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _import_av() -> Any:
    try:
        import av
    except ImportError as error:
        raise ImportError("Decoding MiniMax-H3 video or audio references requires PyAV.") from error
    return av


def _fetch_image_bytes(url: str) -> bytes:
    try:
        import requests
    except ImportError as error:
        raise ImportError("Loading a reference image from a URL requires requests.") from error
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


def load_reference_image(image: str | os.PathLike[str] | Image.Image) -> Image.Image:
    """Load a path, URL, or PIL image to an EXIF-transposed RGB image."""
    if isinstance(image, str | os.PathLike):
        path = str(image)
        if path.startswith(("http://", "https://")):
            loaded = Image.open(io.BytesIO(_fetch_image_bytes(path)))
        elif os.path.isfile(path):
            loaded = Image.open(path)
        else:
            raise ValueError(f"Incorrect path or URL: {path} is not a valid path.")
    elif isinstance(image, Image.Image):
        loaded = image
    else:
        raise ValueError(f"Incorrect format used for the image: {type(image).__name__}.")
    return ImageOps.exif_transpose(loaded).convert("RGB")


def _decode_audio_stream(av_module: Any, container: Any, stream: Any) -> tuple[np.ndarray, int]:
    sample_rate = int(stream.codec_context.sample_rate)
    resampler = av_module.audio.resampler.AudioResampler(format="fltp", layout=stream.layout, rate=sample_rate)
    chunks: list[np.ndarray] = []
    for frame in container.decode(stream):
        chunks.extend(np.asarray(item.to_ndarray()) for item in resampler.resample(frame))
    chunks.extend(np.asarray(item.to_ndarray()) for item in resampler.resample(None))
    if not chunks:
        raise ValueError("The reference audio stream contains no samples.")
    return np.concatenate(chunks, axis=-1).astype(np.float32), sample_rate


def decode_reference_audio(source: str | os.PathLike[str]) -> tuple[np.ndarray, int]:
    """Decode the first audio stream without imposing H3's target sample rate."""
    av_module = _import_av()
    with av_module.open(str(source)) as container:
        if not container.streams.audio:
            raise ValueError(f"Reference media {source!s} contains no audio stream.")
        return _decode_audio_stream(av_module, container, container.streams.audio[0])


def decode_reference_video(source: str | os.PathLike[str]) -> tuple[np.ndarray, float, tuple[np.ndarray, int] | None]:
    """Decode RGB frames, display rotation, frame rate, and an optional soundtrack."""
    av_module = _import_av()
    with av_module.open(str(source)) as container:
        if not container.streams.video:
            raise ValueError(f"Reference media {source!s} contains no video stream.")
        stream = container.streams.video[0]
        rate = stream.average_rate or getattr(stream, "guessed_rate", None)
        if rate is None:
            raise ValueError(f"Reference media {source!s} does not expose a frame rate.")
        # Everything past the trimmed maximum is discarded downstream; stop
        # decoding (and buffering) once that many source frames are in hand.
        max_frames = math.ceil(MINIMAX_H3_MAX_DURATION * float(rate)) + 1
        frames = []
        rotation = 0.0
        for frame in container.decode(stream):
            rotation = float(getattr(frame, "rotation", 0.0) or 0.0)
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= max_frames:
                break
        if not frames:
            raise ValueError(f"Reference media {source!s} contains no video frames.")

        soundtrack = None
        if container.streams.audio:
            container.seek(0)
            soundtrack = _decode_audio_stream(av_module, container, container.streams.audio[0])

    result = np.stack(frames)
    turns = round(rotation / 90.0) % 4
    if turns:
        result = np.ascontiguousarray(np.rot90(result, k=-turns, axes=(1, 2)))
    return result, float(rate), soundtrack


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


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


def resolve_canvas_size(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    """The shared 768-short-edge canvas a reference video is resized onto."""
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


def reference_media_to_uint8(media: Any) -> np.ndarray:
    """Normalize PIL, NumPy, or frame-list media to channel-last uint8."""
    if isinstance(media, list):
        if not media:
            raise ValueError("A reference video must contain at least one frame.")
        return np.stack([reference_media_to_uint8(item) for item in media])
    if isinstance(media, Image.Image):
        return np.asarray(media.convert("RGB"))
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
    waveform: Any,
    sample_rate: int,
    target_sample_rate: int,
    max_duration: float,
) -> np.ndarray:
    """Normalize reference audio to stereo float32 at the audio-VAE rate.

    Resampling uses SciPy's polyphase filter rather than torchaudio's sinc
    kernel: the MLX runtime has no torchaudio, and the audio VAE is an
    approximation target rather than a sample-exact oracle.
    """
    waveform = np.asarray(waveform)
    if waveform.ndim != 2 or waveform.shape[0] not in (1, MINIMAX_H3_AUDIO_CHANNELS):
        raise ValueError(
            f"A reference soundtrack must be mono or stereo [channels, samples], got {tuple(waveform.shape)}.")
    if waveform.shape[-1] == 0:
        raise ValueError("A reference soundtrack must contain samples.")
    if sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("Reference audio sample rates must be positive.")
    waveform = waveform.astype(np.float32)[:, :int(max_duration * sample_rate)]
    if waveform.shape[0] == 1:
        waveform = np.repeat(waveform, MINIMAX_H3_AUDIO_CHANNELS, axis=0)
    if sample_rate == target_sample_rate:
        return np.ascontiguousarray(waveform)
    try:
        from scipy.signal import resample_poly
    except ImportError as error:
        raise ImportError("Resampling a reference soundtrack requires SciPy.") from error
    divisor = math.gcd(sample_rate, target_sample_rate)
    resampled = resample_poly(waveform, up=target_sample_rate // divisor, down=sample_rate // divisor, axis=-1)
    return np.ascontiguousarray(resampled.astype(np.float32))


def trim_reference_num_frames(num_frames: int) -> int:
    """Trim video references to the causal VAE's complete chunk geometry."""
    if num_frames < 1:
        raise ValueError(f"A reference video must have at least one frame, got {num_frames}.")
    return (
        max(1,
            (num_frames - MINIMAX_H3_LATENTS_PER_CHUNK) // MINIMAX_H3_FRAMES_PER_CHUNK) * MINIMAX_H3_FRAMES_PER_CHUNK +
        MINIMAX_H3_LATENTS_PER_CHUNK)


def _resolve_audio_source(source: Any) -> tuple[np.ndarray, int | None]:
    if isinstance(source, str | os.PathLike):
        return decode_reference_audio(source)
    return np.asarray(source), None


def prepare_reference(
    reference: MiniMaxH3Reference,
    num_frames: int,
    target_sample_rate: int,
) -> MiniMaxH3PreparedReference:
    """Decode and normalize one deferred Ref2VA medium."""
    prepared = MiniMaxH3PreparedReference(media_type=reference.media_type)
    if reference.media_type == "image":
        source = reference.source
        image = load_reference_image(source) if isinstance(source, str | os.PathLike) else source
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


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def build_ref2va_presentation(
    tokenizer: Any,
    prompt: str,
    references: list[MiniMaxH3PreparedReference],
    image_token_counts: list[int],
    video_block_token_counts: list[int],
) -> tuple[list[int], list[int]]:
    """Tokenize ordered reference labels, vision blocks, then the prompt.

    Rows of a vision block carry the video tag, exactly as on the CUDA path:
    the DiT selects its modality AdaLN row from that tag, and a reference image
    is a one-frame video there.
    """

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


def _token_ids(tokenized: Any) -> list[int]:
    input_ids = tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized.input_ids
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("MiniMax H3 tokenization must produce exactly one sequence.")
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


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
    "build_ref2va_presentation",
    "decode_reference_audio",
    "decode_reference_video",
    "load_reference_image",
    "prepare_reference",
    "prepare_reference_frames",
    "prepare_reference_image",
    "prepare_reference_waveform",
    "resample_reference_frames",
    "resolve_canvas_size",
    "resolve_reference_image_size",
    "sample_reference_video_frames",
    "trim_reference_num_frames",
    "validate_references",
]
