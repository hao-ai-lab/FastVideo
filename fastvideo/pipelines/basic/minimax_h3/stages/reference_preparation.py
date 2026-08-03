# SPDX-License-Identifier: Apache-2.0
"""Ordered reference validation, decoding, and normalization for H3 Ref2VA."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.vision_utils import load_image
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_FPS,
    audio_latent_num_frames,
    video_latent_num_frames,
)
from fastvideo.pipelines.basic.minimax_h3.packing_ref2va import (
    MINIMAX_H3_MAX_REFERENCES,
    MINIMAX_H3_MAX_REFERENCE_AUDIOS,
    MINIMAX_H3_MAX_REFERENCE_IMAGES,
    MINIMAX_H3_MAX_REFERENCE_VIDEOS,
    prepare_reference_frames,
    prepare_reference_image,
    prepare_reference_waveform,
    reference_media_to_uint8,
    resample_reference_frames,
    resolve_reference_image_size,
)
from fastvideo.pipelines.basic.minimax_h3.stages.input_preparation import (
    prepare_common_request,
    resolve_target_canvas,
    resolve_target_num_frames,
)
from fastvideo.pipelines.basic.minimax_h3.types import (
    MiniMaxH3PreparedReference,
    MiniMaxH3Reference,
    get_minimax_h3_state,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult


def _arch_value(module: Any, name: str) -> Any:
    value = getattr(module, name, None)
    if value is None:
        config = getattr(module, "config", None)
        arch = getattr(config, "arch_config", config)
        value = getattr(arch, name, None)
    if value is None:
        raise ValueError(f"MiniMax-H3 component {type(module).__name__} does not expose `{name}`.")
    return value


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
    av_module = _import_av()
    with av_module.open(str(source)) as container:
        if not container.streams.audio:
            raise ValueError(f"Reference media {source!s} contains no audio stream.")
        return _decode_audio_stream(av_module, container, container.streams.audio[0])


def decode_reference_video(source: str | os.PathLike[str]) -> tuple[np.ndarray, float, tuple[torch.Tensor, int] | None]:
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


def _resolve_audio_source(source: Any) -> tuple[torch.Tensor, int | None]:
    if isinstance(source, str | os.PathLike):
        return decode_reference_audio(source)
    return torch.as_tensor(source), None


class MiniMaxH3ReferencePreparationStage(PipelineStage):
    """Resolve Ref2VA target geometry and prepare each deferred medium in order."""

    def __init__(self, vae: Any, audio_vae: Any) -> None:
        super().__init__()
        self.vae = vae
        self.audio_vae = audio_vae

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, lambda value: isinstance(value, str))
        result.add_check("references", batch.references, V.list_not_empty)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        result.add_check("num_inference_steps", batch.num_inference_steps, V.positive_int)
        result.add_check("latents", batch.latents, V.none_or_tensor)
        result.add_check("audio_latents", batch.audio_latents, V.none_or_tensor)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        state = get_minimax_h3_state(batch)
        result = VerificationResult()
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check("prepared_references", state.prepared_references, V.list_not_empty)
        result.add_check("height", state.height, V.positive_int)
        result.add_check("width", state.width, V.positive_int)
        result.add_check("num_frames", state.num_frames, V.positive_int)
        result.add_check("num_latent_frames", state.num_latent_frames, V.positive_int)
        result.add_check("num_audio_latents", state.num_audio_latents, V.positive_int)
        result.add_check("batch.references", batch.references, lambda value: value is None)
        return result

    @staticmethod
    def _validate_references(references: list[Any]) -> list[MiniMaxH3Reference]:
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

    def _prepare_reference(
        self,
        reference: MiniMaxH3Reference,
        num_frames: int,
        target_sample_rate: int,
    ) -> MiniMaxH3PreparedReference:
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

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        del fastvideo_args
        prepare_common_request(batch)
        if batch.image_path is not None or batch.pil_image is not None or batch.last_image is not None:
            raise ValueError("MiniMax-H3 Ref2VA accepts media through `references`, not FL2VA keyframe fields.")
        references = self._validate_references(list(batch.references or []))

        height, width, ratio = resolve_target_canvas(batch, self.vae, (16, 9))
        num_frames = resolve_target_num_frames(batch.num_frames)
        target_sample_rate = int(_arch_value(self.audio_vae, "sampling_rate"))
        prepared = [self._prepare_reference(reference, num_frames, target_sample_rate) for reference in references]

        state = get_minimax_h3_state(batch)
        state.height = batch.height = height
        state.width = batch.width = width
        state.num_frames = batch.num_frames = num_frames
        state.num_latent_frames = video_latent_num_frames(num_frames)
        state.latent_height = height // ratio
        state.latent_width = width // ratio
        state.num_audio_latents = audio_latent_num_frames(num_frames)
        state.keyframes = []
        state.keyframe_anchors = ()
        state.prepared_references = prepared
        batch.references = None
        return batch


__all__ = [
    "MiniMaxH3ReferencePreparationStage",
    "decode_reference_audio",
    "decode_reference_video",
]
