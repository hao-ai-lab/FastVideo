# SPDX-License-Identifier: Apache-2.0
"""Request validation and geometry resolution for MiniMax H3."""

from __future__ import annotations

from PIL import Image, ImageOps
import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.vision_utils import load_image
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_MAX_DURATION,
    MINIMAX_H3_MIN_DURATION,
    align_num_frames,
    audio_latent_num_frames,
    prepare_keyframe_image,
    resolve_canvas_size,
    video_latent_num_frames,
)
from fastvideo.pipelines.basic.minimax_h3.types import get_minimax_h3_state
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult


def _spatial_compression_ratio(vae: object) -> int:
    ratio = getattr(vae, "spatial_compression_ratio", None)
    if ratio is None:
        config = getattr(vae, "config", None)
        arch = getattr(config, "arch_config", config)
        ratio = getattr(arch, "spatial_compression_ratio", None)
    if ratio is None:
        raise ValueError("MiniMax-H3 video VAE does not expose its spatial compression ratio.")
    return int(ratio)


def _has_negative_prompt(value: str | list[str] | None) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return any(bool(item.strip()) for item in value)


def prepare_common_request(batch: ForwardBatch) -> None:
    """Validate H3's shared one-request, no-CFG contract and initialize RNG."""
    if not isinstance(batch.prompt, str):
        raise ValueError("MiniMax-H3 packs one request, so `prompt` must be a single string.")
    if batch.prompt_embeds:
        raise ValueError("MiniMax-H3 requires conditioner token tags and does not accept standalone prompt_embeds.")
    if batch.num_videos_per_prompt != 1:
        raise ValueError("MiniMax-H3 generates one packed video/audio request at a time.")
    if _has_negative_prompt(batch.negative_prompt):
        raise ValueError("MiniMax-H3 is guidance-distilled and does not accept a negative prompt.")
    if batch.guidance_scale != 1.0 or batch.batch_cfg or batch.do_classifier_free_guidance:
        raise ValueError("MiniMax-H3 does not support classifier-free guidance; guidance_scale must be 1.0.")
    if batch.num_inference_steps < 2:
        raise ValueError("MiniMax-H3 needs at least two sigma grid points, including terminal zero.")

    if batch.generator is None:
        seed = 0 if batch.seed is None else int(batch.seed)
        batch.generator = torch.Generator("cpu").manual_seed(seed)
    elif isinstance(batch.generator, list) and len(batch.generator) != 1:
        raise ValueError("MiniMax-H3 accepts exactly one request generator.")

    fps = MINIMAX_H3_FPS if batch.fps is None else batch.fps
    if not isinstance(fps, int) or fps != MINIMAX_H3_FPS:
        raise ValueError(f"MiniMax-H3 uses a fixed {MINIMAX_H3_FPS} fps, got {batch.fps!r}.")
    batch.fps = MINIMAX_H3_FPS


def resolve_target_canvas(batch: ForwardBatch, vae: object, default_aspect: tuple[int, int]) -> tuple[int, int, int]:
    """Resolve target geometry independently of model-specific conditions."""
    if (batch.height is None) != (batch.width is None):
        raise ValueError("MiniMax-H3 `height` and `width` must be passed together, or neither.")
    if batch.height is None:
        height, width = resolve_canvas_size(*default_aspect)
    else:
        if not isinstance(batch.height, int) or not isinstance(batch.width, int):
            raise TypeError("MiniMax-H3 `height` and `width` must be integers.")
        height, width = batch.height, batch.width
        if height <= 0 or width <= 0 or height % MINIMAX_H3_CANVAS_MULTIPLE or width % MINIMAX_H3_CANVAS_MULTIPLE:
            raise ValueError(f"MiniMax-H3 `height` and `width` must be positive multiples of "
                             f"{MINIMAX_H3_CANVAS_MULTIPLE}, got {height}x{width}.")

    ratio = _spatial_compression_ratio(vae)
    if height % ratio or width % ratio:
        raise ValueError(f"MiniMax-H3 canvas {height}x{width} is not divisible by VAE ratio {ratio}.")
    return height, width, ratio


def resolve_target_num_frames(num_frames: object) -> int:
    if not isinstance(num_frames, int):
        raise TypeError("MiniMax-H3 `num_frames` must be an integer.")
    aligned = align_num_frames(num_frames)
    duration = aligned / MINIMAX_H3_FPS
    if not MINIMAX_H3_MIN_DURATION <= duration <= MINIMAX_H3_MAX_DURATION:
        raise ValueError(f"MiniMax-H3 generates {MINIMAX_H3_MIN_DURATION:g}-{MINIMAX_H3_MAX_DURATION:g} seconds at "
                         f"{MINIMAX_H3_FPS} fps; aligned num_frames={aligned}.")
    return aligned


class MiniMaxH3InputPreparationStage(PipelineStage):
    """Resolve the one-request T2VA/FL2VA plan without generic Wan preprocessing."""

    def __init__(self, vae: object) -> None:
        super().__init__()
        self.vae = vae

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, lambda value: isinstance(value, str))
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        result.add_check("num_inference_steps", batch.num_inference_steps, V.positive_int)
        result.add_check("num_videos_per_prompt", batch.num_videos_per_prompt, lambda value: value == 1)
        result.add_check("latents", batch.latents, V.none_or_tensor)
        result.add_check("audio_latents", batch.audio_latents, V.none_or_tensor)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        state = get_minimax_h3_state(batch)
        result = VerificationResult()
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check("height", state.height, V.positive_int)
        result.add_check("width", state.width, V.positive_int)
        result.add_check("num_frames", state.num_frames, V.positive_int)
        result.add_check("num_latent_frames", state.num_latent_frames, V.positive_int)
        result.add_check("num_audio_latents", state.num_audio_latents, V.positive_int)
        result.add_check("keyframes", state.keyframes, V.is_list)
        return result

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        del fastvideo_args
        prepare_common_request(batch)
        if batch.references:
            raise ValueError("MiniMax-H3 references belong to the Ref2VA pipeline, which starts in Stage 3.")

        if batch.image_path is not None:
            batch.pil_image = load_image(batch.image_path)
        for name, image in (("pil_image", batch.pil_image), ("last_image", batch.last_image)):
            if image is not None and not isinstance(image, Image.Image):
                raise TypeError(f"MiniMax-H3 `{name}` must be a PIL image, got {type(image).__name__}.")

        raw_keyframes = [(anchor, ImageOps.exif_transpose(image).convert("RGB"))
                         for anchor, image in (("first", batch.pil_image), ("last", batch.last_image))
                         if image is not None]
        default_aspect = raw_keyframes[0][1].size if raw_keyframes else (16, 9)
        height, width, ratio = resolve_target_canvas(batch, self.vae, default_aspect)
        num_frames = resolve_target_num_frames(batch.num_frames)

        state = get_minimax_h3_state(batch)
        state.height = batch.height = height
        state.width = batch.width = width
        state.num_frames = batch.num_frames = num_frames
        state.num_latent_frames = video_latent_num_frames(num_frames)
        state.latent_height = height // ratio
        state.latent_width = width // ratio
        state.num_audio_latents = audio_latent_num_frames(num_frames)
        state.keyframe_anchors = tuple(anchor for anchor, _ in raw_keyframes)
        state.keyframes = [
            prepare_keyframe_image(image, height, width, stretch=index == 0)
            for index, (_, image) in enumerate(raw_keyframes)
        ]
        return batch


__all__ = [
    "MiniMaxH3InputPreparationStage",
    "prepare_common_request",
    "resolve_target_canvas",
    "resolve_target_num_frames",
]
