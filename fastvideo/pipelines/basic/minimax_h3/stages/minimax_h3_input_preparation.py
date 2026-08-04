# SPDX-License-Identifier: Apache-2.0
"""Unified request and target-geometry preparation for MiniMax H3."""

from __future__ import annotations

from typing import Any

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
    prepare_keyframe_image,
    resolve_canvas_size,
    video_latent_num_frames,
)
from fastvideo.pipelines.basic.minimax_h3.reference import (
    MiniMaxH3PreparedReference,
    prepare_reference,
    validate_references,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

MINIMAX_H3_KEYFRAMES_KEY = "minimax_h3_keyframes"
MINIMAX_H3_KEYFRAME_ANCHORS_KEY = "minimax_h3_keyframe_anchors"


def _component_value(module: Any, name: str) -> Any:
    value = getattr(module, name, None)
    if value is not None:
        return value
    config = getattr(module, "config", None)
    for owner in (getattr(config, "arch_config", None), config):
        value = getattr(owner, name, None)
        if value is not None:
            return value
    raise ValueError(f"MiniMax-H3 component {type(module).__name__} does not expose `{name}`.")


def _has_negative_prompt(value: str | list[str] | None) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return any(bool(item.strip()) for item in value)


def prepare_common_request(batch: ForwardBatch) -> None:
    """Validate H3's one-request, no-CFG contract and initialize its RNG."""
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
    """Resolve target geometry independently of the selected condition mode."""
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

    ratio = int(_component_value(vae, "spatial_compression_ratio"))
    if height % ratio or width % ratio:
        raise ValueError(f"MiniMax-H3 canvas {height}x{width} is not divisible by VAE ratio {ratio}.")
    return height, width, ratio


def resolve_target_num_frames(num_frames: object) -> int:
    """Align target pixels to the released causal-VAE chunk geometry."""
    if not isinstance(num_frames, int):
        raise TypeError("MiniMax-H3 `num_frames` must be an integer.")
    aligned = align_num_frames(num_frames)
    duration = aligned / MINIMAX_H3_FPS
    if not MINIMAX_H3_MIN_DURATION <= duration <= MINIMAX_H3_MAX_DURATION:
        raise ValueError(f"MiniMax-H3 generates {MINIMAX_H3_MIN_DURATION:g}-{MINIMAX_H3_MAX_DURATION:g} seconds at "
                         f"{MINIMAX_H3_FPS} fps; aligned num_frames={aligned}.")
    return aligned


class MiniMaxH3InputPreparationStage(PipelineStage):
    """Prepare FL2VA/T2VA or Ref2VA inputs without a parallel family state object."""

    def __init__(self, vae: object, audio_vae: object | None = None, *, ref2va: bool = False) -> None:
        super().__init__()
        if ref2va and audio_vae is None:
            raise ValueError("MiniMax-H3 Ref2VA input preparation requires an audio VAE.")
        self.vae = vae
        self.audio_vae = audio_vae
        self.ref2va = ref2va

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, lambda value: isinstance(value, str))
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        result.add_check("num_inference_steps", batch.num_inference_steps, V.positive_int)
        result.add_check("num_videos_per_prompt", batch.num_videos_per_prompt, lambda value: value == 1)
        result.add_check("latents", batch.latents, V.none_or_tensor)
        result.add_check("audio_latents", batch.audio_latents, V.none_or_tensor)
        if self.ref2va:
            result.add_check("references", batch.references, V.list_not_empty)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        result.add_check("height_latents", batch.height_latents, V.positive_int)
        result.add_check("width_latents", batch.width_latents, V.positive_int)
        result.add_check(
            "raw_latent_shape",
            batch.raw_latent_shape,
            lambda value: isinstance(value, tuple) and len(value) == 5 and all(
                isinstance(item, int) and item > 0 for item in value),
        )
        result.add_check("keyframes", batch.extra.get(MINIMAX_H3_KEYFRAMES_KEY), V.is_list)
        result.add_check(
            "keyframe_anchors",
            batch.extra.get(MINIMAX_H3_KEYFRAME_ANCHORS_KEY),
            lambda value: isinstance(value, tuple),
        )
        if self.ref2va:
            result.add_check(
                "prepared_references",
                batch.references,
                lambda value: isinstance(value, list) and bool(value) and all(
                    isinstance(reference, MiniMaxH3PreparedReference) for reference in value),
            )
        return result

    def _write_target_geometry(self, batch: ForwardBatch, height: int, width: int, ratio: int, num_frames: int) -> None:
        latent_height = height // ratio
        latent_width = width // ratio
        num_latent_frames = video_latent_num_frames(num_frames)
        latent_channels = int(_component_value(self.vae, "latent_channels"))

        batch.height = height
        batch.width = width
        batch.num_frames = num_frames
        batch.height_latents = latent_height
        batch.width_latents = latent_width
        batch.raw_latent_shape = (1, latent_channels, num_latent_frames, latent_height, latent_width)

    def _prepare_fl2va(self, batch: ForwardBatch) -> None:
        if batch.references:
            raise ValueError("MiniMax-H3 references belong to the Ref2VA pipeline.")

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
        self._write_target_geometry(batch, height, width, ratio, num_frames)

        batch.extra[MINIMAX_H3_KEYFRAME_ANCHORS_KEY] = tuple(anchor for anchor, _ in raw_keyframes)
        batch.extra[MINIMAX_H3_KEYFRAMES_KEY] = [
            prepare_keyframe_image(image, height, width, stretch=index == 0)
            for index, (_, image) in enumerate(raw_keyframes)
        ]

    def _prepare_ref2va(self, batch: ForwardBatch) -> None:
        if batch.image_path is not None or batch.pil_image is not None or batch.last_image is not None:
            raise ValueError("MiniMax-H3 Ref2VA accepts media through `references`, not FL2VA keyframe fields.")
        references = validate_references(list(batch.references or []))
        if self.audio_vae is None:
            raise RuntimeError("MiniMax-H3 Ref2VA input preparation has no audio VAE.")

        height, width, ratio = resolve_target_canvas(batch, self.vae, (16, 9))
        num_frames = resolve_target_num_frames(batch.num_frames)
        target_sample_rate = int(_component_value(self.audio_vae, "sampling_rate"))
        batch.references = [prepare_reference(reference, num_frames, target_sample_rate) for reference in references]
        self._write_target_geometry(batch, height, width, ratio, num_frames)
        batch.extra[MINIMAX_H3_KEYFRAMES_KEY] = []
        batch.extra[MINIMAX_H3_KEYFRAME_ANCHORS_KEY] = ()

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        del fastvideo_args
        prepare_common_request(batch)
        if self.ref2va:
            self._prepare_ref2va(batch)
        else:
            self._prepare_fl2va(batch)
        return batch


__all__ = [
    "MINIMAX_H3_KEYFRAME_ANCHORS_KEY",
    "MINIMAX_H3_KEYFRAMES_KEY",
    "MiniMaxH3InputPreparationStage",
    "prepare_common_request",
    "resolve_target_canvas",
    "resolve_target_num_frames",
]
