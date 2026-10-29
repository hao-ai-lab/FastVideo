# SPDX-License-Identifier: Apache-2.0
"""Pika H3 Flex input preparation: bucket canvas, cropped keyframes, tiered references."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image, ImageOps

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.vision_utils import load_image
from fastvideo.pipelines.basic.minimax_h3.packing import prepare_keyframe_image
from fastvideo.pipelines.basic.minimax_h3.reference import (
    MiniMaxH3PreparedReference,
    prepare_reference,
    validate_references,
)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import (
    MINIMAX_H3_KEYFRAME_ANCHORS_KEY,
    MINIMAX_H3_KEYFRAMES_KEY,
    MiniMaxH3InputPreparationStage,
    prepare_common_request,
)
from fastvideo.pipelines.basic.pika_h3_flex.buckets import (
    PIKA_TIERS,
    resolve_canvas,
    resolve_frames,
    resolve_tier,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

logger = init_logger(__name__)

# Pika's released reference-video floor: two seconds at 24 fps is 48 frames,
# but the adapter's own validator requires at least 22 decoded frames.
PIKA_MIN_REFERENCE_VIDEO_FRAMES = 22


class PikaH3FlexInputPreparationStage(MiniMaxH3InputPreparationStage):
    """Prepare T2VA/I2VA/L2VA/FL2VA or Ref2VA inputs on a Pika bucket."""

    def __init__(self, vae: Any, audio_vae: Any | None = None, *, ref2va: bool = False) -> None:
        super().__init__(vae, audio_vae, ref2va=ref2va)
        self._tier = "p768"

    def _request_tier(self, fastvideo_args: FastVideoArgs) -> str:
        return resolve_tier(getattr(fastvideo_args.pipeline_config, "pika_resolution_tier", None))

    def _resolve_canvas(
        self,
        batch: ForwardBatch,
        default_aspect: tuple[int, int],
    ) -> tuple[int, int, str, str]:
        height, width, tier, aspect = resolve_canvas(
            batch.height,
            batch.width,
            default_aspect=default_aspect,
            default_tier=self._tier,
        )
        if (batch.height, batch.width) != (height, width):
            logger.info("Pika H3 Flex snapped canvas %sx%s to %sx%s (%s %s).", batch.height, batch.width, height, width,
                        tier, aspect)
        ratio = int(self.vae.spatial_compression_ratio)
        if height % ratio or width % ratio:
            raise ValueError(f"Pika H3 Flex canvas {height}x{width} is not divisible by VAE ratio {ratio}.")
        return height, width, tier, aspect

    def _prepare_fl2va(self, batch: ForwardBatch) -> None:
        if batch.references:
            raise ValueError("Pika H3 Flex references belong to the Ref2VA pipeline.")

        if batch.image_path is not None:
            batch.pil_image = load_image(batch.image_path)
        for name, image in (("pil_image", batch.pil_image), ("last_image", batch.last_image)):
            if image is not None and not isinstance(image, Image.Image):
                raise TypeError(f"Pika H3 Flex `{name}` must be a PIL image, got {type(image).__name__}.")

        raw_keyframes = [(anchor, ImageOps.exif_transpose(image).convert("RGB"))
                         for anchor, image in (("first", batch.pil_image), ("last", batch.last_image))
                         if image is not None]
        default_aspect = raw_keyframes[0][1].size if raw_keyframes else (16, 9)
        height, width, _, _ = self._resolve_canvas(batch, default_aspect)
        num_frames = resolve_frames(batch.num_frames)
        ratio = int(self.vae.spatial_compression_ratio)
        self._write_target_geometry(batch, height, width, ratio, num_frames)

        batch.extra[MINIMAX_H3_KEYFRAME_ANCHORS_KEY] = tuple(anchor for anchor, _ in raw_keyframes)
        # Pika crops both endpoints. MiniMax-H3 stretches the first frame; using
        # that fit here would distort every Pika image on a bucketed canvas.
        batch.extra[MINIMAX_H3_KEYFRAMES_KEY] = [
            prepare_keyframe_image(image, height, width, stretch=False) for _, image in raw_keyframes
        ]

    def _prepare_ref2va(self, batch: ForwardBatch) -> None:
        if batch.image_path is not None or batch.pil_image is not None or batch.last_image is not None:
            raise ValueError("Pika H3 Flex Ref2VA accepts media through `references`, not FL2VA keyframe fields.")
        references = validate_references(list(batch.references or []))
        if self.audio_vae is None:
            raise RuntimeError("Pika H3 Flex Ref2VA input preparation has no audio VAE.")

        height, width, tier, _ = self._resolve_canvas(batch, (16, 9))
        num_frames = resolve_frames(batch.num_frames)
        target_sample_rate = int(self.audio_vae.sampling_rate)
        rule = PIKA_TIERS[tier]
        prepared: list[MiniMaxH3PreparedReference] = []
        for reference in references:
            item = prepare_reference(
                reference,
                num_frames,
                target_sample_rate,
                reference_short_edge=rule.reference_short_edge,
                reference_canvas_short_edge=rule.short_edge,
                reference_canvas_max_pixels=rule.max_pixels,
            )
            if (item.media_type == "video" and item.frames is not None
                    and item.frames.shape[0] < PIKA_MIN_REFERENCE_VIDEO_FRAMES):
                raise ValueError(f"A Pika H3 Flex reference video needs at least "
                                 f"{PIKA_MIN_REFERENCE_VIDEO_FRAMES} decoded frames at 24 fps; got "
                                 f"{item.frames.shape[0]}.")
            prepared.append(item)
        if not prepared:
            raise ValueError("Pika H3 Flex Ref2VA requires at least one non-audio reference.")

        batch.references = prepared
        self._write_target_geometry(batch, height, width, int(self.vae.spatial_compression_ratio), num_frames)
        batch.extra[MINIMAX_H3_KEYFRAMES_KEY] = []
        batch.extra[MINIMAX_H3_KEYFRAME_ANCHORS_KEY] = ()

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        prepare_common_request(batch)
        self._tier = self._request_tier(fastvideo_args)
        if self.ref2va:
            self._prepare_ref2va(batch)
        else:
            self._prepare_fl2va(batch)
        return batch


__all__ = ["PIKA_MIN_REFERENCE_VIDEO_FRAMES", "PikaH3FlexInputPreparationStage"]
