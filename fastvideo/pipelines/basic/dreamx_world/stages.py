# SPDX-License-Identifier: Apache-2.0
"""DreamX-World pipeline stages."""

from __future__ import annotations

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult

from fastvideo.pipelines.basic.dreamx_world.camera_conditioning import (
    build_dreamx_camera_condition,
)


DREAMX_Y_CAMERA_KEY = "dreamx_y_camera"


class DreamXWorldCameraConditioningStage(PipelineStage):
    """Build PRoPE camera conditioning for DreamX-World-5B-Cam."""

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        del fastvideo_args
        if DREAMX_Y_CAMERA_KEY in batch.extra:
            return batch

        action_seq = batch.extra.get("dreamx_action_seq", batch.action_list)
        action_speed_list = batch.extra.get(
            "dreamx_action_speed_list", batch.action_speed_list)
        if action_seq is None:
            action_seq = ["w"]
        if action_speed_list is None:
            action_speed_list = [4]

        if isinstance(action_seq, str):
            action_seq = [action_seq]
        action_speed_list = [float(speed) for speed in action_speed_list]

        height = int(batch.height) if batch.height is not None else 704
        width = int(batch.width) if batch.width is not None else 1280
        num_frames = int(batch.num_frames)
        dtype = batch.latents.dtype if torch.is_tensor(batch.latents) else torch.float32
        device = batch.latents.device if torch.is_tensor(batch.latents) else "cpu"

        y_camera = build_dreamx_camera_condition(
            list(action_seq),
            action_speed_list,
            num_frames=num_frames,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
        )
        batch.extra[DREAMX_Y_CAMERA_KEY] = {
            key: value.unsqueeze(0)
            for key, value in y_camera.items()
        }
        return batch

    def verify_output(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        del fastvideo_args
        result = VerificationResult()
        y_camera = batch.extra.get(DREAMX_Y_CAMERA_KEY)
        result.add_check("dreamx_y_camera", y_camera, lambda value: isinstance(value, dict))
        if isinstance(y_camera, dict):
            result.add_check("dreamx_y_camera.viewmats", y_camera.get("viewmats"), torch.is_tensor)
            result.add_check("dreamx_y_camera.K", y_camera.get("K"), torch.is_tensor)
        return result
