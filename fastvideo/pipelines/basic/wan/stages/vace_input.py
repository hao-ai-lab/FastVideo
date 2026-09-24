# SPDX-License-Identifier: Apache-2.0
"""Wan-VACE input preparation: mask video, reference images, and reference-only pixels."""

import torch
import torch.nn.functional as F

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.vision_utils import load_image, normalize, numpy_to_pt, pil_to_numpy
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.pipelines.stages.video_tensor_utils import load_video_path_to_tensor


def preprocess_vace_reference_images(
    references: list[str] | None,
    image_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    if not references:
        return []
    height, width = image_size
    processed = []
    for path in references:
        image = load_image(path)
        tensor = numpy_to_pt(normalize(pil_to_numpy([image]))).squeeze(0)
        img_height, img_width = tensor.shape[-2:]
        scale = min(height / img_height, width / img_width)
        new_height, new_width = int(img_height * scale), int(img_width * scale)
        resized = F.interpolate(tensor.unsqueeze(0), size=(new_height, new_width), mode="bilinear",
                                align_corners=False).squeeze(0)
        top = (height - new_height) // 2
        left = (width - new_width) // 2
        canvas = torch.ones(3, height, width, device=device, dtype=dtype)
        canvas[:, top:top + new_height, left:left + new_width] = resized.to(device=device, dtype=dtype)
        processed.append(canvas)
    return processed


class WanVACEInputStage(PipelineStage):
    """Prepare VACE-specific inputs after shared validation and text encoding."""

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.mask_path is not None:
            mask_tensor = load_video_path_to_tensor(
                batch.mask_path,
                target_height=batch.height,
                target_width=batch.width,
                target_fps=batch.fps,
                target_num_frames=batch.num_frames,
            )
            expected_frames = (batch.video_latent.shape[2] if batch.video_latent is not None else batch.num_frames)
            if mask_tensor.shape[2] != expected_frames:
                raise ValueError(
                    f"VACE mask has {mask_tensor.shape[2]} frames but video requires {expected_frames} frames")
            batch.mask_video = mask_tensor

        if batch.references:
            batch.vace_reference_images = preprocess_vace_reference_images(
                batch.references,
                (batch.height, batch.width),
                get_local_torch_device(),
                torch.float32,
            )
            batch.vace_num_reference_frames = len(batch.vace_reference_images)

        # Diffusers also supplies zero pixels for unconditional and mask-only VACE.
        if batch.video_latent is None:
            batch.video_latent = torch.zeros(
                1,
                3,
                batch.num_frames,
                batch.height,
                batch.width,
                dtype=torch.float32,
            )
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        return result
