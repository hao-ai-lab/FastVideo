# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for loading PIL video frames into batch tensors."""

from __future__ import annotations

import torch

from fastvideo.models.vision_utils import load_video, normalize, numpy_to_pt, pil_to_numpy, resize


def load_video_path_to_tensor(
    video_path: str,
    *,
    target_height: int,
    target_width: int,
    target_fps: float | None = None,
    target_num_frames: int | None = None,
) -> torch.Tensor:
    """Load a video file and return ``[B, C, T, H, W]`` in [-1, 1]."""
    pil_images, original_fps = load_video(video_path, return_fps=True)
    if target_fps is not None and original_fps is not None:
        frame_skip = max(1, int(original_fps // target_fps))
        if frame_skip > 1:
            pil_images = pil_images[::frame_skip]
    if target_num_frames is not None and len(pil_images) > target_num_frames:
        pil_images = pil_images[:target_num_frames]
    resized_images = [
        resize(img, target_height, target_width, resize_mode="default", resample="lanczos") for img in pil_images
    ]
    video_numpy = normalize(pil_to_numpy(resized_images))
    video_tensor = numpy_to_pt(video_numpy)
    return video_tensor.permute(1, 0, 2, 3).unsqueeze(0)
