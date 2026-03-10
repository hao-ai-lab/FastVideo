# SPDX-License-Identifier: Apache-2.0
"""VideoAlign reward functions for motion quality and
text-video alignment."""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any

import numpy as np
import torch

from fastvideo.logger import init_logger
from fastvideo.train.methods.rl.reward.utils import (
    prepare_images,
)

logger = init_logger(__name__)

# Add VideoAlign submodule to path for importing.
_VIDEOALIGN_ROOT = os.path.join(
    os.path.dirname(__file__), "VideoAlign"
)
if os.path.isdir(_VIDEOALIGN_ROOT):
    if _VIDEOALIGN_ROOT not in sys.path:
        sys.path.insert(0, _VIDEOALIGN_ROOT)

# Global cache of VideoAlign inferencers.
_VIDEOALIGN_INFERENCERS: dict[str, Any] = {}


def _normalize_device_str(device) -> str:
    if isinstance(device, torch.device):
        return str(device)
    return str(torch.device(device))


def set_videoalign_device(device) -> None:
    """Move cached VideoAlign inferencers to device."""
    key = _normalize_device_str(device)
    for old_key, inf in list(
        _VIDEOALIGN_INFERENCERS.items()
    ):
        if old_key != key and old_key.split(":")[0] != key:
            new_key = inf._key_prefix + ":" + key
            inf.to(device)
            _VIDEOALIGN_INFERENCERS[new_key] = inf
            del _VIDEOALIGN_INFERENCERS[old_key]


def _get_inferencer(
    device,
    checkpoint_path: str | None = None,
):
    """Get or create VideoAlign inferencer."""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "..", "..",
            "data", "VideoReward",
        )
    checkpoint_path = os.path.abspath(checkpoint_path)

    key = _normalize_device_str(device)
    cache_key = f"{checkpoint_path}:{key}"
    if cache_key not in _VIDEOALIGN_INFERENCERS:
        try:
            from inference import VideoVLMRewardInference
        except ImportError as exc:
            msg = (
                "VideoAlign not found. Ensure the "
                "VideoAlign submodule is checked out "
                "under fastvideo/train/methods/rl/"
                "reward/VideoAlign"
            )
            raise ImportError(msg) from exc

        inf = VideoVLMRewardInference(
            load_from_pretrained=checkpoint_path,
            device=device,
        )
        inf._key_prefix = checkpoint_path or "default"
        _VIDEOALIGN_INFERENCERS[cache_key] = inf
    return _VIDEOALIGN_INFERENCERS[cache_key]


def _convert_to_grayscale(
    frames: np.ndarray,
) -> np.ndarray:
    """Convert FHWC frames to grayscale FHWC."""
    if frames.ndim == 4 and frames.shape[-1] == 3:
        gray = np.mean(frames, axis=-1, keepdims=True)
        return np.repeat(gray.astype(np.uint8), 3, axis=-1)
    return frames


def _save_video_to_temp(
    frames: np.ndarray,
    fps: int = 8,
) -> str:
    """Save frames to a temporary MP4 file."""
    import cv2

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    h, w = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()
    return path


def videoalign_mq_score(
    device,
    checkpoint_path: str | None = None,
):
    """Return Motion Quality reward fn (grayscale)."""

    def _score(images, prompts, metadata, only_strict=False):
        inf = _get_inferencer(device, checkpoint_path)
        images_np = prepare_images(images)
        batch_scores = []

        for b in range(len(images_np)):
            frames = images_np[b]
            if frames.ndim == 3:
                frames = frames[np.newaxis]
            gray_frames = _convert_to_grayscale(frames)
            path = _save_video_to_temp(gray_frames)
            try:
                results = inf.reward(
                    [path], [""], use_norm=True
                )
                mq = float(results[0].get("MQ", 0))
                batch_scores.append(mq)
            finally:
                os.remove(path)

        reward = torch.tensor(
            batch_scores, device=device
        ).float()
        return {"avg": reward}, {}

    return _score


def videoalign_ta_score(
    device,
    checkpoint_path: str | None = None,
):
    """Return Text-Video Alignment reward fn (color)."""

    def _score(images, prompts, metadata, only_strict=False):
        inf = _get_inferencer(device, checkpoint_path)
        images_np = prepare_images(images)
        batch_scores = []

        for b in range(len(images_np)):
            frames = images_np[b]
            if frames.ndim == 3:
                frames = frames[np.newaxis]
            prompt = (
                prompts[b] if prompts and b < len(prompts)
                else ""
            )
            path = _save_video_to_temp(frames)
            try:
                results = inf.reward(
                    [path], [prompt], use_norm=True
                )
                ta = float(results[0].get("TA", 0))
                batch_scores.append(ta)
            finally:
                os.remove(path)

        reward = torch.tensor(
            batch_scores, device=device
        ).float()
        return {"avg": reward}, {}

    return _score
