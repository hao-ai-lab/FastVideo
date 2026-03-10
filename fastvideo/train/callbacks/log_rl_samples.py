# SPDX-License-Identifier: Apache-2.0
"""Callback to log sampled RL videos to the tracker."""

from __future__ import annotations

import contextlib
import os
import tempfile
from typing import Any, TYPE_CHECKING

import imageio
import numpy as np
import torch

from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)


class LogRLSamplesCallback(Callback):
    """Log RL-sampled videos to the experiment tracker.

    Expects ``outputs`` to contain:

    - ``sample_videos``: uint8 tensor (B, 3, T, H, W).
    - ``sample_prompts``: list of prompt strings.

    Configuration (YAML ``callbacks.log_rl_samples``):

    .. code-block:: yaml

        callbacks:
          log_rl_samples:
            every_steps: 5
            max_videos: 4
            fps: 16
    """

    def __init__(
        self,
        *,
        every_steps: int = 1,
        max_videos: int = 4,
        fps: int = 16,
    ) -> None:
        self._every_steps = int(every_steps)
        self._max_videos = int(max_videos)
        self._fps = int(fps)

    def on_before_optimizer_step(
        self,
        method: TrainingMethod,
        iteration: int = 0,
        outputs: dict[str, Any] | None = None,
    ) -> None:
        if outputs is None:
            return
        if (
            self._every_steps > 0
            and iteration % self._every_steps != 0
        ):
            return

        videos = outputs.get("sample_videos")
        prompts = outputs.get("sample_prompts")
        if videos is None:
            return

        tracker = getattr(method, "tracker", None)
        if tracker is None:
            return

        self._log_videos(tracker, videos, prompts, iteration)

    def _log_videos(
        self,
        tracker: Any,
        videos: torch.Tensor,
        prompts: list[str] | None,
        step: int,
    ) -> None:
        n = min(len(videos), self._max_videos)
        tmp_dir = tempfile.mkdtemp(prefix="rl_samples_")
        video_logs = []

        try:
            for i in range(n):
                # (3, T, H, W) uint8 -> (T, H, W, 3) numpy.
                v = videos[i].permute(1, 2, 3, 0)
                frames = v.numpy().astype(np.uint8)
                fname = os.path.join(
                    tmp_dir, f"sample_{step}_{i}.mp4"
                )
                imageio.mimsave(fname, frames, fps=self._fps)

                caption = (
                    prompts[i]
                    if prompts and i < len(prompts)
                    else None
                )
                art = tracker.video(
                    fname, caption=caption, fps=self._fps
                )
                if art is not None:
                    video_logs.append(art)

            if video_logs:
                tracker.log_artifacts(
                    {"rl_sample_videos": video_logs},
                    step,
                )
                logger.info(
                    "Logged %d RL sample videos at step %d",
                    len(video_logs),
                    step,
                )
        finally:
            # Clean up temp files.
            for f in os.listdir(tmp_dir):
                with contextlib.suppress(OSError):
                    os.remove(os.path.join(tmp_dir, f))
            with contextlib.suppress(OSError):
                os.rmdir(tmp_dir)
