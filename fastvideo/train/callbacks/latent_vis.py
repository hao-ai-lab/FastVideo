# SPDX-License-Identifier: Apache-2.0
"""Intermediate-latent visualization callback for distillation methods.

Port of the legacy ``fastvideo/training/distillation_pipeline.py`` latent
logging to the modular trainer: every ``every_steps`` iterations, rank 0
decodes the method's latest latent snapshots (``method.latent_vis`` — the
student's rollout prediction plus the real- and fake-score predictions on
generator-update steps) through the model's ``decode_vis_latents`` hook and
logs them to the tracker as videos.

Both hooks are optional: methods that never populate ``latent_vis`` or models
without ``decode_vis_latents`` turn this callback into a no-op.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from fastvideo.distributed import get_world_group
from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback
from fastvideo.training.trackers import DummyTracker

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)

_DEFAULT_KEYS = (
    "generator_pred_video",
    "real_score_pred_video",
    "faker_score_pred_video",
)


class LatentVisCallback(Callback):
    """Decode and log intermediate training latents as tracker videos."""

    def __init__(
        self,
        every_steps: int = 100,
        keys: list[str] | None = None,
        fps: int = 24,
    ) -> None:
        self.every_steps = int(every_steps)
        self.keys = tuple(keys) if keys else _DEFAULT_KEYS
        self.fps = int(fps)
        self.tracker: Any = DummyTracker()

    def on_train_start(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        tracker = getattr(method, "tracker", None)
        if tracker is not None:
            self.tracker = tracker

    def on_training_step_end(
        self,
        method: TrainingMethod,
        loss_dict: dict[str, Any],
        iteration: int = 0,
    ) -> None:
        if self.every_steps <= 0 or iteration % self.every_steps != 0:
            return
        if get_world_group().rank != 0:
            return
        vis = getattr(method, "latent_vis", None)
        if not vis:
            return
        decode = getattr(getattr(method, "student", None), "decode_vis_latents", None)
        if decode is None:
            return

        artifacts: dict[str, Any] = {}
        for key in self.keys:
            latent = vis.get(key)
            if not isinstance(latent, torch.Tensor):
                continue
            try:
                clip = decode(latent)
            except Exception as exc:
                logger.warning("Latent visualization decode failed for %r: %s", key, exc)
                continue
            art = self.tracker.video(clip, fps=self.fps, format="mp4")
            if art is not None:
                artifacts[f"latent_vis/{key}"] = art
        if artifacts:
            self.tracker.log_artifacts(artifacts, iteration)
