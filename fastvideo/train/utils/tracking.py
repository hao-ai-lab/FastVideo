# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

from fastvideo.distributed import get_world_group
from fastvideo.training.trackers import (
    initialize_trackers,
    Trackers,
)

if TYPE_CHECKING:
    from fastvideo.distillation.utils.distill_config import (
        CheckpointConfig,
        TrackerConfig,
    )


def build_tracker(
    tracker_config: TrackerConfig,
    checkpoint_config: CheckpointConfig,
    *,
    config: dict[str, Any] | None,
) -> Any:
    """Build a tracker instance for a distillation run."""

    world_group = get_world_group()

    trackers = list(tracker_config.trackers)
    if not trackers and str(tracker_config.project_name):
        trackers.append(Trackers.WANDB.value)
    if world_group.rank != 0:
        trackers = []

    tracker_log_dir = (checkpoint_config.output_dir or os.getcwd())
    if trackers:
        tracker_log_dir = os.path.join(tracker_log_dir, "tracker")

    tracker_config_dict = config if trackers else None
    tracker_run_name = tracker_config.run_name or None
    project = (tracker_config.project_name or "fastvideo")

    return initialize_trackers(
        trackers,
        experiment_name=project,
        config=tracker_config_dict,
        log_dir=tracker_log_dir,
        run_name=tracker_run_name,
    )
