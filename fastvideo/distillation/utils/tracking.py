# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any

from fastvideo.distributed import get_world_group
from fastvideo.training.trackers import initialize_trackers, Trackers


def build_tracker(training_args: Any, *, config: dict[str, Any] | None) -> Any:
    """Build a tracker instance for a distillation run.

    Tracker selection is rank0-only; other ranks get a no-op tracker from
    ``initialize_trackers([])``.
    """

    world_group = get_world_group()

    trackers = list(training_args.trackers)
    if not trackers and str(training_args.tracker_project_name):
        trackers.append(Trackers.WANDB.value)
    if world_group.rank != 0:
        trackers = []

    tracker_log_dir = training_args.output_dir or os.getcwd()
    if trackers:
        tracker_log_dir = os.path.join(tracker_log_dir, "tracker")

    tracker_config = config if trackers else None
    tracker_run_name = training_args.wandb_run_name or None
    project = training_args.tracker_project_name or "fastvideo"

    return initialize_trackers(
        trackers,
        experiment_name=project,
        config=tracker_config,
        log_dir=tracker_log_dir,
        run_name=tracker_run_name,
    )

