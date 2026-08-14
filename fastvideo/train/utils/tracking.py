# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

from fastvideo.distributed import get_world_group
from fastvideo.logger import init_logger
from fastvideo.training.trackers import (
    initialize_trackers,
    Trackers,
)

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        CheckpointConfig,
        TrackerConfig,
    )

logger = init_logger(__name__)


def _wandb_usable() -> bool:
    """True when wandb can actually start a run (importable + credentials)."""
    try:
        import wandb
    except Exception:
        return False
    if os.environ.get("WANDB_API_KEY"):
        return True
    if os.environ.get("WANDB_MODE") in ("offline", "disabled"):
        return True
    try:
        # Covers ~/.netrc logins from a prior `wandb login`.
        return wandb.api.api_key is not None
    except Exception:
        return False


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
    if Trackers.WANDB.value in trackers and not _wandb_usable():
        logger.warning("wandb tracking requested but wandb is not importable "
                       "or no credentials are configured (WANDB_API_KEY); "
                       "continuing without the wandb tracker.")
        trackers = [t for t in trackers if t != Trackers.WANDB.value]

    tracker_log_dir = (checkpoint_config.output_dir or os.getcwd())
    if trackers:
        tracker_log_dir = os.path.join(tracker_log_dir, "tracker")

    tracker_config_dict = config if trackers else None
    tracker_entity = tracker_config.entity or None
    tracker_run_name = tracker_config.run_name or None
    project = (tracker_config.project_name or "fastvideo")

    try:
        return initialize_trackers(
            trackers,
            experiment_name=project,
            config=tracker_config_dict,
            log_dir=tracker_log_dir,
            entity=tracker_entity,
            run_name=tracker_run_name,
        )
    except Exception as exc:
        if Trackers.WANDB.value not in trackers:
            raise
        # A revoked API key or unreachable api.wandb.ai passes _wandb_usable()
        # (it only proves a key exists) and then throws inside wandb.init —
        # which must not kill a multi-node run at boot. Offline init never
        # contacts the API; the run stays syncable later via `wandb sync`.
        logger.warning("Tracker init failed (%s); retrying wandb in offline mode.", exc)
        os.environ["WANDB_MODE"] = "offline"
        try:
            return initialize_trackers(
                trackers,
                experiment_name=project,
                config=tracker_config_dict,
                log_dir=tracker_log_dir,
                entity=tracker_entity,
                run_name=tracker_run_name,
            )
        except Exception as offline_exc:
            logger.warning("Offline tracker init also failed (%s); continuing without trackers.", offline_exc)
            return initialize_trackers(
                [],
                experiment_name=project,
                config=None,
                log_dir=tracker_log_dir,
                entity=None,
                run_name=None,
            )
