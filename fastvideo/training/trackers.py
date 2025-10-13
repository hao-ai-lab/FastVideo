"""Utilities for logging metrics and artifacts to external trackers.

This module is inspired by the trackers implementation in
https://github.com/huggingface/finetrainers and provides a minimal, shared
interface that can be used across all FastVideo training pipelines.
"""

from __future__ import annotations

import contextlib
import copy
import os
import pathlib
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any
from collections.abc import Iterable, Iterator

from fastvideo.logger import init_logger

logger = init_logger(__name__)


@dataclass
class Timer:
    """Simple timer utility used by the trackers."""

    name: str

    _start_time: float | None = None
    _end_time: float | None = None

    def start(self) -> None:
        self._start_time = time.perf_counter()

    def end(self) -> None:
        self._end_time = time.perf_counter()

    @property
    def elapsed_time(self) -> float:
        if self._start_time is None:
            raise RuntimeError(
                "Timer.start() must be called before elapsed_time")
        end_time = self._end_time if self._end_time is not None else time.perf_counter(
        )
        return end_time - self._start_time


class BaseTracker:
    """Base tracker implementation.

    The default tracker stores timing information but does not emit any logs.
    """

    def __init__(self) -> None:
        self._timed_metrics: dict[str, float] = {}

    @contextlib.contextmanager
    def timed(
        self,
        name: str,
    ) -> Iterator[Timer]:
        timer = Timer(name)
        timer.start()
        try:
            yield timer
        finally:
            timer.end()
            elapsed_time = timer.elapsed_time
            if name in self._timed_metrics:
                self._timed_metrics[name] += elapsed_time
            else:
                self._timed_metrics[name] = elapsed_time

    def log(self, metrics: dict[str, Any],
            step: int) -> None:  # pragma: no cover - interface
        """Log metrics for the given step."""
        # Merge timing metrics with provided metrics
        metrics = {**self._timed_metrics, **metrics}
        self._timed_metrics = {}

    def log_artifacts(self, artifacts: dict[str, Any], step: int) -> None:
        """Log artifacts such as videos or images.

        By default this is treated the same as :meth:`log`.
        """

        if artifacts:
            self.log(artifacts, step)

    def finish(self) -> None:  # pragma: no cover - interface
        """Finalize the tracker session."""

    def video(
        self,
        data: Any,
        *,
        caption: str | None = None,
        fps: int | None = None,
        format: str | None = None,
    ) -> Any | None:
        """Create a tracker specific video artifact.

        Trackers that do not support video artifacts should return ``None``.
        """

        return None


class DummyTracker(BaseTracker):
    """Tracker implementation used when logging is disabled."""

    def log(self, metrics: dict[str, Any],
            step: int) -> None:  # pragma: no cover - no-op
        super().log(metrics, step)

    def finish(self) -> None:  # pragma: no cover - no-op
        pass


class WandbTracker(BaseTracker):
    """Tracker implementation for Weights & Biases."""

    def __init__(
        self,
        experiment_name: str,
        log_dir: str,
        *,
        config: dict[str, Any] | None = None,
        run_name: str | None = None,
    ) -> None:
        super().__init__()

        import wandb

        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

        self._wandb = wandb
        self._run = wandb.init(
            project=experiment_name,
            dir=log_dir,
            config=config,
            name=run_name,
        )
        logger.info("Initialized Weights & Biases tracker")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        metrics = {**self._timed_metrics, **metrics}
        if metrics:
            self._run.log(metrics, step=step)
        self._timed_metrics = {}

    def finish(self) -> None:
        self._run.finish()

    def video(
        self,
        data: Any,
        *,
        caption: str | None = None,
        fps: int | None = None,
        format: str | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if caption is not None:
            kwargs["caption"] = caption
        if fps is not None:
            kwargs["fps"] = fps
        if format is not None:
            kwargs["format"] = format
        else:
            kwargs["format"] = "mp4"
        return self._wandb.Video(data, **kwargs)


class SequentialTracker(BaseTracker):
    """A tracker that forwards logging calls to a sequence of trackers."""

    def __init__(self, trackers: Iterable[BaseTracker]) -> None:
        super().__init__()
        self._trackers: list[BaseTracker] = list(trackers)

    @contextlib.contextmanager
    def timed(
        self,
        name: str,
    ) -> Iterator[Timer]:
        with super().timed(name) as timer:
            yield timer
        for tracker in self._trackers:
            tracker._timed_metrics = copy.deepcopy(self._timed_metrics)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for tracker in self._trackers:
            tracker.log({**self._timed_metrics, **metrics}, step)
        self._timed_metrics = {}

    def log_artifacts(self, artifacts: dict[str, Any], step: int) -> None:
        for tracker in self._trackers:
            tracker.log_artifacts(artifacts, step)
        self._timed_metrics = {}

    def finish(self) -> None:
        for tracker in self._trackers:
            tracker.finish()

    def video(
        self,
        data: Any,
        *,
        caption: str | None = None,
        fps: int | None = None,
        format: str | None = None,
    ) -> Any | None:
        for tracker in self._trackers:
            video = tracker.video(data, caption=caption, fps=fps, format=format)
            if video is not None:
                return video
        return None


class Trackers(str, Enum):
    NONE = "none"
    WANDB = "wandb"


SUPPORTED_TRACKERS = {tracker.value for tracker in Trackers}


def initialize_trackers(
    trackers: Iterable[str],
    *,
    experiment_name: str,
    config: dict[str, Any] | None,
    log_dir: str,
    run_name: str | None = None,
) -> BaseTracker:
    """Create tracker instances based on ``trackers`` configuration."""

    tracker_names = [tracker.lower() for tracker in trackers]
    if not tracker_names:
        return DummyTracker()

    unsupported = [
        name for name in tracker_names if name not in SUPPORTED_TRACKERS
    ]
    if unsupported:
        raise ValueError(
            f"Unsupported tracker(s) provided: {unsupported}. Supported trackers: {sorted(SUPPORTED_TRACKERS)}"
        )

    tracker_instances: list[BaseTracker] = []
    for tracker_name in tracker_names:
        if tracker_name == Trackers.NONE.value:
            tracker_instances.append(DummyTracker())
        elif tracker_name == Trackers.WANDB.value:
            tracker_instances.append(
                WandbTracker(
                    experiment_name,
                    os.path.abspath(log_dir),
                    config=config,
                    run_name=run_name,
                ))

    if not tracker_instances:
        return DummyTracker()

    if len(tracker_instances) == 1:
        return tracker_instances[0]

    return SequentialTracker(tracker_instances)


TrackerType = BaseTracker
