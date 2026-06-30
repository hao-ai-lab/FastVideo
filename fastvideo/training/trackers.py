"""Utilities for logging metrics and artifacts to external trackers.

This module is inspired by the trackers implementation in
https://github.com/huggingface/finetrainers and provides a minimal, shared
interface that can be used across all FastVideo training pipelines.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import contextlib
import copy
import math
import os
import pathlib
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch

from fastvideo.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_VIDEO_FPS = 16
_MISSING_ARTIFACT = object()


def _sanitize_wandb_config(value: Any) -> Any:
    """Best-effort conversion of nested config objects to W&B-safe values."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_wandb_config(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_sanitize_wandb_config(v) for v in value]
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(dtype=torch.float32)
        if tensor.ndim == 0 or tensor.numel() == 1:
            return tensor.item()
        if tensor.numel() <= 256:
            return tensor.tolist()
        return {
            "_type": "tensor_summary",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    if callable(value):
        return getattr(value, "__name__", repr(value))
    return repr(value)


def _prepare_video_array(data: Any) -> np.ndarray:
    """Convert W&B-style TCHW/BTCHW video data into GIF-ready frames."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    video = np.asarray(data)
    if video.ndim == 4:
        video = video.reshape(1, *video.shape)
    elif video.ndim != 5:
        raise ValueError("Video data must have shape [T, C, H, W] or [B, T, C, H, W]")

    batch_size, num_frames, channels, height, width = video.shape
    if batch_size == 0 or num_frames == 0:
        raise ValueError("Video data must contain at least one batch item and one frame")
    if channels not in (1, 3, 4):
        raise ValueError(f"Video data must have 1, 3, or 4 channels; got {channels}")
    if video.dtype != np.uint8:
        logger.warning("Converting video data to uint8 for SwanLab")
        video = video.astype(np.uint8)

    # Match wandb.Video's batch tiling so the same input has a familiar layout
    # in either tracker.
    if batch_size & (batch_size - 1):
        padded_batch_size = 1 << batch_size.bit_length()
        padding = np.zeros(
            (padded_batch_size - batch_size, num_frames, channels, height, width),
            dtype=video.dtype,
        )
        video = np.concatenate((video, padding), axis=0)

    num_rows = 1 << ((batch_size.bit_length() - 1) // 2)
    num_columns = video.shape[0] // num_rows
    video = video.reshape(num_rows, num_columns, num_frames, channels, height, width)
    video = np.transpose(video, axes=(2, 0, 4, 1, 5, 3))
    video = video.reshape(num_frames, num_rows * height, num_columns * width, channels)
    if channels == 1:
        video = video[..., 0]
    return np.ascontiguousarray(video)


def _read_video_file(file_path: str) -> tuple[list[np.ndarray], float | None]:
    """Decode a video file and return its frames and encoded frame rate."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(file_path)
    try:
        metadata = reader.get_meta_data() or {}
        frames = [np.asarray(frame) for frame in reader]
    finally:
        reader.close()

    if not frames:
        raise ValueError(f"Video file contains no frames: {file_path}")

    raw_source_fps = metadata.get("fps")
    try:
        source_fps = float(str(raw_source_fps))
    except ValueError:
        source_fps = None
    if source_fps is not None and (not math.isfinite(source_fps) or source_fps <= 0):
        source_fps = None
    return frames, source_fps


def _coerce_video_fps(fps: int | float | None) -> float:
    if fps is None:
        return _DEFAULT_VIDEO_FPS
    value = float(fps)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Video fps must be a positive finite number; got {fps!r}")
    return value


def _write_gif(file_path: str, frames: Any, fps: float) -> None:
    """Encode frames as an animated GIF for SwanLab's video API."""
    from PIL import Image

    images = []
    for frame in frames:
        array = np.asarray(frame)
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[-1] not in (3, 4)):
            raise ValueError(f"GIF frames must be grayscale, RGB, or RGBA; got shape {array.shape}")
        images.append(Image.fromarray(array))

    if not images:
        raise ValueError("Video data must contain at least one frame")

    duration_ms = max(1, round(1000 / fps))
    images[0].save(
        file_path,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


@dataclass(frozen=True)
class _SequentialArtifact:
    """Backend-specific versions of one artifact created by a tracker group."""

    values: tuple[Any | None, ...]


def _select_tracker_artifact(value: Any, tracker_index: int) -> Any:
    """Resolve nested sequential artifacts for one child tracker."""
    if isinstance(value, _SequentialArtifact):
        selected = value.values[tracker_index]
        return _MISSING_ARTIFACT if selected is None else selected
    if isinstance(value, dict):
        selected_dict = {}
        for key, item in value.items():
            selected = _select_tracker_artifact(item, tracker_index)
            if selected is not _MISSING_ARTIFACT:
                selected_dict[key] = selected
        return selected_dict if selected_dict else _MISSING_ARTIFACT
    if isinstance(value, list):
        selected_list = [
            selected for item in value
            if (selected := _select_tracker_artifact(item, tracker_index)) is not _MISSING_ARTIFACT
        ]
        return selected_list if selected_list else _MISSING_ARTIFACT
    if isinstance(value, tuple):
        selected_tuple = tuple(selected for item in value
                               if (selected := _select_tracker_artifact(item, tracker_index)) is not _MISSING_ARTIFACT)
        return selected_tuple if selected_tuple else _MISSING_ARTIFACT
    return value


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
            raise RuntimeError("Timer.start() must be called before elapsed_time")
        end_time = self._end_time if self._end_time is not None else time.perf_counter()
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

    def log(self, metrics: dict[str, Any], step: int) -> None:  # pragma: no cover - interface
        """Log metrics for the given step."""
        # Merge timing metrics with provided metrics
        metrics = {**self._timed_metrics, **metrics}
        self._timed_metrics = {}

    def log_artifacts(self, artifacts: dict[str, Any], step: int) -> None:
        """Log tracker artifacts such as sampled media.

        By default this is treated the same as :meth:`log`.
        """

        if artifacts:
            self.log(artifacts, step)

    def log_file(
        self,
        file_path: str,
        name: str | None = None,
    ) -> None:
        """Attach a file to the tracker run (e.g. config YAML)."""

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

    def log(self, metrics: dict[str, Any], step: int) -> None:  # pragma: no cover - no-op
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
            config=(_sanitize_wandb_config(config) if config is not None else None),
            name=run_name,
        )
        logger.info("Initialized Weights & Biases tracker")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        metrics = {**self._timed_metrics, **metrics}
        if metrics:
            self._run.log(metrics, step=step)
        self._timed_metrics = {}

    def log_artifacts(self, artifacts: dict[str, Any], step: int) -> None:
        if not artifacts:
            return

        normalized_artifacts = dict(artifacts)
        if "validation_ref_videos" in normalized_artifacts:
            normalized_artifacts["reference_video/videos"] = (normalized_artifacts.pop("validation_ref_videos"))

        self.log(normalized_artifacts, step)

    def log_file(
        self,
        file_path: str,
        name: str | None = None,
    ) -> None:
        self._wandb.save(
            file_path,
            base_path=os.path.dirname(file_path),
        )

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
        kwargs["fps"] = fps if fps is not None else _DEFAULT_VIDEO_FPS
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
        for tracker_index, tracker in enumerate(self._trackers):
            tracker_artifacts = _select_tracker_artifact(artifacts, tracker_index)
            if tracker_artifacts is not _MISSING_ARTIFACT:
                tracker.log_artifacts(tracker_artifacts, step)
        self._timed_metrics = {}

    def log_file(
        self,
        file_path: str,
        name: str | None = None,
    ) -> None:
        for tracker in self._trackers:
            tracker.log_file(file_path, name=name)

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
        videos = tuple(tracker.video(data, caption=caption, fps=fps, format=format) for tracker in self._trackers)
        if all(video is None for video in videos):
            return None
        return _SequentialArtifact(videos)


class SwanlabTracker(BaseTracker):
    """Tracker implementation for SwanLab."""

    def __init__(
        self,
        experiment_name: str,
        log_dir: str,
        *,
        config: dict[str, Any] | None = None,
        run_name: str | None = None,
    ) -> None:
        super().__init__()

        try:
            import swanlab
        except ModuleNotFoundError as error:
            if error.name != "swanlab":
                raise
            raise ModuleNotFoundError("SwanLab tracking requires the optional 'swanlab' dependency. "
                                      "Install it with `uv pip install 'fastvideo[swanlab]'` (or "
                                      "`uv pip install -e '.[swanlab]'` from a source checkout).") from error

        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

        self._swanlab = swanlab
        self._run = swanlab.init(
            project=experiment_name,
            experiment_name=run_name,
            config=(_sanitize_wandb_config(config) if config is not None else None),
            logdir=log_dir,
        )
        logger.info("Initialized SwanLab tracker")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        metrics = {**self._timed_metrics, **metrics}
        if metrics:
            self._swanlab.log(metrics, step=step)
        self._timed_metrics = {}

    def finish(self) -> None:
        self._swanlab.finish()

    def video(
        self,
        data: Any,
        *,
        caption: str | None = None,
        fps: int | None = None,
        format: str | None = None,
    ) -> Any:
        """Create a SwanLab GIF artifact from a file or W&B-style array."""
        del format  # SwanLab currently supports GIF artifacts only.

        if isinstance(data, str | os.PathLike):
            source_path = os.fspath(data)
            if pathlib.Path(source_path).suffix.lower() == ".gif":
                return self._swanlab.Video(source_path, caption=caption)
            frames, source_fps = _read_video_file(source_path)
            if fps is not None:
                video_fps = _coerce_video_fps(fps)
            else:
                video_fps = source_fps if source_fps is not None else _coerce_video_fps(None)
        else:
            frames = _prepare_video_array(data)
            video_fps = _coerce_video_fps(fps)

        file_descriptor, gif_path = tempfile.mkstemp(suffix=".gif")
        os.close(file_descriptor)
        try:
            _write_gif(gif_path, frames, video_fps)
            return self._swanlab.Video(gif_path, caption=caption)
        finally:
            pathlib.Path(gif_path).unlink(missing_ok=True)


class Trackers(str, Enum):
    NONE = "none"
    WANDB = "wandb"
    SWANLAB = "swanlab"


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

    unsupported = [name for name in tracker_names if name not in SUPPORTED_TRACKERS]
    if unsupported:
        raise ValueError(
            f"Unsupported tracker(s) provided: {unsupported}. Supported trackers: {sorted(SUPPORTED_TRACKERS)}")

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
        elif tracker_name == Trackers.SWANLAB.value:
            tracker_instances.append(
                SwanlabTracker(
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
