from __future__ import annotations

import builtins
from io import BytesIO
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np
from PIL import Image
import pytest

from fastvideo.training import trackers as trackers_module
from fastvideo.training.trackers import (
    _coerce_video_fps,
    _prepare_video_array,
    BaseTracker,
    SequentialTracker,
    SwanlabTracker,
    WandbTracker,
)


class _FakeSwanlab(ModuleType):
    def __init__(self) -> None:
        super().__init__("swanlab")
        self.init_kwargs: dict[str, Any] | None = None
        self.logged: list[tuple[dict[str, Any], int]] = []
        self.video_paths: list[str] = []
        self.finished = False

    def init(self, **kwargs: Any) -> object:
        self.init_kwargs = kwargs
        return object()

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self.logged.append((metrics, step))

    def finish(self) -> None:
        self.finished = True

    def Video(self, file_path: str, caption: str | None = None) -> dict[str, Any]:  # noqa: N802
        self.video_paths.append(file_path)
        return {
            "content": Path(file_path).read_bytes(),
            "caption": caption,
        }


def _install_fake_swanlab(monkeypatch: pytest.MonkeyPatch) -> _FakeSwanlab:
    swanlab = _FakeSwanlab()
    monkeypatch.setitem(sys.modules, "swanlab", swanlab)
    return swanlab


def test_prepare_video_array_converts_tchw_to_thwc() -> None:
    video = np.arange(2 * 3 * 2 * 4, dtype=np.uint8).reshape(2, 3, 2, 4)

    prepared = _prepare_video_array(video)

    np.testing.assert_array_equal(prepared, video.transpose(0, 2, 3, 1))


def test_default_video_fps_is_16() -> None:
    assert _coerce_video_fps(None) == 16


def test_wandb_video_uses_shared_default_fps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wandb = ModuleType("wandb")
    video_calls: list[dict[str, Any]] = []
    wandb.init = lambda **_: object()

    def video(data: Any, **kwargs: Any) -> dict[str, Any]:
        video_calls.append(kwargs)
        return {"data": data, **kwargs}

    wandb.Video = video
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    tracker = WandbTracker("project", str(tmp_path))

    tracker.video(np.zeros((2, 3, 2, 2), dtype=np.uint8))

    assert video_calls == [{"fps": 16, "format": "mp4"}]


def test_prepare_video_array_tiles_and_pads_batches() -> None:
    video = np.zeros((3, 1, 3, 1, 1), dtype=np.uint8)
    video[0] = 1
    video[1] = 2
    video[2] = 3

    prepared = _prepare_video_array(video)

    assert prepared.shape == (1, 1, 4, 3)
    np.testing.assert_array_equal(
        prepared[0, 0],
        np.array(
            [
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )


def test_swanlab_video_converts_array_to_gif_and_logs_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    swanlab = _install_fake_swanlab(monkeypatch)
    tracker = SwanlabTracker("project", str(tmp_path), run_name="run")
    video = np.zeros((2, 3, 2, 2), dtype=np.uint8)
    video[0, 0] = 255
    video[1, 2] = 255

    artifact = tracker.video(video, caption="validation sample", fps=5, format="mp4")
    tracker.log_artifacts({"validation/videos": [artifact]}, step=7)

    assert artifact["content"].startswith((b"GIF87a", b"GIF89a"))
    assert artifact["caption"] == "validation sample"
    assert not Path(swanlab.video_paths[0]).exists()
    with Image.open(BytesIO(artifact["content"])) as image:
        assert image.n_frames == 2
        assert image.info["duration"] == 200
    assert swanlab.logged == [({"validation/videos": [artifact]}, 7)]


@pytest.mark.parametrize(("requested_fps", "expected_fps"), [(None, 12.0), (2, 2.0)])
def test_swanlab_video_uses_requested_or_source_fps_during_gif_conversion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    requested_fps: int | None,
    expected_fps: float,
) -> None:
    _install_fake_swanlab(monkeypatch)
    tracker = SwanlabTracker("project", str(tmp_path))
    frames = [np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2, 3), dtype=np.uint8)]
    written_fps: list[float] = []
    write_gif = trackers_module._write_gif

    monkeypatch.setattr(trackers_module, "_read_video_file", lambda _: (frames, 12.0))

    def capture_fps(file_path: str, video_frames: Any, fps: float) -> None:
        written_fps.append(fps)
        write_gif(file_path, video_frames, fps)

    monkeypatch.setattr(trackers_module, "_write_gif", capture_fps)

    artifact = tracker.video(tmp_path / "validation.mp4", fps=requested_fps)

    assert artifact["content"].startswith((b"GIF87a", b"GIF89a"))
    assert written_fps == [expected_fps]


def test_swanlab_video_passes_existing_gif_through(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    swanlab = _install_fake_swanlab(monkeypatch)
    tracker = SwanlabTracker("project", str(tmp_path))
    gif_path = tmp_path / "validation.gif"
    Image.new("RGB", (2, 2)).save(gif_path, format="GIF")

    artifact = tracker.video(gif_path, caption="existing")

    assert swanlab.video_paths == [str(gif_path)]
    assert gif_path.exists()
    assert artifact["caption"] == "existing"


def test_swanlab_missing_dependency_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delitem(sys.modules, "swanlab", raising=False)
    real_import = builtins.__import__

    def missing_swanlab(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "swanlab":
            raise ModuleNotFoundError("No module named 'swanlab'", name="swanlab")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_swanlab)

    with pytest.raises(ModuleNotFoundError, match=r"fastvideo\[swanlab\]"):
        SwanlabTracker("project", str(tmp_path))


class _RecordingTracker(BaseTracker):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.artifact_logs: list[dict[str, Any]] = []

    def video(
        self,
        data: Any,
        *,
        caption: str | None = None,
        fps: int | None = None,
        format: str | None = None,
    ) -> str:
        return f"{self.name}:{data}"

    def log_artifacts(self, artifacts: dict[str, Any], step: int) -> None:
        self.artifact_logs.append(artifacts)


def test_sequential_tracker_logs_backend_specific_video_artifacts() -> None:
    first = _RecordingTracker("first")
    second = _RecordingTracker("second")
    tracker = SequentialTracker([first, second])

    artifact = tracker.video("sample.mp4")
    tracker.log_artifacts({"validation/videos": [artifact]}, step=3)

    assert first.artifact_logs == [{"validation/videos": ["first:sample.mp4"]}]
    assert second.artifact_logs == [{"validation/videos": ["second:sample.mp4"]}]
