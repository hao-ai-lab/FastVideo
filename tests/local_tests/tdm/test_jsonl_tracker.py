# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from fastvideo.training.trackers import initialize_trackers


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_jsonl_tracker_writes_metrics_artifacts_and_files(tmp_path: Path) -> None:
    source_file = tmp_path / "run.yaml"
    source_file.write_text("training: {}\n", encoding="utf-8")

    tracker = initialize_trackers(
        ["jsonl"],
        experiment_name="unused",
        config={"tensor_like": [1, 2, 3]},
        log_dir=str(tmp_path / "tracker"),
        run_name="smoke",
    )

    tracker.log(
        {
            "loss": 1.25,
            "nonfinite": math.nan,
        },
        step=3,
    )
    video = tracker.video(
        str(tmp_path / "sample.mp4"),
        caption="prompt",
        fps=16,
    )
    tracker.log_artifacts(
        {"validation": [video]},
        step=3,
    )
    tracker.log_file(
        str(source_file),
        name="run.yaml",
    )

    metric_rows = _read_jsonl(tmp_path / "tracker" / "metrics.jsonl")
    assert metric_rows[0]["step"] == 3
    metrics = metric_rows[0]["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["loss"] == 1.25
    assert metrics["nonfinite"] == {
        "_type": "nonfinite_float",
        "value": "nan",
    }

    artifact_rows = _read_jsonl(tmp_path / "tracker" / "artifacts.jsonl")
    assert artifact_rows[0]["step"] == 3
    assert artifact_rows[1]["file"] == {
        "name": "run.yaml",
        "path": str(tmp_path / "tracker" / "files" / "run.yaml"),
        "source": str(source_file),
    }
    assert (tmp_path / "tracker" / "config.json").is_file()
    saved_run = tmp_path / "tracker" / "files" / "run.yaml"
    assert saved_run.read_text(encoding="utf-8") == "training: {}\n"


class _FakeWandbRun:

    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, Any], int]] = []

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self.logged.append((metrics, step))

    def finish(self) -> None:
        pass


class _FakeWandb:

    def __init__(self) -> None:
        self.run = _FakeWandbRun()

    def init(self, **kwargs: Any) -> _FakeWandbRun:
        return self.run

    @staticmethod
    def Video(data: Any, **kwargs: Any) -> tuple[str, Any, dict[str, Any]]:
        return ("wandb-video", data, kwargs)

    @staticmethod
    def save(*args: Any, **kwargs: Any) -> None:
        pass


def test_sequential_tracker_creates_video_for_each_backend(
        tmp_path: Path, monkeypatch: Any) -> None:
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    tracker = initialize_trackers(
        ["jsonl", "wandb"],
        experiment_name="test",
        config=None,
        log_dir=str(tmp_path / "tracker"),
    )
    video_path = tmp_path / "sample.mp4"

    video = tracker.video(str(video_path), caption="prompt", fps=16)
    tracker.log_artifacts({"validation": [video]}, step=7)

    artifact_rows = _read_jsonl(tmp_path / "tracker" / "artifacts.jsonl")
    assert artifact_rows[0]["artifacts"] == {
        "validation": [{
            "caption": "prompt",
            "format": "mp4",
            "fps": 16,
            "path": str(video_path),
        }]
    }
    assert fake_wandb.run.logged == [({
        "validation": [("wandb-video", str(video_path), {
            "caption": "prompt",
            "format": "mp4",
            "fps": 16,
        })]
    }, 7)]
