# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests for VideoAlign reward inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fastvideo.train.methods.rl.rewards import videoalign


class _FakeInferencer:

    def __init__(self, score_key: str) -> None:
        self.score_key = score_key
        self.calls: list[tuple[list[str], list[str], bool]] = []

    def reward(self, paths: list[str], prompts: list[str], *, use_norm: bool):
        self.calls.append((paths, prompts, use_norm))
        return [{self.score_key: 0.5}]


def _capture_frames(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[np.ndarray]:
    captured: list[np.ndarray] = []

    def save_video(frames: np.ndarray) -> str:
        captured.append(frames.copy())
        path = tmp_path / "sample.mp4"
        path.touch()
        return str(path)

    monkeypatch.setattr(videoalign, "_save_video_to_temp", save_video)
    return captured


@pytest.mark.parametrize(
    ("scorer_type", "score_key"),
    [
        (videoalign.VideoAlignMotionQualityScorer, "MQ"),
        (videoalign.VideoAlignVisualQualityScorer, "VQ"),
    ],
)
def test_videoalign_quality_scorers_preserve_prompts(
    scorer_type,
    score_key: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inferencer = _FakeInferencer(score_key)
    monkeypatch.setattr(videoalign, "_get_inferencer", lambda *_args: inferencer)
    _capture_frames(monkeypatch, tmp_path)

    scores = scorer_type(device="cpu")(torch.zeros(1, 3, 1, 2, 2), ["a red fox runs"])

    assert scores.tolist() == [0.5]
    assert inferencer.calls[0][1] == ["a red fox runs"]
    assert inferencer.calls[0][2] is True


def test_videoalign_motion_quality_uses_weighted_luminance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inferencer = _FakeInferencer("MQ")
    monkeypatch.setattr(videoalign, "_get_inferencer", lambda *_args: inferencer)
    captured = _capture_frames(monkeypatch, tmp_path)
    media = torch.zeros(1, 3, 1, 1, 4)
    media[0, :, 0, 0] = torch.tensor([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ])

    videoalign.VideoAlignMotionQualityScorer(device="cpu")(media, ["primary colors"])

    assert captured[0][0, 0].tolist() == [
        [76, 76, 76],
        [150, 150, 150],
        [29, 29, 29],
        [255, 255, 255],
    ]
