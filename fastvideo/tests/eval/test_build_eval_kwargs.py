"""Contract test for :func:`fastvideo.eval.io.build_eval_kwargs`.

Locks the sample shape handed to ``Evaluator.evaluate(**sample)``:
``video`` must be ``(T, C, H, W)`` float in ``[0, 1]`` — the shape every
metric consumes (see :meth:`BaseMetric.compute`) — with no leading batch
dim. Regression guard for the ``unsqueeze(0)`` bug fixed in #1412.

CPU-only; uses a directory of PNG frames so no video decoder is needed.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from fastvideo.eval.io import build_eval_kwargs

T, H, W = 3, 8, 10


def _write_frames(dir_path) -> None:
    dir_path.mkdir()
    for i in range(T):
        arr = np.full((H, W, 3), i * 40, dtype=np.uint8)
        Image.fromarray(arr).save(dir_path / f"frame_{i:03d}.png")


def test_build_eval_kwargs_video_is_4d_tchw(tmp_path):
    frames = tmp_path / "frames"
    _write_frames(frames)

    row = {"prompt": "a test prompt", "auxiliary_info": {"key": "val"}}
    sample = build_eval_kwargs(row, frames, fps=24.0)

    video = sample["video"]
    assert video.shape == (T, 3, H, W), "video must be (T, C, H, W) with no batch dim"
    assert video.dtype.is_floating_point
    assert 0.0 <= video.min() and video.max() <= 1.0
    assert sample["fps"] == 24.0
    assert sample["text_prompt"] == "a test prompt"
    assert sample["auxiliary_info"] == {"key": "val"}


def test_build_eval_kwargs_omits_absent_row_keys(tmp_path):
    frames = tmp_path / "frames"
    _write_frames(frames)

    sample = build_eval_kwargs({}, frames)

    assert set(sample) == {"video", "fps"}
    assert sample["fps"] == 24.0
