# SPDX-License-Identifier: Apache-2.0
"""Regression test: preprocess_kandinsky5_overfit.load_video must fail with
a path-specific error on an unreadable input, not an opaque IndexError.

``cv2.VideoCapture`` doesn't raise on a missing or corrupt file -- it just
decodes zero frames, and the repeat-last-frame fill then crashed on
``frames[-1]`` with an ``IndexError`` that named neither the file nor the
cause.

CPU-only -- decodes a tiny real clip written with cv2, no GPU or model
load.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from fastvideo.pipelines.preprocess.preprocess_kandinsky5_overfit import load_video


def test_load_video_missing_file_raises_path_specific_error(tmp_path):
    missing = tmp_path / "does_not_exist.mp4"

    with pytest.raises(ValueError, match="does_not_exist.mp4"):
        load_video(str(missing), num_frames=5, height=64, width=64)


def test_load_video_corrupt_file_raises_path_specific_error(tmp_path):
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"this is not an mp4 container")

    with pytest.raises(ValueError, match="corrupt.mp4"):
        load_video(str(corrupt), num_frames=5, height=64, width=64)


def test_load_video_short_clip_still_fills_by_repeating_last_frame(tmp_path):
    """The zero-frame guard must not break the documented short-clip fill."""
    clip = tmp_path / "short.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (64, 64))
    assert writer.isOpened(), "mp4v VideoWriter unavailable in this OpenCV build"
    rng = np.random.default_rng(0)
    for _ in range(3):
        writer.write(rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8))
    writer.release()

    video = load_video(str(clip), num_frames=8, height=64, width=64)

    assert video.shape == (1, 3, 8, 64, 64)
