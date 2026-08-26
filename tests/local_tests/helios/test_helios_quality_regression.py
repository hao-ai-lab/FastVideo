# SPDX-License-Identifier: Apache-2.0
"""Asset-optional video-level quality and container regression checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


def _video_summary(path: Path) -> dict:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        check=True,
        capture_output=True,
    ).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width)
    means = frames.mean(axis=(1, 2))
    return {
        "codec": stream["codec_name"],
        "width": width,
        "height": height,
        "fps": stream["avg_frame_rate"],
        "frames": int(stream["nb_read_frames"]),
        "decoded_frames": len(frames),
        "mean": float(frames.mean()),
        "std": float(frames.std()),
        "black_frame_count": int((means < 1.0).sum()),
        "frame_mean_std": float(means.std()),
    }


def test_helios_quality_candidate_has_valid_384x640_video():
    candidate = os.environ.get("HELIOS_QUALITY_CANDIDATE")
    if not candidate:
        pytest.skip("Set HELIOS_QUALITY_CANDIDATE to run the real-video quality gate")
    path = Path(candidate)
    if not path.is_file():
        pytest.skip(f"Quality candidate does not exist: {path}")
    summary = _video_summary(path)
    assert summary["codec"] in {"h264", "hevc", "av1"}
    assert (summary["width"], summary["height"]) == (640, 384)
    assert summary["fps"] == "24/1"
    assert summary["frames"] == summary["decoded_frames"] == 33
    assert summary["std"] > 5.0
    assert summary["black_frame_count"] == 0


def test_helios_quality_candidate_is_stable_against_reference():
    candidate = os.environ.get("HELIOS_QUALITY_CANDIDATE")
    reference = os.environ.get("HELIOS_QUALITY_REFERENCE")
    if not candidate or not reference:
        pytest.skip("Set HELIOS_QUALITY_CANDIDATE and HELIOS_QUALITY_REFERENCE for comparison")
    candidate_summary = _video_summary(Path(candidate))
    reference_summary = _video_summary(Path(reference))
    assert abs(candidate_summary["mean"] - reference_summary["mean"]) < 35.0
    assert abs(candidate_summary["std"] - reference_summary["std"]) < 35.0
    assert abs(candidate_summary["frame_mean_std"] - reference_summary["frame_mean_std"]) < 25.0
