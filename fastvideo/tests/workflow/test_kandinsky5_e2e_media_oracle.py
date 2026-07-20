# SPDX-License-Identifier: Apache-2.0
"""Known-bad calibration tests for the Kandinsky5 e2e media oracle.

A review pass demonstrated a concrete false-green: a 121-frame solid clip
at the committed reference's mean RGB passed the old degeneracy check
(global RGB std counts channel-mean differences as "variance": 15.4) AND
the old MS-SSIM floor (scoring a mean of ~0.92 against the low-contrast
reference, with the SSIM helper additionally truncating both clips to the
shorter frame count). These tests turn that exploit -- and its frozen and
truncated siblings -- into permanent regressions against the *currently
committed* reference video, so the oracle's thresholds can never silently
drift below a known-bad clip again: if a future reference changes the
similarity landscape, the floor assertions here fail and force
recalibration.

CPU-only; skips when the committed reference is absent (pre-bootstrap
checkouts).
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from fastvideo.tests.nightly.test_e2e_kandinsky5_dmd_t2v_overfit import (
    EXPECTED_FRAME_COUNT,
    MIN_MEAN_MS_SSIM,
    REFERENCE_VIDEO,
    _assert_video_not_degenerate,
    _decode_video,
)

pytestmark = pytest.mark.skipif(
    not REFERENCE_VIDEO.exists(),
    reason="committed Kandinsky5 e2e reference video not present",
)


def _write_clip(path, frames) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (width, height))
    assert writer.isOpened(), "mp4v VideoWriter unavailable in this OpenCV build"
    for frame in frames:
        writer.write(frame)
    writer.release()


def _mean_ms_ssim_vs_reference(path) -> float:
    pytest.importorskip("pytorch_msssim")
    pytest.importorskip("av")
    from fastvideo.tests.utils import compute_video_ssim_torchvision

    mean_ssim, _, _ = compute_video_ssim_torchvision(str(REFERENCE_VIDEO), str(path), use_ms_ssim=True)
    return float(mean_ssim)


def test_committed_reference_passes_the_oracle():
    """Self-consistency: thresholds must never be tightened past what the
    reviewed reference itself exhibits."""
    _assert_video_not_degenerate(REFERENCE_VIDEO)


def test_solid_clip_at_reference_mean_color_is_rejected(tmp_path):
    """The review's exact exploit: solid clip at the reference's own mean
    RGB. Global-std passed it at 15.4; per-frame luma spatial std is 0."""
    reference = _decode_video(REFERENCE_VIDEO)
    mean_color = reference.astype(np.float64).mean(axis=(0, 1, 2)).round().astype(np.uint8)
    solid_frame = np.full(reference.shape[1:], mean_color, dtype=np.uint8)
    solid_path = tmp_path / "solid.mp4"
    _write_clip(solid_path, [solid_frame] * EXPECTED_FRAME_COUNT)

    with pytest.raises(AssertionError, match="solid"):
        _assert_video_not_degenerate(solid_path)

    # The similarity floor must ALSO sit above this clip's score (measured
    # ~0.9189 against the 2026-07-18 reference): the structural check is
    # the primary defense, but the floor must not regress into known-bad
    # territory either.
    assert _mean_ms_ssim_vs_reference(solid_path) < MIN_MEAN_MS_SSIM


def test_frozen_clip_is_rejected(tmp_path):
    """A single real reference frame repeated 121x has genuine spatial
    content (luma std ~12) -- only the temporal check catches it."""
    reference = _decode_video(REFERENCE_VIDEO)
    frozen_path = tmp_path / "frozen.mp4"
    _write_clip(frozen_path, [reference[0]] * EXPECTED_FRAME_COUNT)

    with pytest.raises(AssertionError, match="frozen"):
        _assert_video_not_degenerate(frozen_path)

    # Measured ~0.8597 against the 2026-07-18 reference.
    assert _mean_ms_ssim_vs_reference(frozen_path) < MIN_MEAN_MS_SSIM


def test_truncated_clip_is_rejected(tmp_path):
    """compute_video_ssim_torchvision truncates both clips to the shorter
    frame count, so a partial video could score arbitrarily well -- the
    oracle must reject on frame count before similarity is even
    consulted."""
    reference = _decode_video(REFERENCE_VIDEO)
    truncated_path = tmp_path / "truncated.mp4"
    _write_clip(truncated_path, list(reference[:EXPECTED_FRAME_COUNT // 2]))

    with pytest.raises(AssertionError, match="frame count or geometry"):
        _assert_video_not_degenerate(truncated_path)
