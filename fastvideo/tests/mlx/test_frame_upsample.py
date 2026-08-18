# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for pixel-space frame resampling."""

from __future__ import annotations

import numpy as np
import pytest

from fastvideo.mlx_runtime.frame_upsample import (
    PIXEL_UPSAMPLE_MODES,
    unsharp,
    upsample_frame,
    upsample_frames,
)
from fastvideo.mlx_runtime.rife_interp import aligned_keyframe_count


def _frame(height: int = 24, width: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


@pytest.mark.parametrize("mode", PIXEL_UPSAMPLE_MODES)
def test_upsample_frame_hits_target_size_in_every_mode(mode: str) -> None:
    out = upsample_frame(_frame(), width=80, height=48, mode=mode)
    assert out.shape == (48, 80, 3)
    assert out.dtype == np.uint8


def test_upsample_frame_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported pixel upsample mode"):
        upsample_frame(_frame(), width=80, height=48, mode="mitchell")


def test_upsample_frame_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="HxWx3"):
        upsample_frame(np.zeros((24, 40), dtype=np.uint8), width=80, height=48)


def test_upsample_frame_rejects_nonpositive_size() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        upsample_frame(_frame(), width=0, height=48)


def test_upsample_frame_at_target_size_is_a_passthrough() -> None:
    frame = _frame()
    np.testing.assert_array_equal(upsample_frame(frame, width=40, height=24), frame)


def test_nearest_upsample_is_exact_block_replication() -> None:
    """A 2x nearest resize must duplicate pixels, not blend them."""
    frame = _frame(height=4, width=6)
    out = upsample_frame(frame, width=12, height=8, mode="nearest")
    np.testing.assert_array_equal(out, np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1))


def test_unsharp_zero_amount_is_identity() -> None:
    frame = _frame()
    assert unsharp(frame, 0.0) is frame


def test_unsharp_steepens_a_soft_edge() -> None:
    # A blurred step edge: unsharp masking should raise its peak gradient.
    import cv2

    frame = np.zeros((24, 40, 3), dtype=np.uint8)
    frame[:, 20:] = 200
    frame = cv2.GaussianBlur(frame, (0, 0), 2.0)

    def peak_gradient(image: np.ndarray) -> float:
        return float(np.abs(np.diff(image[:, :, 0].astype(np.int32), axis=1)).max())

    assert peak_gradient(unsharp(frame, 0.8)) > peak_gradient(frame)


def test_upsample_frames_preserves_order_and_leaves_input_alone() -> None:
    frames = [_frame(seed=index) for index in range(3)]
    originals = [frame.copy() for frame in frames]
    out = upsample_frames(frames, width=80, height=48)

    assert len(out) == 3
    assert all(frame.shape == (48, 80, 3) for frame in out)
    # Distinct inputs stay distinct and in order.
    assert not np.array_equal(out[0], out[1])
    for frame, original in zip(frames, originals, strict=True):
        np.testing.assert_array_equal(frame, original)


@pytest.mark.parametrize(
    ("target_frames", "factor", "keyframes"),
    [(1, 999, 1), (81, 2, 41), (81, 3, 29), (82, 2, 45), (121, 4, 33)],
)
def test_aligned_keyframes_expand_to_exact_target(target_frames: int, factor: int, keyframes: int) -> None:
    assert aligned_keyframe_count(target_frames, factor) == keyframes
    interpolated_frames = (keyframes - 1) * factor + 1
    assert interpolated_frames >= target_frames
    assert keyframes % 4 == 1
    if keyframes > 1:
        assert (keyframes - 5) * factor + 1 < target_frames
