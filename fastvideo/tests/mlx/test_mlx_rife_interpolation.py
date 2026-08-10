# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Apple-Silicon MLX RIFE interpolation backend."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX required for RIFE tests")


@pytest.mark.skipif(
    not bool(getattr(mx, "metal", None) and mx.metal.is_available()),
    reason="RIFE MLX regression requires Apple Silicon Metal",
)
def test_rife_interpolation_preserves_keyframes_shape_and_count() -> None:
    if importlib.util.find_spec("rife_mlx") is None:
        pytest.skip("rife-mlx package is not installed")

    from fastvideo.mlx_runtime.rife_interp import interpolate, load_model

    frame0 = np.zeros((64, 96, 3), dtype=np.uint8)
    frame1 = np.zeros((64, 96, 3), dtype=np.uint8)
    frame1[:, :, 0] = 255
    model = load_model()

    frames = interpolate([frame0, frame1], factor=2, model=model)

    assert len(frames) == 3
    assert frames[1].shape == frame0.shape
    assert frames[1].dtype == np.uint8
    np.testing.assert_array_equal(frames[0], frame0)
    np.testing.assert_array_equal(frames[-1], frame1)
