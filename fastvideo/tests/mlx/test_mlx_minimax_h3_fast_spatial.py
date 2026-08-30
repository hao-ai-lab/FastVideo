# SPDX-License-Identifier: Apache-2.0
"""Contracts for MiniMax-H3 spatial fast mode on Apple Silicon."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="MLX is required for MiniMax H3 fast-spatial tests")

from fastvideo.mlx_runtime.frame_upsample import upsample_frames  # noqa: E402
from fastvideo.mlx_runtime.minimax_h3_pipeline import (  # noqa: E402
    _center_crop_frames,
    _preflight_media_dependencies,
    plan_fast_spatial,
)


def test_spatial_plan_rounds_stage1_canvas_up_to_model_grid() -> None:
    plan = plan_fast_spatial(480, 832)

    assert (plan.target_height, plan.target_width) == (480, 832)
    assert (plan.stage1_height, plan.stage1_width) == (240, 416)
    assert (plan.canvas_height, plan.canvas_width) == (256, 416)
    assert plan.scale == 2


def test_spatial_plan_720p_lands_on_384x640_canvas() -> None:
    plan = plan_fast_spatial(720, 1280)

    assert (plan.stage1_height, plan.stage1_width) == (360, 640)
    assert (plan.canvas_height, plan.canvas_width) == (384, 640)


def test_spatial_plan_rejects_scale_below_two() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        plan_fast_spatial(480, 832, scale=1)


def test_spatial_plan_rejects_non_reducing_scale() -> None:
    with pytest.raises(ValueError, match="does not reduce"):
        plan_fast_spatial(32, 32, scale=2)


def test_spatial_plan_rejects_unknown_upsample_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported upsample mode"):
        plan_fast_spatial(480, 832, upsample_mode="metalfx")


def test_spatial_plan_rejects_negative_sharpen() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        plan_fast_spatial(480, 832, sharpen=-0.1)


def test_spatial_preflight_requires_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fastvideo.mlx_runtime.minimax_h3_pipeline.shutil.which", lambda _name: "/opt/ffmpeg")
    monkeypatch.setattr("fastvideo.mlx_runtime.minimax_h3_pipeline.importlib.util.find_spec", lambda _name: None)

    with pytest.raises(RuntimeError, match="OpenCV is required"):
        _preflight_media_dependencies(fast=False, fast_sharpen=0.0, rife_weights_dir=None, fast_spatial=True)


def test_spatial_crop_then_upsample_restores_exact_target_size() -> None:
    pytest.importorskip("cv2", reason="OpenCV backs the pixel-space resample")
    plan = plan_fast_spatial(480, 832)
    frames = np.zeros((2, plan.canvas_height, plan.canvas_width, 3), dtype=np.uint8)

    cropped = _center_crop_frames(frames, plan.stage1_height, plan.stage1_width)
    assert cropped.shape == (2, 240, 416, 3)

    upsampled = upsample_frames(cropped, width=plan.target_width, height=plan.target_height,
                                mode="bilinear", sharpen=0.0)
    assert np.stack(upsampled).shape == (2, 480, 832, 3)
