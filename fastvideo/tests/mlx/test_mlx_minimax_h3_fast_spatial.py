# SPDX-License-Identifier: Apache-2.0
"""Contracts for MiniMax-H3 spatial fast mode on Apple Silicon."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="MLX is required for MiniMax H3 fast-spatial tests")

from fastvideo.mlx_runtime import rife_interp  # noqa: E402
from fastvideo.mlx_runtime.frame_upsample import upsample_frames  # noqa: E402
from fastvideo.mlx_runtime.minimax_h3 import (  # noqa: E402
    build_packed_layout,
    video_latent_num_frames,
)
from fastvideo.mlx_runtime.minimax_h3_pipeline import (  # noqa: E402
    DEFAULT_FAST_SPATIAL_SHARPEN,
    MiniMaxH3MLXPipeline,
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


# -- generation-level orchestration contracts (heavyweight phases mocked) ----


def _generate_with_mocked_phases(monkeypatch, tmp_path, **generate_kwargs):
    """Run real ``generate()`` orchestration with condition/denoise/decode/mux stubbed."""
    events: list[str] = []
    calls: dict = {}

    pipeline = MiniMaxH3MLXPipeline.__new__(MiniMaxH3MLXPipeline)
    pipeline.video_decode_backend = "h3-vae"
    pipeline.dit_checkpoint = tmp_path

    monkeypatch.setattr("fastvideo.mlx_runtime.minimax_h3_pipeline._validate_checkpoint_step_ladder",
                        lambda _checkpoint, _steps: None)
    monkeypatch.setattr("fastvideo.mlx_runtime.minimax_h3_pipeline._preflight_media_dependencies",
                        lambda **_kwargs: None)
    monkeypatch.setattr("fastvideo.mlx_runtime.minimax_h3_pipeline.mlx_h3_checkpoint_vsa_capable",
                        lambda _checkpoint: False)

    def fake_encode_prompt(_prompt):
        return np.zeros((8, 8), dtype=np.float32), np.zeros(8, dtype=np.int64)

    def fake_denoise(_text_rows, _token_tags, **kwargs):
        events.append("denoise")
        calls["denoise"] = kwargs
        return np.zeros((4, 4), dtype=np.float32), np.zeros((4, 4), dtype=np.float32)

    def fake_decode_video(_rows, *, height, width, num_frames, tiled):
        events.append("decode_video")
        calls["decode_video"] = {"height": height, "width": width, "num_frames": num_frames}
        return np.zeros((num_frames, height, width, 3), dtype=np.uint8)

    def fake_decode_audio(_rows, *, num_frames):
        events.append("decode_audio")
        calls["decode_audio"] = {"num_frames": num_frames}
        return np.zeros((2, 64), dtype=np.float32)

    pipeline.encode_prompt = fake_encode_prompt
    pipeline.denoise = fake_denoise
    pipeline.decode_video = fake_decode_video
    pipeline.decode_audio = fake_decode_audio
    pipeline.mux = lambda _frames, _waveform, output_path: Path(output_path)

    def fake_interpolate(frames, target, *, model):
        events.append("rife")
        calls["rife"] = {"target": target}
        return [np.array(frames[0]) for _ in range(target)]

    def fake_load_model(weights_dir=None):
        return object()

    fake_load_model.cache_clear = lambda: None
    monkeypatch.setattr(rife_interp, "interpolate_to_frame_count", fake_interpolate)
    monkeypatch.setattr(rife_interp, "load_model", fake_load_model)

    def fake_sharpen(frames, amount):
        events.append("sharpen")
        calls.setdefault("sharpen", []).append(amount)
        return list(frames)

    def fake_upsample(frames, *, width, height, mode, sharpen):
        events.append("upsample")
        calls["upsample"] = {"width": width, "height": height, "mode": mode, "sharpen": sharpen}
        return [np.zeros((height, width, 3), dtype=np.uint8) for _ in frames]

    monkeypatch.setattr("fastvideo.mlx_runtime.minimax_h3_pipeline._sharpen_frames", fake_sharpen)
    monkeypatch.setattr("fastvideo.mlx_runtime.minimax_h3_pipeline.upsample_frames", fake_upsample)

    result = pipeline.generate(
        "(S1) test prompt",
        output_path=tmp_path / "out.mp4",
        height=480,
        width=832,
        num_frames=124,
        save_frames=True,
        **generate_kwargs,
    )
    return events, calls, result


def test_generate_spatial_only_denoises_reduced_canvas_with_full_audio(
        monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    events, calls, result = _generate_with_mocked_phases(monkeypatch, tmp_path, fast_spatial=True)

    assert calls["denoise"]["height"] == 256
    assert calls["denoise"]["width"] == 416
    assert calls["denoise"]["num_frames"] == 124
    assert calls["denoise"]["audio_num_frames"] is None
    assert calls["denoise"]["video_temporal_scale"] == 1.0
    assert calls["decode_video"] == {"height": 256, "width": 416, "num_frames": 124}
    assert calls["decode_audio"] == {"num_frames": 124}
    assert calls["upsample"] == {
        "width": 832, "height": 480, "mode": "lanczos",
        "sharpen": pytest.approx(DEFAULT_FAST_SPATIAL_SHARPEN),
    }
    assert "rife" not in events
    assert "sharpen" not in events
    assert result.frames.shape == (124, 480, 832, 3)
    assert "spatial_upsample_s" in result.timings


def test_generate_temporal_only_control_keeps_full_canvas(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    events, calls, result = _generate_with_mocked_phases(monkeypatch, tmp_path, fast=True)

    assert calls["denoise"]["height"] == 480
    assert calls["denoise"]["width"] == 832
    assert calls["denoise"]["num_frames"] == 73
    assert calls["denoise"]["audio_num_frames"] == 124
    assert calls["denoise"]["video_temporal_scale"] > 1.0
    assert calls["decode_video"] == {"height": 480, "width": 832, "num_frames": 73}
    assert calls["rife"] == {"target": 124}
    assert calls["decode_audio"] == {"num_frames": 124}
    assert calls["sharpen"] == [pytest.approx(0.6)]
    assert "upsample" not in events
    assert result.frames.shape == (124, 480, 832, 3)


def test_generate_stacked_runs_rife_before_upsample_with_one_sharpen(
        monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    events, calls, result = _generate_with_mocked_phases(monkeypatch, tmp_path, fast=True, fast_spatial=True)

    assert calls["denoise"]["height"] == 256
    assert calls["denoise"]["width"] == 416
    assert calls["denoise"]["num_frames"] == 73
    assert calls["denoise"]["audio_num_frames"] == 124
    assert calls["decode_video"] == {"height": 256, "width": 416, "num_frames": 73}
    assert calls["rife"] == {"target": 124}
    assert events.index("rife") < events.index("upsample")
    assert "sharpen" not in events
    assert calls["upsample"]["sharpen"] == pytest.approx(0.6)
    assert calls["decode_audio"] == {"num_frames": 124}
    assert result.frames.shape == (124, 480, 832, 3)


# -- reduced VSA layout contracts --------------------------------------------


def test_reduced_layout_has_8x13_video_grid_and_unchanged_audio_prefix() -> None:
    reduced = build_packed_layout(8, 37, 16, 26, 207)
    full = build_packed_layout(8, 37, 30, 52, 207)

    assert reduced.video_indices.shape[0] == 37 * 8 * 13
    assert full.video_indices.shape[0] == 37 * 15 * 26
    np.testing.assert_array_equal(reduced.audio_indices, full.audio_indices)
    # A/V sync rides on the audio rows' temporal positions (column 0), which
    # must not move with the canvas. Column 2 is excluded on purpose: audio
    # rows borrow the video width grid's edge coordinates for their spatial
    # tag, so it tracks the canvas the same way any native resolution change
    # does.
    np.testing.assert_array_equal(
        reduced.position_ids[reduced.audio_indices, 0],
        full.position_ids[full.audio_indices, 0],
    )
    np.testing.assert_array_equal(reduced.position_ids[reduced.audio_indices, 1],
                                  np.zeros(reduced.audio_indices.shape[0]))


def test_reduced_layout_stacked_uses_22_latent_frames() -> None:
    assert video_latent_num_frames(73) == 22
    stacked = build_packed_layout(8, 22, 16, 26, 207, video_temporal_scale=1.7)
    baseline = build_packed_layout(8, 22, 16, 26, 207)

    assert stacked.video_indices.shape[0] == 22 * 8 * 13
    np.testing.assert_array_equal(stacked.audio_indices, baseline.audio_indices)
    np.testing.assert_array_equal(
        stacked.position_ids[stacked.audio_indices],
        baseline.position_ids[baseline.audio_indices],
    )
