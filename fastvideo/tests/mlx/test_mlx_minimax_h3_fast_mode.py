# SPDX-License-Identifier: Apache-2.0
"""Contracts for MiniMax-H3 temporal fast mode on Apple Silicon."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="MLX is required for MiniMax H3 fast-mode tests")

from fastvideo.mlx_runtime.minimax_h3 import (  # noqa: E402
    audio_latent_num_frames,
    build_packed_layout,
    video_latent_num_frames,
)
from fastvideo.mlx_runtime.minimax_h3_pipeline import (  # noqa: E402
    MiniMaxH3MLXPipeline,
    _adaln_schedule_union,
    _center_crop_frames,
    _default_metal_wired_limit_gib,
    _preflight_media_dependencies,
    _validate_checkpoint_step_ladder,
    plan_fast_temporal,
)
from fastvideo.mlx_runtime import rife_interp  # noqa: E402


def test_fast_plan_keeps_full_audio_and_reduces_only_video() -> None:
    plan = plan_fast_temporal(124, factor=2)

    assert plan.target_frames == 124
    assert plan.source_frames == 73
    assert plan.source_frames % 17 == 5
    assert video_latent_num_frames(plan.source_frames) == 22
    assert video_latent_num_frames(plan.target_frames) == 37
    assert audio_latent_num_frames(plan.target_frames) == 207
    assert plan.video_temporal_scale > 1.0


def test_fast_layout_stretches_video_positions_without_changing_audio() -> None:
    baseline = build_packed_layout(8, 22, 46, 80, 207)
    fast = build_packed_layout(8, 22, 46, 80, 207, video_temporal_scale=1.7)

    np.testing.assert_array_equal(fast.audio_indices, baseline.audio_indices)
    np.testing.assert_array_equal(
        fast.position_ids[fast.audio_indices],
        baseline.position_ids[baseline.audio_indices],
    )
    baseline_video_time = baseline.position_ids[baseline.video_indices, 0] - 8.0
    fast_video_time = fast.position_ids[fast.video_indices, 0] - 8.0
    np.testing.assert_allclose(fast_video_time, baseline_video_time * 1.7)


def test_rife_interpolates_to_exact_non_multiple_count(monkeypatch: pytest.MonkeyPatch) -> None:
    source = [np.full((4, 6, 3), value, dtype=np.uint8) for value in (0, 100, 200)]
    sentinel_model = object()

    def fake_pair(left, right, timestep, *, model, scale=1.0):
        assert model is sentinel_model
        return np.rint(left * (1.0 - timestep) + right * timestep).astype(np.uint8)

    monkeypatch.setattr(rife_interp, "interpolate_pair", fake_pair)
    result = rife_interp.interpolate_to_frame_count(source, 6, model=sentinel_model)

    assert len(result) == 6
    np.testing.assert_array_equal(result[0], source[0])
    np.testing.assert_array_equal(result[-1], source[-1])
    assert [int(frame[0, 0, 0]) for frame in result] == [0, 40, 80, 120, 160, 200]


def test_fast_mode_center_crops_internal_736p_canvas_to_720p() -> None:
    frames = np.arange(2 * 736 * 1280 * 3, dtype=np.uint8).reshape(2, 736, 1280, 3)
    cropped = _center_crop_frames(frames, 720, 1280)

    assert cropped.shape == (2, 720, 1280, 3)
    np.testing.assert_array_equal(cropped[:, 0], frames[:, 8])
    np.testing.assert_array_equal(cropped[:, -1], frames[:, 727])


def test_fixed_adaln_checkpoint_rejects_different_step_count(tmp_path) -> None:
    manifest = {"adaln_cache": {"timesteps": _adaln_schedule_union(4).tolist()}}
    (tmp_path / "mlx_h3_dit.json").write_text(json.dumps(manifest))

    _validate_checkpoint_step_ladder(tmp_path, 4)
    with pytest.raises(ValueError, match=r"does not support --steps 6"):
        _validate_checkpoint_step_ladder(tmp_path, 6)


def test_media_preflight_checks_ffmpeg_before_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fastvideo.mlx_runtime.minimax_h3_pipeline.shutil.which", lambda _name: None)

    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        _preflight_media_dependencies(fast=False, fast_sharpen=0.0, rife_weights_dir=None)


def test_fast_media_preflight_resolves_rife_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr("fastvideo.mlx_runtime.minimax_h3_pipeline.shutil.which", lambda _name: "/opt/ffmpeg")
    monkeypatch.setattr("fastvideo.mlx_runtime.minimax_h3_pipeline.importlib.util.find_spec",
                        lambda _name: object())
    monkeypatch.setattr(rife_interp, "ensure_weights_available", lambda **kwargs: calls.append(kwargs))

    _preflight_media_dependencies(fast=True, fast_sharpen=0.6, rife_weights_dir="/tmp/rife")

    assert calls == [{"weights_dir": "/tmp/rife"}]


def test_default_wired_limit_scales_down_and_keeps_tested_cap() -> None:
    small = SimpleNamespace(metal=SimpleNamespace(device_info=lambda: {"memory_size": 24 * 2**30}))
    large = SimpleNamespace(metal=SimpleNamespace(device_info=lambda: {"memory_size": 64 * 2**30}))

    assert _default_metal_wired_limit_gib(small) == pytest.approx(20.16)
    assert _default_metal_wired_limit_gib(large) == 30.0


def test_mux_cleans_temporary_files_after_ffmpeg_failure(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = MiniMaxH3MLXPipeline.__new__(MiniMaxH3MLXPipeline)
    output = tmp_path / "output.mp4"
    frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    waveform = np.zeros((2, 32), dtype=np.float32)
    monkeypatch.setattr("fastvideo.mlx_runtime.minimax_h3_pipeline.shutil.which", lambda _name: "/opt/ffmpeg")

    def fail(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, "ffmpeg")

    monkeypatch.setattr("fastvideo.mlx_runtime.minimax_h3_pipeline.subprocess.run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.mux(frames, waveform, output)

    assert not output.with_suffix(".tmp.mp4").exists()
    assert not output.with_suffix(".tmp.wav").exists()
