# SPDX-License-Identifier: Apache-2.0
"""FastH3 8-Step V2 uses its contract ladder; the four-step MLX path stays uniform."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="MLX is required for MiniMax H3 schedule tests")

from fastvideo.mlx_runtime.minimax_h3 import (  # noqa: E402
    MiniMaxH3SchedulerState,
    load_fasth3_inference_contract,
    resolve_h3_denoise_schedule,
    validate_fasth3_dmd_rungs,
)
from fastvideo.mlx_runtime.minimax_h3_pipeline import (  # noqa: E402
    _adaln_schedule_union,
    _require_positive_vae_tiles,
    _validate_checkpoint_step_ladder,
)
from scripts.checkpoint_conversion.convert_minimax_h3_mlx import _adaln_cache_timesteps  # noqa: E402

EIGHT_RUNGS = [999, 874, 749, 624, 500, 375, 250, 125]
FOUR_RUNGS = [999, 749, 500, 250]


def _contract(steps: list[int], *, shifts: bool) -> dict:
    payload = {
        "schema_version": "fasth3-inference-contract-v1",
        "dmd_denoising_steps": steps,
        "num_inference_steps": len(steps) + 1,
        "transformer_forwards": len(steps),
    }
    if shifts:
        payload["video_scheduler_shift"] = 10.0
        payload["audio_scheduler_shift"] = 3.0
    return payload


def _write_contract(tmp_path, payload: dict) -> None:
    (tmp_path / "fastvideo_inference.json").write_text(json.dumps(payload))


def _write_cache(tmp_path, timesteps) -> None:
    values = np.asarray(timesteps, dtype=np.float32).tolist()
    (tmp_path / "mlx_h3_dit.json").write_text(json.dumps({"adaln_cache": {"timesteps": values}}))


def test_four_step_sidecar_keeps_the_uniform_ladder(tmp_path) -> None:
    _write_contract(tmp_path, _contract(FOUR_RUNGS, shifts=False))
    uniform = _adaln_schedule_union(4)
    _write_cache(tmp_path, uniform)

    assert load_fasth3_inference_contract(tmp_path) is None
    schedule = resolve_h3_denoise_schedule(tmp_path, 4, cached_timesteps=uniform)
    assert schedule.source == "uniform"
    assert schedule.num_steps == 4
    assert schedule.video.shift == 12.0
    _validate_checkpoint_step_ladder(tmp_path, 4, model_root=tmp_path)
    with pytest.raises(ValueError, match=r"does not support --steps 8"):
        _validate_checkpoint_step_ladder(tmp_path, 8, model_root=tmp_path)


def test_shifted_contract_rejects_a_different_step_count(tmp_path) -> None:
    _write_contract(tmp_path, _contract(FOUR_RUNGS, shifts=True))
    with pytest.raises(ValueError, match="trains 4 transformer forwards"):
        resolve_h3_denoise_schedule(tmp_path, 8)


def test_eight_step_contract_runs_the_trained_rungs(tmp_path) -> None:
    _write_contract(tmp_path, _contract(EIGHT_RUNGS, shifts=True))
    for requested in (8, 9):
        schedule = resolve_h3_denoise_schedule(tmp_path, requested)
        assert schedule.source == "contract-dmd"
        assert schedule.num_steps == 8
        assert schedule.video.shift == 10.0
        assert schedule.audio.shift == 3.0
        assert schedule.video.timesteps.shape == (8, )
        assert float(schedule.video.sigmas[-1]) == 0.0
        assert bool(np.all(schedule.video.sigmas[1:] < schedule.video.sigmas[:-1]))

    union = schedule.adaln_timesteps
    _write_cache(tmp_path, union)
    _validate_checkpoint_step_ladder(tmp_path, 8, model_root=tmp_path)
    _validate_checkpoint_step_ladder(tmp_path, 9, model_root=tmp_path)

    _write_cache(tmp_path, np.linspace(0.0, 1.0, 17))
    with pytest.raises(ValueError, match=r"does not support --steps 8"):
        _validate_checkpoint_step_ladder(tmp_path, 8, model_root=tmp_path)


def test_dmd_rungs_reject_invalid_ladders() -> None:
    with pytest.raises(ValueError, match="strictly decreasing"):
        validate_fasth3_dmd_rungs([999, 999])
    with pytest.raises(ValueError, match="strictly decreasing"):
        validate_fasth3_dmd_rungs([100, 200])
    with pytest.raises(ValueError, match="strictly decreasing"):
        validate_fasth3_dmd_rungs([0, -1])
    with pytest.raises(ValueError, match="strictly decreasing"):
        validate_fasth3_dmd_rungs([1001])
    with pytest.raises(ValueError, match="positive finite"):
        MiniMaxH3SchedulerState.from_dmd_steps(0.0, EIGHT_RUNGS)


def test_malformed_shifted_contract_fails_closed(tmp_path) -> None:
    (tmp_path / "fastvideo_inference.json").write_text("{")
    with pytest.raises(ValueError, match="Malformed"):
        load_fasth3_inference_contract(tmp_path)
    with pytest.raises(ValueError, match="Malformed"):
        _adaln_cache_timesteps(tmp_path)


def test_converter_cache_follows_only_a_shifted_contract(tmp_path) -> None:
    four_step = _adaln_cache_timesteps(None)
    np.testing.assert_array_equal(four_step, _adaln_schedule_union(4))

    _write_contract(tmp_path, _contract(FOUR_RUNGS, shifts=False))
    np.testing.assert_array_equal(_adaln_cache_timesteps(tmp_path), four_step)

    shifted = tmp_path / "shifted"
    shifted.mkdir()
    _write_contract(shifted, _contract(EIGHT_RUNGS, shifts=True))
    cached = _adaln_cache_timesteps(shifted)
    expected = resolve_h3_denoise_schedule(shifted, 8).adaln_timesteps
    np.testing.assert_array_equal(cached, expected)
    assert not np.array_equal(cached, _adaln_schedule_union(8))


def test_vae_tiles_must_be_positive() -> None:
    _require_positive_vae_tiles(256, 256)
    with pytest.raises(ValueError, match="positive"):
        _require_positive_vae_tiles(0, 256)
    with pytest.raises(ValueError, match="positive"):
        _require_positive_vae_tiles(256, -1)
