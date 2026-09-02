# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = (
    ROOT
    / "examples"
    / "train"
    / "rvm_h3"
    / "calibrate_reward_profile.py"
)
SPEC = importlib.util.spec_from_file_location(
    "fastvideo_test_calibrate_reward_profile",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_robust_component_stats_uses_median_mad() -> None:
    result = MODULE._robust_component_stats(
        [1.0, 2.0, 3.0],
        eps=1e-6,
        constant_scale_fallback=None,
    )

    assert result["center"] == 2.0
    assert result["scale"] == pytest.approx(1.4826)
    assert result["method"] == "median_mad"
    assert result["count"] == 3


def test_constant_component_requires_explicit_fallback() -> None:
    with pytest.raises(
        ValueError,
        match="constant",
    ):
        MODULE._robust_component_stats(
            [1.0, 1.0, 1.0],
            eps=1e-6,
            constant_scale_fallback=None,
        )

    result = MODULE._robust_component_stats(
        [1.0, 1.0, 1.0],
        eps=1e-6,
        constant_scale_fallback=1.0,
    )
    assert result["center"] == 1.0
    assert result["scale"] == 1.0
    assert result["method"] == "constant_override"


def test_discover_inputs_maps_video_indices_to_prompts(
    tmp_path: Path,
) -> None:
    prompts = [
        "prompt zero",
        "prompt one",
        "prompt two",
    ]
    (tmp_path / "prompt-000002.mp4").touch()
    (tmp_path / "prompt-000000.mp4").touch()
    (tmp_path / "ignore.mp4").touch()

    result = MODULE._discover_inputs(
        tmp_path,
        prompts,
        max_videos=10,
    )

    assert [entry["index"] for entry in result] == [0, 2]
    assert [entry["prompt"] for entry in result] == [
        "prompt zero",
        "prompt two",
    ]
