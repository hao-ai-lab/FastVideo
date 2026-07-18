# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest

from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method


def _method_with_ratios(min_ratio, max_ratio) -> DMD2Method:
    method = object.__new__(DMD2Method)
    object.__setattr__(method, "method_config", {
        "min_timestep_ratio": min_ratio,
        "max_timestep_ratio": max_ratio,
    })
    object.__setattr__(method, "student", SimpleNamespace(num_train_timesteps=1000))
    return method


def test_dmd2_score_timestep_bounds_match_legacy_recipe() -> None:
    method = _method_with_ratios(0.02, 0.98)
    assert method._parse_score_timestep_bounds() == (20, 980)


def test_dmd2_score_timestep_bounds_default_to_full_range() -> None:
    method = _method_with_ratios(None, None)
    assert method._parse_score_timestep_bounds() == (0, 1000)


@pytest.mark.parametrize(
    ("min_ratio", "max_ratio"),
    [(-0.1, 0.9), (0.8, 0.2), (0.0, 1.1)],
)
def test_dmd2_score_timestep_bounds_reject_invalid_ranges(
    min_ratio: float,
    max_ratio: float,
) -> None:
    method = _method_with_ratios(min_ratio, max_ratio)
    with pytest.raises(ValueError, match="0 <= min <= max <= 1"):
        method._parse_score_timestep_bounds()
