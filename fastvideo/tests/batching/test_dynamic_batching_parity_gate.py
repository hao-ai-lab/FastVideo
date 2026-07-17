# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from fastvideo.tests.batching.run_dynamic_batching_parity import _parity_gate


def test_parity_gate_accepts_metrics_within_thresholds() -> None:
    gate = _parity_gate(
        {"max_abs_diff": 0.15, "mean_abs_diff": 0.01},
        max_abs_diff=0.2,
        max_mean_abs_diff=0.02,
    )

    assert gate == {
        "passed": True,
        "max_abs_diff": 0.2,
        "max_mean_abs_diff": 0.02,
        "failures": [],
    }


def test_parity_gate_reports_each_exceeded_threshold() -> None:
    gate = _parity_gate(
        {"max_abs_diff": 0.25, "mean_abs_diff": 0.03},
        max_abs_diff=0.2,
        max_mean_abs_diff=0.02,
    )

    assert gate["passed"] is False
    assert gate["failures"] == [
        "max_abs_diff 0.25 exceeds 0.2",
        "mean_abs_diff 0.03 exceeds 0.02",
    ]
