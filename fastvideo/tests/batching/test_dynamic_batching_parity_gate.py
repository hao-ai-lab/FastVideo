# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest

from fastvideo.tests.batching.run_dynamic_batching_parity import (
    _parity_gate,
    _run_dynamic,
)


def test_parity_gate_uses_stable_aggregate_mean() -> None:
    gate = _parity_gate(
        {"max_abs_diff": 10.0, "mean_abs_diff": 0.01},
        max_mean_abs_diff=0.02,
    )

    assert gate == {
        "passed": True,
        "max_mean_abs_diff": 0.02,
        "failures": [],
    }


def test_parity_gate_reports_exceeded_mean_threshold() -> None:
    gate = _parity_gate(
        {"max_abs_diff": 0.25, "mean_abs_diff": 0.03},
        max_mean_abs_diff=0.02,
    )

    assert gate["passed"] is False
    assert gate["failures"] == ["mean_abs_diff 0.03 exceeds 0.02"]


def _parity_args():
    return SimpleNamespace(
        batch_size=2,
        prompts=["one", "two"],
        output_dir="/tmp/dynamic-batching-parity-test",
        height=16,
        width=16,
        num_frames=1,
        num_inference_steps=2,
        guidance_scale=1.0,
        embedded_cfg_scale=6.0,
        seed=1024,
    )


class _FallbackGenerator:

    def _run_forward_batch(self, batch, fastvideo_args):
        raise AssertionError("sequential fallback should not call the merged forward")

    def generate_video_batch(self, requests):
        return requests


class _MergedGenerator:

    def _run_forward_batch(self, batch, fastvideo_args):
        return None

    def generate_video_batch(self, requests):
        self._run_forward_batch(
            SimpleNamespace(prompt=[request["prompt"] for request in requests]),
            SimpleNamespace(),
        )
        return requests


def test_dynamic_parity_rejects_sequential_fallback() -> None:
    generator = _FallbackGenerator()

    with pytest.raises(AssertionError, match="did not execute a multi-request forward"):
        _run_dynamic(generator, _parity_args())


def test_dynamic_parity_accepts_observed_merged_forward() -> None:
    generator = _MergedGenerator()
    original_forward = generator._run_forward_batch

    results, _elapsed = _run_dynamic(generator, _parity_args())

    assert [result["prompt"] for result in results] == ["one", "two"]
    assert generator._run_forward_batch == original_forward
