# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fastvideo.tests.batching.run_dynamic_batching_parity import (
    _parity_gate,
    _run_dynamic,
    _tensor_metrics,
)


def _valid_metrics(*, mean_abs_diff: float = 0.01, tail_mean_abs_diff: float = 0.1) -> dict:
    return {
        "per_request": [{
            "index": 0,
            "expected_prompt": "one",
            "sequential_prompt": "one",
            "dynamic_prompt": "one",
            "prompt_mapping_matches": True,
            "expected_shape": [1, 3, 1, 2, 2],
            "sequential_shape": [1, 3, 1, 2, 2],
            "dynamic_shape": [1, 3, 1, 2, 2],
            "shape_matches": True,
            "sequential_finite": True,
            "dynamic_finite": True,
            "max_abs_diff": 10.0,
            "mean_abs_diff": mean_abs_diff,
            "tail_mean_abs_diff": tail_mean_abs_diff,
        }],
        "tail_fraction": 0.001,
        "max_abs_diff": 10.0,
        "mean_abs_diff": mean_abs_diff,
        "tail_mean_abs_diff": tail_mean_abs_diff,
    }


def test_parity_gate_uses_stable_request_mean_and_tail() -> None:
    gate = _parity_gate(
        _valid_metrics(),
        max_mean_abs_diff=0.02,
        max_tail_mean_abs_diff=0.30,
    )

    assert gate == {
        "passed": True,
        "max_mean_abs_diff": 0.02,
        "max_tail_mean_abs_diff": 0.30,
        "failures": [],
    }


def test_parity_gate_reports_exceeded_mean_threshold() -> None:
    gate = _parity_gate(
        _valid_metrics(mean_abs_diff=0.03),
        max_mean_abs_diff=0.02,
        max_tail_mean_abs_diff=0.30,
    )

    assert gate["passed"] is False
    assert gate["failures"] == ["request 0 mean_abs_diff 0.03 exceeds 0.02"]


def test_parity_gate_rejects_request_above_threshold_when_aggregate_passes() -> None:
    sequential = [
        {"prompts": "one", "samples": torch.zeros((1, 2))},
        {"prompts": "two", "samples": torch.zeros((1, 2))},
    ]
    dynamic = [
        {"prompts": "one", "samples": torch.full((1, 2), 0.039)},
        {"prompts": "two", "samples": torch.zeros((1, 2))},
    ]
    metrics = _tensor_metrics(
        sequential,
        dynamic,
        expected_prompts=["one", "two"],
        expected_shape=[1, 2],
    )

    gate = _parity_gate(metrics, max_mean_abs_diff=0.02, max_tail_mean_abs_diff=0.30)

    assert metrics["per_request"][0]["mean_abs_diff"] == pytest.approx(0.039)
    assert metrics["per_request"][1]["mean_abs_diff"] == 0.0
    assert metrics["mean_abs_diff"] == pytest.approx(0.0195)
    assert metrics["mean_abs_diff"] < gate["max_mean_abs_diff"]
    assert gate["passed"] is False
    assert gate["failures"] == ["request 0 mean_abs_diff 0.039 exceeds 0.02"]


def test_parity_gate_rejects_localized_corruption_when_whole_tensor_mean_passes() -> None:
    sequential_samples = torch.zeros((1, 10_000))
    dynamic_samples = sequential_samples.clone()
    dynamic_samples[0, :10] = 1.0
    metrics = _tensor_metrics(
        [{"prompts": "one", "samples": sequential_samples}],
        [{"prompts": "one", "samples": dynamic_samples}],
        expected_prompts=["one"],
        expected_shape=[1, 10_000],
    )

    gate = _parity_gate(
        metrics,
        max_mean_abs_diff=0.02,
        max_tail_mean_abs_diff=0.30,
    )

    assert metrics["per_request"][0]["mean_abs_diff"] == pytest.approx(0.001)
    assert metrics["per_request"][0]["tail_mean_abs_diff"] == pytest.approx(1.0)
    assert metrics["tail_fraction"] == 0.001
    assert gate["passed"] is False
    assert gate["failures"] == [
        "request 0 tail_mean_abs_diff 1 exceeds 0.3",
    ]


@pytest.mark.parametrize(
    ("non_finite_side", "expected_failure"),
    [
        ("sequential", "request 0 sequential output contains non-finite values"),
        ("dynamic", "request 0 dynamic output contains non-finite values"),
    ],
)
def test_parity_gate_rejects_non_finite_outputs(non_finite_side: str, expected_failure: str) -> None:
    sequential_samples = torch.zeros((1, 3, 1, 2, 2), dtype=torch.float32)
    dynamic_samples = sequential_samples.clone()
    target = sequential_samples if non_finite_side == "sequential" else dynamic_samples
    target[0, 0, 0, 0, 0] = torch.nan
    metrics = _tensor_metrics(
        [{"prompts": "one", "samples": sequential_samples}],
        [{"prompts": "one", "samples": dynamic_samples}],
        expected_prompts=["one"],
        expected_shape=[1, 3, 1, 2, 2],
    )

    gate = _parity_gate(metrics, max_mean_abs_diff=0.02, max_tail_mean_abs_diff=0.30)

    assert gate["passed"] is False
    assert expected_failure in gate["failures"]
    assert "request 0 max_abs_diff is non-finite: nan" in gate["failures"]
    assert "request 0 mean_abs_diff is non-finite: nan" in gate["failures"]
    assert "request 0 tail_mean_abs_diff is non-finite: nan" in gate["failures"]


def test_parity_gate_rejects_non_finite_per_request_metric_before_tolerance() -> None:
    metrics = _valid_metrics()
    metrics["per_request"][0]["mean_abs_diff"] = float("nan")

    gate = _parity_gate(metrics, max_mean_abs_diff=0.02, max_tail_mean_abs_diff=0.30)

    assert gate["passed"] is False
    assert gate["failures"] == ["request 0 mean_abs_diff is non-finite: nan"]


def test_tensor_metrics_and_gate_validate_decoded_shape_and_prompt_mapping() -> None:
    sequential = [{
        "prompts": "one",
        "samples": torch.zeros((1, 3, 1, 2, 2), dtype=torch.float32),
    }]
    dynamic = [{
        "prompts": "one",
        "samples": torch.zeros((1, 3, 1, 2, 2), dtype=torch.float32),
    }]

    metrics = _tensor_metrics(
        sequential,
        dynamic,
        expected_prompts=["one"],
        expected_shape=[1, 3, 1, 2, 2],
    )

    assert metrics["per_request"][0]["shape_matches"] is True
    assert metrics["per_request"][0]["prompt_mapping_matches"] is True
    assert _parity_gate(metrics, max_mean_abs_diff=0.02, max_tail_mean_abs_diff=0.30)["passed"] is True


def test_parity_gate_rejects_decoded_shape_and_prompt_mapping_mismatch() -> None:
    metrics = _tensor_metrics(
        [{"prompts": "one", "samples": torch.zeros((1, 3, 1, 2, 2))}],
        [{"prompts": "two", "samples": torch.zeros((1, 3, 1, 2, 1))}],
        expected_prompts=["one"],
        expected_shape=[1, 3, 1, 2, 2],
    )

    gate = _parity_gate(metrics, max_mean_abs_diff=0.02, max_tail_mean_abs_diff=0.30)

    assert gate["passed"] is False
    assert any("prompt mapping mismatch" in failure for failure in gate["failures"])
    assert any("output shape mismatch" in failure for failure in gate["failures"])


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_parity_gate_rejects_non_finite_aggregate_metrics(bad_value: float) -> None:
    metrics = _valid_metrics()
    metrics["max_abs_diff"] = bad_value
    metrics["mean_abs_diff"] = bad_value
    metrics["tail_mean_abs_diff"] = bad_value

    gate = _parity_gate(metrics, max_mean_abs_diff=0.02, max_tail_mean_abs_diff=0.30)

    assert gate["passed"] is False
    assert any("aggregate max_abs_diff is non-finite" in failure for failure in gate["failures"])
    assert any("aggregate mean_abs_diff is non-finite" in failure for failure in gate["failures"])
    assert any("aggregate tail_mean_abs_diff is non-finite" in failure for failure in gate["failures"])


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
