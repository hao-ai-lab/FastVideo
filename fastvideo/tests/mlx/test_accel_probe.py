# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the Apple-Silicon accelerator probe (tiny shapes)."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from fastvideo.mlx_runtime import accel_probe


def test_detect_platform_fields() -> None:
    info = accel_probe.detect_platform()
    assert info.chip
    assert info.mlx_version
    assert isinstance(info.neural_accel_expected, bool)


def test_run_probe_small_shapes() -> None:
    # Sides divisible by every mode's group size (affine 64, mxfp* 32, nvfp4 16).
    report = accel_probe.run_probe(
        gemm=(128, 128, 128), attn=(1, 2, 64, 32), warmup=1, iters=1
    )
    assert "platform" in report and "results" in report
    kinds = {r["kind"] for r in report["results"]}
    assert {"gemm", "attention"} <= kinds

    # There must be an fp16 GEMM baseline, and every quantized GEMM row must
    # carry a speedup relative to it.
    gemms = [r for r in report["results"] if r["kind"] == "gemm"]
    assert any(r["backend"] == "fp16" for r in gemms)
    for r in gemms:
        if r["backend"] != "fp16":
            assert r["speedup_vs_fp16"] is not None and r["speedup_vs_fp16"] > 0

    # On non-M5 silicon the engaged flag must stay False even if a bandwidth-
    # driven quant speedup exists.
    if not report["platform"]["neural_accel_expected"]:
        assert report["accelerator_likely_engaged"] is False


def test_bench_gemm_reports_positive_throughput() -> None:
    res = accel_probe.bench_gemm("fp16", 128, 128, 128, warmup=1, iters=1)
    assert res.tflops > 0
    assert res.seconds_per_iter > 0
