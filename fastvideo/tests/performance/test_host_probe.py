# SPDX-License-Identifier: Apache-2.0
"""CPU-runnable smoke test for the host CPU health probe (~2s)."""

import pytest

from fastvideo.tests.performance import host_probe


def test_probe_reports_positive_normalized_profile(monkeypatch):
    monkeypatch.setenv("PERF_HOST_CPU_ST_REF_KOPS", "1000")

    profile = host_probe.measure_host_cpu_profile()

    assert profile["host_cpu_single_thread_kops"] > 0
    assert profile["host_cpu_multi_thread_gflops"] > 0
    # The gated score is the single-thread axis normalized by the reference.
    assert profile["host_cpu_score"] == pytest.approx(
        profile["host_cpu_single_thread_kops"] / 1000.0, rel=0.05)
