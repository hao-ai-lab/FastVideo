# SPDX-License-Identifier: Apache-2.0

from fastvideo.tests.performance import compare_baseline


def _raw_result():
    return {
        "benchmark_id": "wan-t2v-1.3b-2gpu",
        "device": "NVIDIA L40S",
        "avg_generation_time_s": 10.0,
        "throughput_fps": 4.5,
        "max_peak_memory_mb": 10000.0,
        "commit": "a" * 40,
        "timestamp": "2026-06-16T00:00:00+00:00",
        "pr_number": "123",
    }


def test_detect_run_source_prefers_explicit_env(monkeypatch):
    monkeypatch.setenv("PERF_RUN_SOURCE", "local")
    monkeypatch.setenv("BUILDKITE_PULL_REQUEST", "123")

    assert compare_baseline._detect_run_source() == "local"


def test_detect_run_source_infers_pr(monkeypatch):
    monkeypatch.delenv("PERF_RUN_SOURCE", raising=False)
    monkeypatch.setenv("BUILDKITE_PULL_REQUEST", "123")

    assert compare_baseline._detect_run_source() == "pr"


def test_detect_run_source_infers_scheduled_main(monkeypatch):
    monkeypatch.delenv("PERF_RUN_SOURCE", raising=False)
    monkeypatch.setenv("BUILDKITE_PULL_REQUEST", "false")
    monkeypatch.setenv("BUILDKITE_BRANCH", "main")
    monkeypatch.setenv("TEST_SCOPE", "full")

    assert compare_baseline._detect_run_source() == "scheduled_main"


def test_upload_policy_pass_requires_success(monkeypatch):
    monkeypatch.setattr(compare_baseline, "UPLOAD_POLICY", "pass")

    assert compare_baseline._upload_allowed({"success": True}) is True
    assert compare_baseline._upload_allowed({"success": False}) is False


def test_upload_policy_always_uploads_failures(monkeypatch):
    monkeypatch.setattr(compare_baseline, "UPLOAD_POLICY", "always")

    assert compare_baseline._upload_allowed({"success": False}) is True


def test_normalized_record_includes_source_metadata(monkeypatch):
    monkeypatch.setenv("PERF_RUN_SOURCE", "pr")
    monkeypatch.setenv("BUILDKITE_BRANCH", "feature/perf")
    monkeypatch.setenv("TEST_SCOPE", "direct")
    monkeypatch.setenv("BUILDKITE_BUILD_URL", "https://buildkite.example/build")
    monkeypatch.setenv("BUILDKITE_BUILD_ID", "build-1")
    monkeypatch.setenv("BUILDKITE_JOB_ID", "job-1")

    record = compare_baseline.normalize_performance_result(_raw_result())

    assert record["run_source"] == "pr"
    assert record["baseline_eligible"] is False
    assert record["branch"] == "feature/perf"
    assert record["pr_number"] == "123"
    assert record["test_scope"] == "direct"
    assert record["build_url"] == "https://buildkite.example/build"
    assert record["build_id"] == "build-1"
    assert record["job_id"] == "job-1"


def test_baseline_eligibility_only_for_successful_scheduled_main():
    assert compare_baseline._is_baseline_eligible("scheduled_main", True) is True
    assert compare_baseline._is_baseline_eligible("scheduled_main", False) is False
    assert compare_baseline._is_baseline_eligible("pr", True) is False
    assert compare_baseline._is_baseline_eligible("local", True) is False

