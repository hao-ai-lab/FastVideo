# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from fastvideo.tests.performance import compare_baseline
from fastvideo.performance.metric_policy import resolve_metric_policies


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


def test_normalized_record_includes_effective_regression_thresholds(monkeypatch):
    monkeypatch.setenv("PERF_RUN_SOURCE", "pr")
    raw = _raw_result()
    raw["regression_thresholds"] = {
        "latency": {
            "threshold_percent": 0.09,
            "threshold_absolute": 0.75,
            "gated": True,
        },
        "throughput": {
            "gated": False,
        },
    }

    record = compare_baseline.normalize_performance_result(raw)

    assert record["regression_thresholds"]["latency"] == {
        "threshold_percent": 0.09,
        "threshold_absolute": 0.75,
        "gated": True,
    }
    assert record["regression_thresholds"]["throughput"]["gated"] is False


def test_invalid_regression_threshold_container_uses_defaults():
    policies = resolve_metric_policies(["not", "a", "mapping"])

    latency = next(policy for policy in policies if policy.key == "latency")
    assert latency.threshold_percent == 0.08
    assert latency.threshold_absolute == 0.5
    assert latency.gated is True


def test_boolean_regression_threshold_values_are_ignored():
    policies = resolve_metric_policies({
        "latency": {
            "threshold_percent": True,
            "threshold_absolute": False,
            "gated": "false",
        }
    })

    latency = next(policy for policy in policies if policy.key == "latency")
    assert latency.threshold_percent == 0.08
    assert latency.threshold_absolute == 0.5
    assert latency.gated is False
def test_normalized_record_preserves_identity_metadata(monkeypatch):
    monkeypatch.setenv("PERF_RUN_SOURCE", "pr")
    raw = _raw_result()
    raw.update({
        "result_schema_version": 2,
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "recipe": {
            "recipe_schema_version": 1,
        },
        "recipe_fingerprint": "recipe-1",
        "hardware_profile": {
            "gpu_count": 1,
        },
        "hardware_profile_id": "hw-1",
        "software_profile": {
            "python": "3.12",
        },
        "software_profile_id": "sw-1",
        "environment_metadata": {
            "env": {
                "IMAGE_VERSION": "latest",
            },
        },
        "environment_fingerprint": "env-1",
        "quality_metadata": {
            "quality_status": "canonical",
        },
    })

    record = compare_baseline.normalize_performance_result(raw)

    assert record["result_schema_version"] == 2
    assert record["workload_id"] == "wan-t2v"
    assert record["variant_id"] == "1.3b-sp2"
    assert record["benchmark_version"] == 2
    assert record["recipe"] == {"recipe_schema_version": 1}
    assert record["recipe_fingerprint"] == "recipe-1"
    assert record["hardware_profile"] == {"gpu_count": 1}
    assert record["hardware_profile_id"] == "hw-1"
    assert record["software_profile"] == {"python": "3.12"}
    assert record["software_profile_id"] == "sw-1"
    assert record["environment_metadata"] == {"env": {"IMAGE_VERSION": "latest"}}
    assert record["environment_fingerprint"] == "env-1"
    assert record["quality_metadata"] == {"quality_status": "canonical"}


def test_normalized_record_prefers_raw_v2_provenance(monkeypatch):
    monkeypatch.setenv("PERF_RUN_SOURCE", "pr")
    monkeypatch.setenv("BUILDKITE_BRANCH", "feature/from-env")
    monkeypatch.setenv("TEST_SCOPE", "direct")
    monkeypatch.setenv("BUILDKITE_BUILD_URL", "https://buildkite.example/env")
    monkeypatch.setenv("BUILDKITE_BUILD_ID", "env-build")
    monkeypatch.setenv("BUILDKITE_JOB_ID", "env-job")
    raw = _raw_result()
    raw.update({
        "result_schema_version": 2,
        "run_source": "scheduled_main",
        "branch": "main",
        "test_scope": "full",
        "build_url": "https://buildkite.example/raw",
        "build_id": "raw-build",
        "job_id": "raw-job",
        "pr_number": "false",
    })

    record = compare_baseline.normalize_performance_result(raw)

    assert record["result_schema_version"] == 2
    assert record["run_source"] == "scheduled_main"
    assert record["branch"] == "main"
    assert record["test_scope"] == "full"
    assert record["build_url"] == "https://buildkite.example/raw"
    assert record["build_id"] == "raw-build"
    assert record["job_id"] == "raw-job"
    assert record["pr_number"] == ""


def test_v1_normalized_record_has_no_result_schema_version(monkeypatch):
    monkeypatch.setenv("PERF_RUN_SOURCE", "pr")

    record = compare_baseline.normalize_performance_result(_raw_result())

    assert "result_schema_version" not in record


def test_main_writes_v1_current_artifact_without_comparison_identity(monkeypatch, tmp_path, capsys):
    results_dir = tmp_path / "results"
    reports_dir = tmp_path / "reports"
    tracking_root = tmp_path / "tracking"
    results_dir.mkdir()
    (results_dir / "perf_legacy.json").write_text(json.dumps(_raw_result()), encoding="utf-8")
    uploaded_records = []

    monkeypatch.setenv("PERF_RUN_SOURCE", "scheduled_main")
    monkeypatch.delenv("PERF_PYTEST_RC", raising=False)
    monkeypatch.setattr(compare_baseline, "RESULTS_DIR", str(results_dir))
    monkeypatch.setattr(compare_baseline, "PERF_REPORTS_DIR", str(reports_dir))
    monkeypatch.setattr(compare_baseline, "TRACKING_ROOT", str(tracking_root))
    monkeypatch.setattr(compare_baseline, "UPLOAD_POLICY", "always")
    monkeypatch.setattr(compare_baseline, "sync_from_hf", lambda local_dir, strict=False: local_dir)

    def fail_load_records_for_model(*_args, **_kwargs):
        raise AssertionError("legacy current records should skip v2 baseline lookup")

    def fake_upload_record(_path, record, *, strict=False):
        uploaded_records.append(record.copy())

    monkeypatch.setattr(compare_baseline, "load_records_for_model", fail_load_records_for_model)
    monkeypatch.setattr(compare_baseline, "upload_record", fake_upload_record)

    assert compare_baseline.main() == 0

    output = capsys.readouterr().out
    normalized_files = list((reports_dir / "results").glob("normalized_perf_*.json"))
    assert "Skipping rolling baseline comparison" in output
    assert "workload_id" in output
    assert len(normalized_files) == 1

    normalized = json.loads(normalized_files[0].read_text(encoding="utf-8"))
    assert "result_schema_version" not in normalized
    assert normalized["success"] is True
    assert normalized["baseline_eligible"] is False
    assert len(uploaded_records) == 1
    assert uploaded_records[0]["baseline_eligible"] is False


def test_normalized_record_reads_identity_labels_from_recipe(monkeypatch):
    monkeypatch.setenv("PERF_RUN_SOURCE", "pr")
    raw = _raw_result()
    raw["recipe"] = {
        "benchmark": {
            "workload_id": "wan-t2v",
            "variant_id": "1.3b-sp2",
            "benchmark_version": 2,
        },
    }

    record = compare_baseline.normalize_performance_result(raw)

    assert record["workload_id"] == "wan-t2v"
    assert record["variant_id"] == "1.3b-sp2"
    assert record["benchmark_version"] == 2


def test_comparison_identity_filters_use_full_issue_key():
    record = {
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "recipe_fingerprint": "recipe-1",
        "hardware_profile_id": "hw-1",
        "software_profile_id": "sw-1",
    }

    assert compare_baseline._comparison_identity_filters(record) == {
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": "2",
        "recipe_fingerprint": "recipe-1",
        "hardware_profile_id": "hw-1",
        "software_profile_id": "sw-1",
    }


def test_comparison_identity_filters_require_full_issue_key():
    record = {
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 0,
        "recipe_fingerprint": "recipe-1",
        "hardware_profile_id": "hw-1",
    }

    with pytest.raises(ValueError, match="software_profile_id"):
        compare_baseline._comparison_identity_filters(record)


def test_comparison_identity_filters_keep_zero_version():
    record = {
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 0,
        "recipe_fingerprint": "recipe-1",
        "hardware_profile_id": "hw-1",
        "software_profile_id": "sw-1",
    }

    assert compare_baseline._comparison_identity_filters(record)["benchmark_version"] == "0"


def test_baseline_eligibility_only_for_successful_scheduled_main():
    assert compare_baseline._is_baseline_eligible("scheduled_main", True) is True
    assert compare_baseline._is_baseline_eligible("scheduled_main", False) is False
    assert compare_baseline._is_baseline_eligible("pr", True) is False
    assert compare_baseline._is_baseline_eligible("local", True) is False


def test_latency_regression_requires_percent_and_absolute_floors():
    baseline = [{"latency": 10.0}]
    current = {"model_id": "wan", "latency": 10.6}

    percent_only = resolve_metric_policies({
        "latency": {
            "threshold_percent": 0.05,
            "threshold_absolute": 0.75,
        }
    })
    absolute_only = resolve_metric_policies({
        "latency": {
            "threshold_percent": 0.10,
            "threshold_absolute": 0.5,
        }
    })
    both = resolve_metric_policies({
        "latency": {
            "threshold_percent": 0.05,
            "threshold_absolute": 0.5,
        }
    })

    assert compare_baseline._check_regressions(current, baseline, percent_only) == []
    assert compare_baseline._check_regressions(current, baseline, absolute_only) == []

    failures = compare_baseline._check_regressions(current, baseline, both)
    assert len(failures) == 1
    assert "latency regressed by 6.0% and 0.600" in failures[0]


def test_throughput_regression_uses_higher_is_better_direction():
    baseline = [{"throughput": 10.0}]
    current = {"model_id": "wan", "throughput": 9.0}
    policies = resolve_metric_policies({
        "throughput": {
            "threshold_percent": 0.05,
            "threshold_absolute": 0.5,
        }
    })

    failures = compare_baseline._check_regressions(current, baseline, policies)

    assert len(failures) == 1
    assert "throughput regressed by 10.0% and 1.000" in failures[0]


def test_memory_regression_uses_metric_specific_absolute_floor():
    baseline = [{"memory": 10000.0}]
    current = {"model_id": "wan", "memory": 10600.0}
    policies = resolve_metric_policies({
        "memory": {
            "threshold_percent": 0.05,
            "threshold_absolute": 256.0,
        }
    })

    failures = compare_baseline._check_regressions(current, baseline, policies)

    assert len(failures) == 1
    assert "memory regressed by 6.0% and 600.0" in failures[0]


def test_component_metric_can_gate_independently():
    baseline = [{"dit_time_s": 8.0}]
    current = {"model_id": "wan", "dit_time_s": 8.6}
    policies = resolve_metric_policies({
        "dit_time_s": {
            "threshold_percent": 0.05,
            "threshold_absolute": 0.25,
        }
    })

    failures = compare_baseline._check_regressions(current, baseline, policies)

    assert len(failures) == 1
    assert "dit_time_s regressed by 7.5% and 0.600" in failures[0]


def test_informational_metric_remains_visible_without_failing():
    baseline = [{"throughput": 10.0}]
    current = {"model_id": "wan", "gpu_type": "NVIDIA L40S", "throughput": 8.0}
    policies = resolve_metric_policies({
        "throughput": {
            "threshold_percent": 0.01,
            "threshold_absolute": 0.01,
            "gated": False,
        }
    })

    row = compare_baseline._build_summary_row(current, baseline, policies, False)

    assert compare_baseline._check_regressions(current, baseline, policies) == []
    assert row["metrics"]["throughput"]["regression_pct"] == 20.0
    assert row["metrics"]["throughput"]["gated"] is False
    assert row["metrics"]["throughput"]["threshold_exceeded"] is True
    assert row["metrics"]["throughput"]["regressed"] is False
    assert row["threshold_exceeded_metrics"] == ["throughput"]
    assert row["failing_metrics"] == []
