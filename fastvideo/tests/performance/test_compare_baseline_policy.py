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


def _v2_raw_result(**overrides):
    raw = _raw_result()
    raw.update({
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "recipe_fingerprint": "recipe-1",
        "hardware_profile_id": "hw-1",
        "software_profile_id": "sw-1",
    })
    raw.update(overrides)
    return raw


def _v2_baseline_record(**overrides):
    record = {
        "gpu_type": "NVIDIA L40S",
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "recipe_fingerprint": "recipe-1",
        "hardware_profile_id": "hw-1",
        "software_profile_id": "sw-1",
        "latency": 10.0,
        "throughput": 4.5,
        "memory": 10000.0,
        "timestamp": "2026-06-15T00:00:00+00:00",
        "success": True,
        "run_source": "scheduled_main",
        "baseline_eligible": True,
    }
    record.update(overrides)
    return record


def _run_compare(monkeypatch, tmp_path, raw_result, baseline_records):
    """Run compare_baseline.main() against a local tracking root.

    Returns (exit_code, normalized_record, markdown_report).
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "perf_current.json").write_text(json.dumps(raw_result))

    model_dir = tmp_path / "tracking" / raw_result["benchmark_id"]
    model_dir.mkdir(parents=True)
    for index, record in enumerate(baseline_records):
        (model_dir / f"rec{index}.json").write_text(json.dumps(record))

    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(compare_baseline, "RESULTS_DIR", str(results_dir))
    monkeypatch.setattr(compare_baseline, "TRACKING_ROOT", str(tmp_path / "tracking"))
    monkeypatch.setattr(compare_baseline, "PERF_REPORTS_DIR", str(reports_dir))
    monkeypatch.setattr(compare_baseline, "UPLOAD_POLICY", "never")
    monkeypatch.setattr(compare_baseline, "sync_from_hf", lambda *args, **kwargs: None)
    monkeypatch.setenv("PERF_RUN_SOURCE", "scheduled_main")
    monkeypatch.delenv("PERF_PYTEST_RC", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    exit_code = compare_baseline.main()

    normalized_paths = list((reports_dir / "results").glob("normalized_perf_*.json"))
    assert len(normalized_paths) == 1
    markdown = "\n".join(path.read_text() for path in reports_dir.glob("perf_*.md"))
    return exit_code, json.loads(normalized_paths[0].read_text()), markdown


def test_main_pass_with_comparable_baseline(monkeypatch, tmp_path):
    exit_code, record, markdown = _run_compare(
        monkeypatch, tmp_path, _v2_raw_result(), [_v2_baseline_record()])

    assert exit_code == 0
    assert record["comparator_status"] == "PASS"
    assert record["baseline_status"] == "compared"
    assert record["success"] is True
    # Scheduled-main PASS records keep advancing the rolling baseline.
    assert record["baseline_eligible"] is True
    assert "| PASS |" in markdown


def test_main_regression_on_gated_metric_fails_ci(monkeypatch, tmp_path):
    exit_code, record, markdown = _run_compare(
        monkeypatch, tmp_path, _v2_raw_result(avg_generation_time_s=20.0),
        [_v2_baseline_record()])

    assert exit_code == 1
    assert record["comparator_status"] == "REGRESSION"
    assert record["success"] is False
    assert record["baseline_eligible"] is False
    assert "| REGRESSION |" in markdown


def test_main_missing_baseline_is_calibration_needed_and_does_not_seed(monkeypatch, tmp_path):
    exit_code, record, markdown = _run_compare(
        monkeypatch, tmp_path, _v2_raw_result(), [])

    assert exit_code == 0
    assert record["comparator_status"] == "CALIBRATION_NEEDED"
    assert record["baseline_status"] == "initialized_new_cohort"
    assert record["success"] is True
    # Visible, but never silently seeds a passing baseline.
    assert record["baseline_eligible"] is False
    assert "| CALIBRATION_NEEDED |" in markdown


def test_main_recipe_change_without_new_variant_is_recipe_mismatch(monkeypatch, tmp_path):
    exit_code, record, markdown = _run_compare(
        monkeypatch, tmp_path, _v2_raw_result(recipe_fingerprint="recipe-2"),
        [_v2_baseline_record()])

    assert exit_code == 1
    assert record["comparator_status"] == "RECIPE_MISMATCH"
    assert record["success"] is False
    assert record["baseline_eligible"] is False
    assert "| RECIPE_MISMATCH |" in markdown


def test_main_recipe_change_as_new_variant_is_calibration_needed(monkeypatch, tmp_path):
    exit_code, record, _ = _run_compare(
        monkeypatch, tmp_path,
        _v2_raw_result(variant_id="1.3b-sp2-r2", recipe_fingerprint="recipe-2"),
        [_v2_baseline_record()])

    assert exit_code == 0
    assert record["comparator_status"] == "CALIBRATION_NEEDED"


def test_main_v2_record_never_compares_against_v1_baselines(monkeypatch, tmp_path):
    legacy_v1_baseline = {
        "gpu_type": "NVIDIA L40S",
        # Would be a >100% latency regression if the comparator (wrongly)
        # matched the v2 record against v1 history.
        "latency": 1.0,
        "timestamp": "2026-06-15T00:00:00+00:00",
        "success": True,
    }

    exit_code, record, _ = _run_compare(
        monkeypatch, tmp_path, _v2_raw_result(), [legacy_v1_baseline])

    assert exit_code == 0
    assert record["comparator_status"] == "CALIBRATION_NEEDED"


def test_main_legacy_v1_record_skips_comparison_with_pass_verdict(monkeypatch, tmp_path):
    legacy_v1_baseline = {
        "gpu_type": "NVIDIA L40S",
        # Would be a >100% latency regression if the comparator (wrongly)
        # compared the identity-less v1 record against this history.
        "latency": 1.0,
        "timestamp": "2026-06-15T00:00:00+00:00",
        "success": True,
    }

    exit_code, record, markdown = _run_compare(
        monkeypatch, tmp_path, _raw_result(), [legacy_v1_baseline])

    assert exit_code == 0
    assert record["comparator_status"] == "PASS"
    assert record["baseline_status"] == "skipped_missing_identity"
    assert record["baseline_eligible"] is False
    assert "| PASS |" in markdown


def test_main_partial_identity_skips_comparison(monkeypatch, tmp_path):
    raw = _v2_raw_result()
    del raw["software_profile_id"]

    exit_code, record, _ = _run_compare(monkeypatch, tmp_path, raw, [])

    assert exit_code == 0
    assert record["comparator_status"] == "PASS"
    assert record["baseline_status"] == "skipped_missing_identity"
    assert record["success"] is True
    assert record["baseline_eligible"] is False


def test_main_baseline_load_failure_is_infra_error_and_fails_ci(monkeypatch, tmp_path):
    def _boom(*_args, **_kwargs):
        raise OSError("hf store unavailable")

    monkeypatch.setattr(compare_baseline, "load_records_for_model", _boom)

    exit_code, record, markdown = _run_compare(
        monkeypatch, tmp_path, _v2_raw_result(), [])

    assert exit_code == 1
    assert record["comparator_status"] == "INFRA_ERROR"
    assert record["success"] is False
    assert record["baseline_eligible"] is False
    assert "| INFRA_ERROR |" in markdown


def test_main_host_below_profile_skips_gate_and_never_seeds_baseline(
        monkeypatch, tmp_path, capsys):
    # A 2x latency "regression" measured on a packed host must not fail CI.
    exit_code, record, markdown = _run_compare(
        monkeypatch, tmp_path,
        _v2_raw_result(avg_generation_time_s=20.0, host_cpu_score=0.55),
        [_v2_baseline_record()])

    assert exit_code == 0
    assert record["comparator_status"] == "HOST_BELOW_PROFILE"
    assert record["success"] is True
    assert record["baseline_eligible"] is False
    assert "| HOST_BELOW_PROFILE |" in markdown
    assert ("host below profile — measurements not comparable"
            in capsys.readouterr().out)


def test_main_healthy_host_score_gates_normally(monkeypatch, tmp_path):
    exit_code, record, _ = _run_compare(
        monkeypatch, tmp_path,
        _v2_raw_result(avg_generation_time_s=20.0, host_cpu_score=0.97),
        [_v2_baseline_record()])

    assert exit_code == 1
    assert record["comparator_status"] == "REGRESSION"


def test_main_healthy_host_pass_keeps_baseline_eligibility_and_score(
        monkeypatch, tmp_path):
    exit_code, record, _ = _run_compare(
        monkeypatch, tmp_path, _v2_raw_result(host_cpu_score=1.02),
        [_v2_baseline_record()])

    assert exit_code == 0
    assert record["comparator_status"] == "PASS"
    assert record["baseline_eligible"] is True
    assert record["host_cpu_score"] == 1.02


def test_host_cpu_min_score_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PERF_HOST_CPU_MIN_SCORE", "0.5")

    exit_code, record, _ = _run_compare(
        monkeypatch, tmp_path,
        _v2_raw_result(avg_generation_time_s=20.0, host_cpu_score=0.55),
        [_v2_baseline_record()])

    assert exit_code == 1
    assert record["comparator_status"] == "REGRESSION"
