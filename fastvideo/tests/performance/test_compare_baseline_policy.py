# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from fastvideo.tests.performance import compare_baseline
from fastvideo.tests.performance import seed_baseline
from fastvideo.tests.performance import test_inference_performance as perf_test
from fastvideo.performance.hf_store import (
    load_records_for_identity,
    load_records_for_model,
    sanitize,
)
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


def _v2_raw_result(**overrides):
    raw = _raw_result()
    raw.update({
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "recipe_fingerprint": "recipe-1",
        "hardware_profile_id": "hw-l40s-2",
        "software_profile_id": "sw-cuda",
    })
    raw.update(overrides)
    return raw


def _v2_record(**overrides):
    return compare_baseline.normalize_performance_result(_v2_raw_result(**overrides))


def _write_record(root, model_id, filename, record):
    model_dir = root / sanitize(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / filename, "w", encoding="utf-8") as f:
        json.dump(record, f)


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
    monkeypatch.setenv("PERF_RUN_SOURCE", "scheduled_main")
    monkeypatch.delenv("PERF_PYTEST_RC", raising=False)
    raw_result = perf_test._build_result_record(
        cfg={"benchmark_id": "wan-t2v-1.3b-2gpu"},
        model_info={},
        init_kwargs={},
        gen_kwargs={},
        num_warmup=1,
        num_measure=1,
        thresholds={},
        times=[10.0],
        peak_memories=[10000.0],
        all_component_times=[],
        prompt="A cinematic video.",
        runtime_identity={},
        device_name="NVIDIA L40S",
        timestamp="2026-06-16T00:00:00+00:00",
    )
    assert "result_schema_version" not in raw_result
    (results_dir / "perf_legacy.json").write_text(json.dumps(raw_result), encoding="utf-8")
    uploaded_records = []

    monkeypatch.setattr(compare_baseline, "RESULTS_DIR", str(results_dir))
    monkeypatch.setattr(compare_baseline, "PERF_REPORTS_DIR", str(reports_dir))
    monkeypatch.setattr(compare_baseline, "TRACKING_ROOT", str(tracking_root))
    monkeypatch.setattr(compare_baseline, "UPLOAD_POLICY", "always")
    monkeypatch.setattr(compare_baseline, "sync_from_hf", lambda local_dir, strict=False: local_dir)

    def fail_load_records_for_identity(*_args, **_kwargs):
        raise AssertionError("legacy current records should skip v2 baseline lookup")

    def fake_upload_record(_path, record, *, strict=False):
        uploaded_records.append(record.copy())

    monkeypatch.setattr(compare_baseline, "load_records_for_identity", fail_load_records_for_identity)
    monkeypatch.setattr(compare_baseline, "upload_record", fake_upload_record)

    assert compare_baseline.main() == 0

    output = capsys.readouterr().out
    normalized_files = list((reports_dir / "results").glob("normalized_perf_*.json"))
    assert "Skipping rolling baseline comparison" in output
    assert "workload_id" in output
    assert len(normalized_files) == 1

    normalized = json.loads(normalized_files[0].read_text(encoding="utf-8"))
    assert "result_schema_version" not in normalized
    assert normalized["comparison_status"] == compare_baseline.STATUS_PASS
    assert normalized["baseline_status"] == "skipped_missing_identity"
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
    assert compare_baseline._is_baseline_eligible(
        "scheduled_main", True, compare_baseline.STATUS_PASS) is True
    assert compare_baseline._is_baseline_eligible(
        "scheduled_main", False, compare_baseline.STATUS_PASS) is False
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


def test_normalized_record_preserves_v2_comparison_identity(monkeypatch):
    monkeypatch.setenv("PERF_RUN_SOURCE", "pr")

    record = _v2_record()

    assert record["workload_id"] == "wan-t2v"
    assert record["variant_id"] == "1.3b-sp2"
    assert record["benchmark_version"] == 2
    assert record["recipe_fingerprint"] == "recipe-1"
    assert record["hardware_profile_id"] == "hw-l40s-2"
    assert record["software_profile_id"] == "sw-cuda"


def test_comparison_identity_filters_require_full_issue_key():
    record = {
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "recipe_fingerprint": "recipe-1",
        "hardware_profile_id": "hw-l40s-2",
    }

    with pytest.raises(ValueError, match="software_profile_id"):
        compare_baseline._comparison_identity_filters(record)


def test_comparison_identity_filters_keep_zero_version():
    record = {
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 0,
        "recipe_fingerprint": "recipe-1",
        "hardware_profile_id": "hw-l40s-2",
        "software_profile_id": "sw-cuda",
    }

    assert compare_baseline._comparison_identity_filters(record)["benchmark_version"] == "0"


def test_exact_comparable_baseline_without_regression_reports_pass(monkeypatch):
    monkeypatch.setenv("PERF_RUN_SOURCE", "pr")
    record = _v2_record()
    baseline_records = [{
        "latency": 10.0,
        "throughput": 4.5,
        "memory": 10000.0,
    }]

    failures, status, reason = compare_baseline._evaluate_record_comparison(
        record,
        baseline_records,
        [],
        resolve_metric_policies(None),
        [],
        False,
    )

    assert failures == []
    assert status == compare_baseline.STATUS_PASS
    assert "no gated regressions" in reason


def test_slower_gated_metric_reports_regression(monkeypatch):
    monkeypatch.setenv("PERF_RUN_SOURCE", "pr")
    record = _v2_record(avg_generation_time_s=11.0)
    baseline_records = [{"latency": 10.0}]

    failures, status, reason = compare_baseline._evaluate_record_comparison(
        record,
        baseline_records,
        [],
        resolve_metric_policies(None),
        [],
        False,
    )

    assert status == compare_baseline.STATUS_REGRESSION
    assert len(failures) == 1
    assert "latency regressed by 10.0%" in failures[0]
    assert reason == failures[0]


def test_missing_exact_v2_baseline_reports_calibration_needed(monkeypatch):
    monkeypatch.setenv("PERF_RUN_SOURCE", "pr")
    record = _v2_record()

    failures, status, reason = compare_baseline._evaluate_record_comparison(
        record,
        [],
        [],
        resolve_metric_policies(None),
        [],
        False,
    )

    assert failures == []
    assert status == compare_baseline.STATUS_CALIBRATION_NEEDED
    assert "No baseline found for exact comparable identity" in reason


def test_seeded_v2_calibration_artifact_enables_next_compare_pass(
    monkeypatch,
    tmp_path,
):
    tracking_root = tmp_path / "tracking"
    results_dir = tmp_path / "results"
    reports_dir = tmp_path / "reports"
    results_dir.mkdir()

    calibration_record = _v2_record()
    calibration_record.update({
        "comparison_status": compare_baseline.STATUS_CALIBRATION_NEEDED,
        "comparison_status_reason": "No baseline found for exact comparable identity",
        "success": True,
        "baseline_eligible": False,
        "run_source": "scheduled_main",
        "branch": "main",
        "test_scope": "full",
        "pr_number": "",
    })
    seed_record = seed_baseline.build_baseline_seed_record(
        calibration_record,
        reason="reviewed first v2 baseline",
        timestamp="2026-06-17T00:00:00+00:00",
        operator="test",
    )
    seed_baseline.write_seed_record(str(tracking_root), seed_record)

    assert seed_record["baseline_eligible"] is True
    assert seed_record["comparison_status"] == compare_baseline.STATUS_PASS
    assert seed_record["baseline_seed_source_status"] == compare_baseline.STATUS_CALIBRATION_NEEDED
    assert load_records_for_identity(
        str(tracking_root),
        compare_baseline._comparison_identity_filters(calibration_record),
        baseline_eligible_only=True,
    ) == [seed_record]

    with open(results_dir / "perf_current.json", "w", encoding="utf-8") as f:
        json.dump(_v2_raw_result(
            avg_generation_time_s=10.1,
            timestamp="2026-06-18T00:00:00+00:00",
        ), f)

    monkeypatch.setenv("PERF_RUN_SOURCE", "pr")
    monkeypatch.setattr(compare_baseline, "TRACKING_ROOT", str(tracking_root))
    monkeypatch.setattr(compare_baseline, "RESULTS_DIR", str(results_dir))
    monkeypatch.setattr(compare_baseline, "PERF_REPORTS_DIR", str(reports_dir))
    monkeypatch.setattr(compare_baseline, "UPLOAD_POLICY", "never")
    monkeypatch.setattr(compare_baseline, "sync_from_hf", lambda local_dir, strict=False: local_dir)
    monkeypatch.delenv("PERF_PYTEST_RC", raising=False)

    assert compare_baseline.main() == 0

    artifacts = list((reports_dir / "results").glob("normalized_perf_*.json"))
    assert len(artifacts) == 1
    with open(artifacts[0], encoding="utf-8") as f:
        normalized = json.load(f)

    assert normalized["comparison_status"] == compare_baseline.STATUS_PASS
    assert normalized["baseline_eligible"] is False


def test_baseline_seed_rejects_non_calibration_sources(monkeypatch):
    monkeypatch.setenv("PERF_RUN_SOURCE", "scheduled_main")
    record = _v2_record()
    record.update({
        "comparison_status": compare_baseline.STATUS_PASS,
        "success": True,
    })

    with pytest.raises(ValueError, match="CALIBRATION_NEEDED"):
        seed_baseline.build_baseline_seed_record(
            record,
            reason="not a calibration",
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({
            "run_source": "pr",
            "branch": "main",
            "test_scope": "full",
            "pr_number": "123",
        }, "scheduled_main"),
        ({
            "run_source": "local",
            "branch": "",
            "test_scope": "",
            "pr_number": "",
        }, "scheduled_main"),
        ({
            "run_source": "scheduled_main",
            "branch": "main",
            "test_scope": "full",
            "pr_number": "123",
        }, "non-PR"),
        ({
            "run_source": "scheduled_main",
            "branch": "feature",
            "test_scope": "full",
            "pr_number": "",
        }, "main-branch full-suite"),
        ({
            "run_source": "scheduled_main",
            "branch": "main",
            "test_scope": "direct",
            "pr_number": "",
        }, "main-branch full-suite"),
    ],
)
def test_baseline_seed_rejects_untrusted_calibration_sources(monkeypatch, overrides, match):
    monkeypatch.setenv("PERF_RUN_SOURCE", "scheduled_main")
    record = _v2_record()
    record.update({
        "comparison_status": compare_baseline.STATUS_CALIBRATION_NEEDED,
        "success": True,
        **overrides,
    })

    with pytest.raises(ValueError, match=match):
        seed_baseline.build_baseline_seed_record(
            record,
            reason="not trusted",
        )


def test_baseline_seed_rejects_mixed_exact_identities(monkeypatch):
    monkeypatch.setenv("PERF_RUN_SOURCE", "scheduled_main")
    first = _v2_record()
    second = _v2_record(variant_id="different-variant")
    for record in (first, second):
        record.update({
            "comparison_status": compare_baseline.STATUS_CALIBRATION_NEEDED,
            "success": True,
        })

    with pytest.raises(ValueError, match="exact comparable identity"):
        seed_baseline._validate_same_identity([first, second])


def test_baseline_seed_validates_entire_batch_before_writing_or_uploading(monkeypatch, tmp_path):
    valid_source = _v2_record()
    valid_source.update({
        "comparison_status": compare_baseline.STATUS_CALIBRATION_NEEDED,
        "success": True,
        "run_source": "scheduled_main",
        "branch": "main",
        "test_scope": "full",
        "pr_number": "",
    })
    invalid_source = dict(valid_source)
    invalid_source.update({
        "run_source": "pr",
        "pr_number": "123",
    })
    source_paths = []
    for index, source in enumerate((valid_source, invalid_source), start=1):
        source_path = tmp_path / f"source_{index}.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        source_paths.append(source_path)

    tracking_root = tmp_path / "tracking"
    uploaded_records = []
    monkeypatch.setattr(
        seed_baseline,
        "upload_record",
        lambda path, record, *, strict=False: uploaded_records.append((path, record)),
    )

    with pytest.raises(ValueError, match="scheduled_main"):
        seed_baseline.main([
            "--source-result",
            str(source_paths[0]),
            "--source-result",
            str(source_paths[1]),
            "--intent-rationale",
            "reviewed first v2 baseline",
            "--tracking-root",
            str(tracking_root),
            "--upload",
        ])

    assert not tracking_root.exists()
    assert uploaded_records == []


def test_baseline_seed_rejects_duplicate_sources_before_writing_or_uploading(monkeypatch, tmp_path):
    source = _v2_record()
    source.update({
        "comparison_status": compare_baseline.STATUS_CALIBRATION_NEEDED,
        "success": True,
        "run_source": "scheduled_main",
        "branch": "main",
        "test_scope": "full",
        "pr_number": "",
    })
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    tracking_root = tmp_path / "tracking"
    uploaded_records = []
    monkeypatch.setattr(
        seed_baseline,
        "upload_record",
        lambda path, record, *, strict=False: uploaded_records.append((path, record)),
    )

    with pytest.raises(ValueError, match="duplicates a resolved source path"):
        seed_baseline.main([
            "--source-result",
            str(source_path),
            "--source-result",
            str(source_path),
            "--intent-rationale",
            "reviewed first v2 baseline",
            "--tracking-root",
            str(tracking_root),
            "--upload",
        ])

    assert not tracking_root.exists()
    assert uploaded_records == []


def test_baseline_seed_rejects_later_missing_model_id_before_persistence(monkeypatch, tmp_path):
    valid_source = _v2_record()
    valid_source.update({
        "comparison_status": compare_baseline.STATUS_CALIBRATION_NEEDED,
        "success": True,
        "run_source": "scheduled_main",
        "branch": "main",
        "test_scope": "full",
        "pr_number": "",
    })
    invalid_source = dict(valid_source)
    invalid_source.pop("model_id")
    invalid_source["timestamp"] = "2026-06-17T00:00:00+00:00"
    source_paths = []
    for index, source in enumerate((valid_source, invalid_source), start=1):
        source_path = tmp_path / f"source_{index}.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        source_paths.append(source_path)

    tracking_root = tmp_path / "tracking"
    uploaded_records = []
    monkeypatch.setattr(
        seed_baseline,
        "upload_record",
        lambda path, record, *, strict=False: uploaded_records.append((path, record)),
    )

    with pytest.raises(ValueError, match="non-empty string model_id"):
        seed_baseline.main([
            "--source-result",
            str(source_paths[0]),
            "--source-result",
            str(source_paths[1]),
            "--intent-rationale",
            "reviewed first v2 baseline",
            "--tracking-root",
            str(tracking_root),
            "--upload",
        ])

    assert not tracking_root.exists()
    assert uploaded_records == []


@pytest.mark.parametrize(
    ("metric", "divergent_value"),
    [
        ("latency", 12.0),
        ("throughput", 3.5),
    ],
)
def test_baseline_seed_rejects_inconsistent_batch_before_persistence(
    monkeypatch,
    tmp_path,
    metric,
    divergent_value,
):
    first_source = _v2_record()
    first_source.update({
        "comparison_status": compare_baseline.STATUS_CALIBRATION_NEEDED,
        "success": True,
        "run_source": "scheduled_main",
        "branch": "main",
        "test_scope": "full",
        "pr_number": "",
    })
    second_source = dict(first_source)
    second_source.update({
        metric: divergent_value,
        "timestamp": "2026-06-17T00:00:00+00:00",
    })
    source_paths = []
    for index, source in enumerate((first_source, second_source), start=1):
        source_path = tmp_path / f"source_{index}.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        source_paths.append(source_path)

    tracking_root = tmp_path / "tracking"
    uploaded_records = []
    monkeypatch.setattr(
        seed_baseline,
        "upload_record",
        lambda path, record, *, strict=False: uploaded_records.append((path, record)),
    )

    with pytest.raises(ValueError, match=rf"source artifact 2 {metric} regresses"):
        seed_baseline.main([
            "--source-result",
            str(source_paths[0]),
            "--source-result",
            str(source_paths[1]),
            "--intent-rationale",
            "reviewed first v2 baseline",
            "--tracking-root",
            str(tracking_root),
            "--max-intra-batch-regression",
            "0.05",
            "--upload",
        ])

    assert not tracking_root.exists()
    assert uploaded_records == []


def test_same_variant_changed_recipe_reports_recipe_mismatch(monkeypatch):
    monkeypatch.setenv("PERF_RUN_SOURCE", "pr")
    record = _v2_record(recipe_fingerprint="recipe-2")
    recipe_mismatch_records = [{
        "recipe_fingerprint": "recipe-1",
    }]

    failures, status, reason = compare_baseline._evaluate_record_comparison(
        record,
        [],
        recipe_mismatch_records,
        resolve_metric_policies(None),
        [],
        False,
    )

    assert status == compare_baseline.STATUS_RECIPE_MISMATCH
    assert len(failures) == 1
    assert "recipe_fingerprint=recipe-2" in failures[0]
    assert "recipe-1" in reason


def test_same_recipe_calibration_record_does_not_report_recipe_mismatch(monkeypatch):
    monkeypatch.setenv("PERF_RUN_SOURCE", "pr")
    record = _v2_record()

    assert compare_baseline._recipe_mismatch_records(
        record,
        [{
            "recipe_fingerprint": "recipe-1",
            "run_source": "scheduled_main",
        }],
    ) == []


def test_main_recipe_mismatch_takes_precedence_over_static_regression(
    monkeypatch,
    tmp_path,
):
    tracking_root = tmp_path / "tracking"
    results_dir = tmp_path / "results"
    reports_dir = tmp_path / "reports"
    results_dir.mkdir()
    _write_record(
        tracking_root,
        "renamed-benchmark-id",
        "prior-calibration.json",
        {
            "model_id": "renamed-benchmark-id",
            "gpu_type": "NVIDIA L40S PCIe",
            "timestamp": "2026-06-15T00:00:00+00:00",
            "success": True,
            "baseline_eligible": False,
            "run_source": "scheduled_main",
            "comparison_status": compare_baseline.STATUS_CALIBRATION_NEEDED,
            "latency": 10.0,
            "workload_id": "wan-t2v",
            "variant_id": "1.3b-sp2",
            "benchmark_version": 2,
            "recipe_fingerprint": "recipe-1",
            "hardware_profile_id": "hw-l40s-2",
            "software_profile_id": "sw-cuda",
        },
    )
    with open(results_dir / "perf_current.json", "w", encoding="utf-8") as f:
        json.dump(
            _v2_raw_result(
                recipe_fingerprint="recipe-2",
                avg_generation_time_s=20.0,
                thresholds={"max_generation_time_s": 15.0},
            ),
            f,
        )

    monkeypatch.setattr(compare_baseline, "TRACKING_ROOT", str(tracking_root))
    monkeypatch.setattr(compare_baseline, "RESULTS_DIR", str(results_dir))
    monkeypatch.setattr(compare_baseline, "PERF_REPORTS_DIR", str(reports_dir))
    monkeypatch.setattr(compare_baseline, "UPLOAD_POLICY", "never")
    monkeypatch.setattr(compare_baseline, "sync_from_hf", lambda local_dir, strict=False: local_dir)
    monkeypatch.delenv("PERF_PYTEST_RC", raising=False)

    assert compare_baseline.main() == 1

    artifacts = list((reports_dir / "results").glob("normalized_perf_*.json"))
    assert len(artifacts) == 1
    with open(artifacts[0], encoding="utf-8") as f:
        normalized = json.load(f)

    assert normalized["comparison_status"] == compare_baseline.STATUS_RECIPE_MISMATCH
    assert "recipe-1" in normalized["comparison_status_reason"]
    assert "fixed threshold" not in normalized["comparison_status_reason"]


def test_prior_pr_calibration_record_does_not_report_recipe_mismatch(
    monkeypatch,
    tmp_path,
):
    tracking_root = tmp_path / "tracking"
    results_dir = tmp_path / "results"
    reports_dir = tmp_path / "reports"
    results_dir.mkdir()
    _write_record(
        tracking_root,
        "renamed-benchmark-id",
        "prior-pr-calibration.json",
        {
            "model_id": "renamed-benchmark-id",
            "gpu_type": "NVIDIA L40S PCIe",
            "timestamp": "2026-06-15T00:00:00+00:00",
            "success": True,
            "baseline_eligible": False,
            "run_source": "pr",
            "comparison_status": compare_baseline.STATUS_CALIBRATION_NEEDED,
            "latency": 10.0,
            "workload_id": "wan-t2v",
            "variant_id": "1.3b-sp2",
            "benchmark_version": 2,
            "recipe_fingerprint": "recipe-1",
            "hardware_profile_id": "hw-l40s-2",
            "software_profile_id": "sw-cuda",
        },
    )
    with open(results_dir / "perf_current.json", "w", encoding="utf-8") as f:
        json.dump(_v2_raw_result(recipe_fingerprint="recipe-2"), f)

    monkeypatch.setattr(compare_baseline, "TRACKING_ROOT", str(tracking_root))
    monkeypatch.setattr(compare_baseline, "RESULTS_DIR", str(results_dir))
    monkeypatch.setattr(compare_baseline, "PERF_REPORTS_DIR", str(reports_dir))
    monkeypatch.setattr(compare_baseline, "UPLOAD_POLICY", "never")
    monkeypatch.setattr(compare_baseline, "sync_from_hf", lambda local_dir, strict=False: local_dir)
    monkeypatch.delenv("PERF_PYTEST_RC", raising=False)

    assert compare_baseline.main() == 0

    artifacts = list((reports_dir / "results").glob("normalized_perf_*.json"))
    assert len(artifacts) == 1
    with open(artifacts[0], encoding="utf-8") as f:
        normalized = json.load(f)

    assert normalized["comparison_status"] == compare_baseline.STATUS_CALIBRATION_NEEDED


def test_v2_records_do_not_compare_against_v1_records_by_default(tmp_path):
    model_id = "wan-t2v-1.3b-2gpu"
    model_dir = tmp_path / sanitize(model_id)
    model_dir.mkdir()
    legacy_record = {
        "model_id": model_id,
        "gpu_type": "NVIDIA L40S",
        "timestamp": "2026-06-15T00:00:00+00:00",
        "success": True,
        "baseline_eligible": True,
        "latency": 10.0,
    }
    with open(model_dir / "legacy.json", "w", encoding="utf-8") as f:
        json.dump(legacy_record, f)

    record = _v2_record()
    identity_filters = compare_baseline._comparison_identity_filters(record)

    assert load_records_for_identity(
        str(tmp_path),
        identity_filters,
        baseline_eligible_only=True,
    ) == []


def test_v2_identity_lookup_ignores_model_id_and_gpu_display_string(tmp_path):
    model_id = "wan-t2v-1.3b-2gpu"
    model_dir = tmp_path / sanitize(model_id)
    model_dir.mkdir()
    baseline = {
        "model_id": "renamed-benchmark-id",
        "gpu_type": "NVIDIA L40S PCIe",
        "timestamp": "2026-06-15T00:00:00+00:00",
        "success": True,
        "baseline_eligible": True,
        "latency": 10.0,
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "recipe_fingerprint": "recipe-1",
        "hardware_profile_id": "hw-l40s-2",
        "software_profile_id": "sw-cuda",
    }
    with open(model_dir / "baseline.json", "w", encoding="utf-8") as f:
        json.dump(baseline, f)

    identity_filters = compare_baseline._comparison_identity_filters(_v2_record())

    records = load_records_for_identity(
        str(tmp_path),
        identity_filters,
        baseline_eligible_only=True,
    )

    assert records == [baseline]


def test_main_v2_identity_lookup_ignores_model_id_and_gpu_display_string(
    monkeypatch,
    tmp_path,
):
    tracking_root = tmp_path / "tracking"
    results_dir = tmp_path / "results"
    reports_dir = tmp_path / "reports"
    results_dir.mkdir()
    _write_record(
        tracking_root,
        "renamed-benchmark-id",
        "baseline.json",
        {
            "model_id": "renamed-benchmark-id",
            "gpu_type": "NVIDIA L40S PCIe",
            "timestamp": "2026-06-15T00:00:00+00:00",
            "success": True,
            "baseline_eligible": True,
            "latency": 10.0,
            "workload_id": "wan-t2v",
            "variant_id": "1.3b-sp2",
            "benchmark_version": 2,
            "recipe_fingerprint": "recipe-1",
            "hardware_profile_id": "hw-l40s-2",
            "software_profile_id": "sw-cuda",
        },
    )
    with open(results_dir / "perf_current.json", "w", encoding="utf-8") as f:
        json.dump(_v2_raw_result(), f)

    monkeypatch.setenv("PERF_RUN_SOURCE", "pr")
    monkeypatch.setattr(compare_baseline, "TRACKING_ROOT", str(tracking_root))
    monkeypatch.setattr(compare_baseline, "RESULTS_DIR", str(results_dir))
    monkeypatch.setattr(compare_baseline, "PERF_REPORTS_DIR", str(reports_dir))
    monkeypatch.setattr(compare_baseline, "UPLOAD_POLICY", "never")
    monkeypatch.setattr(compare_baseline, "sync_from_hf", lambda local_dir, strict=False: local_dir)
    monkeypatch.delenv("PERF_PYTEST_RC", raising=False)

    assert compare_baseline.main() == 0

    artifacts = list((reports_dir / "results").glob("normalized_perf_*.json"))
    assert len(artifacts) == 1
    with open(artifacts[0], encoding="utf-8") as f:
        normalized = json.load(f)

    assert normalized["comparison_status"] == compare_baseline.STATUS_PASS
    assert normalized["baseline_status"] == "compared"


def test_scheduled_main_static_regression_does_not_contaminate_passing_record(
    monkeypatch,
    tmp_path,
):
    tracking_root = tmp_path / "tracking"
    results_dir = tmp_path / "results"
    reports_dir = tmp_path / "reports"
    results_dir.mkdir()

    for benchmark_id, variant_id in (
        ("regressed-benchmark", "regressed-variant"),
        ("passing-benchmark", "passing-variant"),
    ):
        _write_record(
            tracking_root,
            benchmark_id,
            "baseline.json",
            {
                "model_id": benchmark_id,
                "gpu_type": "NVIDIA L40S",
                "timestamp": "2026-06-15T00:00:00+00:00",
                "success": True,
                "baseline_eligible": True,
                "latency": 10.0,
                "throughput": 4.5,
                "memory": 10000.0,
                "workload_id": "wan-t2v",
                "variant_id": variant_id,
                "benchmark_version": 2,
                "recipe_fingerprint": "recipe-1",
                "hardware_profile_id": "hw-l40s-2",
                "software_profile_id": "sw-cuda",
            },
        )

    common_thresholds = {
        "max_generation_time_s": 15.0,
        "max_peak_memory_mb": 11000.0,
    }
    with open(results_dir / "perf_regressed.json", "w", encoding="utf-8") as f:
        json.dump(
            _v2_raw_result(
                benchmark_id="regressed-benchmark",
                variant_id="regressed-variant",
                avg_generation_time_s=20.0,
                thresholds=common_thresholds,
            ),
            f,
        )
    with open(results_dir / "perf_passing.json", "w", encoding="utf-8") as f:
        json.dump(
            _v2_raw_result(
                benchmark_id="passing-benchmark",
                variant_id="passing-variant",
                thresholds=common_thresholds,
            ),
            f,
        )

    monkeypatch.setenv("PERF_RUN_SOURCE", "scheduled_main")
    monkeypatch.setenv("PERF_PYTEST_RC", "1")
    monkeypatch.setattr(compare_baseline, "TRACKING_ROOT", str(tracking_root))
    monkeypatch.setattr(compare_baseline, "RESULTS_DIR", str(results_dir))
    monkeypatch.setattr(compare_baseline, "PERF_REPORTS_DIR", str(reports_dir))
    monkeypatch.setattr(compare_baseline, "UPLOAD_POLICY", "never")
    monkeypatch.setattr(compare_baseline, "sync_from_hf", lambda local_dir, strict=False: local_dir)

    assert compare_baseline.main() == 1

    normalized = {}
    for artifact in (reports_dir / "results").glob("normalized_perf_*.json"):
        with open(artifact, encoding="utf-8") as f:
            record = json.load(f)
        normalized[record["model_id"]] = record

    assert normalized["regressed-benchmark"]["comparison_status"] == compare_baseline.STATUS_REGRESSION
    assert "avg_generation_time_s exceeded fixed threshold" in normalized[
        "regressed-benchmark"
    ]["comparison_status_reason"]
    assert normalized["regressed-benchmark"]["success"] is False
    assert normalized["regressed-benchmark"]["baseline_eligible"] is False
    assert normalized["passing-benchmark"]["comparison_status"] == compare_baseline.STATUS_PASS
    assert normalized["passing-benchmark"]["success"] is True
    assert normalized["passing-benchmark"]["baseline_eligible"] is True


def test_unattributed_performance_pytest_failure_reports_infra_error(monkeypatch):
    monkeypatch.setenv("PERF_PYTEST_RC", "2")
    record = _v2_record()

    failures, status, reason = compare_baseline._evaluate_record_comparison(
        record,
        [{"latency": 10.0}],
        [],
        resolve_metric_policies(None),
        [],
        True,
    )

    assert status == compare_baseline.STATUS_INFRA_ERROR
    assert failures == [reason]
    assert "without an attributable static-threshold regression" in reason


def test_main_partial_v2_identity_takes_precedence_over_static_regression(monkeypatch, tmp_path):
    results_dir = tmp_path / "results"
    reports_dir = tmp_path / "reports"
    tracking_root = tmp_path / "tracking"
    results_dir.mkdir()
    raw = _v2_raw_result(
        result_schema_version=2,
        avg_generation_time_s=20.0,
        thresholds={"max_generation_time_s": 15.0},
    )
    del raw["software_profile_id"]
    with open(results_dir / "perf_current.json", "w", encoding="utf-8") as f:
        json.dump(raw, f)

    monkeypatch.setenv("PERF_RUN_SOURCE", "pr")
    monkeypatch.setattr(compare_baseline, "TRACKING_ROOT", str(tracking_root))
    monkeypatch.setattr(compare_baseline, "RESULTS_DIR", str(results_dir))
    monkeypatch.setattr(compare_baseline, "PERF_REPORTS_DIR", str(reports_dir))
    monkeypatch.setattr(compare_baseline, "UPLOAD_POLICY", "never")
    monkeypatch.setattr(compare_baseline, "sync_from_hf", lambda local_dir, strict=False: local_dir)
    monkeypatch.delenv("PERF_PYTEST_RC", raising=False)

    assert compare_baseline.main() == 1

    artifacts = list((reports_dir / "results").glob("normalized_perf_*.json"))
    assert len(artifacts) == 1
    with open(artifacts[0], encoding="utf-8") as f:
        normalized = json.load(f)

    assert normalized["comparison_status"] == compare_baseline.STATUS_INFRA_ERROR
    assert "software_profile_id" in normalized["comparison_status_reason"]
    assert "fixed threshold" not in normalized["comparison_status_reason"]
    assert normalized["baseline_eligible"] is False


def test_v2_identity_lookup_last_n_uses_timestamp_across_model_dirs(tmp_path):
    identity_fields = {
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "recipe_fingerprint": "recipe-1",
        "hardware_profile_id": "hw-l40s-2",
        "software_profile_id": "sw-cuda",
    }
    records = [
        ("aaa-renamed", "newer-6.json", "2026-06-06T00:00:00+00:00", 6.0),
        ("aaa-renamed", "newer-7.json", "2026-06-07T00:00:00+00:00", 7.0),
        ("zzz-original", "older-1.json", "2026-06-01T00:00:00+00:00", 1.0),
        ("zzz-original", "older-2.json", "2026-06-02T00:00:00+00:00", 2.0),
        ("zzz-original", "older-3.json", "2026-06-03T00:00:00+00:00", 3.0),
        ("zzz-original", "older-4.json", "2026-06-04T00:00:00+00:00", 4.0),
        ("zzz-original", "older-5.json", "2026-06-05T00:00:00+00:00", 5.0),
    ]
    for model_id, filename, timestamp, latency in records:
        _write_record(
            tmp_path,
            model_id,
            filename,
            {
                "model_id": model_id,
                "gpu_type": "NVIDIA L40S",
                "timestamp": timestamp,
                "success": True,
                "baseline_eligible": True,
                "latency": latency,
                **identity_fields,
            },
        )

    loaded = load_records_for_identity(
        str(tmp_path),
        compare_baseline._comparison_identity_filters(_v2_record()),
        last_n=5,
        baseline_eligible_only=True,
    )

    assert [record["latency"] for record in loaded] == [3.0, 4.0, 5.0, 6.0, 7.0]


def test_load_records_filters_same_variant_changed_recipe(tmp_path):
    model_id = "wan-t2v-1.3b-2gpu"
    model_dir = tmp_path / sanitize(model_id)
    model_dir.mkdir()
    baseline = {
        "model_id": model_id,
        "gpu_type": "NVIDIA L40S",
        "timestamp": "2026-06-15T00:00:00+00:00",
        "success": True,
        "baseline_eligible": True,
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "recipe_fingerprint": "recipe-1",
        "hardware_profile_id": "hw-l40s-2",
        "software_profile_id": "sw-cuda",
    }
    with open(model_dir / "baseline.json", "w", encoding="utf-8") as f:
        json.dump(baseline, f)

    exact_filters = compare_baseline._comparison_identity_filters(_v2_record(recipe_fingerprint="recipe-2"))
    recipe_cohort_filters = compare_baseline._recipe_cohort_filters(_v2_record(recipe_fingerprint="recipe-2"))

    assert load_records_for_identity(
        str(tmp_path),
        exact_filters,
        baseline_eligible_only=True,
    ) == []
    mismatch_records = load_records_for_identity(
        str(tmp_path),
        recipe_cohort_filters,
        baseline_eligible_only=True,
    )

    assert len(mismatch_records) == 1


def test_legacy_record_lookup_still_uses_model_and_gpu(tmp_path):
    model_id = "wan-t2v-1.3b-2gpu"
    model_dir = tmp_path / sanitize(model_id)
    model_dir.mkdir()
    baseline = {
        "model_id": model_id,
        "gpu_type": "NVIDIA L40S",
        "timestamp": "2026-06-15T00:00:00+00:00",
        "success": True,
        "baseline_eligible": True,
        "latency": 10.0,
    }
    with open(model_dir / "baseline.json", "w", encoding="utf-8") as f:
        json.dump(baseline, f)

    assert load_records_for_model(
        str(tmp_path),
        model_id,
        "NVIDIA L40S",
        baseline_eligible_only=True,
    ) == [baseline]


def test_non_pass_status_is_not_baseline_eligible():
    assert compare_baseline._is_baseline_eligible(
        "scheduled_main", True, compare_baseline.STATUS_CALIBRATION_NEEDED) is False
    assert compare_baseline._is_baseline_eligible("pr", True, compare_baseline.STATUS_PASS) is False
    assert compare_baseline._is_baseline_eligible("local", True, compare_baseline.STATUS_PASS) is False


def test_upload_policy_pass_allows_calibration_record(monkeypatch):
    monkeypatch.setattr(compare_baseline, "UPLOAD_POLICY", "pass")

    assert compare_baseline._upload_allowed({
        "success": True,
        "comparison_status": compare_baseline.STATUS_CALIBRATION_NEEDED,
    }) is True
