# SPDX-License-Identifier: Apache-2.0
from fastvideo.performance import hf_store
from fastvideo.performance_dashboard.service import build_latest_summary, build_trends, filter_records


def _record(ts, commit, latency, throughput, success=True, **metadata):
    record = {
        "model_id": "wan-t2v-1.3b-2gpu",
        "gpu_type": "NVIDIA L40S",
        "timestamp": ts,
        "commit_sha": commit,
        "latency": latency,
        "throughput": throughput,
        "memory": 10000.0,
        "text_encoder_time_s": None,
        "dit_time_s": 8.0,
        "vae_decode_time_s": 3.0,
        "success": success,
    }
    record.update(metadata)
    return record


def test_build_latest_summary_uses_previous_successful_records_for_baseline():
    records = [
        _record("2026-01-01T00:00:00+00:00", "a" * 40, 10.0, 10.0),
        _record("2026-01-02T00:00:00+00:00", "b" * 40, 12.0, 8.0, success=False),
        _record("2026-01-03T00:00:00+00:00", "c" * 40, 11.0, 9.0),
    ]

    rows = build_latest_summary(records)

    assert len(rows) == 1
    row = rows[0]
    assert row["baseline_n"] == 1
    assert row["metrics"]["latency"]["baseline"] == 10.0
    assert row["metrics"]["latency"]["regression_pct"] == 10.0
    assert row["metrics"]["latency"]["absolute_delta"] == 1.0
    assert row["metrics"]["latency"]["threshold_percent"] == 8.0
    assert row["metrics"]["latency"]["threshold_absolute"] == 0.5
    assert row["metrics"]["latency"]["threshold_exceeded"] is True
    assert row["metrics"]["latency"]["regressed"] is True
    assert row["metrics"]["throughput"]["regression_pct"] == 10.0
    assert row["status"] == "pass"
    assert row["computed_regression_status"] == "fail"
    assert row["threshold_exceeded_metrics"] == ["latency", "throughput"]
    assert row["failing_metrics"] == ["latency", "throughput"]


def test_build_latest_summary_status_uses_latest_record_success_field():
    records = [
        _record("2026-01-01T00:00:00+00:00", "a" * 40, 10.0, 10.0),
        _record("2026-01-02T00:00:00+00:00", "b" * 40, 10.0, 10.0, success=False),
    ]

    rows = build_latest_summary(records)

    assert rows[0]["status"] == "fail"
    assert rows[0]["success"] is False


def test_build_latest_summary_run_source_filter_keeps_canonical_baseline():
    records = [
        _record(
            "2026-01-01T00:00:00+00:00",
            "a" * 40,
            10.0,
            10.0,
            run_source="scheduled_main",
            baseline_eligible=True,
        ),
        _record(
            "2026-01-02T00:00:00+00:00",
            "b" * 40,
            11.0,
            9.0,
            run_source="pr",
            baseline_eligible=False,
            pr_number="123",
        ),
    ]

    rows = build_latest_summary(records, run_source="pr")

    assert len(rows) == 1
    assert rows[0]["run_source"] == "pr"
    assert rows[0]["pr_number"] == "123"
    assert rows[0]["baseline_n"] == 1
    assert rows[0]["metrics"]["latency"]["baseline"] == 10.0
    assert rows[0]["computed_regression_status"] == "fail"


def test_build_latest_summary_requires_absolute_floor_for_computed_regression():
    records = [
        _record("2026-01-01T00:00:00+00:00", "a" * 40, 10.0, 10.0),
        _record(
            "2026-01-02T00:00:00+00:00",
            "b" * 40,
            10.6,
            10.0,
            regression_thresholds={
                "latency": {
                    "threshold_percent": 0.05,
                    "threshold_absolute": 0.75,
                    "gated": True,
                }
            },
        ),
    ]

    rows = build_latest_summary(records)

    assert round(rows[0]["metrics"]["latency"]["regression_pct"], 1) == 6.0
    assert round(rows[0]["metrics"]["latency"]["absolute_delta"], 3) == 0.6
    assert rows[0]["metrics"]["latency"]["threshold_exceeded"] is False
    assert rows[0]["metrics"]["latency"]["regressed"] is False
    assert rows[0]["computed_regression_status"] == "pass"


def test_build_latest_summary_separates_informational_threshold_crossing():
    records = [
        _record("2026-01-01T00:00:00+00:00", "a" * 40, 10.0, 10.0),
        _record(
            "2026-01-02T00:00:00+00:00",
            "b" * 40,
            10.6,
            10.0,
            regression_thresholds={
                "latency": {
                    "threshold_percent": 0.05,
                    "threshold_absolute": 0.5,
                    "gated": False,
                }
            },
        ),
    ]

    rows = build_latest_summary(records)

    assert rows[0]["metrics"]["latency"]["threshold_exceeded"] is True
    assert rows[0]["metrics"]["latency"]["regressed"] is False
    assert rows[0]["threshold_exceeded_metrics"] == ["latency"]
    assert rows[0]["failing_metrics"] == []
    assert rows[0]["computed_regression_status"] == "pass"
def test_build_latest_summary_keeps_identity_cohorts_separate():
    records = [
        _record(
            "2026-01-01T00:00:00+00:00",
            "a" * 40,
            10.0,
            10.0,
            recipe_fingerprint="recipe-a",
            workload_id="wan-t2v",
            variant_id="1.3b-sp2",
            benchmark_version=2,
            hardware_profile_id="hw-l40s",
            software_profile_id="sw-cu130",
            run_source="scheduled_main",
            baseline_eligible=True,
        ),
        _record(
            "2026-01-02T00:00:00+00:00",
            "b" * 40,
            20.0,
            5.0,
            recipe_fingerprint="recipe-b",
            workload_id="wan-t2v",
            variant_id="1.3b-sp2",
            benchmark_version=2,
            hardware_profile_id="hw-l40s",
            software_profile_id="sw-cu130",
            run_source="scheduled_main",
            baseline_eligible=True,
        ),
        _record(
            "2026-01-03T00:00:00+00:00",
            "c" * 40,
            22.0,
            4.5,
            recipe_fingerprint="recipe-b",
            workload_id="wan-t2v",
            variant_id="1.3b-sp2",
            benchmark_version=2,
            hardware_profile_id="hw-l40s",
            software_profile_id="sw-cu130",
        ),
    ]

    rows = build_latest_summary(records, max_regression=0.05)
    recipe_b_row = next(row for row in rows if row["recipe_fingerprint"] == "recipe-b")

    assert len(rows) == 2
    assert recipe_b_row["baseline_n"] == 1
    assert recipe_b_row["metrics"]["latency"]["baseline"] == 20.0
    assert recipe_b_row["workload_id"] == "wan-t2v"
    assert recipe_b_row["variant_id"] == "1.3b-sp2"
    assert recipe_b_row["benchmark_version"] == 2


def test_build_latest_summary_keeps_variant_versions_separate():
    records = [
        _record(
            "2026-01-01T00:00:00+00:00",
            "a" * 40,
            10.0,
            10.0,
            workload_id="wan-t2v",
            variant_id="1.3b-sp2",
            benchmark_version=1,
            recipe_fingerprint="recipe-a",
            hardware_profile_id="hw-l40s",
            software_profile_id="sw-cu130",
            run_source="scheduled_main",
            baseline_eligible=True,
        ),
        _record(
            "2026-01-02T00:00:00+00:00",
            "b" * 40,
            20.0,
            5.0,
            workload_id="wan-t2v",
            variant_id="1.3b-sp2",
            benchmark_version=2,
            recipe_fingerprint="recipe-a",
            hardware_profile_id="hw-l40s",
            software_profile_id="sw-cu130",
            run_source="scheduled_main",
            baseline_eligible=True,
        ),
        _record(
            "2026-01-03T00:00:00+00:00",
            "c" * 40,
            22.0,
            4.5,
            workload_id="wan-t2v",
            variant_id="1.3b-sp2",
            benchmark_version=2,
            recipe_fingerprint="recipe-a",
            hardware_profile_id="hw-l40s",
            software_profile_id="sw-cu130",
        ),
    ]

    rows = build_latest_summary(records, max_regression=0.05)
    version_2_row = next(row for row in rows if row["benchmark_version"] == 2)

    assert len(rows) == 2
    assert version_2_row["baseline_n"] == 1
    assert version_2_row["metrics"]["latency"]["baseline"] == 20.0


def test_filter_records_and_trends_preserve_metric_points():
    records = [
        _record("2026-01-01T00:00:00+00:00", "a" * 40, 10.0, 10.0),
        _record("2026-01-02T00:00:00+00:00", "b" * 40, 12.0, 8.0, success=False),
    ]

    failed = filter_records(records, success=False)
    trends = build_trends(records)

    assert [record["commit_sha"] for record in failed] == ["b" * 40]
    assert len(trends) == 1
    assert trends[0]["points"][1]["metrics"]["latency"] == 12.0


def test_trends_include_source_metadata_with_legacy_defaults():
    records = [
        _record(
            "2026-01-01T00:00:00+00:00",
            "a" * 40,
            10.0,
            10.0,
            run_source="pr",
            baseline_eligible=False,
            pr_number="123",
            branch="feature/dashboard",
            build_url="https://buildkite.example/build",
        ),
        _record("2026-01-02T00:00:00+00:00", "b" * 40, 12.0, 8.0),
    ]

    filtered = filter_records(records, run_source="pr")
    trends = build_trends(records)

    assert len(filtered) == 1
    assert trends[0]["points"][0]["run_source"] == "pr"
    assert trends[0]["points"][0]["pr_number"] == "123"
    assert trends[0]["points"][0]["branch"] == "feature/dashboard"
    assert trends[0]["points"][0]["build_url"] == "https://buildkite.example/build"
    assert trends[0]["points"][1]["run_source"] == "unknown"
    assert trends[0]["points"][1]["baseline_eligible"] is True


def test_hf_token_resolution_accepts_standard_env_names(monkeypatch):
    for env_var in hf_store.HF_TOKEN_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setenv("HF_TOKEN", "hf_local")

    assert hf_store.resolve_hf_token() == "hf_local"


def test_load_records_can_filter_baseline_eligible_records(tmp_path):
    model_dir = tmp_path / "wan"
    model_dir.mkdir()
    (model_dir / "pr.json").write_text(
        '{"timestamp": "2026-01-01T00:00:00+00:00", "success": true, "baseline_eligible": false}',
        encoding="utf-8",
    )
    (model_dir / "main.json").write_text(
        '{"timestamp": "2026-01-02T00:00:00+00:00", "success": true, "baseline_eligible": true}',
        encoding="utf-8",
    )
    (model_dir / "legacy.json").write_text(
        '{"timestamp": "2026-01-03T00:00:00+00:00", "success": true}',
        encoding="utf-8",
    )

    records = hf_store.load_records(str(tmp_path), successful_only=True, baseline_eligible_only=True)

    assert len(records) == 2
    assert {record["timestamp"] for record in records} == {
        "2026-01-02T00:00:00+00:00",
        "2026-01-03T00:00:00+00:00",
    }


def test_load_records_for_model_filters_identity_cohort(tmp_path):
    model_dir = tmp_path / "wan"
    model_dir.mkdir()
    (model_dir / "matching.json").write_text(
        """
        {
          "model_id": "wan",
          "gpu_type": "NVIDIA L40S",
          "timestamp": "2026-01-01T00:00:00+00:00",
          "success": true,
          "baseline_eligible": true,
          "workload_id": "wan-t2v",
          "variant_id": "1.3b-sp2",
          "benchmark_version": 2,
          "recipe_fingerprint": "recipe-a",
          "hardware_profile_id": "hw-l40s",
          "software_profile_id": "sw-cu130"
        }
        """,
        encoding="utf-8",
    )
    (model_dir / "other_recipe.json").write_text(
        """
        {
          "model_id": "wan",
          "gpu_type": "NVIDIA L40S",
          "timestamp": "2026-01-02T00:00:00+00:00",
          "success": true,
          "baseline_eligible": true,
          "workload_id": "wan-t2v",
          "variant_id": "1.3b-sp2",
          "benchmark_version": 2,
          "recipe_fingerprint": "recipe-b",
          "hardware_profile_id": "hw-l40s",
          "software_profile_id": "sw-cu130"
        }
        """,
        encoding="utf-8",
    )
    (model_dir / "other_version.json").write_text(
        """
        {
          "model_id": "wan",
          "gpu_type": "NVIDIA L40S",
          "timestamp": "2026-01-03T00:00:00+00:00",
          "success": true,
          "baseline_eligible": true,
          "workload_id": "wan-t2v",
          "variant_id": "1.3b-sp2",
          "benchmark_version": 3,
          "recipe_fingerprint": "recipe-a",
          "hardware_profile_id": "hw-l40s",
          "software_profile_id": "sw-cu130"
        }
        """,
        encoding="utf-8",
    )

    records = hf_store.load_records_for_model(
        str(tmp_path),
        "wan",
        "NVIDIA L40S",
        workload_id="wan-t2v",
        variant_id="1.3b-sp2",
        benchmark_version="2",
        recipe_fingerprint="recipe-a",
        hardware_profile_id="hw-l40s",
        software_profile_id="sw-cu130",
        baseline_eligible_only=True,
    )

    assert len(records) == 1
    assert records[0]["recipe_fingerprint"] == "recipe-a"
