# SPDX-License-Identifier: Apache-2.0
from fastvideo.performance import hf_store
from fastvideo.performance_dashboard.service import (
    ADVANCED_FILTER_LEGACY,
    build_cohort_catalog,
    build_latest_summary,
    build_trends,
    filter_records,
)


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


def test_build_latest_summary_run_source_filter_excludes_future_baselines():
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
        _record(
            "2026-01-03T00:00:00+00:00",
            "c" * 40,
            30.0,
            3.0,
            run_source="scheduled_main",
            baseline_eligible=True,
        ),
    ]

    rows = build_latest_summary(records, run_source="pr")

    assert len(rows) == 1
    assert rows[0]["run_source"] == "pr"
    assert rows[0]["baseline_n"] == 1
    assert rows[0]["metrics"]["latency"]["baseline"] == 10.0
    assert rows[0]["metrics"]["throughput"]["baseline"] == 10.0



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

    rows = build_latest_summary(records)
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

    rows = build_latest_summary(records)
    version_2_row = next(row for row in rows if row["benchmark_version"] == 2)

    assert len(rows) == 2
    assert version_2_row["baseline_n"] == 1
    assert version_2_row["metrics"]["latency"]["baseline"] == 20.0


def test_v2_dashboard_cohort_spans_display_name_changes():
    identity = {
        "result_schema_version": 2,
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "recipe_fingerprint": "recipe-a",
        "hardware_profile_id": "hw-l40s",
        "software_profile_id": "sw-cu130",
    }
    records = [
        _record(
            "2026-01-01T00:00:00+00:00",
            "a" * 40,
            10.0,
            10.0,
            model_id="old-display-name",
            gpu_type="NVIDIA L40S old label",
            run_source="scheduled_main",
            baseline_eligible=True,
            **identity,
        ),
        _record(
            "2026-01-02T00:00:00+00:00",
            "b" * 40,
            11.0,
            9.0,
            model_id="new-display-name",
            gpu_type="NVIDIA L40S",
            **identity,
        ),
    ]

    rows = build_latest_summary(records)
    trends = build_trends(records)

    assert len(rows) == 1
    assert rows[0]["model_id"] == "new-display-name"
    assert rows[0]["gpu_type"] == "NVIDIA L40S"
    assert rows[0]["baseline_n"] == 1
    assert rows[0]["metrics"]["latency"]["baseline"] == 10.0
    assert len(trends) == 1
    assert trends[0]["model_id"] == "new-display-name"
    assert len(trends[0]["points"]) == 2


def test_legacy_dashboard_cohorts_still_use_model_and_gpu():
    records = [
        _record("2026-01-01T00:00:00+00:00", "a" * 40, 10.0, 10.0, model_id="wan"),
        _record("2026-01-02T00:00:00+00:00", "b" * 40, 20.0, 5.0, model_id="ltx"),
    ]

    rows = build_latest_summary(records)
    trends = build_trends(records)

    assert {row["model_id"] for row in rows} == {"wan", "ltx"}
    assert len(trends) == 2


def test_partial_v2_identity_does_not_cross_model_or_gpu():
    partial_identity = {
        "result_schema_version": 2,
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
    }
    records = [
        _record(
            "2026-01-01T00:00:00+00:00",
            "a" * 40,
            10.0,
            10.0,
            model_id="wan",
            gpu_type="NVIDIA L40S",
            **partial_identity,
        ),
        _record(
            "2026-01-02T00:00:00+00:00",
            "b" * 40,
            20.0,
            5.0,
            model_id="ltx",
            gpu_type="NVIDIA H100",
            **partial_identity,
        ),
    ]

    rows = build_latest_summary(records)
    trends = build_trends(records)

    assert {(row["model_id"], row["gpu_type"]) for row in rows} == {
        ("wan", "NVIDIA L40S"),
        ("ltx", "NVIDIA H100"),
    }
    assert len(trends) == 2


def test_dashboard_identity_preserves_zero_benchmark_version():
    records = [
        _record(
            "2026-01-01T00:00:00+00:00",
            "a" * 40,
            10.0,
            10.0,
            workload_id="wan-t2v",
            variant_id="1.3b-sp2",
            benchmark_version=0,
            recipe_fingerprint="recipe-a",
            hardware_profile_id="hw-l40s",
            software_profile_id="sw-cu130",
            run_source="scheduled_main",
            baseline_eligible=True,
        ),
        _record(
            "2026-01-02T00:00:00+00:00",
            "b" * 40,
            11.0,
            9.0,
            workload_id="wan-t2v",
            variant_id="1.3b-sp2",
            benchmark_version=0,
            recipe_fingerprint="recipe-a",
            hardware_profile_id="hw-l40s",
            software_profile_id="sw-cu130",
        ),
        _record(
            "2026-01-03T00:00:00+00:00",
            "c" * 40,
            20.0,
            5.0,
        ),
    ]

    rows = build_latest_summary(records)
    version_zero_row = next(row for row in rows if row["benchmark_version"] == 0)
    legacy_row = next(row for row in rows if row["benchmark_version"] == "")
    trends = build_trends(records)
    version_zero_trend = next(trend for trend in trends if trend["benchmark_version"] == 0)
    legacy_trend = next(trend for trend in trends if trend["benchmark_version"] == "")

    assert len(rows) == 2
    assert version_zero_row["baseline_n"] == 1
    assert version_zero_row["metrics"]["latency"]["baseline"] == 10.0
    assert legacy_row["baseline_n"] == 0
    assert len(trends) == 2
    assert version_zero_trend["points"][0]["benchmark_version"] == 0
    assert legacy_trend["points"][0]["benchmark_version"] == ""


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


def test_build_cohort_catalog_cascades_gpu_options_and_preserves_exact_cohorts():
    shared_identity = {
        "result_schema_version": 2,
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "software_profile_id": "sw-cu126",
        "software_profile": {"cuda": "12.6", "pytorch": "2.7"},
    }
    records = [
        _record(
            "2026-01-01T00:00:00+00:00",
            "a" * 40,
            10.0,
            10.0,
            recipe_fingerprint="recipe-a",
            hardware_profile_id="hw-l40s-2",
            hardware_profile={
                "gpu_count": 2,
                "gpus": [{"name": "NVIDIA L40S", "memory_gb": 48}] * 2,
                "interconnect": "full_nvlink",
            },
            run_source="scheduled_main",
            baseline_eligible=True,
            **shared_identity,
        ),
        _record(
            "2026-01-02T00:00:00+00:00",
            "b" * 40,
            11.0,
            9.0,
            recipe_fingerprint="recipe-b",
            hardware_profile_id="hw-l40s-2",
            hardware_profile={
                "gpu_count": 2,
                "gpus": [{"name": "NVIDIA L40S", "memory_gb": 48}] * 2,
                "interconnect": "full_nvlink",
            },
            **shared_identity,
        ),
        _record(
            "2026-01-03T00:00:00+00:00",
            "c" * 40,
            12.0,
            8.0,
            model_id="ltx-video",
            gpu_type="NVIDIA H100",
            workload_id="ltx-t2v",
            variant_id="2b-tp1",
            benchmark_version=2,
            recipe_fingerprint="recipe-c",
            hardware_profile_id="hw-h100-1",
            software_profile_id="sw-cu126",
            hardware_profile={
                "gpu_count": 1,
                "gpus": [{"name": "NVIDIA H100", "memory_gb": 80}],
                "interconnect": "single_gpu",
            },
            software_profile={"cuda": "12.6", "pytorch": "2.7"},
            result_schema_version=2,
        ),
    ]

    all_options = build_cohort_catalog(records)
    wan_options = build_cohort_catalog(records, model_id="wan-t2v-1.3b-2gpu")

    assert all_options["models"] == ["ltx-video", "wan-t2v-1.3b-2gpu"]
    assert len(all_options["gpus"]) == 2
    assert len(all_options["cohorts"]) == 3
    assert len(wan_options["gpus"]) == 1
    assert len(wan_options["cohorts"]) == 2
    assert wan_options["gpus"][0]["label"] == "2× NVIDIA L40S · 48 GB · full nvlink"


def test_build_cohort_catalog_prefers_most_recent_baseline_eligible_cohort():
    identity = {
        "result_schema_version": 2,
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "hardware_profile_id": "hw-l40s",
        "software_profile_id": "sw-cu126",
    }
    baseline = _record(
        "2026-01-01T00:00:00+00:00",
        "a" * 40,
        10.0,
        10.0,
        recipe_fingerprint="recipe-baseline",
        run_source="scheduled_main",
        baseline_eligible=True,
        **identity,
    )
    newer_pr = _record(
        "2026-01-03T00:00:00+00:00",
        "b" * 40,
        11.0,
        9.0,
        recipe_fingerprint="recipe-pr",
        run_source="pr",
        baseline_eligible=False,
        **identity,
    )

    catalog = build_cohort_catalog([baseline, newer_pr])
    baseline_option = next(option for option in catalog["cohorts"] if option["baseline_eligible"])

    assert catalog["default_cohort_key"] == baseline_option["key"]


def test_build_cohort_catalog_falls_back_to_most_recent_record():
    identity = {
        "result_schema_version": 2,
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "hardware_profile_id": "hw-l40s",
        "software_profile_id": "sw-cu126",
        "run_source": "pr",
        "baseline_eligible": False,
    }
    older = _record(
        "2026-01-01T00:00:00+00:00",
        "a" * 40,
        10.0,
        10.0,
        recipe_fingerprint="recipe-older",
        **identity,
    )
    newer = _record(
        "2026-01-02T00:00:00+00:00",
        "b" * 40,
        11.0,
        9.0,
        recipe_fingerprint="recipe-newer",
        **identity,
    )

    catalog = build_cohort_catalog([older, newer])
    newer_option = next(
        option for option in catalog["cohorts"]
        if option["raw_ids"]["recipe_fingerprint"] == "recipe-newer")

    assert catalog["default_cohort_key"] == newer_option["key"]


def test_build_cohort_catalog_returns_cascading_advanced_filter_options():
    common = {
        "result_schema_version": 2,
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "model_id": "wan",
        "gpu_type": "NVIDIA L40S",
    }
    profile_a = _record(
        "2026-01-01T00:00:00+00:00",
        "a" * 40,
        10.0,
        10.0,
        hardware_profile_id="hw-a",
        software_profile_id="sw-a",
        recipe_fingerprint="recipe-a",
        run_source="pr",
        **common,
    )
    profile_b = _record(
        "2026-01-02T00:00:00+00:00",
        "b" * 40,
        11.0,
        9.0,
        hardware_profile_id="hw-b",
        software_profile_id="sw-b",
        recipe_fingerprint="recipe-b",
        run_source="scheduled_main",
        **common,
    )
    legacy = _record(
        "2026-01-03T00:00:00+00:00",
        "c" * 40,
        12.0,
        8.0,
        model_id="legacy-model",
        gpu_type="NVIDIA A100",
    )

    catalog = build_cohort_catalog([profile_a, profile_b, legacy], model_id="wan")
    software_filtered = build_cohort_catalog(
        [profile_a, profile_b, legacy],
        model_id="wan",
        software_profile_id="sw-a",
    )
    source_filtered = build_cohort_catalog([profile_a, profile_b, legacy], model_id="wan", run_source="pr")
    legacy_catalog = build_cohort_catalog([profile_a, profile_b, legacy], model_id="legacy-model")

    assert {option["value"] for option in catalog["advanced_filters"]["hardware_profiles"]} == {
        "hw-a",
        "hw-b",
    }
    assert [option["value"] for option in software_filtered["advanced_filters"]["hardware_profiles"]
            ] == ["hw-a"]
    assert [cohort["raw_ids"]["software_profile_id"] for cohort in software_filtered["cohorts"]] == ["sw-a"]
    assert [option["value"] for option in source_filtered["advanced_filters"]["recipes"]] == ["recipe-a"]
    assert legacy_catalog["advanced_filters"]["hardware_profiles"][0]["value"] == ADVANCED_FILTER_LEGACY
