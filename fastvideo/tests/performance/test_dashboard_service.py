# SPDX-License-Identifier: Apache-2.0
from fastvideo.performance_dashboard.service import build_latest_summary, build_trends, filter_records
from fastvideo.tests.performance import hf_store


def _record(ts, commit, latency, throughput, success=True):
    return {
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


def test_build_latest_summary_uses_previous_successful_records_for_baseline():
    records = [
        _record("2026-01-01T00:00:00+00:00", "a" * 40, 10.0, 10.0),
        _record("2026-01-02T00:00:00+00:00", "b" * 40, 12.0, 8.0, success=False),
        _record("2026-01-03T00:00:00+00:00", "c" * 40, 11.0, 9.0),
    ]

    rows = build_latest_summary(records, max_regression=0.05)

    assert len(rows) == 1
    row = rows[0]
    assert row["baseline_n"] == 1
    assert row["metrics"]["latency"]["baseline"] == 10.0
    assert row["metrics"]["latency"]["regression_pct"] == 10.0
    assert row["metrics"]["throughput"]["regression_pct"] == 10.0
    assert row["status"] == "pass"
    assert row["computed_regression_status"] == "fail"


def test_build_latest_summary_status_uses_latest_record_success_field():
    records = [
        _record("2026-01-01T00:00:00+00:00", "a" * 40, 10.0, 10.0),
        _record("2026-01-02T00:00:00+00:00", "b" * 40, 10.0, 10.0, success=False),
    ]

    rows = build_latest_summary(records)

    assert rows[0]["status"] == "fail"
    assert rows[0]["success"] is False


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


def test_hf_token_resolution_accepts_standard_env_names(monkeypatch):
    for env_var in hf_store.HF_TOKEN_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setenv("HF_TOKEN", "hf_local")

    assert hf_store.resolve_hf_token() == "hf_local"
