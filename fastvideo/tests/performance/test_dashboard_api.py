# SPDX-License-Identifier: Apache-2.0
from fastapi.testclient import TestClient
from datetime import datetime, timedelta

from fastvideo.performance_dashboard.api import PerformanceDataStore, create_app


class FakeStore(PerformanceDataStore):
    def __init__(self, records):
        super().__init__(tracking_root="/tmp/fake-fastvideo-perf-dashboard")
        self._records = records
        self.last_sync_at = "2026-01-03T00:00:00+00:00"

    @property
    def repo_id(self):
        return "FastVideo/performance-tracking"

    def sync(self):
        return {
            "ok": True,
            "repo_id": self.repo_id,
            "tracking_root": self.tracking_root,
            "last_sync_at": self.last_sync_at,
            "last_sync_error": None,
        }

    def load_records(self, *, days=None, successful_only=False):
        records = list(self._records)
        if days is not None:
            latest_ts = max(datetime.fromisoformat(record["timestamp"]) for record in records) if records else None
            if latest_ts is not None:
                cutoff = latest_ts - timedelta(days=days)
                records = [record for record in records if datetime.fromisoformat(record["timestamp"]) >= cutoff]
        if successful_only:
            return [record for record in records if record.get("success", True)]
        return records


def _record(model_id, gpu_type, ts, commit, latency, throughput, success=True, **metadata):
    record = {
        "model_id": model_id,
        "gpu_type": gpu_type,
        "timestamp": ts,
        "commit_sha": commit,
        "latency": latency,
        "throughput": throughput,
        "memory": 10000.0,
        "text_encoder_time_s": 2.0,
        "dit_time_s": 8.0,
        "vae_decode_time_s": 3.0,
        "success": success,
    }
    record.update(metadata)
    return record


def test_summary_endpoint_returns_latest_group_status():
    app = create_app(FakeStore([
        _record(
            "wan",
            "NVIDIA L40S",
            "2026-01-01T00:00:00+00:00",
            "a" * 40,
            10.0,
            10.0,
            workload_id="wan-t2v",
            variant_id="1.3b-sp2",
            benchmark_version=2,
            recipe_fingerprint="recipe-a",
            hardware_profile_id="hw-l40s",
            software_profile_id="sw-cu130",
        ),
        _record(
            "wan",
            "NVIDIA L40S",
            "2026-01-02T00:00:00+00:00",
            "b" * 40,
            11.0,
            9.0,
            workload_id="wan-t2v",
            variant_id="1.3b-sp2",
            benchmark_version=2,
            recipe_fingerprint="recipe-a",
            hardware_profile_id="hw-l40s",
            software_profile_id="sw-cu130",
        ),
    ]))
    client = TestClient(app)

    response = client.get("/api/performance/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["status_counts"] == {"pass": 1, "fail": 0}
    assert body["rows"][0]["metrics"]["latency"]["baseline"] == 10.0
    assert body["rows"][0]["metrics"]["latency"]["threshold_exceeded"] is True
    assert body["rows"][0]["threshold_exceeded_metrics"] == ["latency", "throughput"]
    assert body["rows"][0]["computed_regression_status"] == "fail"
    assert body["rows"][0]["workload_id"] == "wan-t2v"
    assert body["rows"][0]["variant_id"] == "1.3b-sp2"
    assert body["rows"][0]["benchmark_version"] == 2
    assert body["rows"][0]["recipe_fingerprint"] == "recipe-a"


def test_summary_status_is_independent_of_days_window():
    app = create_app(FakeStore([
        _record("wan", "NVIDIA L40S", "2026-01-01T00:00:00+00:00", "a" * 40, 10.0, 10.0),
        _record("wan", "NVIDIA L40S", "2026-02-15T00:00:00+00:00", "b" * 40, 11.0, 9.0),
    ]))
    client = TestClient(app)

    narrow = client.get("/api/performance/summary", params={"days": 1}).json()
    wide = client.get("/api/performance/summary", params={"days": 365}).json()
    trends = client.get("/api/performance/trends", params={"days": 1}).json()

    assert narrow["rows"][0]["status"] == wide["rows"][0]["status"]
    assert narrow["rows"][0]["baseline_n"] == wide["rows"][0]["baseline_n"] == 1
    assert narrow["filters"]["days"] is None
    assert narrow["filters"]["trend_window_days"] == 1
    assert len(trends["groups"][0]["points"]) == 1


def test_dashboard_endpoints_filter_and_return_run_source_metadata():
    app = create_app(FakeStore([
        _record(
            "wan",
            "NVIDIA L40S",
            "2026-01-01T00:00:00+00:00",
            "a" * 40,
            10.0,
            10.0,
            run_source="pr",
            pr_number="123",
            branch="feature/perf",
            baseline_eligible=False,
        ),
        _record(
            "wan",
            "NVIDIA L40S",
            "2026-01-02T00:00:00+00:00",
            "b" * 40,
            11.0,
            9.0,
            run_source="scheduled_main",
            baseline_eligible=True,
        ),
    ]))
    client = TestClient(app)

    summary = client.get("/api/performance/summary", params={"run_source": "pr"}).json()
    trends = client.get("/api/performance/trends", params={"run_source": "pr"}).json()

    assert summary["count"] == 1
    assert summary["rows"][0]["run_source"] == "pr"
    assert summary["rows"][0]["pr_number"] == "123"
    assert summary["rows"][0]["baseline_n"] == 0
    assert summary["rows"][0]["metrics"]["latency"]["baseline"] is None
    assert summary["filters"]["run_source"] == "pr"
    assert trends["count"] == 1
    assert trends["groups"][0]["points"][0]["run_source"] == "pr"


def test_records_and_trends_endpoints_filter_by_model_and_gpu():
    app = create_app(FakeStore([
        _record("wan", "NVIDIA L40S", "2026-01-01T00:00:00+00:00", "a" * 40, 10.0, 10.0),
        _record("ltx", "NVIDIA A100", "2026-01-01T00:00:00+00:00", "b" * 40, 20.0, 5.0),
    ]))
    client = TestClient(app)

    records = client.get("/api/performance/records", params={"model_id": "wan"}).json()
    trends = client.get("/api/performance/trends", params={"gpu_type": "NVIDIA L40S"}).json()

    assert records["count"] == 1
    assert records["records"][0]["model_id"] == "wan"
    assert trends["count"] == 1
    assert trends["groups"][0]["gpu_type"] == "NVIDIA L40S"


def test_refresh_endpoint_reports_sync_metadata():
    app = create_app(FakeStore([]))
    client = TestClient(app)

    response = client.post("/api/performance/refresh")

    assert response.status_code == 200
    assert response.json()["ok"] is True
