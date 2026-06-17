# SPDX-License-Identifier: Apache-2.0
"""Pure data transforms for the local performance dashboard.

The functions in this module operate on normalized records from
``fastvideo/tests/performance/compare_baseline.py``. They intentionally avoid
network and FastAPI concerns so they can be tested with in-memory fixtures.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from fastvideo.tests.performance.hf_store import safe_float

from .metrics import METRICS

Record = dict[str, Any]


def parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        ts = value
    else:
        try:
            ts = datetime.fromisoformat(str(value))
        except ValueError:
            return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def record_sort_key(record: Record) -> tuple[datetime, str]:
    ts = parse_timestamp(record.get("timestamp"))
    return (ts or datetime.min.replace(tzinfo=timezone.utc), str(record.get("commit_sha") or ""))


def filter_records(
    records: list[Record],
    *,
    model_id: str | None = None,
    gpu_type: str | None = None,
    success: bool | None = None,
) -> list[Record]:
    filtered = records
    if model_id:
        filtered = [record for record in filtered if record.get("model_id") == model_id]
    if gpu_type:
        filtered = [record for record in filtered if record.get("gpu_type") == gpu_type]
    if success is not None:
        filtered = [record for record in filtered if bool(record.get("success", True)) == success]
    return sorted(filtered, key=record_sort_key)


def group_by_model_gpu(records: list[Record]) -> dict[tuple[str, str], list[Record]]:
    groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for record in records:
        model_id = str(record.get("model_id") or "unknown")
        gpu_type = str(record.get("gpu_type") or "unknown")
        groups[(model_id, gpu_type)].append(record)
    return {key: sorted(value, key=record_sort_key) for key, value in groups.items()}


def baseline_value(records: list[Record], metric_key: str) -> float | None:
    values = [safe_float(record.get(metric_key)) for record in records]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(statistics.median(values))


def regression_percent(metric_key: str, current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None or baseline <= 0:
        return None
    metric = next(metric for metric in METRICS if metric.key == metric_key)
    if metric.lower_is_better:
        return (current - baseline) / baseline * 100.0
    return (baseline - current) / baseline * 100.0


def build_latest_summary(records: list[Record],
                         *,
                         baseline_window: int = 5,
                         max_regression: float = 0.05) -> list[Record]:
    rows: list[Record] = []
    for (model_id, gpu_type), group in group_by_model_gpu(records).items():
        latest = group[-1]
        earlier_successes = [record for record in group[:-1] if record.get("success", True)]
        baseline_records = earlier_successes[-baseline_window:]

        metrics: dict[str, Record] = {}
        regressions: list[float] = []
        for metric in METRICS:
            current = safe_float(latest.get(metric.key))
            baseline = baseline_value(baseline_records, metric.key)
            regression = regression_percent(metric.key, current, baseline)
            metrics[metric.key] = {
                "current": current,
                "baseline": baseline,
                "regression_pct": regression,
                "label": metric.label,
                "lower_is_better": metric.lower_is_better,
                "precision": metric.precision,
            }
            if regression is not None:
                regressions.append(regression)

        worst_regression = max(regressions) if regressions else None
        success = bool(latest.get("success", True))
        status = "pass" if success else "fail"

        rows.append({
            "model_id":
            model_id,
            "gpu_type":
            gpu_type,
            "timestamp":
            latest.get("timestamp"),
            "commit_sha":
            latest.get("commit_sha"),
            "success":
            success,
            "baseline_n":
            len(baseline_records),
            "worst_regression_pct":
            worst_regression,
            "regression_threshold_pct":
            max_regression * 100.0,
            "computed_regression_status":
            "fail" if worst_regression is not None and worst_regression > max_regression * 100.0 else "pass",
            "status":
            status,
            "metrics":
            metrics,
        })

    return sorted(rows, key=lambda row: (row["status"] != "fail", row["model_id"], row["gpu_type"]))


def build_trends(records: list[Record]) -> list[Record]:
    trends: list[Record] = []
    for (model_id, gpu_type), group in group_by_model_gpu(records).items():
        points = []
        for record in group:
            point = {
                "timestamp": record.get("timestamp"),
                "commit_sha": record.get("commit_sha"),
                "success": bool(record.get("success", True)),
                "metrics": {
                    metric.key: safe_float(record.get(metric.key))
                    for metric in METRICS
                },
            }
            points.append(point)
        trends.append({
            "model_id": model_id,
            "gpu_type": gpu_type,
            "points": points,
        })
    return sorted(trends, key=lambda trend: (trend["model_id"], trend["gpu_type"]))
