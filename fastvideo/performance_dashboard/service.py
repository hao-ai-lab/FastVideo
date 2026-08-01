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

from fastvideo.performance.cohort import (
    COMPARISON_IDENTITY_KEYS,
    cohort_descriptor,
    cohort_key,
    cohort_value,
)
from fastvideo.performance.hf_store import is_baseline_eligible_record, safe_float
from fastvideo.performance.metric_policy import regression_delta, resolve_metric_policies

Record = dict[str, Any]
CohortKey = str
COMPARISON_COHORT_KEYS = COMPARISON_IDENTITY_KEYS


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
    run_source: str | None = None,
    success: bool | None = None,
) -> list[Record]:
    filtered = records
    if model_id:
        filtered = [record for record in filtered if record.get("model_id") == model_id]
    if gpu_type:
        filtered = [record for record in filtered if record.get("gpu_type") == gpu_type]
    if run_source:
        filtered = [record for record in filtered if record_run_source(record) == run_source]
    if success is not None:
        filtered = [record for record in filtered if bool(record.get("success", True)) == success]
    return sorted(filtered, key=record_sort_key)


def record_run_source(record: Record) -> str:
    value = str(record.get("run_source") or "unknown")
    return value if value in {"pr", "local", "scheduled_main", "unknown"} else "unknown"


def record_metadata(record: Record) -> Record:
    return {
        "run_source": record_run_source(record),
        "baseline_eligible": is_baseline_eligible_record(record),
        "branch": record.get("branch") or "",
        "pr_number": record.get("pr_number") or "",
        "test_scope": record.get("test_scope") or "",
        "build_url": record.get("build_url") or "",
        "build_id": record.get("build_id") or "",
        "job_id": record.get("job_id") or "",
    }


def _cohort_metadata_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, str) and not value.strip():
        return ""
    return value


def record_comparison_metadata(record: Record) -> Record:
    return {
        **{
            key: _cohort_metadata_value(record.get(key))
            for key in COMPARISON_COHORT_KEYS
        },
        "cohort": cohort_descriptor(record),
    }


def comparison_cohort_key(record: Record) -> CohortKey:
    return cohort_key(record)


def group_by_comparison_cohort(records: list[Record]) -> dict[CohortKey, list[Record]]:
    groups: dict[CohortKey, list[Record]] = defaultdict(list)
    for record in records:
        groups[comparison_cohort_key(record)].append(record)
    return {key: sorted(value, key=record_sort_key) for key, value in groups.items()}


def comparison_sort_key(record: Record) -> tuple[str, ...]:
    return (
        str(record.get("model_id") or "unknown"),
        str(record.get("gpu_type") or "unknown"),
        *(cohort_value(record.get(key)) for key in COMPARISON_COHORT_KEYS),
    )


def latest_row_sort_key(row: Record) -> tuple[Any, ...]:
    return (row["status"] != "fail", *comparison_sort_key(row))


def _latest_baseline_record(records: list[Record]) -> Record | None:
    eligible = [record for record in records if record.get("success", True) and is_baseline_eligible_record(record)]
    return eligible[-1] if eligible else None


def build_cohort_catalog(
    records: list[Record],
    *,
    model_id: str | None = None,
    gpu_key: str | None = None,
) -> Record:
    """Build cascading model/GPU/cohort options and a deterministic default."""
    groups = list(group_by_comparison_cohort(records).values())
    latest_records = [group[-1] for group in groups]
    models = sorted({str(record.get("model_id") or "unknown") for record in latest_records})

    model_groups = [
        group for group in groups if not model_id or str(group[-1].get("model_id") or "unknown") == model_id
    ]
    gpu_records: dict[str, Record] = {}
    for group in model_groups:
        latest = group[-1]
        descriptor = cohort_descriptor(latest)
        current = gpu_records.get(descriptor["gpu_key"])
        if current is None or record_sort_key(latest) > record_sort_key(current):
            gpu_records[descriptor["gpu_key"]] = latest
    gpus = sorted(
        ({
            "key": descriptor["gpu_key"],
            "label": descriptor["gpu_label"],
            "gpu_type": str(record.get("gpu_type") or "unknown"),
        } for record in gpu_records.values() for descriptor in (cohort_descriptor(record), )),
        key=lambda option: (option["label"], option["key"]),
    )

    selected_groups = [
        group for group in model_groups if not gpu_key or cohort_descriptor(group[-1])["gpu_key"] == gpu_key
    ]
    cohorts = []
    for group in selected_groups:
        latest = group[-1]
        baseline_record = _latest_baseline_record(group)
        cohorts.append({
            **cohort_descriptor(latest),
            "model_id":
            str(latest.get("model_id") or "unknown"),
            "gpu_type":
            str(latest.get("gpu_type") or "unknown"),
            "latest_timestamp":
            latest.get("timestamp"),
            "latest_baseline_timestamp":
            None if baseline_record is None else baseline_record.get("timestamp"),
            "baseline_eligible":
            baseline_record is not None,
        })
    cohorts.sort(key=lambda option: (option["title"], option["gpu_label"], option["key"]))

    baseline_groups = [group for group in selected_groups if _latest_baseline_record(group) is not None]
    if baseline_groups:
        default_group = max(baseline_groups, key=lambda group: record_sort_key(_latest_baseline_record(group) or {}))
    elif selected_groups:
        default_group = max(selected_groups, key=lambda group: record_sort_key(group[-1]))
    else:
        default_group = None

    return {
        "models": models,
        "gpus": gpus,
        "cohorts": cohorts,
        "default_cohort_key": None if default_group is None else cohort_key(default_group[-1]),
    }


def baseline_value(records: list[Record], metric_key: str) -> float | None:
    values = [safe_float(record.get(metric_key)) for record in records]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(statistics.median(values))


def build_latest_summary(records: list[Record],
                         *,
                         baseline_window: int = 5,
                         run_source: str | None = None) -> list[Record]:
    rows: list[Record] = []
    for group in group_by_comparison_cohort(records).values():
        latest_candidates = group
        if run_source:
            latest_candidates = [record for record in group if record_run_source(record) == run_source]
        if not latest_candidates:
            continue

        latest = latest_candidates[-1]
        model_id = str(latest.get("model_id") or "unknown")
        gpu_type = str(latest.get("gpu_type") or "unknown")
        latest_index = next(index for index, record in enumerate(group) if record is latest)
        baseline_pool = [
            record for record in group[:latest_index]
            if record.get("success", True) and is_baseline_eligible_record(record)
        ]
        baseline_records = baseline_pool[-baseline_window:]
        metric_policies = resolve_metric_policies(latest.get("regression_thresholds"))

        metrics: dict[str, Record] = {}
        regressions: list[float] = []
        failing_metrics: list[str] = []
        threshold_exceeded_metrics: list[str] = []
        for policy in metric_policies:
            current = safe_float(latest.get(policy.key))
            baseline = baseline_value(baseline_records, policy.key)
            delta = None
            if current is not None and baseline is not None:
                delta = regression_delta(policy, current, baseline)
            regression = None if delta is None else delta.percent * 100.0
            metrics[policy.key] = {
                "current": current,
                "baseline": baseline,
                "regression_pct": regression,
                "absolute_delta": None if delta is None else delta.absolute,
                "threshold_percent": policy.threshold_percent * 100.0,
                "threshold_absolute": policy.threshold_absolute,
                "gated": policy.gated,
                "threshold_exceeded": False if delta is None else delta.threshold_exceeded,
                "regressed": False if delta is None else delta.regressed,
                "label": policy.label,
                "lower_is_better": policy.lower_is_better,
                "precision": policy.precision,
            }
            if regression is not None:
                regressions.append(regression)
            if delta is not None and delta.threshold_exceeded:
                threshold_exceeded_metrics.append(policy.key)
            if delta is not None and delta.regressed:
                failing_metrics.append(policy.key)

        worst_regression = max(regressions) if regressions else None
        success = bool(latest.get("success", True))
        status = "pass" if success else "fail"

        rows.append({
            "model_id": model_id,
            "gpu_type": gpu_type,
            "timestamp": latest.get("timestamp"),
            "commit_sha": latest.get("commit_sha"),
            **record_metadata(latest),
            **record_comparison_metadata(latest),
            "success": success,
            "baseline_n": len(baseline_records),
            "worst_regression_pct": worst_regression,
            "threshold_exceeded_metrics": threshold_exceeded_metrics,
            "failing_metrics": failing_metrics,
            "computed_regression_status": "fail" if failing_metrics else "pass",
            "status": status,
            "metrics": metrics,
        })

    return sorted(rows, key=latest_row_sort_key)


def build_trends(records: list[Record]) -> list[Record]:
    trends: list[Record] = []
    for group in group_by_comparison_cohort(records).values():
        latest = group[-1]
        model_id = str(latest.get("model_id") or "unknown")
        gpu_type = str(latest.get("gpu_type") or "unknown")
        points = []
        for record in group:
            metric_policies = resolve_metric_policies(record.get("regression_thresholds"))
            point = {
                "timestamp": record.get("timestamp"),
                "commit_sha": record.get("commit_sha"),
                **record_metadata(record),
                **record_comparison_metadata(record),
                "success": bool(record.get("success", True)),
                "metrics": {
                    policy.key: safe_float(record.get(policy.key))
                    for policy in metric_policies
                },
            }
            points.append(point)
        trends.append({
            "model_id": model_id,
            "gpu_type": gpu_type,
            **record_comparison_metadata(latest),
            "points": points,
        })
    return sorted(trends, key=comparison_sort_key)
