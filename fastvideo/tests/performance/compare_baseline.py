# SPDX-License-Identifier: Apache-2.0
"""Track performance results and compare against historical baseline.

This script:
1) reads current benchmark results from fastvideo/tests/performance/results,
2) writes normalized tracking records to the Modal volume path,
3) compares each current record against the mean of up to 5 prior records,
4) exits non-zero if any metric regresses by more than 15%.
"""

import glob
import json
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from typing import Any


RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
)
TRACKING_ROOT = os.environ.get(
    "PERFORMANCE_TRACKING_ROOT",
    "/root/data/performance-tracking",
)
MAX_REGRESSION = float(os.environ.get("PERF_MAX_REGRESSION", "0.15"))


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_current_results() -> list[dict[str, Any]]:
    pattern = os.path.join(RESULTS_DIR, "perf_*.json")
    records: list[dict[str, Any]] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def _normalize_record(result: dict[str, Any]) -> dict[str, Any]:
    benchmark_id = result.get("benchmark_id", "unknown")
    model_id = benchmark_id

    timestamp = result.get("timestamp")
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()

    commit_sha = result.get("commit") or os.environ.get("BUILDKITE_COMMIT", "")
    latency = _safe_float(result.get("avg_generation_time_s"))
    throughput = _safe_float(result.get("throughput_fps"))
    memory = _safe_float(result.get("max_peak_memory_mb"))

    return {
        "model_id": model_id,
        "timestamp": timestamp,
        "commit_sha": commit_sha,
        "gpu_type": result.get("device", "unknown"),
        "latency": latency,
        "throughput": throughput,
        "memory": memory,
    }


def _write_tracking_record(record: dict[str, Any]) -> str:
    model_dir = os.path.join(TRACKING_ROOT, _sanitize(record["model_id"]))
    os.makedirs(model_dir, exist_ok=True)

    timestamp = _sanitize(record["timestamp"])
    commit = _sanitize(record["commit_sha"] or "unknown")
    out_path = os.path.join(model_dir, f"{timestamp}_{commit}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return out_path


def _load_baseline_records(model_id: str, current_path: str) -> list[dict[str, Any]]:
    model_dir = os.path.join(TRACKING_ROOT, _sanitize(model_id))
    if not os.path.exists(model_dir):
        return []

    all_paths = sorted(glob.glob(os.path.join(model_dir, "*.json")))
    prior_paths = [p for p in all_paths if os.path.abspath(p) != os.path.abspath(current_path)]

    baseline: list[dict[str, Any]] = []
    for path in prior_paths:
        try:
            with open(path, encoding="utf-8") as f:
                baseline.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return baseline


def _filter_by_gpu_type(
    records: list[dict[str, Any]],
    gpu_type: str,
) -> list[dict[str, Any]]:
    return [r for r in records if r.get("gpu_type") == gpu_type]


def _mean_metric(records: list[dict[str, Any]], key: str) -> float | None:
    values = [
        _safe_float(r.get(key))
        for r in records
    ]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return statistics.mean(values)


def _check_regressions(
    current: dict[str, Any],
    baseline_records: list[dict[str, Any]],
    max_regression: float,
) -> list[str]:
    failures: list[str] = []

    for metric in ("latency", "memory"):
        baseline = _mean_metric(baseline_records, metric)
        curr = _safe_float(current.get(metric))
        if baseline is None or curr is None or baseline <= 0:
            continue
        regression = (curr - baseline) / baseline
        if regression > max_regression:
            failures.append(
                f"{current['model_id']} {metric} regressed by {regression * 100:.1f}% "
                f"(current={curr:.3f}, baseline_mean={baseline:.3f})"
            )

    baseline_tp = _mean_metric(baseline_records, "throughput")
    curr_tp = _safe_float(current.get("throughput"))
    if baseline_tp is not None and curr_tp is not None and baseline_tp > 0:
        regression = (baseline_tp - curr_tp) / baseline_tp
        if regression > max_regression:
            failures.append(
                f"{current['model_id']} throughput regressed by {regression * 100:.1f}% "
                f"(current={curr_tp:.3f}, baseline_mean={baseline_tp:.3f})"
            )

    return failures


def _metric_delta_percent(
    metric: str,
    current: dict[str, Any],
    baseline_records: list[dict[str, Any]],
) -> float | None:
    curr = _safe_float(current.get(metric))
    baseline = _mean_metric(baseline_records, metric)
    if curr is None or baseline is None or baseline <= 0:
        return None

    if metric in ("latency", "memory"):
        return (curr - baseline) / baseline * 100.0
    if metric == "throughput":
        return (baseline - curr) / baseline * 100.0
    return None


def _compact_value(value: float | None, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def _build_markdown_summary(
    summary_rows: list[dict[str, Any]],
    max_regression: float,
) -> str:
    lines = [
        "## Performance Baseline Comparison",
        "",
        f"Threshold: regressions greater than {max_regression * 100:.1f}% fail",
        "",
        "| Model | GPU | Baseline N | Latency (curr/base) | Throughput (curr/base) | Memory (curr/base) | Worst Regression | Status |",
        "|---|---|---:|---|---|---|---:|---|",
    ]

    for row in summary_rows:
        latency = (
            f"{_compact_value(row['latency_curr'])}/{_compact_value(row['latency_base'])}"
            if row["latency_base"] is not None else "n/a"
        )
        throughput = (
            f"{_compact_value(row['throughput_curr'])}/{_compact_value(row['throughput_base'])}"
            if row["throughput_base"] is not None else "n/a"
        )
        memory = (
            f"{_compact_value(row['memory_curr'], 1)}/{_compact_value(row['memory_base'], 1)}"
            if row["memory_base"] is not None else "n/a"
        )

        worst_reg = "n/a" if row["worst_regression_pct"] is None else f"{row['worst_regression_pct']:.1f}%"
        status = "FAIL" if row["failed"] else "PASS"

        lines.append(
            f"| {row['model_id']} | {row['gpu_type']} | {row['baseline_n']} | "
            f"{latency} | {throughput} | {memory} | {worst_reg} | {status} |"
        )

    return "\n".join(lines) + "\n"


def _emit_markdown_summary(markdown: str) -> None:
    print("\n" + markdown)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(markdown)
            f.write("\n")


def main() -> int:
    current_results = _load_current_results()
    if not current_results:
        print(f"No performance result files found in {RESULTS_DIR}")
        return 0

    all_failures: list[str] = []
    summary_rows: list[dict[str, Any]] = []

    for raw in current_results:
        record = _normalize_record(raw)
        current_path = _write_tracking_record(record)
        print(f"Tracked performance record: {current_path}")

        baseline_records = _load_baseline_records(record["model_id"], current_path)
        baseline_records = _filter_by_gpu_type(baseline_records, record["gpu_type"])
        baseline_records = baseline_records[-5:]

        latency_base = _mean_metric(baseline_records, "latency")
        throughput_base = _mean_metric(baseline_records, "throughput")
        memory_base = _mean_metric(baseline_records, "memory")
        latency_reg = _metric_delta_percent("latency", record, baseline_records)
        throughput_reg = _metric_delta_percent("throughput", record,
                                               baseline_records)
        memory_reg = _metric_delta_percent("memory", record, baseline_records)
        regressions = [
            v for v in (latency_reg, throughput_reg, memory_reg) if v is not None
        ]
        worst_regression_pct = max(regressions) if regressions else None

        summary_row = {
            "model_id": record["model_id"],
            "gpu_type": record["gpu_type"],
            "baseline_n": len(baseline_records),
            "latency_curr": _safe_float(record.get("latency")),
            "latency_base": latency_base,
            "throughput_curr": _safe_float(record.get("throughput")),
            "throughput_base": throughput_base,
            "memory_curr": _safe_float(record.get("memory")),
            "memory_base": memory_base,
            "worst_regression_pct": worst_regression_pct,
            "failed": False,
        }

        if not baseline_records:
            print(f"No baseline records for {record['model_id']}; skipping regression check")
            summary_rows.append(summary_row)
            continue

        failures = _check_regressions(record, baseline_records, MAX_REGRESSION)
        summary_row["failed"] = bool(failures)
        summary_rows.append(summary_row)
        all_failures.extend(failures)

    markdown = _build_markdown_summary(summary_rows, MAX_REGRESSION)
    _emit_markdown_summary(markdown)

    if all_failures:
        print("Performance regression check failed:")
        for item in all_failures:
            print(f"  - {item}")
        return 1

    print("Performance baseline comparison passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
