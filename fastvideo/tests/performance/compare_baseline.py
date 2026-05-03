# SPDX-License-Identifier: Apache-2.0
"""Track performance results and compare against historical baseline.

This script:
1) reads current benchmark results from fastvideo/tests/performance/results,
2) syncs the canonical baseline from the configured HF dataset repo,
3) compares each current record against the median of up to 5 prior records
   (filtered by gpu_type, successful only),
4) on persist runs (full-suite on main branch), writes the normalized record
   back to the HF dataset repo,
5) exits non-zero if any metric regresses by more than PERF_MAX_REGRESSION
   (default 5%).
"""

import glob
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from typing import Any

from hf_store import (
    load_records_for_model,
    safe_float,
    sanitize,
    sync_from_hf,
    upload_record,
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
)
TRACKING_ROOT = os.environ.get(
    "PERFORMANCE_TRACKING_ROOT",
    "/tmp/perf-tracking",
)
MAX_REGRESSION = float(os.environ.get("PERF_MAX_REGRESSION", "0.05"))
METRICS = (
    ("latency", "Latency", 3),
    ("throughput", "Throughput", 3),
    ("memory", "Memory", 1),
    ("text_encoder_time_s", "Text Enc", 3),
    ("dit_time_s", "DiT", 3),
    ("vae_decode_time_s", "VAE Decode", 3),
)
LOWER_IS_BETTER_METRICS = {
    "latency",
    "memory",
    "text_encoder_time_s",
    "dit_time_s",
    "vae_decode_time_s",
}


def _should_persist_tracking() -> bool:
    test_scope = os.environ.get("TEST_SCOPE", "")
    branch = os.environ.get("BUILDKITE_BRANCH", "")
    return test_scope == "full" and branch == "main"

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
    latency = safe_float(result.get("avg_generation_time_s"))
    throughput = safe_float(result.get("throughput_fps"))
    memory = safe_float(result.get("max_peak_memory_mb"))
    text_encoder_time = safe_float(result.get("text_encoder_time_s"))
    dit_time = safe_float(result.get("dit_time_s"))
    vae_decode_time = safe_float(result.get("vae_decode_time_s"))

    return {
        "model_id": model_id,
        "timestamp": timestamp,
        "commit_sha": commit_sha,
        "gpu_type": result.get("device", "unknown"),
        "latency": latency,
        "throughput": throughput,
        "memory": memory,
        "text_encoder_time_s": text_encoder_time,
        "dit_time_s": dit_time,
        "vae_decode_time_s": vae_decode_time,
        "success": True,
    }


def _write_tracking_record(record: dict[str, Any]) -> str:
    model_dir = os.path.join(TRACKING_ROOT, sanitize(record["model_id"]))
    os.makedirs(model_dir, exist_ok=True)

    timestamp = sanitize(record["timestamp"])
    commit = sanitize(record["commit_sha"] or "unknown")
    out_path = os.path.join(model_dir, f"{timestamp}_{commit}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return out_path


def _baseline_metric(records: list[dict[str, Any]], key: str) -> float | None:
    values = [safe_float(r.get(key)) for r in records]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return statistics.median(values)


def _check_regressions(
    current: dict[str, Any],
    baseline_records: list[dict[str, Any]],
    max_regression: float,
) -> list[str]:
    failures: list[str] = []

    for metric in ("latency", "memory", "text_encoder_time_s", "dit_time_s",
                   "vae_decode_time_s"):
        baseline = _baseline_metric(baseline_records, metric)
        curr = safe_float(current.get(metric))
        if baseline is None or curr is None or baseline <= 0:
            continue
        regression = (curr - baseline) / baseline
        if regression > max_regression:
            failures.append(f"{current['model_id']} {metric} regressed by "
                            f"{regression * 100:.1f}% "
                            f"(current={curr:.3f}, baseline_median={baseline:.3f})")

    baseline_tp = _baseline_metric(baseline_records, "throughput")
    curr_tp = safe_float(current.get("throughput"))
    if baseline_tp is not None and curr_tp is not None and baseline_tp > 0:
        regression = (baseline_tp - curr_tp) / baseline_tp
        if regression > max_regression:
            failures.append(f"{current['model_id']} throughput regressed by "
                            f"{regression * 100:.1f}% "
                            f"(current={curr_tp:.3f}, baseline_median={baseline_tp:.3f})")

    return failures


def _metric_delta_percent(
    metric: str,
    current: dict[str, Any],
    baseline_records: list[dict[str, Any]],
) -> float | None:
    curr = safe_float(current.get(metric))
    baseline = _baseline_metric(baseline_records, metric)
    if curr is None or baseline is None or baseline <= 0:
        return None

    if metric in LOWER_IS_BETTER_METRICS:
        return (curr - baseline) / baseline * 100.0
    if metric == "throughput":
        return (baseline - curr) / baseline * 100.0
    return None


def _compact_value(value: float | None, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def _build_summary_row(
    record: dict[str, Any],
    baseline_records: list[dict[str, Any]],
    has_failed: bool,
) -> dict[str, Any]:
    """Format a single benchmark result as a row for the Markdown table."""

    metric_values: dict[str, dict[str, float | None]] = {}
    regressions: list[float] = []
    for metric, _label, _precision in METRICS:
        curr = safe_float(record.get(metric))
        baseline = _baseline_metric(baseline_records, metric)
        regression = _metric_delta_percent(metric, record, baseline_records)
        metric_values[metric] = {
            "curr": curr,
            "base": baseline,
            "regression_pct": regression,
        }
        if regression is not None:
            regressions.append(regression)

    worst_regression_pct = max(regressions) if regressions else None

    return {
        "model_id": record["model_id"],
        "gpu_type": record["gpu_type"],
        "baseline_n": len(baseline_records),
        "metrics": metric_values,
        "worst_regression_pct": worst_regression_pct,
        "failed": has_failed,
    }


def _build_markdown_summary(
    summary_rows: list[dict[str, Any]],
    max_regression: float,
) -> str:
    lines = [
        "## Performance Baseline Comparison",
        "",
        f"Threshold: regressions greater than {max_regression * 100:.1f}% fail",
        "",
        ("| Model | GPU | Baseline N | Latency (curr/base) | "
         "Throughput (curr/base) | Memory (curr/base) | "
         "Text Enc (curr/base) | DiT (curr/base) | "
         "VAE Decode (curr/base) | Worst Regression | Status |"),
        "|---|---|---:|---|---|---|---|---|---|---:|---|",
    ]

    for row in summary_rows:
        metric_cells = []
        for metric, _label, precision in METRICS:
            values = row["metrics"][metric]
            metric_cells.append(f"{_compact_value(values['curr'], precision)} / "
                                f"{_compact_value(values['base'], precision)}")

        worst_reg = ("n/a" if row["worst_regression_pct"] is None else f"{row['worst_regression_pct']:.1f}%")
        status = "FAIL" if row["failed"] else "PASS"

        lines.append(f"| {row['model_id']} | {row['gpu_type']} | "
                     f"{row['baseline_n']} | "
                     f"{' | '.join(metric_cells)} | "
                     f"{worst_reg} | {status} |")

    return "\n".join(lines) + "\n"


def _emit_markdown_summary(markdown: str, commit_sha: str) -> None:
    print("\n" + markdown)

    # 1. Existing GitHub logic (safe to keep)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(markdown + "\n")

    # 2. Write to Modal volume for Buildkite to pick up in post-run hook
    try:
        perf_reports_dir = "/root/data/perf_reports"
        os.makedirs(perf_reports_dir, exist_ok=True)
        short_sha = commit_sha[:7] if commit_sha else "unknown"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(perf_reports_dir, f"perf_{short_sha}_{timestamp}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown + "\n")
        print(f"Performance report written to {report_path}")
    except Exception as e:
        print(f"Failed to write performance report to Modal volume: {e}")


def main() -> int:
    persist_tracking = _should_persist_tracking()

    # Strict on persist: a silent sync failure would pollute the baseline.
    sync_from_hf(TRACKING_ROOT, strict=persist_tracking)

    current_results = _load_current_results()
    if not current_results:
        print(f"No performance result files found in {RESULTS_DIR}")
        return 0

    all_failures: list[str] = []
    summary_rows: list[dict[str, Any]] = []

    if persist_tracking:
        print("Tracking persistence enabled: full-suite run on main branch")
    else:
        print("Tracking persistence disabled: "
              "only full-suite runs on main branch are persisted")

    for raw in current_results:
        record = _normalize_record(raw)

        baseline_records = load_records_for_model(
            TRACKING_ROOT,
            record["model_id"],
            record["gpu_type"],
            last_n=5,
            successful_only=True,
        )

        if not baseline_records:
            print(f"No baseline for {record['model_id']} on "
                  f"{record['gpu_type']}. Initializing...")
            failures: list[str] = []
            record["success"] = True
        else:
            failures = _check_regressions(record, baseline_records, MAX_REGRESSION)
            record["success"] = not failures
            all_failures.extend(failures)

        # Strict upload: a silent failure would freeze the rolling baseline.
        if persist_tracking:
            current_path = _write_tracking_record(record)
            upload_record(current_path, record, strict=True)

        summary_rows.append(_build_summary_row(record, baseline_records, bool(failures)))

    commit_sha = os.environ.get("BUILDKITE_COMMIT", "unknown")[:7]
    markdown = _build_markdown_summary(summary_rows, MAX_REGRESSION)
    _emit_markdown_summary(markdown, commit_sha)

    if all_failures:
        print("Performance regression check failed:")
        for item in all_failures:
            print(f"  - {item}")
        return 1

    print("Performance baseline comparison passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
