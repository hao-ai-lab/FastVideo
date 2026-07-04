# SPDX-License-Identifier: Apache-2.0
"""Track performance results and compare against historical baseline.

This script:
1) reads current benchmark results from fastvideo/tests/performance/results,
2) syncs the canonical baseline from the configured HF dataset repo,
3) compares each current record against the median of up to 5 prior
   baseline-eligible successful records (filtered by gpu_type),
4) writes normalized records back to the HF dataset repo according to
   PERF_UPLOAD_POLICY,
5) exits non-zero if any gated metric exceeds both its percent and absolute
   regression floors.
"""

import glob
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from typing import Any

try:
    from .hf_store import (
        load_records_for_model,
        safe_float,
        sanitize,
        sync_from_hf,
        upload_record,
    )
    from .metric_policy import (
        MetricDelta,
        MetricPolicy,
        regression_delta,
        resolve_metric_policies,
        serialize_metric_thresholds,
    )
except ImportError:
    from hf_store import (
        load_records_for_model,
        safe_float,
        sanitize,
        sync_from_hf,
        upload_record,
    )
    from metric_policy import (
        MetricDelta,
        MetricPolicy,
        regression_delta,
        resolve_metric_policies,
        serialize_metric_thresholds,
    )

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
)
TRACKING_ROOT = os.environ.get(
    "PERFORMANCE_TRACKING_ROOT",
    "/tmp/perf-tracking",
)
PERF_REPORTS_DIR = os.environ.get("PERF_REPORTS_DIR", "/root/data/perf_reports")
UPLOAD_POLICY = os.environ.get("PERF_UPLOAD_POLICY", "never").strip().lower()
VALID_UPLOAD_POLICIES = {"never", "pass", "always"}
VALID_RUN_SOURCES = {"pr", "local", "scheduled_main", "unknown"}


def _should_persist_tracking() -> bool:
    return _normalized_upload_policy() != "never"


def _normalized_upload_policy() -> str:
    if UPLOAD_POLICY in VALID_UPLOAD_POLICIES:
        return UPLOAD_POLICY
    print(f"Invalid PERF_UPLOAD_POLICY={UPLOAD_POLICY!r}; using 'never'")
    return "never"


def _truthy_pr_number(value: str | None) -> bool:
    return bool(value and value not in {"false", "0", "None", "none"})


def _detect_run_source() -> str:
    explicit = os.environ.get("PERF_RUN_SOURCE", "").strip().lower()
    if explicit in VALID_RUN_SOURCES:
        return explicit
    if explicit:
        print(f"Invalid PERF_RUN_SOURCE={explicit!r}; inferring run source")

    if _truthy_pr_number(os.environ.get("BUILDKITE_PULL_REQUEST")):
        return "pr"
    if os.environ.get("BUILDKITE_BRANCH") == "main" and os.environ.get("TEST_SCOPE") == "full":
        return "scheduled_main"
    if not os.environ.get("BUILDKITE_COMMIT"):
        return "local"
    return "unknown"


def _is_baseline_eligible(run_source: str, success: bool) -> bool:
    return run_source == "scheduled_main" and success


def _upload_allowed(record: dict[str, Any]) -> bool:
    policy = _normalized_upload_policy()
    if policy == "always":
        return True
    if policy == "pass":
        return bool(record.get("success", True))
    return False


def _result_failed_static_thresholds() -> bool:
    value = os.environ.get("PERF_PYTEST_RC", "")
    if not value:
        return False
    try:
        return int(value) != 0
    except ValueError:
        return False


def _record_metadata(run_source: str, result: dict[str, Any]) -> dict[str, Any]:
    pr_number = result.get("pr_number") or os.environ.get("BUILDKITE_PULL_REQUEST", "")
    if not _truthy_pr_number(str(pr_number)):
        pr_number = ""
    return {
        "run_source": run_source,
        "baseline_eligible": False,
        "branch": os.environ.get("BUILDKITE_BRANCH", ""),
        "pr_number": pr_number,
        "test_scope": os.environ.get("TEST_SCOPE", ""),
        "build_url": os.environ.get("BUILDKITE_BUILD_URL", ""),
        "build_id": os.environ.get("BUILDKITE_BUILD_ID", ""),
        "job_id": os.environ.get("BUILDKITE_JOB_ID", ""),
    }


def _load_current_results() -> list[dict[str, Any]]:
    pattern = os.path.join(RESULTS_DIR, "perf_*.json")
    records: list[dict[str, Any]] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def normalize_performance_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw perf_*.json result into the HF tracking schema.

    The Buildkite artifact intentionally keeps the raw benchmark output from
    test_inference_performance.py. Baseline comparison, main-branch persistence,
    and manual baseline reseeds should all use this mapping so the stored HF
    records do not drift from the artifact schema.
    """
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
    metric_policies = resolve_metric_policies(result.get("regression_thresholds"))

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
        "regression_thresholds": serialize_metric_thresholds(metric_policies),
        "success": True,
        **_record_metadata(_detect_run_source(), result),
    }


def _normalize_record(result: dict[str, Any]) -> dict[str, Any]:
    return normalize_performance_result(result)


def _write_tracking_record(record: dict[str, Any]) -> str:
    model_dir = os.path.join(TRACKING_ROOT, sanitize(record["model_id"]))
    os.makedirs(model_dir, exist_ok=True)

    timestamp = sanitize(record["timestamp"])
    commit = sanitize(record["commit_sha"] or "unknown")
    out_path = os.path.join(model_dir, f"{timestamp}_{commit}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return out_path


def _write_normalized_artifact(record: dict[str, Any]) -> None:
    try:
        results_dir = os.path.join(PERF_REPORTS_DIR, "results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = sanitize(record["timestamp"])
        model_id = sanitize(record["model_id"])
        commit = sanitize(record["commit_sha"] or "unknown")
        path = os.path.join(
            results_dir,
            f"normalized_perf_{model_id}_{timestamp}_{commit}.json",
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(f"Normalized performance result written to {path}")
    except Exception as e:
        print(f"Failed to write normalized performance result artifact: {e}")


def _baseline_metric(records: list[dict[str, Any]], key: str) -> float | None:
    values = [safe_float(r.get(key)) for r in records]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return statistics.median(values)


def _metric_delta(
    policy: MetricPolicy,
    current: dict[str, Any],
    baseline_records: list[dict[str, Any]],
) -> MetricDelta | None:
    curr = safe_float(current.get(policy.key))
    baseline = _baseline_metric(baseline_records, policy.key)
    if curr is None or baseline is None:
        return None
    return regression_delta(policy, curr, baseline)


def _metric_policy_summary(policy: MetricPolicy) -> str:
    gated = "gated" if policy.gated else "info"
    return (
        f"{gated}, >{policy.threshold_percent * 100:.1f}% "
        f"and >{policy.threshold_absolute:.{policy.precision}f}"
    )


def _check_regressions(
    current: dict[str, Any],
    baseline_records: list[dict[str, Any]],
    metric_policies: tuple[MetricPolicy, ...],
) -> list[str]:
    failures: list[str] = []

    for policy in metric_policies:
        baseline = _baseline_metric(baseline_records, policy.key)
        curr = safe_float(current.get(policy.key))
        if baseline is None or curr is None:
            continue
        delta = regression_delta(policy, curr, baseline)
        if delta is None or not delta.regressed:
            continue
        failures.append(
            f"{current['model_id']} {policy.key} regressed by "
            f"{delta.percent * 100:.1f}% and "
            f"{delta.absolute:.{policy.precision}f} "
            f"(current={curr:.{policy.precision}f}, "
            f"baseline_median={baseline:.{policy.precision}f}, "
            f"threshold={_metric_policy_summary(policy)})"
        )

    return failures


def _compact_value(value: float | None, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def _build_summary_row(
    record: dict[str, Any],
    baseline_records: list[dict[str, Any]],
    metric_policies: tuple[MetricPolicy, ...],
    has_failed: bool,
) -> dict[str, Any]:
    """Format a single benchmark result as a row for the Markdown table."""

    metric_values: dict[str, dict[str, Any]] = {}
    regressions: list[float] = []
    failing_metrics: list[str] = []
    for policy in metric_policies:
        curr = safe_float(record.get(policy.key))
        baseline = _baseline_metric(baseline_records, policy.key)
        delta = _metric_delta(policy, record, baseline_records)
        regression = None if delta is None else delta.percent * 100.0
        absolute_delta = None if delta is None else delta.absolute
        metric_values[policy.key] = {
            "curr": curr,
            "base": baseline,
            "regression_pct": regression,
            "absolute_delta": absolute_delta,
            "threshold_percent": policy.threshold_percent * 100.0,
            "threshold_absolute": policy.threshold_absolute,
            "gated": policy.gated,
            "regressed": False if delta is None else delta.regressed,
        }
        if regression is not None:
            regressions.append(regression)
        if delta is not None and delta.regressed:
            failing_metrics.append(policy.key)

    worst_regression_pct = max(regressions) if regressions else None

    return {
        "model_id": record["model_id"],
        "gpu_type": record["gpu_type"],
        "baseline_n": len(baseline_records),
        "metrics": metric_values,
        "worst_regression_pct": worst_regression_pct,
        "failing_metrics": failing_metrics,
        "failed": has_failed,
    }


def _build_markdown_summary(
    summary_rows: list[dict[str, Any]],
    metric_policies: tuple[MetricPolicy, ...],
) -> str:
    lines = [
        "## Performance Baseline Comparison",
        "",
        "Threshold: gated metrics fail only when both percent and absolute "
        "regression floors are exceeded.",
        "",
        ("| Model | GPU | Baseline N | Latency (curr/base) | "
         "Throughput (curr/base) | Memory (curr/base) | "
         "Text Enc (curr/base) | DiT (curr/base) | "
         "VAE Decode (curr/base) | Worst Regression | Failing Metrics | Status |"),
        "|---|---|---:|---|---|---|---|---|---|---:|---|---|",
    ]

    for row in summary_rows:
        metric_cells = []
        for policy in metric_policies:
            values = row["metrics"][policy.key]
            metric_cells.append(f"{_compact_value(values['curr'], policy.precision)} / "
                                f"{_compact_value(values['base'], policy.precision)}")

        worst_reg = ("n/a" if row["worst_regression_pct"] is None else f"{row['worst_regression_pct']:.1f}%")
        failing_metrics = ", ".join(row["failing_metrics"]) if row["failing_metrics"] else "none"
        status = "FAIL" if row["failed"] else "PASS"

        lines.append(f"| {row['model_id']} | {row['gpu_type']} | "
                     f"{row['baseline_n']} | "
                     f"{' | '.join(metric_cells)} | "
                     f"{worst_reg} | {failing_metrics} | {status} |")

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
        os.makedirs(PERF_REPORTS_DIR, exist_ok=True)
        short_sha = commit_sha[:7] if commit_sha else "unknown"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(PERF_REPORTS_DIR, f"perf_{short_sha}_{timestamp}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown + "\n")
        print(f"Performance report written to {report_path}")
    except Exception as e:
        print(f"Failed to write performance report to Modal volume: {e}")


def main() -> int:
    persist_tracking = _should_persist_tracking()
    upload_policy = _normalized_upload_policy()
    static_threshold_failed = _result_failed_static_thresholds()

    # Strict on upload-enabled runs: silent sync failure would make comparison
    # and upload state ambiguous.
    sync_from_hf(TRACKING_ROOT, strict=persist_tracking)

    current_results = _load_current_results()
    if not current_results:
        print(f"No performance result files found in {RESULTS_DIR}")
        return 0

    all_failures: list[str] = []
    summary_rows: list[dict[str, Any]] = []

    if persist_tracking:
        print(f"Tracking persistence enabled: PERF_UPLOAD_POLICY={upload_policy}")
    else:
        print("Tracking persistence disabled: PERF_UPLOAD_POLICY=never")

    if static_threshold_failed:
        print(f"Static-threshold phase failed: PERF_PYTEST_RC={os.environ.get('PERF_PYTEST_RC')}")

    for raw in current_results:
        record = _normalize_record(raw)
        metric_policies = resolve_metric_policies(record.get("regression_thresholds"))

        baseline_records = load_records_for_model(
            TRACKING_ROOT,
            record["model_id"],
            record["gpu_type"],
            last_n=5,
            successful_only=True,
            baseline_eligible_only=True,
        )

        if not baseline_records:
            print(f"No baseline for {record['model_id']} on "
                  f"{record['gpu_type']}. Initializing...")
            failures: list[str] = []
            record["success"] = True
        else:
            failures = _check_regressions(record, baseline_records, metric_policies)
        if static_threshold_failed:
            failures.append(f"{record['model_id']} fixed-threshold phase failed "
                            f"(PERF_PYTEST_RC={os.environ.get('PERF_PYTEST_RC')})")

        record["success"] = not failures
        record["baseline_eligible"] = _is_baseline_eligible(record["run_source"], record["success"])
        all_failures.extend(failures)

        _write_normalized_artifact(record)

        if _upload_allowed(record):
            current_path = _write_tracking_record(record)
            upload_record(current_path, record, strict=True)
        else:
            print("Tracking upload skipped for "
                  f"{record['model_id']} ({record['run_source']}, success={record['success']})")

        summary_rows.append(_build_summary_row(record, baseline_records, metric_policies, bool(failures)))

    commit_sha = os.environ.get("BUILDKITE_COMMIT", "unknown")[:7]
    markdown = _build_markdown_summary(summary_rows, resolve_metric_policies(None))
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
