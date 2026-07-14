# SPDX-License-Identifier: Apache-2.0
"""Prepare reviewed v2 calibration artifacts as baseline seed records."""

import argparse
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from typing import Any

try:
    from fastvideo.tests.performance.compare_baseline import (
        STATUS_CALIBRATION_NEEDED,
        STATUS_PASS,
        _comparison_identity_filters,
        _record_uses_v2_identity,
    )
    from fastvideo.performance.hf_store import safe_float, sanitize
    from fastvideo.performance.metric_policy import DEFAULT_METRIC_POLICIES
except ImportError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from fastvideo.tests.performance.compare_baseline import (
        STATUS_CALIBRATION_NEEDED,
        STATUS_PASS,
        _comparison_identity_filters,
        _record_uses_v2_identity,
    )
    from fastvideo.performance.hf_store import safe_float, sanitize
    from fastvideo.performance.metric_policy import DEFAULT_METRIC_POLICIES

TRACKING_ROOT = os.environ.get("PERFORMANCE_TRACKING_ROOT", "/tmp/perf-tracking")
DEFAULT_MAX_INTRA_BATCH_REGRESSION = 0.05
_SOURCE_PROVENANCE_FIELDS = ("model_id", "commit_sha", "timestamp", "build_id", "job_id")
_CORE_MEASUREMENT_FIELDS = frozenset({"latency", "throughput", "memory"})


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _source_identity(record: dict[str, Any]) -> dict[str, str]:
    if not _record_uses_v2_identity(record):
        raise ValueError("baseline seeds require a v2 record with exact comparable identity fields")
    return _comparison_identity_filters(record)


def _require_nonempty_string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"baseline seed records require a non-empty string {field}")
    return value


def _source_provenance_identity(record: dict[str, Any]) -> tuple[str, ...] | None:
    identity = tuple(str(record.get(field) or "") for field in _SOURCE_PROVENANCE_FIELDS)
    return identity if any(identity[1:]) else None


def _validate_unique_sources(source_paths: list[str], records: list[dict[str, Any]]) -> None:
    seen_paths: set[str] = set()
    seen_contents: set[str] = set()
    seen_provenance: set[tuple[str, ...]] = set()

    for index, (source_path, record) in enumerate(zip(source_paths, records), start=1):
        resolved_path = os.path.realpath(os.path.abspath(source_path))
        if resolved_path in seen_paths:
            raise ValueError(f"source artifact {index} duplicates a resolved source path")
        seen_paths.add(resolved_path)

        content_identity = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if content_identity in seen_contents:
            raise ValueError(f"source artifact {index} duplicates source content")
        seen_contents.add(content_identity)

        provenance_identity = _source_provenance_identity(record)
        if provenance_identity is not None:
            if provenance_identity in seen_provenance:
                raise ValueError(f"source artifact {index} duplicates source provenance")
            seen_provenance.add(provenance_identity)


def _truthy_pr_number(value: Any) -> bool:
    return bool(value and str(value) not in {"false", "0", "None", "none"})


def _validate_source_measurements(record: dict[str, Any]) -> None:
    for policy in DEFAULT_METRIC_POLICIES:
        raw_value = record.get(policy.key)
        if raw_value is None:
            if policy.key in _CORE_MEASUREMENT_FIELDS:
                raise ValueError(f"baseline seed records require a finite positive {policy.key} measurement")
            continue

        value = safe_float(raw_value)
        if value is None or not math.isfinite(value):
            raise ValueError(f"baseline seed records require a finite {policy.key} measurement")
        if policy.key in _CORE_MEASUREMENT_FIELDS:
            if value <= 0:
                raise ValueError(f"baseline seed records require a positive {policy.key} measurement")
        elif value < 0:
            raise ValueError(f"baseline seed records require a non-negative {policy.key} measurement")


def _validate_calibration_source(record: dict[str, Any]) -> None:
    _require_nonempty_string(record, "model_id")
    _source_identity(record)
    _validate_source_measurements(record)
    if record.get("comparison_status") != STATUS_CALIBRATION_NEEDED:
        raise ValueError("baseline seeds require a CALIBRATION_NEEDED source artifact")
    if record.get("success") is not True:
        raise ValueError("baseline seeds require a successful source artifact")
    if record.get("run_source") != "scheduled_main":
        raise ValueError("baseline seeds require a scheduled_main source artifact")
    if _truthy_pr_number(record.get("pr_number")):
        raise ValueError("baseline seeds require a non-PR source artifact")
    if record.get("branch") != "main" or record.get("test_scope") != "full":
        raise ValueError("baseline seeds require a main-branch full-suite source artifact")


def build_baseline_seed_record(
    source_record: dict[str, Any],
    *,
    reason: str,
    source_result: str | None = None,
    operator: str | None = None,
    timestamp: str | None = None,
    batch_size: int = 1,
    batch_index: int = 1,
) -> dict[str, Any]:
    """Return an approved v2 baseline seed derived from a calibration artifact."""
    if not reason.strip():
        raise ValueError("baseline seed reason must be a non-empty string")
    _validate_calibration_source(source_record)

    seed = dict(source_record)
    seed.update({
        "timestamp": timestamp or _now_utc_iso(),
        "success": True,
        "baseline_eligible": True,
        "comparison_status": STATUS_PASS,
        "comparison_status_reason": "Approved baseline seed from CALIBRATION_NEEDED source artifact",
        "baseline_seed": True,
        "baseline_seed_reason": reason,
        "baseline_seed_source_status": source_record.get("comparison_status"),
        "baseline_seed_source_timestamp": source_record.get("timestamp"),
        "baseline_seed_source_success": source_record.get("success"),
        "baseline_seed_source_run_source": source_record.get("run_source"),
        "baseline_seed_source_branch": source_record.get("branch"),
        "baseline_seed_source_test_scope": source_record.get("test_scope"),
        "baseline_seed_source_pr_number": source_record.get("pr_number"),
        "baseline_seed_batch_size": batch_size,
        "baseline_seed_batch_index": batch_index,
    })
    if source_result:
        seed["baseline_seed_source_result"] = source_result
    if operator:
        seed["baseline_seed_operator"] = operator
    return seed


def write_seed_record(
    local_dir: str,
    record: dict[str, Any],
    *,
    suffix: str | None = None,
) -> str:
    """Write *record* under the tracking root and return the local JSON path."""
    out_path = _seed_record_path(local_dir, record, suffix=suffix)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _write_json(out_path, record)
    return out_path


def _seed_record_path(
    local_dir: str,
    record: dict[str, Any],
    *,
    suffix: str | None = None,
) -> str:
    model_id = _require_nonempty_string(record, "model_id")
    timestamp = _require_nonempty_string(record, "timestamp")
    model_dir = os.path.join(local_dir, sanitize(model_id))
    commit = sanitize(record.get("commit_sha") or "unknown")
    suffix_part = f"_{sanitize(suffix)}" if suffix else ""
    return os.path.join(model_dir, f"{sanitize(timestamp)}_{commit}_seed{suffix_part}.json")


def _validate_same_identity(records: list[dict[str, Any]]) -> dict[str, str]:
    first = _source_identity(records[0])
    for index, record in enumerate(records[1:], start=2):
        identity = _source_identity(record)
        if identity != first:
            raise ValueError(f"source artifact {index} does not match the first exact comparable identity")
    return first


def _order_sources_by_timestamp(
    source_paths: list[str],
    records: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    timestamped = []
    undated = []
    for index, (source_path, record) in enumerate(zip(source_paths, records), start=1):
        try:
            timestamp = datetime.fromisoformat(str(record.get("timestamp")))
        except (TypeError, ValueError):
            timestamp = None

        if timestamp is None:
            print(
                f"Warning: source artifact {index} has a missing or unparsable timestamp; "
                "preserving its input order after timestamped sources."
            )
            undated.append((source_path, record))
            continue

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        timestamped.append((timestamp, index, source_path, record))

    timestamped.sort(key=lambda item: (item[0], item[1]))
    return [(source_path, record) for _, _, source_path, record in timestamped] + undated


def _nonnegative_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be a finite non-negative number")
    return parsed


def _validate_batch_consistency(
    records: list[dict[str, Any]],
    max_regression: float,
) -> None:
    print(f"Source batch consistency (maximum regression {max_regression * 100:.1f}%):")
    failures = []
    for policy in DEFAULT_METRIC_POLICIES:
        values = [
            (index, value)
            for index, record in enumerate(records, start=1)
            if (value := safe_float(record.get(policy.key))) is not None
        ]
        if len(values) < 2:
            continue

        batch_median = statistics.median(value for _, value in values)
        if batch_median <= 0:
            raise ValueError(f"cannot validate {policy.key} consistency against non-positive batch median")

        regressions = []
        for source_index, value in values:
            if policy.lower_is_better:
                regression = (value - batch_median) / batch_median
            else:
                regression = (batch_median - value) / batch_median
            regressions.append((source_index, regression))
            print(
                f"  {policy.key} source={source_index} value={value:.6f} "
                f"median={batch_median:.6f} regression={regression * 100:.1f}%")

        worst_source, worst_regression = max(regressions, key=lambda item: item[1])
        print(f"  {policy.key} worst regression: {worst_regression * 100:.1f}% (source={worst_source})")
        if worst_regression > max_regression:
            failures.append(
                f"source artifact {worst_source} {policy.key} regresses by "
                f"{worst_regression * 100:.1f}% against the batch median")

    if failures:
        raise ValueError("source batch exceeds maximum intra-batch regression: " + "; ".join(failures))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reviewed baseline seed records from v2 CALIBRATION_NEEDED artifacts.",
    )
    parser.add_argument(
        "--source-result",
        dest="source_results",
        action="append",
        required=True,
        help="Path to a normalized_perf_*.json artifact. Repeat for multiple reviewed source artifacts.",
    )
    parser.add_argument(
        "--intent-rationale",
        required=True,
        help="Reviewed reason this calibration artifact should seed the baseline.",
    )
    parser.add_argument(
        "--tracking-root",
        default=TRACKING_ROOT,
        help="Local performance tracking root to write seed records under.",
    )
    parser.add_argument(
        "--max-intra-batch-regression",
        type=_nonnegative_finite_float,
        default=DEFAULT_MAX_INTRA_BATCH_REGRESSION,
        help="Maximum regression of any source against the batch median (default: 0.05).",
    )
    parser.add_argument(
        "--operator",
        default=os.environ.get("USER"),
        help="Operator name recorded in seed provenance.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source_records = [_load_json(path) for path in args.source_results]
    for source_record in source_records:
        _validate_calibration_source(source_record)
    _validate_unique_sources(args.source_results, source_records)
    identity = _validate_same_identity(source_records)
    _validate_batch_consistency(source_records, args.max_intra_batch_regression)
    ordered_sources = _order_sources_by_timestamp(args.source_results, source_records)
    print("Seeding exact comparable identity:")
    for key, value in identity.items():
        print(f"  {key}: {value}")

    prepared_seeds = []
    for index, (source_path, source_record) in enumerate(ordered_sources, start=1):
        seed_record = build_baseline_seed_record(
            source_record,
            reason=args.intent_rationale,
            source_result=source_path,
            operator=args.operator,
            batch_size=len(source_records),
            batch_index=index,
        )
        suffix = f"{index:02d}" if len(source_records) > 1 else None
        seed_path = _seed_record_path(args.tracking_root, seed_record, suffix=suffix)
        prepared_seeds.append((seed_path, seed_record, suffix))

    for seed_path, seed_record, suffix in prepared_seeds:
        write_seed_record(args.tracking_root, seed_record, suffix=suffix)
        print(f"Prepared baseline seed: {seed_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
