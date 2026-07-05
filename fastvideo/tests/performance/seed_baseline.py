# SPDX-License-Identifier: Apache-2.0
"""Prepare reviewed v2 calibration artifacts as baseline seed records."""

import argparse
import json
import os
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
    from fastvideo.performance.hf_store import sanitize, upload_record
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
    from fastvideo.performance.hf_store import sanitize, upload_record

TRACKING_ROOT = os.environ.get("PERFORMANCE_TRACKING_ROOT", "/tmp/perf-tracking")


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


def _validate_calibration_source(record: dict[str, Any]) -> None:
    _source_identity(record)
    if record.get("comparison_status") != STATUS_CALIBRATION_NEEDED:
        raise ValueError("baseline seeds require a CALIBRATION_NEEDED source artifact")
    if record.get("success") is not True:
        raise ValueError("baseline seeds require a successful source artifact")


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
    model_dir = os.path.join(local_dir, sanitize(record["model_id"]))
    os.makedirs(model_dir, exist_ok=True)

    timestamp = sanitize(record["timestamp"])
    commit = sanitize(record.get("commit_sha") or "unknown")
    suffix_part = f"_{sanitize(suffix)}" if suffix else ""
    out_path = os.path.join(model_dir, f"{timestamp}_{commit}_seed{suffix_part}.json")
    _write_json(out_path, record)
    return out_path


def _validate_same_identity(records: list[dict[str, Any]]) -> dict[str, str]:
    first = _source_identity(records[0])
    for index, record in enumerate(records[1:], start=2):
        identity = _source_identity(record)
        if identity != first:
            raise ValueError(f"source artifact {index} does not match the first exact comparable identity")
    return first


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
        "--operator",
        default=os.environ.get("USER"),
        help="Operator name recorded in seed provenance.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload prepared seed records to the configured HF performance tracking dataset.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source_records = [_load_json(path) for path in args.source_results]
    identity = _validate_same_identity(source_records)
    print("Seeding exact comparable identity:")
    for key, value in identity.items():
        print(f"  {key}: {value}")

    for index, (source_path, source_record) in enumerate(zip(args.source_results, source_records), start=1):
        seed_record = build_baseline_seed_record(
            source_record,
            reason=args.intent_rationale,
            source_result=source_path,
            operator=args.operator,
            batch_size=len(source_records),
            batch_index=index,
        )
        suffix = f"{index:02d}" if len(source_records) > 1 else None
        seed_path = write_seed_record(args.tracking_root, seed_record, suffix=suffix)
        print(f"Prepared baseline seed: {seed_path}")
        if args.upload:
            upload_record(seed_path, seed_record, strict=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
