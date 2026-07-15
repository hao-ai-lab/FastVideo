# SPDX-License-Identifier: Apache-2.0
"""Prepare reviewed v2 calibration artifacts as baseline seed records."""

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.constants import ENDPOINT

try:
    from fastvideo.tests.performance.compare_baseline import (
        STATUS_CALIBRATION_NEEDED,
        STATUS_PASS,
        _comparison_identity_filters,
        _recipe_cohort_filters,
        _recipe_mismatch_records,
        _record_uses_v2_identity,
    )
    from fastvideo.performance.hf_store import (
        HF_REPO_ID,
        load_records_for_identity,
        resolve_hf_token,
        safe_float,
        sanitize,
        sync_from_hf,
    )
    from fastvideo.performance.metric_policy import DEFAULT_METRIC_POLICIES
except ImportError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from fastvideo.tests.performance.compare_baseline import (
        STATUS_CALIBRATION_NEEDED,
        STATUS_PASS,
        _comparison_identity_filters,
        _recipe_cohort_filters,
        _recipe_mismatch_records,
        _record_uses_v2_identity,
    )
    from fastvideo.performance.hf_store import (
        HF_REPO_ID,
        load_records_for_identity,
        resolve_hf_token,
        safe_float,
        sanitize,
        sync_from_hf,
    )
    from fastvideo.performance.metric_policy import DEFAULT_METRIC_POLICIES

TRACKING_ROOT = os.environ.get("PERFORMANCE_TRACKING_ROOT", "/tmp/perf-tracking")
STAGING_ROOT = os.environ.get("PERFORMANCE_RESEED_STAGING_ROOT", "/tmp/performance_reseed_prepared")
DEFAULT_MAX_INTRA_BATCH_REGRESSION = 0.05
_CORE_MEASUREMENT_FIELDS = frozenset({"latency", "throughput", "memory"})


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    job_id = str(record.get("job_id") or "")
    if job_id:
        return ("job_id", job_id)
    build_id = str(record.get("build_id") or "")
    if build_id:
        return ("build_id", build_id)
    commit_sha = str(record.get("commit_sha") or "")
    timestamp = str(record.get("timestamp") or "")
    if commit_sha or timestamp:
        return ("commit_timestamp", commit_sha, timestamp)
    return None


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

        value = None if isinstance(raw_value, bool) else safe_float(raw_value)
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
    if record.get("result_schema_version") != 2:
        raise ValueError("baseline seeds require a normalized v2 source artifact")
    if record.get("comparison_status") != STATUS_CALIBRATION_NEEDED:
        raise ValueError("baseline seeds require a CALIBRATION_NEEDED source artifact")
    if record.get("success") is not True:
        raise ValueError("baseline seeds require a successful source artifact")
    if record.get("baseline_eligible") is not False:
        raise ValueError("baseline seeds require a baseline_eligible=false source artifact")
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
    """Write *record* under the staging root and return the local JSON path."""
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


def _validate_separate_roots(tracking_root: str, staging_root: str) -> None:
    tracking_root = os.path.realpath(os.path.abspath(tracking_root))
    staging_root = os.path.realpath(os.path.abspath(staging_root))
    try:
        common_root = os.path.commonpath((tracking_root, staging_root))
    except ValueError:
        return
    if common_root in {tracking_root, staging_root}:
        raise ValueError("baseline seed staging and canonical tracking roots must be separate and non-nested")


def _validate_no_existing_seed(
    records_root: str,
    record: dict[str, Any],
    *,
    source: str,
) -> None:
    identity = _source_identity(record)
    if load_records_for_identity(
        records_root,
        identity,
        last_n=1,
        successful_only=True,
        baseline_eligible_only=True,
    ):
        raise ValueError(
            f"{source} already has a baseline-eligible record for the exact comparable identity; "
            "the CALIBRATION_NEEDED source artifact is stale"
        )

    cohort_records = load_records_for_identity(
        records_root,
        _recipe_cohort_filters(record),
        successful_only=True,
    )
    mismatches = _recipe_mismatch_records(record, cohort_records)
    if mismatches:
        recipes = sorted({str(item["recipe_fingerprint"]) for item in mismatches})
        raise ValueError(
            f"{source} already has trusted records for another recipe in this workload, "
            f"variant, and benchmark version: {', '.join(recipes)}"
        )


def _repository_descriptor() -> dict[str, str]:
    return {
        "endpoint": ENDPOINT,
        "repo_id": HF_REPO_ID,
        "repo_type": "dataset",
    }


def _reservation_path(staging_root: str, identity: dict[str, str]) -> str:
    payload = json.dumps({
        "identity": identity,
        "repository": _repository_descriptor(),
    }, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return os.path.join(staging_root, ".seed-reservations", digest)


def _reserve_staging_identity(staging_root: str, identity: dict[str, str]) -> str:
    reservation = _reservation_path(staging_root, identity)
    os.makedirs(os.path.dirname(reservation), exist_ok=True)
    try:
        os.mkdir(reservation, 0o700)
    except FileExistsError as exc:
        raise ValueError(
            "staging already has a reservation for the exact comparable identity; "
            "reuse or explicitly clean the existing preparation"
        ) from exc
    return reservation


def _release_failed_reservation(reservation: str, prepared_paths: list[str]) -> None:
    reservation = os.path.realpath(os.path.abspath(reservation))
    prepared_dirs: set[str] = set()
    for path in prepared_paths:
        path = os.path.realpath(os.path.abspath(path))
        if os.path.commonpath((reservation, path)) != reservation:
            continue
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        prepared_dirs.add(os.path.dirname(path))
    for name in ("manifest", "manifest.tmp"):
        try:
            os.unlink(os.path.join(reservation, name))
        except FileNotFoundError:
            pass
    for directory in sorted(prepared_dirs, key=len, reverse=True):
        try:
            os.rmdir(directory)
        except OSError:
            pass
    try:
        os.rmdir(reservation)
    except OSError:
        pass


def _write_preparation_manifest(
    reservation: str,
    staging_root: str,
    identity: dict[str, str],
    source_paths: list[str],
    source_records: list[dict[str, Any]],
    prepared_paths: list[str],
) -> str:
    manifest_path = os.path.join(reservation, "manifest")
    temp_path = f"{manifest_path}.tmp"
    source_manifest_entries = []
    for path, expected_record in zip(source_paths, source_records, strict=True):
        with open(path, "rb") as f:
            blob = f.read()
        if json.loads(blob) != expected_record:
            raise ValueError(f"source record changed during preparation: {path}")
        source_manifest_entries.append({
            "path": os.path.realpath(os.path.abspath(path)),
            "sha256": hashlib.sha256(blob).hexdigest(),
        })
    manifest = {
        "manifest_schema_version": 1,
        "repository": _repository_descriptor(),
        "identity": identity,
        "recipe_cohort": {
            key: identity[key]
            for key in ("workload_id", "variant_id", "benchmark_version")
        },
        "staging_root": os.path.realpath(os.path.abspath(staging_root)),
        "source_records": source_manifest_entries,
        "prepared_records": [{
            "path": os.path.realpath(os.path.abspath(path)),
            "sha256": _file_sha256(path),
        } for path in prepared_paths],
    }
    _write_json(temp_path, manifest)
    os.replace(temp_path, manifest_path)
    return manifest_path


def _validate_prepared_seed(record: dict[str, Any]) -> None:
    _source_identity(record)
    _require_nonempty_string(record, "model_id")
    _validate_source_measurements(record)
    if record.get("baseline_seed") is not True:
        raise ValueError("prepared records must have baseline_seed=true")
    if record.get("success") is not True or record.get("baseline_eligible") is not True:
        raise ValueError("prepared records must be successful and baseline eligible")
    if record.get("comparison_status") != STATUS_PASS:
        raise ValueError("prepared records must have comparison_status=PASS")
    if record.get("baseline_seed_source_status") != STATUS_CALIBRATION_NEEDED:
        raise ValueError("prepared records must retain CALIBRATION_NEEDED source provenance")
    if not str(record.get("baseline_seed_reason") or "").strip():
        raise ValueError("prepared records must retain a non-empty approval rationale")
    if (
        record.get("baseline_seed_source_success") is not True
        or record.get("baseline_seed_source_run_source") != "scheduled_main"
        or record.get("baseline_seed_source_branch") != "main"
        or record.get("baseline_seed_source_test_scope") != "full"
        or _truthy_pr_number(record.get("baseline_seed_source_pr_number"))
    ):
        raise ValueError("prepared records must retain trusted scheduled-main source provenance")


def upload_prepared_seed_manifest(manifest_path: str) -> str:
    """Conditionally upload one reviewed manifest in a single Hub commit."""
    manifest_path = os.path.realpath(os.path.abspath(manifest_path))
    manifest = _load_json(manifest_path)
    if manifest.get("manifest_schema_version") != 1:
        raise ValueError("unsupported or missing prepared-seed manifest schema")
    expected_repository = _repository_descriptor()
    if manifest.get("repository") != expected_repository:
        raise ValueError("prepared-seed manifest repository does not match the configured Hub destination")

    manifest_staging_root = manifest.get("staging_root")
    if not isinstance(manifest_staging_root, str) or not manifest_staging_root:
        raise ValueError("prepared-seed manifest is missing its staging root")
    staging_root = os.path.realpath(os.path.abspath(manifest_staging_root))
    source_entries = manifest.get("source_records")
    if not isinstance(source_entries, list) or not source_entries:
        raise ValueError("prepared-seed manifest must retain its reviewed source records")
    source_paths: list[str] = []
    source_records: list[dict[str, Any]] = []
    for entry in source_entries:
        if not isinstance(entry, dict):
            raise ValueError("prepared-seed manifest source entries must be objects")
        path = os.path.realpath(os.path.abspath(str(entry.get("path") or "")))
        with open(path, "rb") as f:
            blob = f.read()
        if hashlib.sha256(blob).hexdigest() != entry.get("sha256"):
            raise ValueError(f"source record changed after review: {path}")
        record = json.loads(blob)
        if not isinstance(record, dict):
            raise ValueError(f"source record must contain a JSON object: {path}")
        _validate_calibration_source(record)
        source_paths.append(path)
        source_records.append(record)
    _validate_unique_sources(source_paths, source_records)
    source_identity = _validate_same_identity(source_records)
    reservation = _reservation_path(staging_root, source_identity)
    if os.path.dirname(manifest_path) != reservation:
        raise ValueError("prepared-seed manifest is outside its identity reservation")

    prepared_entries = manifest.get("prepared_records")
    if not isinstance(prepared_entries, list) or not prepared_entries:
        raise ValueError("prepared-seed manifest must contain at least one record")

    prepared_paths: list[str] = []
    prepared_blobs: list[bytes] = []
    records: list[dict[str, Any]] = []
    for entry in prepared_entries:
        if not isinstance(entry, dict):
            raise ValueError("prepared-seed manifest record entries must be objects")
        path = os.path.realpath(os.path.abspath(str(entry.get("path") or "")))
        if os.path.commonpath((reservation, path)) != reservation:
            raise ValueError(f"prepared record is outside its identity reservation: {path}")
        with open(path, "rb") as f:
            blob = f.read()
        if hashlib.sha256(blob).hexdigest() != entry.get("sha256"):
            raise ValueError(f"prepared record changed after review: {path}")
        record = json.loads(blob)
        if not isinstance(record, dict):
            raise ValueError(f"prepared record must contain a JSON object: {path}")
        prepared_paths.append(path)
        prepared_blobs.append(blob)
        records.append(record)

    for record in records:
        _validate_prepared_seed(record)
    if len(set(prepared_paths)) != len(prepared_paths):
        raise ValueError("prepared-seed manifest repeats a prepared record path")
    identity = _validate_same_identity(records)
    if source_identity != identity:
        raise ValueError("prepared records do not match their reviewed source identity")
    if manifest.get("identity") != identity:
        raise ValueError("prepared records no longer match the manifest identity")
    if manifest.get("recipe_cohort") != _recipe_cohort_filters(records[0]):
        raise ValueError("prepared records no longer match the manifest recipe cohort")

    batch_size = len(records)
    if len(source_records) != batch_size:
        raise ValueError("prepared-seed manifest source and prepared batch sizes differ")
    if len({str(record.get("baseline_seed_reason")) for record in records}) != 1:
        raise ValueError("prepared records have inconsistent approval rationales")
    raw_batch_indices = [record.get("baseline_seed_batch_index") for record in records]
    if any(record.get("baseline_seed_batch_size") != batch_size for record in records):
        raise ValueError("prepared records have inconsistent baseline seed batch sizes")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in raw_batch_indices):
        raise ValueError("prepared records have invalid baseline seed batch indices")
    batch_indices = sorted(raw_batch_indices)
    if batch_indices != list(range(1, batch_size + 1)):
        raise ValueError("prepared records have inconsistent baseline seed batch indices")
    for path, record in zip(prepared_paths, records, strict=True):
        index = record["baseline_seed_batch_index"]
        suffix = f"{index:02d}" if batch_size > 1 else None
        if path != _seed_record_path(reservation, record, suffix=suffix):
            raise ValueError(f"prepared record path does not match its reserved identity: {path}")

    ordered_sources = _order_sources_by_timestamp(source_paths, source_records)
    records_by_index = sorted(records, key=lambda record: record["baseline_seed_batch_index"])
    for index, ((source_path, source_record), record) in enumerate(
        zip(ordered_sources, records_by_index, strict=True),
        start=1,
    ):
        expected = build_baseline_seed_record(
            source_record,
            reason=str(record["baseline_seed_reason"]),
            source_result=source_path,
            operator=record.get("baseline_seed_operator"),
            timestamp=str(record["timestamp"]),
            batch_size=batch_size,
            batch_index=index,
        )
        if record != expected:
            raise ValueError(f"prepared record {index} no longer matches its reviewed source")

    token = resolve_hf_token()
    if not token:
        raise RuntimeError("a Hugging Face write token is required to upload baseline seeds")
    api = HfApi(token=token)
    repo_info = api.repo_info(repo_id=HF_REPO_ID, repo_type="dataset", revision="main")
    parent_commit = getattr(repo_info, "sha", None)
    if not parent_commit:
        raise RuntimeError("could not resolve the current performance-tracking repository revision")

    operations = []
    with tempfile.TemporaryDirectory(prefix="performance-seed-remote-") as remote_root:
        sync_from_hf(remote_root, strict=True, revision=parent_commit)
        _validate_no_existing_seed(remote_root, records[0], source="remote tracking history")

        seen_destinations: set[str] = set()
        for path, blob, record in zip(prepared_paths, prepared_blobs, records, strict=True):
            destination = f"{sanitize(record['model_id'])}/{os.path.basename(path)}"
            if destination in seen_destinations:
                raise ValueError(f"prepared records collide at Hub path {destination}")
            if os.path.exists(os.path.join(remote_root, destination)):
                raise ValueError(f"Hub path already exists: {destination}")
            seen_destinations.add(destination)
            operations.append(CommitOperationAdd(path_in_repo=destination, path_or_fileobj=blob))

    commit = api.create_commit(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision="main",
        parent_commit=parent_commit,
        create_pr=False,
        operations=operations,
        commit_message=f"Perf: seed {records[0]['model_id']} baseline",
    )

    commit_id = getattr(commit, "oid", None) or getattr(commit, "commit_id", None)
    commit_id = str(commit_id or commit)
    try:
        _write_json(os.path.join(os.path.dirname(manifest_path), "uploaded"), {
            "commit_id": commit_id,
            "parent_commit": parent_commit,
        })
    except OSError as exc:
        print(f"Warning: seed commit {commit_id} succeeded but local upload marker failed: {exc}")
    return commit_id


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
        help=(
            "Operator tracking-mirror path used only to enforce staging separation; "
            "remote validation uses a fresh temporary snapshot."
        ),
    )
    parser.add_argument(
        "--staging-root",
        default=STAGING_ROOT,
        help="Separate local root for prepared seed records.",
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
    _validate_separate_roots(args.tracking_root, args.staging_root)
    source_records = [_load_json(path) for path in args.source_results]
    for source_record in source_records:
        _validate_calibration_source(source_record)
    _validate_unique_sources(args.source_results, source_records)
    identity = _validate_same_identity(source_records)
    with tempfile.TemporaryDirectory(prefix="performance-seed-prepare-remote-") as remote_root:
        sync_from_hf(remote_root, strict=True)
        _validate_no_existing_seed(remote_root, source_records[0], source="remote tracking history")
    _validate_no_existing_seed(args.staging_root, source_records[0], source="local staging")
    _validate_batch_consistency(source_records, args.max_intra_batch_regression)
    ordered_sources = _order_sources_by_timestamp(args.source_results, source_records)
    print("Seeding exact comparable identity:")
    for key, value in identity.items():
        print(f"  {key}: {value}")

    seed_records = []
    for index, (source_path, source_record) in enumerate(ordered_sources, start=1):
        seed_record = build_baseline_seed_record(
            source_record,
            reason=args.intent_rationale,
            source_result=os.path.realpath(os.path.abspath(source_path)),
            operator=args.operator,
            batch_size=len(source_records),
            batch_index=index,
        )
        suffix = f"{index:02d}" if len(source_records) > 1 else None
        seed_records.append((seed_record, suffix))

    reservation = _reserve_staging_identity(args.staging_root, identity)
    prepared_seeds = [
        (_seed_record_path(reservation, seed_record, suffix=suffix), seed_record, suffix)
        for seed_record, suffix in seed_records
    ]
    prepared_paths = [seed_path for seed_path, _seed_record, _suffix in prepared_seeds]
    try:
        for seed_path, seed_record, suffix in prepared_seeds:
            write_seed_record(reservation, seed_record, suffix=suffix)
            print(f"Prepared baseline seed: {seed_path}")
        manifest_path = _write_preparation_manifest(
            reservation,
            args.staging_root,
            identity,
            args.source_results,
            source_records,
            prepared_paths,
        )
    except Exception:
        _release_failed_reservation(reservation, prepared_paths)
        raise
    print(f"Reserved exact identity in staging: {reservation}")
    print(f"Prepared upload manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
