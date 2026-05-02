# SPDX-License-Identifier: Apache-2.0
"""Shared HuggingFace storage utilities for performance tracking.

Provides a single place for:
- Syncing the HF dataset repo to a local directory
- Loading raw JSON records (with optional recency filter)
- Loading records as a normalized pandas DataFrame
- Uploading individual result files back to HF
- Common helpers: sanitize, safe_float
"""

import glob
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from huggingface_hub import HfApi, snapshot_download

# ---------------------------------------------------------------------------
# Configuration — read once at import time, shared across both consumers
# ---------------------------------------------------------------------------

HF_REPO_ID: str = os.environ.get("HF_REPO_ID", "FastVideo/performance-tracking")
HF_TOKEN: str | None = os.environ.get("HF_API_KEY")

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def sanitize(value: str) -> str:
    """Return a filesystem- and HF-path-safe version of *value*."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def safe_float(value: Any) -> float | None:
    """Coerce *value* to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# HF I/O
# ---------------------------------------------------------------------------


def sync_from_hf(local_dir: str, *, strict: bool = False) -> str:
    """Download the HF dataset repo snapshot to *local_dir*.

    Returns *local_dir* so callers can chain: ``load_records(sync_from_hf(...))``.

    By default (``strict=False``) failures are logged and *local_dir* is
    returned unchanged, so dashboard / PR consumers stay resilient when HF is
    unavailable. Callers that depend on the sync for correctness (e.g. the
    main-branch baseline writer) must pass ``strict=True`` so that misconfig
    or transient HF errors fail loud rather than silently reset the baseline.
    """
    if not HF_REPO_ID:
        msg = "hf_store: HF_REPO_ID not set"
        if strict:
            raise RuntimeError(f"{msg}; cannot sync.")
        print(f"{msg}, skipping sync.")
        return local_dir

    print(f"hf_store: syncing from {HF_REPO_ID} → {local_dir}")
    try:
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=local_dir,
            token=HF_TOKEN,
            allow_patterns="*.json",
        )
    except Exception as exc:
        if strict:
            raise
        print(f"hf_store: sync skipped — {exc}")

    return local_dir


def upload_record(
    local_path: str,
    record: dict[str, Any],
    *,
    strict: bool = False,
) -> None:
    """Upload *local_path* to the HF repo under ``<model_id>/<filename>``.

    By default failures (missing token, network errors) are logged and
    swallowed. Pass ``strict=True`` when the upload is part of a write-path
    that must not silently lose records — otherwise the rolling baseline can
    stop advancing without any signal in the build log.
    """
    if not HF_TOKEN:
        msg = "hf_store: HF_API_KEY not set"
        if strict:
            raise RuntimeError(f"{msg}; cannot upload.")
        print(f"{msg}, skipping upload.")
        return

    model_id = record.get("model_id", "unknown")
    path_in_repo = f"{sanitize(model_id)}/{os.path.basename(local_path)}"
    commit_sha = (record.get("commit_sha") or "unknown")[:7]

    api = HfApi(token=HF_TOKEN)
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"Perf: {model_id} at {commit_sha}",
        )
        print(f"hf_store: uploaded → {HF_REPO_ID}/{path_in_repo}")
    except Exception as exc:
        if strict:
            raise
        print(f"hf_store: upload failed — {exc}")


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------


def load_records(
    local_dir: str,
    *,
    days: int | None = None,
    successful_only: bool = False,
) -> list[dict[str, Any]]:
    """Return raw JSON dicts from *local_dir*.

    Args:
        local_dir: Root directory previously populated by :func:`sync_from_hf`.
        days: When set, discard records whose ``timestamp`` is older than this
            many days. Records with a missing/unparsable timestamp are kept.
        successful_only: When True, only records with ``success=True`` are
            returned. Useful when building a regression baseline.

    Returns:
        List of raw dicts sorted by ``timestamp`` ascending (records that could
        not be parsed are silently skipped).
    """
    cutoff: datetime | None = None
    if days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    records: list[dict[str, Any]] = []

    for path in sorted(glob.glob(os.path.join(local_dir, "**", "*.json"), recursive=True)):
        try:
            with open(path, encoding="utf-8") as fh:
                data: dict[str, Any] = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue

        if successful_only and not data.get("success", True):
            continue

        if cutoff is not None:
            raw_ts = data.get("timestamp")
            if raw_ts:
                try:
                    ts = datetime.fromisoformat(raw_ts)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts < cutoff:
                        continue
                except ValueError:
                    pass  # keep records with unparsable timestamps

        records.append(data)

    return records


def load_records_for_model(
    local_dir: str,
    model_id: str,
    gpu_type: str | None = None,
    *,
    last_n: int | None = None,
    successful_only: bool = True,
) -> list[dict[str, Any]]:
    """Return records for a specific *model_id*, optionally filtered by GPU.

    Args:
        local_dir: Root directory previously populated by :func:`sync_from_hf`.
        model_id: Matches the ``model_id`` field inside each JSON record.
        gpu_type: When set, only records whose ``gpu_type`` matches are returned.
        last_n: When set, return only the most recent *n* records (after all
            other filters). Useful for sliding-window baseline calculations.
        successful_only: Passed through to :func:`load_records`.

    Returns:
        List of matching dicts sorted by timestamp ascending.
    """
    model_dir = os.path.join(local_dir, sanitize(model_id))
    if not os.path.isdir(model_dir):
        return []

    records = load_records(model_dir, successful_only=successful_only)

    if gpu_type is not None:
        records = [r for r in records if r.get("gpu_type") == gpu_type]

    if last_n is not None:
        records = records[-last_n:]

    return records


# ---------------------------------------------------------------------------
# DataFrame helpers (dashboard / analytics consumers)
# ---------------------------------------------------------------------------

_NUMERIC_COLS = ("latency", "throughput", "memory")


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard type coercions to a raw records DataFrame.

    - Parses ``timestamp`` to UTC-aware datetime.
    - Coerces ``latency``, ``throughput``, ``memory`` to float.
    - Adds a ``config_id`` column (first 7 chars of ``commit_sha``).

    Returns the mutated DataFrame (also modifies in place for efficiency).
    """
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["config_id"] = df.get("commit_sha", pd.Series(dtype=str)).fillna("unknown").str[:7]

    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_as_dataframe(
    local_dir: str,
    *,
    days: int | None = None,
    successful_only: bool = False,
) -> pd.DataFrame:
    """Load and normalize records from *local_dir* into a pandas DataFrame.

    Combines :func:`load_records` + :func:`normalize_dataframe` into a single
    call for consumers (e.g. the dashboard) that work exclusively with
    DataFrames.

    Args:
        local_dir: Root directory previously populated by :func:`sync_from_hf`.
        days: Passed through to :func:`load_records`.
        successful_only: Passed through to :func:`load_records`.

    Returns:
        Normalized DataFrame, or an empty DataFrame if no records were found.
    """
    records = load_records(local_dir, days=days, successful_only=successful_only)
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return normalize_dataframe(df)
