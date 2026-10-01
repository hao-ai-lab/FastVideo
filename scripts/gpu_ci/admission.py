# SPDX-License-Identifier: Apache-2.0
"""Persistent GPU admission for cooperating processes on one trusted host.

Keep this SQLite database on local disk, not a shared/network filesystem.
Reservations include backend handles before external creation. Neither stale
heartbeats nor cancellation free active reservations: the caller must stop the
backend worker, confirm it stopped, and then call ``release``. The dispatcher
must also serialize creation and recovery for each build outside this ledger.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class AdmissionStore:
    """Bound distinct PRs and GPU reservations across all selected backends.

    ``pr_key`` identifies a repository and PR together. ``None`` or an empty
    string gives a non-PR build its own slot. Once admitted, all unfinished
    builds for a PR share and retain its slot, even between lanes. Completed
    build and lane records remain as tombstones; retries must use new lane IDs.
    """

    def __init__(self, path: str | Path, max_prs: int = 2, max_gpus_per_pr: int = 4, max_gpus: int = 8):
        for name, value in (("max_prs", max_prs), ("max_gpus_per_pr", max_gpus_per_pr), ("max_gpus", max_gpus)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if str(path) == ":memory:":
            raise ValueError("Admission requires a persistent database on local disk")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_prs = max_prs
        self.max_gpus_per_pr = max_gpus_per_pr
        self.max_gpus = max_gpus
        with self._transaction() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    schema_version INTEGER NOT NULL,
                    max_prs INTEGER NOT NULL,
                    max_gpus_per_pr INTEGER NOT NULL,
                    max_gpus INTEGER NOT NULL
                )
            """)
            connection.execute("INSERT OR IGNORE INTO settings VALUES (1, 1, ?, ?, ?)",
                               (max_prs, max_gpus_per_pr, max_gpus))
            settings = connection.execute("SELECT * FROM settings WHERE id = 1").fetchone()
            expected = (1, max_prs, max_gpus_per_pr, max_gpus)
            actual = tuple(settings[name] for name in ("schema_version", "max_prs", "max_gpus_per_pr", "max_gpus"))
            if actual != expected:
                raise ValueError("Admission database schema or limits conflict with this process")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS builds (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    build_id TEXT NOT NULL UNIQUE,
                    pr_key TEXT,
                    workload_key TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('queued', 'admitted', 'finished')),
                    cancelled INTEGER NOT NULL DEFAULT 0 CHECK (cancelled IN (0, 1)),
                    created_at REAL NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    admitted_at REAL,
                    finished_at REAL
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS allocations (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    build_id TEXT NOT NULL REFERENCES builds(build_id),
                    lane_id TEXT NOT NULL,
                    gpus INTEGER NOT NULL CHECK (gpus > 0),
                    handle_json TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('waiting', 'active', 'released', 'cancelled')),
                    created_at REAL NOT NULL,
                    acquired_at REAL,
                    released_at REAL,
                    UNIQUE (build_id, lane_id)
                )
            """)
            connection.execute("CREATE INDEX IF NOT EXISTS builds_workload ON builds(workload_key, state)")
            connection.execute("CREATE INDEX IF NOT EXISTS allocations_state ON allocations(state, sequence)")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _identifier(value: str, name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")

    @staticmethod
    def _build(connection: sqlite3.Connection, build_id: str) -> sqlite3.Row:
        build = connection.execute("SELECT * FROM builds WHERE build_id = ?", (build_id, )).fetchone()
        if build is None:
            raise KeyError(f"Unknown build: {build_id}")
        return build

    def register(self, build_id: str, pr_key: str | None, backend: str, commit: str) -> None:
        """Register once; an identical duplicate never resets state or queue age."""
        for name, value in (("build_id", build_id), ("backend", backend), ("commit", commit)):
            self._identifier(value, name)
        if pr_key is not None and not isinstance(pr_key, str):
            raise ValueError("pr_key must be a string or None")
        pr_key = pr_key or None
        if pr_key is not None:
            self._identifier(pr_key, "pr_key")
        workload_key = f"pr:{pr_key}" if pr_key is not None else f"build:{build_id}"
        with self._transaction() as connection:
            existing = connection.execute("SELECT * FROM builds WHERE build_id = ?", (build_id, )).fetchone()
            if existing is not None:
                if (existing["pr_key"], existing["backend"], existing["commit_sha"]) != (pr_key, backend, commit):
                    raise ValueError(f"Build identity changed: {build_id}")
                return
            active = connection.execute("SELECT 1 FROM builds WHERE workload_key = ? AND state = 'admitted'",
                                        (workload_key, )).fetchone()
            now = time.time()
            connection.execute("""
                INSERT INTO builds (build_id, pr_key, workload_key, backend, commit_sha,
                                    state, created_at, heartbeat_at, admitted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (build_id, pr_key, workload_key, backend, commit,
                  "admitted" if active else "queued", now, now, now if active else None))

    def try_admit(self, build_id: str) -> bool:
        """Admit the oldest waiting PRs, reserving space for earlier waiters."""
        with self._transaction() as connection:
            build = self._build(connection, build_id)
            if build["cancelled"] or build["state"] == "finished":
                return False
            if build["state"] == "admitted":
                return True
            active_count = connection.execute(
                "SELECT COUNT(DISTINCT workload_key) FROM builds WHERE state = 'admitted'").fetchone()[0]
            available = self.max_prs - active_count
            if available <= 0:
                return False
            eligible = connection.execute("""
                SELECT workload_key FROM builds WHERE state = 'queued' AND cancelled = 0
                GROUP BY workload_key ORDER BY MIN(sequence) LIMIT ?
            """, (available, )).fetchall()
            if build["workload_key"] not in {row["workload_key"] for row in eligible}:
                return False
            connection.execute("""
                UPDATE builds SET state = 'admitted', admitted_at = ?
                WHERE workload_key = ? AND state = 'queued' AND cancelled = 0
            """, (time.time(), build["workload_key"]))
            return True

    def try_acquire(self, build_id: str, lane_id: str, gpus: int, handle: dict[str, Any]) -> bool:
        """Persist a handle and reserve GPUs before the caller creates a worker.

        Pending requests keep FIFO order. A PR whose oldest request cannot fit
        its own budget does not block the other PR. Once a request fits its PR
        budget it waits for global capacity without smaller requests bypassing
        it. Repeating an active request returns True with the same reservation;
        released/cancelled lane IDs cannot acquire again.
        """
        self._identifier(lane_id, "lane_id")
        if type(gpus) is not int or not 1 <= gpus <= min(self.max_gpus_per_pr, self.max_gpus):
            raise ValueError("Requested GPUs must fit both the per-PR and total GPU limits")
        if not isinstance(handle, dict):
            raise ValueError("Backend handle must be a JSON object")
        try:
            encoded_handle = json.dumps(handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("Backend handle must be JSON serializable") from error
        with self._transaction() as connection:
            build = self._build(connection, build_id)
            allocation = connection.execute("SELECT * FROM allocations WHERE build_id = ? AND lane_id = ?",
                                            (build_id, lane_id)).fetchone()
            if allocation is not None:
                if allocation["gpus"] != gpus or allocation["handle_json"] != encoded_handle:
                    raise ValueError(f"Lane reservation identity changed: {build_id}/{lane_id}")
            if build["cancelled"] or build["state"] != "admitted":
                return False
            if allocation is not None and allocation["state"] != "waiting":
                return allocation["state"] == "active"
            if allocation is None:
                connection.execute("""
                    INSERT INTO allocations (build_id, lane_id, gpus, handle_json, state, created_at)
                    VALUES (?, ?, ?, ?, 'waiting', ?)
                """, (build_id, lane_id, gpus, encoded_handle, time.time()))

            usage = connection.execute("""
                SELECT b.workload_key, SUM(a.gpus) AS gpus FROM allocations a
                JOIN builds b ON b.build_id = a.build_id WHERE a.state = 'active'
                GROUP BY b.workload_key
            """).fetchall()
            gpu_usage = {row["workload_key"]: row["gpus"] for row in usage}
            total_gpus = sum(gpu_usage.values())
            waiting = connection.execute("""
                SELECT a.*, b.workload_key FROM allocations a
                JOIN builds b ON b.build_id = a.build_id
                WHERE a.state = 'waiting' AND b.state = 'admitted' AND b.cancelled = 0
                ORDER BY a.sequence
            """).fetchall()
            seen_workloads = set()
            for request in waiting:
                workload_key = request["workload_key"]
                if workload_key in seen_workloads:
                    continue
                seen_workloads.add(workload_key)
                if gpu_usage.get(workload_key, 0) + request["gpus"] > self.max_gpus_per_pr:
                    continue
                if (request["build_id"], request["lane_id"]) != (build_id, lane_id):
                    return False
                if total_gpus + gpus > self.max_gpus:
                    return False
                connection.execute("""
                    UPDATE allocations SET state = 'active', acquired_at = ?
                    WHERE build_id = ? AND lane_id = ?
                """, (time.time(), build_id, lane_id))
                return True
            return False

    def release(self, build_id: str, lane_id: str) -> None:
        """Release only after the caller confirms the worker has stopped.

        Missing/already released reservations are harmless for reconciliation.
        This method does not stop workers or infer their status from timestamps.
        """
        with self._transaction() as connection:
            self._build(connection, build_id)
            connection.execute("""
                UPDATE allocations SET state = 'released', released_at = ?
                WHERE build_id = ? AND lane_id = ? AND state IN ('active', 'waiting')
            """, (time.time(), build_id, lane_id))

    def finish(self, build_id: str) -> None:
        """Finish or cancel a queued build, refusing any active reservations."""
        with self._transaction() as connection:
            build = self._build(connection, build_id)
            if build["state"] == "finished":
                return
            active = connection.execute("SELECT 1 FROM allocations WHERE build_id = ? AND state = 'active'",
                                        (build_id, )).fetchone()
            if active is not None:
                raise RuntimeError(f"Cannot finish build with active GPU allocations: {build_id}")
            now = time.time()
            connection.execute("""
                UPDATE allocations SET state = 'cancelled', released_at = ?
                WHERE build_id = ? AND state = 'waiting'
            """, (now, build_id))
            connection.execute("UPDATE builds SET state = 'finished', finished_at = ? WHERE build_id = ?",
                               (now, build_id))

    def heartbeat(self, build_id: str) -> None:
        """Record liveness for operators; no timeout ever frees a reservation."""
        with self._transaction() as connection:
            self._build(connection, build_id)
            connection.execute("UPDATE builds SET heartbeat_at = ? WHERE build_id = ? AND state != 'finished'",
                               (time.time(), build_id))

    def request_cancel(self, build_id: str) -> None:
        """Block further admission/launch reservations without freeing GPUs."""
        with self._transaction() as connection:
            self._build(connection, build_id)
            connection.execute("UPDATE builds SET cancelled = 1 WHERE build_id = ?", (build_id, ))
            connection.execute("""
                UPDATE allocations SET state = 'cancelled', released_at = ?
                WHERE build_id = ? AND state = 'waiting'
            """, (time.time(), build_id))

    def is_cancelled(self, build_id: str) -> bool:
        with self._transaction() as connection:
            return bool(self._build(connection, build_id)["cancelled"])

    def snapshot(self) -> dict[str, Any]:
        """Return a consistent reconciliation view, including terminal records."""
        with self._transaction() as connection:
            builds = [dict(row) for row in connection.execute("SELECT * FROM builds ORDER BY sequence")]
            allocations = [dict(row) for row in connection.execute("SELECT * FROM allocations ORDER BY sequence")]
        for build in builds:
            build["commit"] = build.pop("commit_sha")
            build["cancelled"] = bool(build["cancelled"])
        for allocation in allocations:
            allocation["handle"] = json.loads(allocation.pop("handle_json"))
        return {
            "limits": {
                "max_prs": self.max_prs,
                "max_gpus_per_pr": self.max_gpus_per_pr,
                "max_gpus": self.max_gpus,
            },
            "builds": builds,
            "allocations": allocations,
        }
