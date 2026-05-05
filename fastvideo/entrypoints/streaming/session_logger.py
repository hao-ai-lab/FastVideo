# SPDX-License-Identifier: Apache-2.0
"""Per-session JSONL event logger.

Each session gets its own JSONL file under the configured log root so
post-hoc analytics (enhancer latency, GPU assignment, segment timings)
can be recovered without a tracing backend. The internal UI uses this
format; keeping the same shape makes log tooling portable.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, TextIO

_FILENAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]")


@dataclass
class SessionLogEvent:
    """One line in the session JSONL file."""

    session_id: str
    event: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class SessionLogger:
    """Append-only JSONL logger keyed by session id.

    Thread-safe; the server may be writing from multiple asyncio tasks
    (fMP4 encoder thread + control-frame handler) for the same session.
    """

    def __init__(self, log_dir: str | None) -> None:
        self._log_dir = log_dir
        self._files: dict[str, TextIO] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()
        self._ensure_dir()

    def log(self, event: SessionLogEvent) -> None:
        if self._log_dir is None:
            return
        opened = self._get_file(event.session_id)
        if opened is None:
            return
        handle, lock = opened
        line = json.dumps({
            "session_id": event.session_id,
            "event": event.event,
            "ts": event.ts,
            "payload": event.payload,
        })
        with lock, contextlib.suppress(ValueError):
            handle.write(line + "\n")
            handle.flush()

    def close(self, session_id: str) -> None:
        with self._registry_lock:
            handle = self._files.pop(session_id, None)
            lock = self._locks.pop(session_id, None)
        if handle is None or lock is None:
            return
        with lock, contextlib.suppress(Exception):
            handle.close()

    def close_all(self) -> None:
        with self._registry_lock:
            sids = list(self._files)
        for sid in sids:
            self.close(sid)

    def _ensure_dir(self) -> None:
        if self._log_dir is None:
            return
        os.makedirs(self._log_dir, exist_ok=True)

    def _get_file(self, session_id: str) -> tuple[TextIO, threading.Lock] | None:
        if self._log_dir is None:
            return None
        with self._registry_lock:
            handle = self._files.get(session_id)
            lock = self._locks.get(session_id)
            if handle is not None and lock is not None:
                return handle, lock
            # Defense-in-depth: session_id is server-generated UUID today,
            # but sanitize against path traversal in case future code paths
            # allow client-supplied ids.
            safe_id = _FILENAME_SANITIZE_RE.sub("_", session_id) or "unknown"
            path = os.path.join(
                self._log_dir,
                f"session-{safe_id}.jsonl",
            )
            try:
                handle = open(path, "a", encoding="utf-8")  # noqa: SIM115
            except OSError:
                return None
            lock = threading.Lock()
            self._files[session_id] = handle
            self._locks[session_id] = lock
            return handle, lock


__all__ = [
    "SessionLogEvent",
    "SessionLogger",
]
