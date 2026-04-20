# SPDX-License-Identifier: Apache-2.0
"""Session state store for the FastVideo streaming server.

The streaming server keeps continuation state (decoded frames + audio
latents from the previous segment) server-side so the client doesn't
re-upload multi-megabyte tensors each WebSocket message. Two operations
are needed:

* ``snapshot(session_id) -> ContinuationState`` — serialize the current
  state so it can be exported (e.g. over HTTP) or migrated to a
  different server.
* ``hydrate(state) -> session_id`` — load a previously serialized state
  into a new session (for resume-after-disconnect flows).

The store is an ABC with an :class:`InMemorySessionStore` default; Redis
or other backends can drop in without touching the pipeline.

Large tensor payloads (video frames, audio latents) are kept out of the
JSON payload via an accompanying :class:`BlobStore`. Both stores share a
process today; they are separate types so that a future implementation
can put blobs on S3 while keeping session metadata in Redis.
"""
from __future__ import annotations

import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

from fastvideo.api.schema import ContinuationState


class BlobStore(ABC):
    """Opaque byte-blob storage keyed by id.

    A :class:`ContinuationState` payload can reference large tensors
    stored in a :class:`BlobStore` rather than inlining them, so the
    JSON payload stays small when the state travels over the wire.
    """

    @abstractmethod
    def put(self, data: bytes, *, mime: str = "application/octet-stream") -> str:
        """Store ``data`` and return a blob id for later retrieval."""

    @abstractmethod
    def get(self, blob_id: str) -> bytes:
        """Load a previously stored blob. Raises ``KeyError`` if absent."""

    @abstractmethod
    def drop(self, blob_id: str) -> None:
        """Remove a blob. Missing ids are a no-op."""

    @abstractmethod
    def __contains__(self, blob_id: str) -> bool:
        ...


@dataclass(frozen=True)
class _BlobRecord:
    data: bytes
    mime: str


class InMemoryBlobStore(BlobStore):
    """Thread-safe in-memory :class:`BlobStore` for single-process servers."""

    def __init__(self) -> None:
        self._blobs: dict[str, _BlobRecord] = {}
        self._lock = threading.Lock()

    def put(self, data: bytes, *, mime: str = "application/octet-stream") -> str:
        blob_id = uuid.uuid4().hex
        with self._lock:
            self._blobs[blob_id] = _BlobRecord(data=data, mime=mime)
        return blob_id

    def get(self, blob_id: str) -> bytes:
        with self._lock:
            record = self._blobs.get(blob_id)
        if record is None:
            raise KeyError(f"Unknown blob id: {blob_id}")
        return record.data

    def drop(self, blob_id: str) -> None:
        with self._lock:
            self._blobs.pop(blob_id, None)

    def __contains__(self, blob_id: str) -> bool:
        with self._lock:
            return blob_id in self._blobs

    def __len__(self) -> int:
        with self._lock:
            return len(self._blobs)


class SessionStore(ABC):
    """Keyed store for per-session continuation state.

    Implementations own the session-id → state mapping. The streaming
    server calls :meth:`store` after each segment and :meth:`snapshot`
    when a client explicitly asks for an exportable state handle.
    """

    @abstractmethod
    def store(self, session_id: str, state: ContinuationState) -> None:
        """Persist ``state`` for ``session_id``, replacing any prior value."""

    @abstractmethod
    def snapshot(self, session_id: str) -> ContinuationState | None:
        """Return the current state for ``session_id`` (or ``None``)."""

    @abstractmethod
    def hydrate(
        self,
        state: ContinuationState,
        *,
        session_id: str | None = None,
    ) -> str:
        """Install ``state`` as the starting point for a session.

        When ``session_id`` is ``None`` the store allocates a fresh id
        (UUID4); when provided the store uses it verbatim, overwriting
        any prior state at that id.
        """

    @abstractmethod
    def drop(self, session_id: str) -> None:
        """Forget a session. Missing ids are a no-op."""

    @abstractmethod
    def __contains__(self, session_id: str) -> bool:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        ...


class InMemorySessionStore(SessionStore):
    """Thread-safe in-memory :class:`SessionStore`.

    Default implementation used by single-process deployments; a future
    Redis-backed store can be dropped in without changes to the server.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, ContinuationState] = {}
        self._lock = threading.Lock()

    def store(self, session_id: str, state: ContinuationState) -> None:
        with self._lock:
            self._sessions[session_id] = state

    def snapshot(self, session_id: str) -> ContinuationState | None:
        with self._lock:
            return self._sessions.get(session_id)

    def hydrate(
        self,
        state: ContinuationState,
        *,
        session_id: str | None = None,
    ) -> str:
        sid = session_id or uuid.uuid4().hex
        with self._lock:
            self._sessions[sid] = state
        return sid

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def __contains__(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._sessions))

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


__all__ = [
    "BlobStore",
    "InMemoryBlobStore",
    "InMemorySessionStore",
    "SessionStore",
]
