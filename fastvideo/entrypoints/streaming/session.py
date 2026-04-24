# SPDX-License-Identifier: Apache-2.0
"""Per-connection session lifecycle for the streaming server.

Each WebSocket opens exactly one :class:`Session`. :class:`SessionManager`
enforces the ``generation_segment_cap`` and ``session_timeout_seconds``
budgets from :class:`fastvideo.api.StreamingConfig`.
"""
from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastvideo.api.schema import ContinuationState


class SessionState(enum.Enum):
    """State-machine positions for a streaming session.

    Transitions are server-owned. See
    ``docs/design/server_contracts/streaming.md`` for the full diagram.
    """

    INITIALIZING = "initializing"
    QUEUED = "queued"
    GPU_BINDING = "gpu_binding"
    ACTIVE = "active"
    COMPLETE = "complete"
    ERROR = "error"
    TIMEOUT = "timeout"
    REJECTED = "rejected"


_VALID_TRANSITIONS: dict[SessionState, frozenset[SessionState]] = {
    SessionState.INITIALIZING:
    frozenset({
        SessionState.QUEUED,
        SessionState.GPU_BINDING,
        SessionState.REJECTED,
        SessionState.ERROR,
    }),
    SessionState.QUEUED:
    frozenset({
        SessionState.GPU_BINDING,
        SessionState.ERROR,
        SessionState.TIMEOUT,
        SessionState.REJECTED,
    }),
    SessionState.GPU_BINDING:
    frozenset({
        SessionState.ACTIVE,
        SessionState.ERROR,
        SessionState.TIMEOUT,
    }),
    SessionState.ACTIVE:
    frozenset({
        SessionState.ACTIVE,
        SessionState.COMPLETE,
        SessionState.ERROR,
        SessionState.TIMEOUT,
    }),
    SessionState.COMPLETE:
    frozenset(),
    SessionState.ERROR:
    frozenset(),
    SessionState.TIMEOUT:
    frozenset(),
    SessionState.REJECTED:
    frozenset(),
}


class InvalidSessionTransition(RuntimeError):
    """Raised when a session is asked to transition along an illegal edge."""


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: SessionState = SessionState.INITIALIZING
    created_at: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)

    client_id: str | None = None
    preset: str | None = None
    preset_label: str | None = None

    curated_prompts: list[str] = field(default_factory=list)

    segment_idx: int = 0

    enhancement_enabled: bool = False
    auto_extension_enabled: bool = False
    loop_generation_enabled: bool = False
    single_clip_mode: bool = False
    generation_paused: bool = False

    stream_mode: str = "av_fmp4"
    gpu_id: int | None = None

    continuation_state: ContinuationState | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def transition(self, target: SessionState) -> None:
        """Move to ``target`` if the edge is allowed.

        Raises :class:`InvalidSessionTransition` on illegal moves. The
        self-loop on ``ACTIVE`` is legal so the server can re-assert
        ACTIVE on segment completion without special casing.
        """
        allowed = _VALID_TRANSITIONS.get(self.state, frozenset())
        if target not in allowed and target is not self.state:
            raise InvalidSessionTransition(f"{self.state.value} -> {target.value} is not a valid "
                                           f"session transition")
        self.state = target
        self.last_activity = time.monotonic()

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def is_active(self) -> bool:
        return self.state is SessionState.ACTIVE

    def segment_cap_reached(self, cap: int) -> bool:
        return self.segment_idx >= cap


class SessionManager:
    """Registers sessions and enforces per-server session limits."""

    def __init__(
        self,
        *,
        segment_cap: int,
        session_timeout_seconds: int,
        max_sessions: int = 1,
    ) -> None:
        self._segment_cap = segment_cap
        self._session_timeout_seconds = session_timeout_seconds
        self._max_sessions = max_sessions
        self._sessions: dict[str, Session] = {}

    @property
    def segment_cap(self) -> int:
        return self._segment_cap

    @property
    def session_timeout_seconds(self) -> int:
        return self._session_timeout_seconds

    def create(self) -> Session:
        if len(self._sessions) >= self._max_sessions:
            raise SessionRejected(f"max sessions reached ({self._max_sessions})")
        session = Session()
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def close(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._sessions

    def __len__(self) -> int:
        return len(self._sessions)

    def active_sessions(self) -> list[Session]:
        return [s for s in self._sessions.values() if s.is_active()]

    def reap_timed_out(self, now: float | None = None) -> list[str]:
        """Return the ids of sessions that have exceeded the idle timeout.

        The caller is responsible for actually closing them — this
        method only *identifies* dead sessions so the server can emit
        ``session_timeout`` frames before dropping the WebSocket.
        """
        now = now if now is not None else time.monotonic()
        dead: list[str] = []
        for sid, session in self._sessions.items():
            if session.state in {
                    SessionState.COMPLETE,
                    SessionState.ERROR,
                    SessionState.TIMEOUT,
                    SessionState.REJECTED,
            }:
                continue
            if now - session.last_activity > self._session_timeout_seconds:
                dead.append(sid)
        return dead


class SessionRejected(RuntimeError):
    """Raised when session creation fails (queue full, auth, etc.)."""


__all__ = [
    "InvalidSessionTransition",
    "Session",
    "SessionManager",
    "SessionRejected",
    "SessionState",
]
