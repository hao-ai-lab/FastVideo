"""Structured cancellation.

Cancellation is a common-path action, not exceptional. Cancel takes effect at the
next loop step boundary: the driver checks the scope between steps, drops queued
work, releases LoopState + cache handles, and reports ``cancelled``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CancelKind(str, Enum):
    REQUEST = "request"
    LOOP = "loop"
    STREAM = "stream"
    SESSION = "session"


class Cancelled(Exception):
    """Raised at a step boundary when a CancelScope is tripped."""


@dataclass
class CancelScope:
    kind: CancelKind = CancelKind.REQUEST
    target_id: str = ""
    _flag: list[bool] = field(default_factory=lambda: [False])  # shared mutable cell

    def cancel(self) -> None:
        self._flag[0] = True

    @property
    def cancelled(self) -> bool:
        return self._flag[0]

    def check(self) -> None:
        """Called by the driver at each step boundary."""
        if self._flag[0]:
            raise Cancelled(f"{self.kind.value}:{self.target_id}")
