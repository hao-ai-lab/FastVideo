# SPDX-License-Identifier: Apache-2.0
"""Opt-in, process-local startup timings without importing or initializing CUDA."""
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import os
import time
from typing import Any


@dataclass
class StartupTimings:
    """Nested wall-clock intervals; parent durations include their children.

    Without a synchronization callback these measure host submission time only.
    Passing a callback (for example torch.cuda.synchronize) waits at both phase
    boundaries and changes execution/overlap; use a separate uninstrumented run
    for an end-to-end latency comparison.
    """
    synchronize: Callable[[], None] | None = field(default=None, repr=False)
    phases: list[dict[str, Any]] = field(default_factory=list)
    _start: float = field(default_factory=time.perf_counter, repr=False)
    _end: float | None = field(default=None, repr=False)
    _next_id: int = field(default=0, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "pid": os.getpid(),
            "clock": "perf_counter",
            "timing_mode": "synchronized_boundaries" if self.synchronize else "host",
            "recording_seconds": (self._end if self._end is not None else time.perf_counter()) - self._start,
            "phases": sorted(self.phases, key=lambda item: item["id"]),
        }


_active: ContextVar[StartupTimings | None] = ContextVar("fastvideo_startup_timings", default=None)
_stack: ContextVar[tuple[tuple[StartupTimings, int], ...]] = ContextVar("fastvideo_startup_stack", default=())


@contextmanager
def collect_startup_timings(*, synchronize: Callable[[], None] | None = None) -> Iterator[StartupTimings]:
    """Collect only work in this context/process. Spawned workers need their own collector."""
    timings = StartupTimings(synchronize=synchronize)
    token = _active.set(timings)
    try:
        yield timings
    finally:
        timings._end = time.perf_counter()
        _active.reset(token)


@contextmanager
def startup_phase(name: str) -> Iterator[None]:
    """Record a named phase when collection is enabled; otherwise just execute it."""
    timings = _active.get()
    if timings is None:
        yield
        return
    if timings.synchronize is not None:
        timings.synchronize()
    phase_id = timings._next_id
    timings._next_id += 1
    stack = _stack.get()
    parent = next((index for owner, index in reversed(stack) if owner is timings), None)
    token = _stack.set((*stack, (timings, phase_id)))
    start = time.perf_counter()
    status = "error"
    try:
        yield
        if timings.synchronize is not None:
            timings.synchronize()
        status = "ok"
    finally:
        timings.phases.append({
            "id": phase_id,
            "parent_id": parent,
            "name": name,
            "start_seconds": start - timings._start,
            "duration_seconds": time.perf_counter() - start,
            "status": status,
        })
        _stack.reset(token)
