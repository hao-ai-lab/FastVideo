"""AsyncEngine — the serving runtime.

A request queue, lifecycle state machine, live streaming, and step-level concurrency over the same
step scheduler. Each request runs as an asyncio task that ticks its runner one step at a time and
yields the event loop (`await asyncio.sleep(0)`), so concurrent requests interleave at step
granularity — and a per-request stall backs off instead of busy-spinning. Drives BOTH inline
``ProgramRunner`` and ``DisaggregatedRunner`` (capacity-aware pool dispatch) through one driver.

Cancellation is common-path: ``cancel(request_id)`` trips the runner's CancelScope, which raises at
the next step boundary and delivers partial artifacts + a structured ``cancelled``.
"""
from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from collections.abc import AsyncIterator

from v2.request.artifacts import Output
from v2.request.cancel import Cancelled
from v2.request.streams import OmniEvent
from v2.runtime.disaggregated import DisaggregatedRunner
from v2.runtime.engine import Engine

_SENTINEL = object()


def _safe_put(q: asyncio.Queue, item: Any) -> None:
    """Non-blocking put on an unbounded queue — safe to call from finally/cancellation paths."""
    with contextlib.suppress(Exception):
        q.put_nowait(item)


class RequestState:
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class AsyncEngine:

    def __init__(self, engine: Engine | None = None, *, max_stall_ticks: int = 200_000, max_history: int = 512):
        self.engine = engine if engine is not None else Engine()
        self._disagg: dict[str, tuple] = {}  # model_id -> (PoolSet, Program)
        self._events: dict[str, asyncio.Queue] = {}
        self._states: dict[str, str] = {}
        self._results: dict[str, Output] = {}
        self._runners: dict[str, Any] = {}
        self._order: list[str] = []  # submission order, for bounded eviction
        self.max_stall_ticks = max_stall_ticks
        self.max_history = max_history

    def _evict(self) -> None:
        """Bound retained per-request state to the most recent ``max_history`` terminal requests."""
        terminal = (RequestState.COMPLETED, RequestState.CANCELLED, RequestState.FAILED)
        i = 0
        while len(self._order) > self.max_history and i < len(self._order):
            rid = self._order[i]
            if self._states.get(rid) in terminal:
                self._order.pop(i)
                self._events.pop(rid, None)
                self._states.pop(rid, None)
                self._results.pop(rid, None)
                self._runners.pop(rid, None)
            else:
                i += 1

    # --- registration ------------------------------------------------------- #
    def register(self, model_id: str, instance: Any, program: Any) -> None:
        self.engine.register(model_id, instance, program)

    def register_disaggregated(self, model_id: str, pools: Any, program: Any) -> None:
        self._disagg[model_id] = (pools, program)

    def register_workflow(self, workflow: Any) -> Any:
        return self.engine.register_workflow(workflow)

    def is_disaggregated(self, model_id: str) -> bool:
        return model_id in self._disagg

    def _make_runner(self, request: Any) -> Any:
        if request.model_id in self._disagg:
            pools, program = self._disagg[request.model_id]
            return DisaggregatedRunner(pools,
                                       program,
                                       request,
                                       observers=self.engine.observers,
                                       interceptors=self.engine.interceptors)
        return self.engine._make_runner(request)

    # --- lifecycle / control ------------------------------------------------ #
    def state(self, request_id: str) -> str | None:
        return self._states.get(request_id)

    def cancel(self, request_id: str) -> bool:
        runner = self._runners.get(request_id)
        if runner is not None and getattr(runner, "cancel_scope", None) is not None:
            runner.cancel_scope.cancel()
            return True
        return False

    def result(self, request_id: str) -> Output | None:
        return self._results.get(request_id)

    @property
    def in_flight(self) -> int:
        return sum(1 for s in self._states.values() if s == RequestState.RUNNING)

    @property
    def queue_depth(self) -> int:
        return sum(1 for s in self._states.values() if s == RequestState.WAITING)

    def serves(self, model_id: str) -> bool:
        return (model_id in self.engine._registry or model_id in self._disagg or model_id in self.engine._workflows)

    # --- the driver: one asyncio task per request --------------------------- #
    async def _run(self, request: Any) -> None:
        rid = request.request_id
        evq = self._events[rid]
        runner = None
        stream_cursor = 0
        stall = 0
        try:
            runner = self._make_runner(request)
            self._runners[rid] = runner
            self._states[rid] = RequestState.RUNNING
            await evq.put(OmniEvent("request.start", rid, payload={"task": request.task.value}))
            while not runner.done:
                runner.cancel_scope.check()  # common-path cancellation (step boundary)
                p0 = runner._progress
                runner.tick()
                # surface any newly emitted media chunks as live events
                events = runner.stream.events
                while stream_cursor < len(events):
                    await evq.put(events[stream_cursor])
                    stream_cursor += 1
                if not runner.done and runner._progress == p0:
                    stall += 1
                    if stall > self.max_stall_ticks:
                        raise RuntimeError(f"request {rid} stalled (state={getattr(runner,'state','?')})")
                    await asyncio.sleep(0.0005)  # backpressure backoff (capacity wait)
                else:
                    stall = 0
                    await asyncio.sleep(0)  # cooperative yield → step interleaving
            out = runner.output()
            self._results[rid] = out
            for name in out.artifacts:
                await evq.put(OmniEvent("artifact.ready", rid, payload={"name": name}))
            self._states[rid] = RequestState.COMPLETED
            await evq.put(OmniEvent("request.complete", rid, payload={"metrics": dict(out.metrics)}))
        except Cancelled:
            self._states[rid] = RequestState.CANCELLED
            if runner is not None and hasattr(runner, "output"):
                try:
                    partial = runner.output()
                    partial.error = "cancelled"
                    self._results[rid] = partial
                except Exception:
                    pass
            _safe_put(evq, OmniEvent("request.cancelled", rid))
        except asyncio.CancelledError:  # driver task cancelled (e.g. client gone)
            self._states[rid] = RequestState.CANCELLED
            raise
        except Exception as e:  # noqa: BLE001 — request-fatal isolation: one request's error
            self._states[rid] = RequestState.FAILED
            self._results[rid] = Output(request_id=rid, error=str(e))
            _safe_put(evq, OmniEvent("request.failed", rid, payload={"error": str(e)}))
        finally:
            # release any pool this request occupied — a cancel/error mid-loop must NOT leak capacity
            if runner is not None and hasattr(runner, "close"):
                with contextlib.suppress(Exception):
                    runner.close()
            _safe_put(evq, _SENTINEL)
            self._evict()

    async def submit(self, request: Any) -> AsyncIterator[OmniEvent]:
        """Submit a request and stream its events live (request.start → media.chunk* → artifact.ready*
        → request.complete | request.cancelled | request.failed). Cancels the driver if the consumer
        abandons the stream (e.g. client disconnect), so no orphaned compute keeps running."""
        rid = request.request_id
        if rid in self._events and self._states.get(rid) in (RequestState.WAITING, RequestState.RUNNING):
            raise ValueError(f"request_id {rid!r} is already in flight")
        self._events[rid] = asyncio.Queue()
        self._states[rid] = RequestState.WAITING
        self._order.append(rid)
        task = asyncio.create_task(self._run(request))
        evq = self._events[rid]
        try:
            while True:
                ev = await evq.get()
                if ev is _SENTINEL:
                    break
                yield ev
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def generate(self, request: Any) -> Output:
        """Non-streaming: drive to completion, return the Output (the offline path over the queue)."""
        async for _ in self.submit(request):
            pass
        return self._results[request.request_id]

    async def generate_many(self, requests: list[Any]) -> dict[str, Output]:
        """Run requests concurrently (they interleave at step granularity through the event loop)."""
        await asyncio.gather(*(self._drain(r) for r in requests))
        return {r.request_id: self._results[r.request_id] for r in requests}

    async def _drain(self, request: Any) -> None:
        async for _ in self.submit(request):
            pass
