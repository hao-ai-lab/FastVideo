"""RuntimeLoopContext — the concrete inversion point (design_v3 §5.1, §11).

This is the single seam: ``execute(plan)`` runs the kernel thunk built by the model's
``next()``, wrapped by interceptors (compute-altering) and observers (read-only). The model
never sees the scheduler; the scheduler never sees the model's math. Per-request loop state
is bound here so interceptors read state from ``LoopState`` (never module globals).
"""
from __future__ import annotations

from time import perf_counter
from typing import Any

from .._enums import ExecutionProfile
from ..loop.contracts import StepResult, WorkPlan
from ..request.streams import OmniEvent, Stream, StreamChunk


class RuntimeLoopContext:
    def __init__(self, instance: Any, *, observers, interceptors, slots: dict[str, Any],
                 stream: Stream, cancel_scope, profile: ExecutionProfile, metrics: dict,
                 request_id: str):
        self.instance = instance
        self.observers = observers
        self.interceptors = interceptors
        self.slots = slots                  # the program's typed named edges
        self.stream = stream
        self.cancel_scope = cancel_scope
        self.profile = profile
        self.metrics = metrics
        self.request_id = request_id
        self.state = None                   # bound per step by the LoopRunner

    def bind_state(self, state) -> None:
        self.state = state

    def execute(self, plan: WorkPlan) -> StepResult:
        self.observers.emit("step_start", plan=plan)
        # Interceptors may supply an override for the *expensive forward* (e.g. a cached
        # prediction). The step body still runs the cheap scheduler step (combine + solver),
        # so the override changes compute, not the loop's state-transition shape.
        override = None
        if self.interceptors.active and self.state is not None:
            override = self.interceptors.before(plan, self.state)
            # only count a skip the step body will actually honor (must carry a forward result)
            if override is not None and "noise_pred" in override:
                self.metrics["skipped_steps"] = self.metrics.get("skipped_steps", 0) + 1

        t0 = perf_counter()
        out = plan.run(self.instance, override) if plan.run is not None else {}
        dt = perf_counter() - t0
        if isinstance(out, StepResult):
            result = out
            if result.actual_seconds == 0.0:
                result.actual_seconds = dt
        else:
            result = StepResult(output=dict(out), actual_seconds=dt)

        if self.interceptors.active and self.state is not None:
            self.interceptors.after(plan, self.state, result)

        self.observers.emit("step_complete", plan=plan, result=result)
        self.metrics["steps"] = self.metrics.get("steps", 0) + 1
        self.metrics["gpu_seconds"] = self.metrics.get("gpu_seconds", 0.0) + result.actual_seconds
        return result

    def emit(self, chunk: StreamChunk) -> None:
        self.stream.emit(OmniEvent(type="media.chunk", request_id=self.request_id,
                                   seq=chunk.seq, payload={"chunk": chunk}))
        self.metrics["stream_chunks"] = self.metrics.get("stream_chunks", 0) + 1

    def check_cancel(self) -> None:
        if self.cancel_scope is not None:
            self.cancel_scope.check()              # raises request.Cancelled at the step boundary

    def observe(self, event: str, **kw) -> None:
        self.observers.emit(event, **kw)
