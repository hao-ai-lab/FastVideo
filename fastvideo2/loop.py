"""The driven-loop contract — iteration the runtime can see.

A loop is a state machine the runner drives:

    state = loop.init(request, instance, inputs)
    while True:
        plan = loop.next(state)          # describe the next step; NO kernels run here
        if isinstance(plan, Done):
            break
        result = plan.run()              # the runner executes the thunk
        state = loop.advance(state, result)
    outputs = loop.finalize(state)

Two properties this buys, both load-bearing for the product:

* every denoise step is a named unit (``plan.label``) — the identity chain
  ``request/stage/loop.step`` reaches profilers (NVTX) and the trace without
  any model code knowing about either;
* all per-request mutable state lives in ``LoopState``, never on the loop
  object — a loop instance is shareable and a session/rollout engine can hold
  many states against one instance.

Loop classes declare a ``semantics`` id (e.g. ``"wan.flow_euler.cfg/v1"``).
Cards pin the semantics their weights assume; ``ModelCard.validate()`` enforces
the match. Rename-proof, and a distilled student cannot silently run under a
base sampler.

This module is stdlib-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


@dataclass
class LoopState:
    """All per-request mutable loop state. Nothing else may hold any."""
    loop_id: str
    request_id: str
    step_idx: int = 0
    latents: Any = None
    sigmas: list[float] = field(default_factory=list)
    cond: dict[str, Any] = field(default_factory=dict)
    trajectory: list[Any] = field(default_factory=list)  # filled when capture is on
    scratch: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkPlan:
    """A typed description of the next step. ``run`` is the kernel thunk,
    built by ``next()`` but not called there — planning stays kernel-free."""
    label: str                      # e.g. "denoise.7" — joins the identity chain
    step: int
    run: Callable[[], dict]
    meta: dict[str, Any] = field(default_factory=dict)


class Done:
    """Sentinel returned by ``next`` when the loop is finished."""


@runtime_checkable
class Loop(Protocol):
    """The model-owned control flow. Four methods; ``next`` is kernel-free."""
    semantics: str

    def init(self, request: Any, instance: Any, inputs: Mapping[str, Any]) -> LoopState: ...
    def next(self, state: LoopState) -> "WorkPlan | Done": ...
    def advance(self, state: LoopState, result: dict) -> LoopState: ...
    def finalize(self, state: LoopState) -> dict: ...


class LoopRunner:
    """Drives one loop for one request. ``observe(label, seconds, meta)`` is the
    trace hook; the engine uses it for NVTX ranges and step timings."""

    def __init__(self, loop: Loop, request: Any, instance: Any, inputs: Mapping[str, Any],
                 observe: Callable[[str, float, dict], None] | None = None):
        self.loop = loop
        self.state = loop.init(request, instance, inputs)
        self.observe = observe
        self._done = False
        self._outputs: dict | None = None

    @property
    def done(self) -> bool:
        return self._done

    def step(self) -> bool:
        """Advance one step. Returns True when the loop has just finished."""
        if self._done:
            return True
        plan = self.loop.next(self.state)
        if isinstance(plan, Done):
            self._outputs = self.loop.finalize(self.state)
            self._done = True
            return True
        import time
        nvtx = _nvtx_push(plan.label)
        t0 = time.perf_counter()
        try:
            result = plan.run()
        finally:
            _nvtx_pop(nvtx)
        if self.observe is not None:
            self.observe(plan.label, time.perf_counter() - t0, plan.meta)
        self.state = self.loop.advance(self.state, result)
        return False

    def run(self) -> dict:
        while not self._done:
            self.step()
        assert self._outputs is not None
        return self._outputs


def _nvtx_push(label: str) -> bool:
    """Open an NVTX range for one step when CUDA is live. Because the engine
    already opened a ``request/stage`` range, step ranges nest under it — the
    NVTX hierarchy *is* the identity chain, with zero model-code awareness."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(label)
            return True
    except ImportError:
        pass
    return False


def _nvtx_pop(pushed: bool) -> None:
    if pushed:
        import torch
        torch.cuda.nvtx.range_pop()


def build_loop(spec: Any) -> Loop:
    """Construct the loop a ``LoopSpec`` declares: resolve the class ref,
    apply the (plain-data) params."""
    from fastvideo2.card import resolve_ref
    cls = resolve_ref(spec.loop)
    return cls(loop_id=spec.loop_id, **spec.params)
