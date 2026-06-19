"""LoopRunner — the only place iteration lives.

The runtime's driver, factored so the engine can either run it to completion
(``run``) or advance it one step at a time (``peek`` + ``step``) to **interleave** the steps
of concurrent requests. Per-request state is entirely in ``LoopState``; the runner holds no
hidden iteration state beyond a cached pending plan, so interleaving is safe by construction.
"""
from __future__ import annotations

from typing import Any

from v2.loop.contracts import Done, LoopContext, LoopResult, LoopState, WorkPlan


class LoopRunner:

    def __init__(self, loop: Any, ctx: LoopContext, request: Any, model: Any):
        self.loop = loop
        self.ctx = ctx
        self.state: LoopState = loop.init(request, model, ctx)
        self._done = False
        self._result: LoopResult | None = None
        self._pending: WorkPlan | None = None

    @property
    def done(self) -> bool:
        return self._done

    @property
    def result(self) -> LoopResult | None:
        return self._result

    def peek(self) -> WorkPlan | None:
        """Compute (and cache) the next WorkPlan. ``next`` is kernel-free, so this is cheap.
        Returns None when the loop is finished (and runs ``finalize`` exactly once)."""
        if self._done:
            return None
        if self._pending is None:
            self.ctx.check_cancel()
            nxt = self.loop.next(self.state)
            if isinstance(nxt, Done):
                self._result = self.loop.finalize(self.state)
                self._done = True
                return None
            self._pending = nxt
        return self._pending

    def step(self) -> bool:
        """Execute the pending plan (the inversion point) and fold the result in.
        Returns True when the loop has just finished."""
        plan = self.peek()
        if self._done or plan is None:
            return True
        # the engine binds the current state onto ctx so interceptors see per-request state
        if hasattr(self.ctx, "bind_state"):
            self.ctx.bind_state(self.state)
        result = self.ctx.execute(plan)
        for chunk in plan.emits:
            self.ctx.emit(chunk)
        self.state = self.loop.advance(self.state, result)
        self._pending = None
        return self._done

    def run(self) -> LoopResult:
        while not self._done:
            self.step()
        assert self._result is not None
        return self._result
