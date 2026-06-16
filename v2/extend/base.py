"""Observers and interceptors (design_v3 §11) — the optimization/debug/parity surface.

They compose with §5 cleanly: the hooks wrap ``ctx.execute(plan)``.
  * Observers (read-only): cannot mutate state. An unused hook is *literally absent* from
    the hot path (the bus only iterates when observers are attached).
  * Interceptors (compute-altering): ``before_step`` may supply a cached prediction (skip);
    ``after_step`` updates calibration state. State lives in ``LoopState.plugin_state[id]``,
    keyed per request AND per CFG branch — the structural fix for module-global residual
    state that corrupts cache-dit/TeaCache forks under concurrency.

Trust boundary: plugins are enabled at deploy scope only (registry), never via a per-request
``plugins=[...]`` field. Requests only *parameterize* pre-enabled plugins.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Observer(Protocol):
    def observe(self, event: str, **kw) -> None: ...


@runtime_checkable
class Interceptor(Protocol):
    plugin_id: str
    distribution_altering: bool
    graph_safe: bool

    def before_step(self, plan: Any, state: Any) -> Any | None: ...   # return override output to skip, or None
    def after_step(self, plan: Any, state: Any, result: Any) -> None: ...


class ObserverBus:
    """Read-only event fan-out. Cheap when empty (the absent-hook rule)."""

    def __init__(self, observers: list[Observer] | None = None):
        self._observers = list(observers or [])

    def add(self, observer: Observer) -> None:
        self._observers.append(observer)

    @property
    def active(self) -> bool:
        return bool(self._observers)

    def emit(self, event: str, **kw) -> None:
        if not self._observers:          # absent from the hot path when off
            return
        for obs in self._observers:
            obs.observe(event, **kw)


class InterceptorConflict(ValueError):
    pass


class InterceptorChain:
    """Ordered interceptor chain; conflicting interceptors rejected pre-flight (§11)."""

    def __init__(self, interceptors: list[Interceptor] | None = None, *, exact_mode: bool = False):
        self._chain = list(interceptors or [])
        self._validate(exact_mode)

    def _validate(self, exact_mode: bool) -> None:
        skippers = [i for i in self._chain if getattr(i, "distribution_altering", False)]
        if len(skippers) > 1:
            raise InterceptorConflict(
                f"multiple distribution-altering interceptors {[i.plugin_id for i in skippers]} "
                "conflict — only one step-skipper allowed (design_v3 §11)")
        if exact_mode and skippers:
            raise InterceptorConflict(
                f"exact-mode rejects distribution_altering interceptors {[i.plugin_id for i in skippers]}")

    @property
    def active(self) -> bool:
        return bool(self._chain)

    def before(self, plan: Any, state: Any) -> Any | None:
        for i in self._chain:
            override = i.before_step(plan, state)
            if override is not None:
                return override
        return None

    def after(self, plan: Any, state: Any, result: Any) -> None:
        for i in self._chain:
            i.after_step(plan, state, result)
