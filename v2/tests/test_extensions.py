"""Observers + interceptors. Per-request interceptor state is interleave-safe; a module-global
version provably is NOT — which is why state is scoped to ``LoopState.plugin_state`` per request
and per CFG branch.
"""
from __future__ import annotations

from v2.runtime.extend import (
    InterceptorChain,
    InterceptorConflict,
    NaNWatch,
    ResidualSkipInterceptor,
)
from v2.recipes import build_default_engine
from v2.core.request import DiffusionParams, TaskType, make_request
from v2.runtime import Engine


def _req(prompt, seed):
    return make_request(TaskType.T2V, "wan2.1-1.3b", prompt,
                        diffusion=DiffusionParams(num_steps=6, seed=seed))


def test_residual_skip_interceptor_skips():
    eng = build_default_engine(Engine(interceptors=InterceptorChain([ResidualSkipInterceptor(interval=2)])))
    out = eng.run(_req("c", 3))
    assert out.metrics.get("skipped_steps", 0) > 0      # the cache-dit-style skip actually fired


def test_conflicting_distribution_altering_interceptors_rejected():
    try:
        InterceptorChain([ResidualSkipInterceptor(), ResidualSkipInterceptor()])
        assert False, "two step-skippers must be rejected pre-flight"
    except InterceptorConflict:
        pass


def test_exact_mode_rejects_distribution_altering():
    try:
        InterceptorChain([ResidualSkipInterceptor()], exact_mode=True)
        assert False
    except InterceptorConflict:
        pass


def test_nanwatch_clean_on_good_run():
    eng = build_default_engine()
    nan = NaNWatch()
    eng.observers.add(nan)
    eng.run(_req("x", 1))
    assert not nan.tripped              # clean run, no NaNs
