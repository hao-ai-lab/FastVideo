"""Observers + interceptors (design_v3 §11). The headline: per-request interceptor state is
interleave-safe; a module-global version provably is NOT — which is *why* §11 scopes state to
``LoopState.plugin_state`` per request and per CFG branch.
"""
from __future__ import annotations

from mini_fastvideo._enums import WorkUnitKind
from mini_fastvideo.card import CostModel
from mini_fastvideo.extend import (
    InterceptorChain,
    InterceptorConflict,
    NaNWatch,
    Profiler,
    ResidualSkipInterceptor,
)
from mini_fastvideo.models import build_default_engine
from mini_fastvideo.parity import assert_interleave_parity
from mini_fastvideo.request import DiffusionParams, TaskType, make_request
from mini_fastvideo.runtime import Engine


def _req(prompt, seed):
    return make_request(TaskType.T2V, "wan2.1-1.3b", prompt,
                        diffusion=DiffusionParams(num_steps=6, seed=seed))


def test_residual_skip_interceptor_is_interleave_safe_and_skips():
    eng = build_default_engine(Engine(interceptors=InterceptorChain([ResidualSkipInterceptor(interval=2)])))
    assert not assert_interleave_parity(eng, [_req("a", 1), _req("b", 2)])
    out = eng.run(_req("c", 3))
    assert out.metrics.get("skipped_steps", 0) > 0      # the cache-dit-style skip actually fired


class _GlobalResidualBug:
    """A DELIBERATELY BUGGY interceptor: caches in a CLASS global keyed only by branch (the bug the
    cache-dit/TeaCache forks have). Under interleaving, request A reads request B's cached forward."""
    plugin_id = "global_bug"
    distribution_altering = True
    graph_safe = False
    _CACHE: dict = {}

    def before_step(self, plan, state):
        b = str(plan.payload.get("branch", "combined"))
        cached = self._CACHE.get(b)
        if cached is not None and state.step_idx % 2 != 0:
            return {"noise_pred": cached}
        return None

    def after_step(self, plan, state, result):
        self._CACHE[str(plan.payload.get("branch", "combined"))] = result.output.get("noise_pred")


def test_global_state_interceptor_breaks_interleave_parity():
    _GlobalResidualBug._CACHE.clear()
    eng = build_default_engine(Engine(interceptors=InterceptorChain([_GlobalResidualBug()])))
    divs = assert_interleave_parity(eng, [_req("alpha", 1), _req("beta", 2)])
    assert divs, "module-global interceptor state MUST break interleave parity (the §11 hazard)"


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


def test_profiler_calibrates_cost_model_and_nanwatch_clean():
    eng = build_default_engine()
    prof, nan = Profiler(), NaNWatch()
    eng.observers.add(prof)
    eng.observers.add(nan)
    eng.run(_req("x", 1))
    assert len(prof.samples) > 0
    cm = CostModel(kind=WorkUnitKind.DIFFUSION_STEP)
    prof.calibrate(cm)
    assert cm.base_seconds >= 0.0
    assert not nan.tripped              # clean run, no NaNs
