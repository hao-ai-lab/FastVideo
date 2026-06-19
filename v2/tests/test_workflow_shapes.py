"""Non-linear workflow shapes — fan-out + best-of-N feedback.

Every workflow so far was a linear chain. These exercise the two non-linear shapes:

  * **Fan-out** (`ParallelWorkflow`): one input → N models in parallel → merged result (namespaced by
    branch). The engine can interleave the branches' steps.
  * **Best-of-N** (`BestOfNWorkflow`): generate N candidates, score each with a reward scorer, return
    the best — inference-time scaling / rejection sampling, a feedback loop composing a generator with
    the served REWARD_BATCH card.
"""
from __future__ import annotations

import numpy as np

from v2.recipes import build_default_engine, build_reward_card
from v2.program import BestOfNWorkflow, ParallelWorkflow, WorkflowStage
from v2.request import DiffusionParams, TaskType, make_request
from v2.training.methods.base import new_instance
from v2.training.rewards import ServedRewardScorer


def _t2v(model_id):
    return lambda s: make_request(TaskType.T2V, model_id, s["prompt"],
                                  diffusion=DiffusionParams(num_steps=4, seed=s.get("seed", 0)))


def test_fan_out_runs_branches_and_merges():
    eng = build_default_engine()
    wf = ParallelWorkflow("variants.fanout", [
        WorkflowStage("wan2.1-1.3b", _t2v("wan2.1-1.3b"), label="wan"),
        WorkflowStage("ltx2-2stage-distilled", _t2v("ltx2-2stage-distilled"), label="ltx"),
    ])
    out = wf.run(eng, prompt="a comet over the sea", seed=1)
    # both branches' artifacts present, namespaced by label
    assert "wan:video" in out.artifacts and "ltx:video" in out.artifacts
    wan_v = np.asarray(out.artifacts["wan:video"].frames)
    ltx_v = np.asarray(out.artifacts["ltx:video"].frames)
    assert wan_v.ndim == 4 and ltx_v.ndim == 4
    # two different models on the same prompt → different videos (genuine fan-out, not duplication)
    assert wan_v.shape != ltx_v.shape or not np.array_equal(wan_v, ltx_v)
    assert wf.requires == ["wan2.1-1.3b", "ltx2-2stage-distilled"]


def _best_of_n(n=4):
    eng = build_default_engine()
    scorer = ServedRewardScorer(new_instance(build_reward_card(batch=2)))
    wf = BestOfNWorkflow("wan.best_of_n", WorkflowStage("wan2.1-1.3b", _t2v("wan2.1-1.3b")),
                         scorer=scorer, n=n, score_key="latents")
    return eng, scorer, wf


def test_best_of_n_returns_the_highest_scored_candidate():
    eng, scorer, wf = _best_of_n(n=4)
    out = wf.run(eng, prompt="a fox", seed=2)
    assert out.metrics["best_of_n"] == 4.0
    # re-score the n candidates independently and confirm the returned one is the argmax
    cands = [eng.run(make_request(TaskType.T2V, "wan2.1-1.3b", "a fox",
                                  diffusion=DiffusionParams(num_steps=4, seed=2 * 100 + i))) for i in range(4)]
    scores = scorer.score([np.asarray(c.artifacts["latents"].latent) for c in cands], ["a fox"] * 4)["avg"]
    assert out.metrics["best_score"] == float(np.max(scores))
    assert out.metrics["best_index"] == float(int(np.argmax(scores)))


def test_best_of_n_is_deterministic():
    eng, _, wf = _best_of_n(n=3)
    a = wf.run(eng, prompt="a city", seed=5).metrics["best_index"]
    b = wf.run(eng, prompt="a city", seed=5).metrics["best_index"]
    assert a == b


def test_best_of_n_requires_only_the_generator():
    _, _, wf = _best_of_n()
    assert wf.requires == ["wan2.1-1.3b"]      # the reward scorer is a held instance, not a chained stage
