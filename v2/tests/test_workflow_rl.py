"""End-to-end RL over a cross-model workflow (design_v3 §10 + §13).

Probes the training-plane / Workflow boundary: can "rollout == serve + capture" span *two model
instances*, and can one final reward train an *earlier* model? These tests train both stages of the
T2I→I2V workflow (`flux-t2i` and `wan-i2v`) from a single final-video reward and assert:

  * the rollout spans both instances with SDE capture in each (a workflow-level rollout);
  * the SAME final-video advantage trains BOTH generators — including the T2I one, which never sees
    the video directly (end-to-end credit across the model boundary);
  * that credit is *caused by the video reward* — a constant reward (zero advantage) moves nothing;
  * the two stages version independently (two WeightSyncPlans on two instances).
"""
from __future__ import annotations

import numpy as np

from v2.models import build_flux_t2i_card, build_wan_i2v_card
from v2.training.methods import build_workflow_rl


def _method(**kw):
    return build_workflow_rl(build_flux_t2i_card(), build_wan_i2v_card(),
                             num_samples_per_prompt=4, rollout_steps=4, t2i_lr=0.03, i2v_lr=0.03, **kw)


def _batch():
    return {"prompts": ["a fox", "blue sky", "a city"], "seeds": [1, 2, 3]}


def test_rollout_spans_two_instances_with_sde_capture():
    """A workflow-level rollout: stage 1 on one instance, stage 2 on another, each capturing an SDE
    trajectory (so FlowGRPO can credit both)."""
    m = _method()
    assert m.t2i is not m.i2v
    assert m.t2i.component("transformer") is not m.i2v.component("transformer")
    res_t, _emb_t, image = m._rollout_t2i("a fox", seed=1)
    res_i, _emb_i, video = m._rollout_i2v("a fox", image, seed=1)
    assert any("sde_logprob" in r for r in (res_t.behavior or []))     # T2I stage captured
    assert any("sde_logprob" in r for r in (res_i.behavior or []))     # I2V stage captured
    assert np.asarray(video).ndim == 4                                 # produced a video


def test_one_final_reward_trains_both_generators():
    m = _method()
    t2i_w0 = m.t2i.component("transformer").w_x.copy()
    i2v_w0 = m.i2v.component("transformer").w_x.copy()
    last = None
    for it in range(20):
        _, last = m.managed_train_step(_batch(), it)
    # the EARLIER model (T2I) is trained by a reward on the FINAL video — credit across the boundary
    assert not np.array_equal(t2i_w0, m.t2i.component("transformer").w_x)
    assert not np.array_equal(i2v_w0, m.i2v.component("transformer").w_x)
    assert last["grad_norm/t2i"] > 0.0 and last["grad_norm/i2v"] > 0.0
    assert 0.5 < last["ppo_ratio_mean"] < 2.0


def test_t2i_credit_is_caused_by_the_video_reward():
    """Causal check: a *constant* final reward ⇒ zero group advantage ⇒ neither generator moves. So the
    T2I movement in the test above is genuinely caused by the video reward, not by the rollout itself."""
    m = _method()
    m._reward = lambda video: 0.5                                     # constant ⇒ std 0 ⇒ advantage 0
    t2i_w0 = m.t2i.component("transformer").w_x.copy()
    i2v_w0 = m.i2v.component("transformer").w_x.copy()
    for it in range(10):
        m.managed_train_step(_batch(), it)
    assert np.array_equal(t2i_w0, m.t2i.component("transformer").w_x)  # no signal ⇒ no T2I update
    assert np.array_equal(i2v_w0, m.i2v.component("transformer").w_x)  # no signal ⇒ no I2V update


def test_stages_version_independently_on_two_instances():
    m = _method()
    _, last = m.managed_train_step(_batch(), 0)
    assert last["t2i_weights_version"] != last["i2v_weights_version"]
    assert m.t2i.version_of("transformer") != "v0"
    assert m.i2v.version_of("transformer") != "v0"
    # each stage's sync touched only its own instance's transformer (frozen text encoders intact)
    assert m.t2i.version_of("text_encoder") == "v0"
    assert m.i2v.version_of("text_encoder") == "v0"
