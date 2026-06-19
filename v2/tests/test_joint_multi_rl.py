"""N-way joint RL over arbitrary experts — generalizing the UniRL stress test.

Answers "can we do joint RL over MORE than two experts, or does it need a rewrite?": no rewrite. The
substrate is already N-ready (a card holds N components/loops — Qwen-Omni shipped three; WeightSyncPlan
is per-component; get_grad_clip_targets returns a dict). Only the *method* body looped over two experts;
``JointMultiExpertRL`` loops over N. These tests drive N refiner LMs + one generator from one reward.

They also surface the real (non-architectural) subtlety — credit assignment: ``per_expert`` reward
decomposition learns cleanly; ``shared`` reward is noisier multi-agent RL. Both are method-config, not
substrate changes.
"""
from __future__ import annotations

import numpy as np

from v2.recipes.multi_expert import build_multi_expert_card, refiner_ids
from v2.training.methods import build_joint_multi_rl


def _method(n_refiners, *, joint=True, credit="per_expert"):
    card = build_multi_expert_card(n_refiners=n_refiners)
    rids = refiner_ids(n_refiners)
    targets = {rid: (i * 2 + 1) % 6 for i, rid in enumerate(rids)}     # distinct target per refiner
    m = build_joint_multi_rl(card, refiner_ids=rids, target_actions=targets, joint_generator=joint,
                             credit=credit, num_samples_per_prompt=4, num_skip_refinement=1)
    return m, rids, targets


def _train(m, iters=60):
    batch = {"prompts": ["a fox", "blue sky", "a city at night"], "seeds": [1, 2, 3]}
    last = None
    for it in range(iters):
        _, last = m.managed_train_step(batch, it)
    return last


# --- N-way learning --------------------------------------------------------------- #

def test_three_refiners_plus_generator_all_learn_per_expert_credit():
    m, rids, targets = _method(3, credit="per_expert")
    assert len(rids) == 3
    w0 = m.dit.w_x.copy()
    p0 = {r: float(m.refiner(r)._probs()[targets[r]]) for r in rids}
    last = _train(m)
    # 1) every refiner learned its OWN target (clean credit assignment)
    for r in rids:
        p1 = float(m.refiner(r)._probs()[targets[r]])
        assert p1 > 0.5 and p1 > p0[r] + 0.2, f"{r} did not learn: {p0[r]:.3f} -> {p1:.3f}"
    # 2) four experts, four INDEPENDENT weight versions
    assert last["n_experts"] == 4.0
    vers = [last[f"weights_version/{r}"] for r in rids] + [last["weights_version/transformer"]]
    assert len(set(vers)) == 4
    # 3) the generator actually moved (FlowGRPO PPO produced real updates)
    assert not np.array_equal(w0, m.dit.w_x)
    assert 0.5 < last["ppo_ratio_mean"] < 2.0


def test_shared_credit_runs_and_versions_independently():
    """Shared single reward (faithful to UniRL) over 3 experts: it RUNS and the experts version
    independently. Unlike per-expert credit, shared-reward joint RL is noisy — some experts learn,
    others may not (multi-agent credit-assignment variance). That noisiness is the honest result, not
    an architectural limit: per-expert reward decomposition (the other mode) fixes it, no rewrite."""
    m, rids, targets = _method(3, credit="shared")
    p0 = {r: float(m.refiner(r)._probs()[targets[r]]) for r in rids}
    last = _train(m, iters=60)
    assert last["credit_mode"] == "shared"
    vers = [last[f"weights_version/{r}"] for r in rids] + [last["weights_version/transformer"]]
    assert len(set(vers)) == 4                                        # independent versions regardless
    # at least one expert learns under the shared reward (the signal is real, just noisily credited)
    gained = [float(m.refiner(r)._probs()[targets[r]]) - p0[r] for r in rids]
    assert max(gained) > 0.2


# --- arbitrary N, substrate is N-ready -------------------------------------------- #

def test_scales_to_arbitrary_n_experts():
    for n in (1, 4):
        m, rids, _ = _method(n)
        _, met = m.managed_train_step({"prompts": ["x", "y"], "seeds": [1, 2]}, 0)
        assert met["n_experts"] == float(n + 1)                      # N refiners + 1 generator
        assert len(m.sync) == n + 1                                  # one WeightSyncPlan per expert
        assert len(m.get_grad_clip_targets()) == n + 1               # the dict was always N-ary


def test_prompt_only_freezes_generator_all_refiners_still_learn():
    m, rids, targets = _method(3, joint=False)
    w0 = m.dit.w_x.copy()
    last = _train(m)
    assert np.array_equal(w0, m.dit.w_x)                             # generator frozen
    assert "dit_pg_loss" not in last
    assert "weights_version/transformer" not in last
    for r in rids:                                                   # refiners still learn (prompt-only)
        assert float(m.refiner(r)._probs()[targets[r]]) > 0.5


def test_each_expert_versions_independently_text_encoder_cache_preserved():
    m, rids, _ = _method(2)
    _train(m, iters=10)
    inst = m.student
    assert inst.version_of("text_encoder") == "v0"                  # frozen encoder cache survives
    expert_vers = {inst.version_of(r) for r in rids} | {inst.version_of("transformer")}
    assert "v0" not in expert_vers                                  # every trained expert bumped
    assert len(expert_vers) == 3                                    # and each independently
