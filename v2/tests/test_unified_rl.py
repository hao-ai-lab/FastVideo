"""UniRL/PromptRL joint LM+generator RL — the design stress test (design_v3 §4, §10; arXiv 2510.17937).

These tests assert the *design holds*: a joint two-expert RL recipe fits inside the existing
Card/Loop/Program + training-plane vocabulary with no new primitive. Specifically:

  * the unified card is two SEPARATE experts (llm + transformer) on two loops — the topological
    opposite of the Cosmos3 MoT card (one shared module on two loops), same vocabulary;
  * the serve path is unchanged ODE; SDE-with-logprob capture is gated on ``sde_rollout``;
  * likelihood-BASED C2: a per-step log-prob identity (the other half of NFT's likelihood-free C2);
  * one reward → two updates (LM token-PG + DiT FlowGRPO-PPO) under one group-relative advantage;
  * two WeightSyncPlans version + cache-scope the experts independently;
  * prompt-only mode freezes the generator;
  * the two-loop serve program still passes the interleave parity gate.
"""
from __future__ import annotations

import numpy as np

from v2._enums import ExecutionProfile, LoopKind
from v2.loop.sampler import flow_sde_step_with_logprob
from v2.recipes import build_unified_engine
from v2.recipes.common import cached_text_encode
from v2.recipes.unified import build_unified_card
from v2.parity import assert_interleave_parity
from v2.request import DiffusionParams, TaskType, make_request
from v2.training.methods import build_unified_rl
from v2.training.rollout import rollout_loop

C2_TOL = 1e-5            # float32 trajectory-storage precision (bf16/fp32 replay buffers in practice)


def _inst(eng):
    return eng._registry["unirl-qwenflux"][0]


def _rollout(inst, prompt, *, sde, seed=7, steps=4):
    emb = cached_text_encode(inst, prompt)
    neg = cached_text_encode(inst, "")
    req = make_request(TaskType.T2V, inst.card.model_id, prompt,
                       diffusion=DiffusionParams(num_steps=steps, seed=seed, guidance_scale=1.0,
                                                 sde_rollout=sde))
    return rollout_loop(inst, "diffusion_denoise", req,
                        slots={"text_embeds": emb, "neg_text_embeds": neg},
                        profile=ExecutionProfile.ROLLOUT), emb


# --- card / program shape --------------------------------------------------------- #

def test_unified_card_validates_and_serves():
    eng = build_unified_engine()
    req = make_request(TaskType.T2V, "unirl-qwenflux", "a red fox in snow")
    res = eng.run(req)
    assert res.artifacts["video"].frames is not None
    assert np.asarray(res.artifacts["video"].frames).ndim == 4


def test_two_separate_experts_not_a_shared_mot():
    """UniRL keeps the refiner and generator as DISTINCT experts — the two loops bind different
    components. (Contrast Cosmos3, where ar_decode + diffusion_denoise bind the SAME transformer.)"""
    card = build_unified_card()
    assert card.loops["ar_decode"].kind == LoopKind.AR_DECODE
    assert card.loops["diffusion_denoise"].kind == LoopKind.DIFFUSION_DENOISE
    assert card.loops["ar_decode"].shared_weight_components == ["llm"]
    assert card.loops["diffusion_denoise"].shared_weight_components == ["transformer"]
    eng = build_unified_engine()
    inst = _inst(eng)
    assert inst.component("llm") is not inst.component("transformer")     # genuinely separate weights


# --- serve ODE vs rollout SDE (gated, §9.4) --------------------------------------- #

def test_serve_is_ode_rollout_is_sde():
    eng = build_unified_engine()
    inst = _inst(eng)
    res_ode, _ = _rollout(inst, "a fox", sde=False)
    res_sde, _ = _rollout(inst, "a fox", sde=True)
    assert all("sde_logprob" not in r for r in res_ode.behavior)          # serve sampler: no log-probs
    sde_steps = [r for r in res_sde.behavior if "sde_logprob" in r]
    assert sde_steps, "SDE rollout must capture per-step log-probs"
    for r in sde_steps:                                                   # the PPO slice is complete
        assert {"sde_logprob", "prev", "sample", "sigma_t", "sigma_next"} <= set(r)


def test_sde_rollout_is_deterministic_given_seed():
    eng = build_unified_engine()
    inst = _inst(eng)
    a, _ = _rollout(inst, "a fox", sde=True, seed=123)
    b, _ = _rollout(inst, "a fox", sde=True, seed=123)
    assert np.array_equal(a.outputs["latents"], b.outputs["latents"])     # seeded ⇒ bit-identical
    assert [r["sde_logprob"] for r in a.behavior if "sde_logprob" in r] == \
           [r["sde_logprob"] for r in b.behavior if "sde_logprob" in r]


def test_c2_likelihood_identity_at_rollout_weights():
    """Likelihood-BASED C2: recomputing each step's log-prob under the UNCHANGED rollout weights
    reproduces the captured log-prob (⇒ PPO ratio == 1). This is the parity gate this method
    lives on, the counterpart to NFT's likelihood-FREE C2."""
    eng = build_unified_engine()
    inst = _inst(eng)
    res, emb = _rollout(inst, "a fox", sde=True, seed=7)
    dit = inst.component("transformer")
    max_diff = 0.0
    for r in res.behavior:
        if "sde_logprob" not in r:
            continue
        v = dit(r["prev"], emb, r["sigma_t"])
        _, logp, _, _ = flow_sde_step_with_logprob(r["prev"], v, r["sigma_t"], r["sigma_next"],
                                                   prev_sample=r["sample"])
        max_diff = max(max_diff, abs(logp - r["sde_logprob"]))
    assert max_diff < C2_TOL, f"C2 likelihood identity broken: {max_diff:.2e}"


# --- joint vs prompt-only RL ------------------------------------------------------ #

def _train(joint, target, iters=40, **kw):
    m = build_unified_rl(build_unified_card(), joint=joint, target_action=target,
                         num_samples_per_prompt=4, num_skip_refinement=1, **kw)
    batch = {"prompts": ["a fox", "blue sky", "a city at night"], "seeds": [1, 2, 3]}
    p_start = float(m.llm._probs()[target])
    last = None
    for it in range(iters):
        _, last = m.managed_train_step(batch, it)
    return m, p_start, float(m.llm._probs()[target]), last


def test_joint_rl_updates_both_experts():
    m, p0, p1, last = _train(joint=True, target=3)
    # 1) the LM learned to pick the reward-favored refinement action
    assert p1 > p0 + 0.2, f"LM did not learn the target action: {p0:.3f} -> {p1:.3f}"
    # 2) the generator moved (FlowGRPO PPO produced real grads)
    assert last["grad_norm/transformer"] > 0.0
    assert "dit_pg_loss" in last and np.isfinite(last["dit_pg_loss"])
    # 3) PPO ratio is near 1 (small on-policy steps), KL finite
    assert 0.5 < last["ppo_ratio_mean"] < 2.0
    assert np.isfinite(last["kl_div_loss"])
    # 4) two experts versioned INDEPENDENTLY; the frozen text-encoder cache is NOT invalidated
    inst = m.student
    assert last["llm_weights_version"] != last["transformer_weights_version"]
    assert inst.version_of("text_encoder") == "v0"                       # §7.1 cache scope preserved
    assert inst.version_of("llm") != "v0" and inst.version_of("transformer") != "v0"


def test_prompt_only_freezes_the_generator():
    card = build_unified_card()
    m = build_unified_rl(card, joint=False, target_action=5,
                         num_samples_per_prompt=4, num_skip_refinement=1)
    w0 = m.dit.w_x.copy()
    batch = {"prompts": ["a fox", "blue sky", "a city"], "seeds": [1, 2, 3]}
    p0 = float(m.llm._probs()[5])
    last = None
    for it in range(40):
        _, last = m.managed_train_step(batch, it)
    p1 = float(m.llm._probs()[5])
    assert p1 > p0 + 0.2, "prompt-only: LM still must learn"
    assert np.array_equal(w0, m.dit.w_x), "prompt-only: generator weights must be frozen"
    assert last["transformer_weights_version"] == "frozen"
    assert "dit_pg_loss" not in last


def test_one_reward_drives_both_via_one_advantage():
    """The structural claim: a single reward per sample → one group-relative advantage → BOTH a
    token-PG (LM) and a FlowGRPO-PPO (DiT) update. Joint moves the DiT; prompt-only does not —
    yet the LM learns in both, from the same advantage."""
    _, _, p1_joint, _ = _train(joint=True, target=2)
    _, _, p1_prompt, _ = _train(joint=False, target=2)
    assert p1_joint > 0.4 and p1_prompt > 0.4                            # LM learns either way


# --- the two-loop serve program still passes the core parity gate ----------------- #

def test_unified_serve_interleave_parity():
    eng = build_unified_engine()
    reqs = [make_request(TaskType.T2V, "unirl-qwenflux", p,
                         diffusion=DiffusionParams(num_steps=4, seed=s))
            for p, s in [("alpha", 11), ("beta", 22), ("alpha", 11)]]
    divs = assert_interleave_parity(eng, reqs)
    assert not divs, f"unified two-loop program failed interleave parity: {divs}"
