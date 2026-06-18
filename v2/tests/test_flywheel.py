"""The (recipe, runtime) flywheel — RL → distill → faster card (design_v3 §16; §9.18).

design_v3's central product claim, made concrete: RL the base (DiffusionNFT), then distill from the
RL'd model (a DMD2 student whose teacher IS the RL'd policy), producing a faster card whose recipe
records the provenance chain base → rl → distilled. Both stages drive the same denoise loop the engine
serves — the flywheel is real, not aspirational.
"""
from __future__ import annotations

import numpy as np

from v2.recipes import build_wan_t2v_program
from v2.recipes.wan21 import build_wan21_card
from v2.request import DiffusionParams, TaskType, make_request
from v2.runtime import Engine
from v2.training.flywheel import run_flywheel
from v2.training.methods.base import new_instance


def _w(inst):
    return inst.component("transformer").w_x


def test_flywheel_chains_rl_then_distill():
    base = build_wan21_card()
    base_w = _w(new_instance(base)).copy()
    fw = run_flywheel(base, rl_iters=8, distill_iters=20)
    assert not np.allclose(base_w, _w(fw["rl"]))                 # RL moved the base policy
    assert np.allclose(_w(fw["teacher"]), _w(fw["rl"]))          # the distill teacher IS the RL'd model
    assert not np.allclose(base_w, _w(fw["distilled"]))          # the student distilled (moved)


def test_distilled_student_approaches_the_rl_teacher():
    base = build_wan21_card()
    base_w = _w(new_instance(base)).copy()
    fw = run_flywheel(base, rl_iters=8, distill_iters=20)
    teach_w, dist_w = _w(fw["teacher"]), _w(fw["distilled"])
    # the distilled student is closer to the RL'd teacher than the (un-distilled) base is
    assert np.linalg.norm(dist_w - teach_w) < np.linalg.norm(base_w - teach_w)


def test_provenance_chain_is_recorded():
    base = build_wan21_card()
    fw = run_flywheel(base, rl_iters=4, distill_iters=6)
    assert fw["rl_card"].recipe.method == "diffusion_nft"
    assert fw["rl_card"].recipe.parents == ["wan2.1-1.3b"]                 # rl ← base
    assert fw["distilled_card"].recipe.method == "dmd2"
    assert fw["distilled_card"].recipe.parents == ["wan2.1-1.3b-rl"]       # distilled ← rl
    # the full auditable chain: base → rl → distilled
    assert fw["distilled_card"].model_id == "wan2.1-1.3b-rl-distilled"


def test_distilled_card_is_a_servable_few_step_pair():
    base = build_wan21_card()
    fw = run_flywheel(base, rl_iters=4, distill_iters=6)
    card, inst = fw["distilled_card"], fw["distilled"]
    eng = Engine()
    eng.register(card.model_id, inst, build_wan_t2v_program())             # the distilled (recipe,runtime) pair serves
    out = eng.run(make_request(TaskType.T2V, card.model_id, "a fox",
                               diffusion=DiffusionParams(num_steps=4, seed=1)))   # few-step
    assert out.artifacts["video"].frames is not None
    assert out.metrics["denoise_steps"] == 4.0
