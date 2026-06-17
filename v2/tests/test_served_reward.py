"""Reward-model-as-a-served-card — REWARD_BATCH work units (design_v3 §10).

In real RL the reward is a *model* (PickScore/CLIP/a VLM judge), not a heuristic. These tests make the
reward a first-class card: a `scorer` component + a `score` loop emitting `REWARD_BATCH` work units, and
a `ServedRewardScorer` that drop-in replaces the numpy scorer so any RL method becomes RLHF/RLAIF with a
learned reward — no method change. Exercises the `REWARD_BATCH` WorkUnit kind (previously zero coverage).
"""
from __future__ import annotations

import math

import numpy as np

from v2._enums import ExecutionProfile, LoopKind, WorkUnitKind
from v2.models.reward import build_reward_card
from v2.models.wan21 import build_wan21_card
from v2.request import TaskType, make_request
from v2.training.methods import build_diffusion_nft
from v2.training.methods.base import new_instance
from v2.training.rewards import ServedRewardScorer
from v2.training.rollout import rollout_loop


def _media(n):
    return [np.random.default_rng(i).standard_normal((4, 2, 4, 6)).astype("float32") for i in range(n)]


def test_reward_card_is_a_distinct_servable_with_a_reward_batch_loop():
    card = build_reward_card()
    assert "scorer" in card.components
    assert card.loops["score"].kind == LoopKind.ENCODER
    assert card.loops["score"].work_unit_kind == WorkUnitKind.REWARD_BATCH


def test_score_loop_emits_reward_batch_units():
    inst = new_instance(build_reward_card(batch=2))
    media = _media(5)
    req = make_request(TaskType.REASON, inst.card.model_id, "")
    res = rollout_loop(inst, "score", req, slots={"media": media}, profile=ExecutionProfile.SERVE)
    assert len(res.outputs["rewards"]) == 5                       # one score per sample
    assert res.metrics["reward_batches"] == math.ceil(5 / 2)      # batched into REWARD_BATCH units


def test_served_scorer_interface_and_determinism():
    scorer = ServedRewardScorer(new_instance(build_reward_card()))
    media = _media(4)
    out = scorer.score(media, ["p"] * 4)
    assert set(out) >= {"avg"} and out["avg"].shape == (4,)       # MultiRewardScorer-compatible
    assert np.array_equal(out["avg"], scorer.score(media, ["p"] * 4)["avg"])   # deterministic


def test_served_reward_drives_an_rl_method():
    """Swap the heuristic scorer for the served reward MODEL — DiffusionNFT runs unchanged and the
    learned reward drives a real group-relative advantage."""
    nft = build_diffusion_nft(build_wan21_card(), num_video_per_prompt=4, num_inner_timesteps=2)
    nft.scorer = ServedRewardScorer(new_instance(build_reward_card(batch=2)))
    loss, m = nft.train_step({"prompts": ["a red car", "a blue boat"], "seeds": [1, 2]}, 0)
    assert np.isfinite(loss["policy_loss"])
    assert m["advantage_std"] > 0.0                               # the served reward produced signal
