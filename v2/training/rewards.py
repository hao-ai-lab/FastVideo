"""Reward layer: weighted multi-reward over duck-typed scorers
(``RewardScorer = Callable[[media, prompts], Tensor]``).

Toy scorers here are deterministic functions of (media, prompt) — enough to exercise group-relative
advantages and the reward->advantage->update loop without a real reward model.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from v2.cache.keys import content_hash

# RewardScorer: (list of media arrays, list of prompts) -> per-sample scores
RewardScorer = Callable[[list, list], np.ndarray]


def toy_pickscore(media: list, prompts: list) -> np.ndarray:
    """Deterministic 'preference' score: blends a prompt-seeded target with media statistics."""
    out = []
    for m, p in zip(media, prompts, strict=False):
        target = np.random.default_rng(int(content_hash(p)[:8], 16)).standard_normal(1)[0]
        score = -abs(float(np.mean(np.asarray(m))) - 0.1 * target)
        out.append(score)
    return np.asarray(out, dtype="float64")


def toy_clipscore(media: list, prompts: list) -> np.ndarray:
    out = [float(np.tanh(np.std(np.asarray(m)))) for m in media]
    return np.asarray(out, dtype="float64")


_REGISTRY: dict[str, RewardScorer] = {"pickscore": toy_pickscore, "clipscore": toy_clipscore}


class MultiRewardScorer:
    """Weighted aggregation of named scorers."""

    def __init__(self, weights: dict[str, float]):
        self.weights = weights
        self.scorers = {name: _REGISTRY[name] for name in weights}

    def score(self, media: list, prompts: list) -> dict[str, np.ndarray]:
        per: dict[str, np.ndarray] = {name: fn(media, prompts) for name, fn in self.scorers.items()}
        agg = sum(self.weights[name] * per[name] for name in per)
        per["avg"] = np.asarray(agg, dtype="float64")
        return per


def build_multi_reward_scorer(config: dict[str, float]) -> MultiRewardScorer:
    return MultiRewardScorer(config)


class ServedRewardScorer:
    """A reward scorer backed by a served reward-model card.

    Drop-in for ``MultiRewardScorer`` (same ``score(media, prompts) -> {"avg": ...}`` interface), but
    instead of a numpy heuristic it runs a reward MODEL through its ``score`` loop — so the K rollout
    samples are scored as ``REWARD_BATCH`` work units (admitted, priced, interleavable). Set it as an RL
    method's ``.scorer`` to turn any method into RLHF/RLAIF with a learned reward, no method change."""

    def __init__(self, reward_instance, *, loop_id: str = "score"):
        self.inst = reward_instance
        self.loop_id = loop_id

    def score(self, media: list, prompts: list | None = None) -> dict[str, np.ndarray]:
        from v2._enums import ExecutionProfile
        from v2.request import TaskType, make_request
        from v2.training.rollout import rollout_loop
        req = make_request(TaskType.REASON, self.inst.card.model_id, "")  # task is cosmetic here
        res = rollout_loop(self.inst, self.loop_id, req, slots={"media": list(media)}, profile=ExecutionProfile.SERVE)
        rewards = np.asarray(res.outputs["rewards"], dtype="float64")
        return {"served": rewards, "avg": rewards}
