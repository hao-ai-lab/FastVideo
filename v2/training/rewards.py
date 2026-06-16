"""Reward layer (design_v3 §10; verl-omni's shape): weighted multi-reward, duck-typed scorers.

> Landed ``train/methods/rl/rewards/`` is the right seed: thin PickScore/CLIP scorers, duck-typed
> ``RewardScorer = Callable[[media, prompts], Tensor]``, weighted aggregation in MultiRewardScorer.

Toy scorers here are deterministic functions of (media, prompt) — enough to exercise group-relative
advantages and the reward→advantage→update loop without a real reward model.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..cache.keys import content_hash

# RewardScorer: (list of media arrays, list of prompts) -> per-sample scores
RewardScorer = Callable[[list, list], np.ndarray]


def toy_pickscore(media: list, prompts: list) -> np.ndarray:
    """Deterministic 'preference' score: blends a prompt-seeded target with media statistics."""
    out = []
    for m, p in zip(media, prompts):
        target = np.random.default_rng(int(content_hash(p)[:8], 16)).standard_normal(1)[0]
        score = -abs(float(np.mean(np.asarray(m))) - 0.1 * target)
        out.append(score)
    return np.asarray(out, dtype="float64")


def toy_clipscore(media: list, prompts: list) -> np.ndarray:
    out = [float(np.tanh(np.std(np.asarray(m)))) for m in media]
    return np.asarray(out, dtype="float64")


_REGISTRY: dict[str, RewardScorer] = {"pickscore": toy_pickscore, "clipscore": toy_clipscore}


class MultiRewardScorer:
    """Weighted aggregation of named scorers (design_v3 §10)."""

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
