"""The catalog — one place that maps names to (card, pipeline builder).

Identity policy (two levels, nothing else resolves):

* ``model_id`` is THE identity — it names a *servable* (weights plus loop
  semantics, precision regime, conventions, defaults). HF repo strings are
  ingredients on the card (``card.weights``, ``ComponentSpec.source``), never
  load keys: one repo can back many cards (base / distilled / precision
  variants) and one card can draw from many repos (the aligned Wan card loads
  the official-layout DiT and the diffusers-layout encoder/VAE).
* ``card.digest()`` is the machine identity: evidence records, T1 baselines,
  and environment manifests key on it, never on names.

There is deliberately no second catalog anywhere else. Adding a model family
= its package declares card constants and a pipeline builder, listed here.
"""
from __future__ import annotations

from typing import Any, Callable

from fastvideo2.card import ModelCard
from fastvideo2.wan21 import WAN21_T2V_1_3B, build_wan_t2v_pipeline

CARDS: dict[str, ModelCard] = {
    WAN21_T2V_1_3B.model_id: WAN21_T2V_1_3B,
}

PIPELINES: dict[str, Callable[[], Any]] = {
    WAN21_T2V_1_3B.model_id: build_wan_t2v_pipeline,
}


def resolve(model_id: str) -> tuple[ModelCard, Callable[[], Any]]:
    """model_id -> (card, pipeline builder). Only model ids resolve — pass an
    HF repo string and you get the catalog, not a guess."""
    if model_id not in CARDS:
        raise KeyError(f"unknown model {model_id!r}; known ids: {sorted(CARDS)} "
                       f"(HF repo strings are card ingredients, not load keys)")
    return CARDS[model_id], PIPELINES[model_id]
