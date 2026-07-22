"""The catalog — one place that maps names to (card, pipeline builder).

Model ids and HF repo ids resolve here; there is deliberately no second
catalog anywhere else. Adding a model family = its package declares card
constants and a pipeline builder, and this module lists them.
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

_HF_IDS: dict[str, str] = {c.weights: c.model_id for c in CARDS.values()}


def resolve(name: str) -> tuple[ModelCard, Callable[[], Any]]:
    """model_id or HF repo id -> (card, pipeline builder)."""
    model_id = _HF_IDS.get(name, name)
    if model_id not in CARDS:
        known = sorted(CARDS) + sorted(_HF_IDS)
        raise KeyError(f"unknown model {name!r}; known: {known}")
    return CARDS[model_id], PIPELINES[model_id]
