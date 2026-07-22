"""The catalog — one place that maps names to (card, pipeline builder).

Identity policy (three levels):

* ``model_id`` is the PRIMARY identity — it names a *servable* (weights plus
  loop semantics, precision regime, conventions, defaults). One HF repo can
  back many cards (base / distilled / precision variants) and one card can
  draw from many repos (the aligned Wan card loads the official-layout DiT
  and the diffusers-layout encoder/VAE), so repo strings cannot be identities.
* HF repo strings are *ingredients* (``card.weights``, ``ComponentSpec.source``)
  accepted here as a convenience alias — but only while unambiguous. As soon
  as two cards reference the same repo, resolving by that repo fails with the
  candidate list; it never picks silently.
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


def _build_aliases(cards: dict[str, Any]) -> dict[str, list[str]]:
    """HF repo string -> model_ids referencing it (as card weights or as a
    per-component source). Values with several ids are ambiguous aliases."""
    out: dict[str, list[str]] = {}
    for card in cards.values():
        refs = {card.weights} | {s.source for s in card.components.values() if s.source}
        for ref in sorted(refs):
            ids = out.setdefault(ref, [])
            if card.model_id not in ids:
                ids.append(card.model_id)
    return out


_ALIASES = _build_aliases(CARDS)


def resolve(name: str) -> tuple[ModelCard, Callable[[], Any]]:
    """model_id (primary), or an unambiguous HF repo alias -> (card, pipeline
    builder)."""
    if name in CARDS:
        return CARDS[name], PIPELINES[name]
    hits = _ALIASES.get(name, [])
    if len(hits) == 1:
        return CARDS[hits[0]], PIPELINES[hits[0]]
    if hits:
        raise KeyError(f"{name!r} is a weights repo referenced by several cards "
                       f"{hits} — load by model_id")
    raise KeyError(f"unknown model {name!r}; known ids: {sorted(CARDS)}; "
                   f"known repo aliases: {sorted(_ALIASES)}")
