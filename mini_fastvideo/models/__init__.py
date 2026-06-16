"""Model definitions — concrete (recipe, runtime) cards (design_v3 §4).

Phase 1: Wan2.1-1.3B (T2V), LTX2.3 (2-stage distilled), Wan-causal (self-forcing student).
``build_default_engine`` loads all three onto one engine (one resident instance per card).
"""
from __future__ import annotations

from typing import Any

from .ltx2 import build_ltx2_card, build_ltx2_program
from .wan21 import build_wan21_card, build_wan_t2v_program
from .wan_causal import build_wan_causal_card, build_wan_causal_program

__all__ = [
    "build_wan21_card", "build_wan_t2v_program",
    "build_ltx2_card", "build_ltx2_program",
    "build_wan_causal_card", "build_wan_causal_program",
    "build_default_engine",
]

_BUILDERS = [
    (build_wan21_card, build_wan_t2v_program),
    (build_ltx2_card, build_ltx2_program),
    (build_wan_causal_card, build_wan_causal_program),
]


def build_default_engine(engine: Any = None) -> Any:
    """Register Wan2.1, LTX2.3, and Wan-causal onto one engine (one resident instance each)."""
    from ..cache import CacheManager
    from ..card import load_card
    from ..runtime import Engine
    eng = engine if engine is not None else Engine()
    for build_card, build_program in _BUILDERS:
        card = build_card()
        inst = load_card(card, cache_manager=CacheManager.from_card(card))
        eng.register(card.model_id, inst, build_program())
    return eng
