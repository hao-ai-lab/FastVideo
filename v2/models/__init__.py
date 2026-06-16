"""Model definitions — concrete (recipe, runtime) cards (design_v3 §4).

Phase 1: Wan2.1-1.3B (T2V), LTX2.3 (2-stage distilled), Wan-causal (self-forcing student).
``build_default_engine`` loads all three onto one engine (one resident instance per card).
"""
from __future__ import annotations

from typing import Any

from .bagel import build_bagel_card, build_bagel_program
from .cosmos3 import build_cosmos3_card, build_cosmos3_program
from .ltx2 import build_ltx2_card, build_ltx2_program
from .unified import build_unified_card, build_unified_program
from .wan21 import build_wan21_card, build_wan_t2v_program
from .wan_causal import build_wan_causal_card, build_wan_causal_program

__all__ = [
    "build_wan21_card", "build_wan_t2v_program",
    "build_ltx2_card", "build_ltx2_program",
    "build_wan_causal_card", "build_wan_causal_program",
    "build_cosmos3_card", "build_cosmos3_program",
    "build_bagel_card", "build_bagel_program",
    "build_unified_card", "build_unified_program",
    "build_default_engine", "build_omni_engine", "build_unified_engine",
]

_BUILDERS = [
    (build_wan21_card, build_wan_t2v_program),
    (build_ltx2_card, build_ltx2_program),
    (build_wan_causal_card, build_wan_causal_program),
]

# Phase-2 omni cards (MoT, one resident instance running AR + diffusion loops on shared weights)
_OMNI_BUILDERS = [
    (build_cosmos3_card, build_cosmos3_program),
    (build_bagel_card, build_bagel_program),
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


def build_omni_engine(engine: Any = None) -> Any:
    """Register the phase-2 omni cards (Cosmos3 + BAGEL/lance) onto one engine.

    Each is ONE resident MoT instance whose ``transformer`` runs BOTH an ar_decode loop and a
    diffusion_denoise loop (shared weights) — the §16 'true omni/MoT serving' claim made native.
    """
    from ..cache import CacheManager
    from ..card import load_card
    from ..runtime import Engine
    eng = engine if engine is not None else Engine()
    for build_card, build_program in _OMNI_BUILDERS:
        card = build_card()
        inst = load_card(card, cache_manager=CacheManager.from_card(card))
        eng.register(card.model_id, inst, build_program())
    return eng


def build_unified_engine(engine: Any = None) -> Any:
    """Register the unified LM+generator card (UniRL/PromptRL): a prompt-refiner ``llm`` expert + a
    flow ``transformer`` expert on the SAME engine, each driving its own loop. Two separate experts,
    one request, both trainable under one RL reward (the §10 joint-RL stress test)."""
    from ..cache import CacheManager
    from ..card import load_card
    from ..runtime import Engine
    eng = engine if engine is not None else Engine()
    card = build_unified_card()
    inst = load_card(card, cache_manager=CacheManager.from_card(card))
    eng.register(card.model_id, inst, build_unified_program())
    return eng
