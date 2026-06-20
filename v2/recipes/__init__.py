"""Model definitions — concrete (recipe, runtime) cards.

The kept v2 recipes: Wan2.1 (T2V) + Wan-causal (self-forcing student) + LTX-2 (distilled / base /
2.3 joint A/V), the omni MoT/cascade cards (Cosmos3 / BAGEL / Qwen-Omni), and the bucket-C ports
flux2 + matrixgame2 (resolved through ``v2.registry``). ``build_default_engine`` loads the core
diffusion cards onto one engine; ``build_omni_engine`` the omni cards.
"""
from __future__ import annotations

from typing import Any

from v2.recipes.bagel import build_bagel_card, build_bagel_program
from v2.recipes.cosmos3 import build_cosmos3_card, build_cosmos3_program
from v2.recipes.ltx2 import build_ltx2_av_program, build_ltx2_card, build_ltx2_program
from v2.recipes.qwen_omni import build_qwen_omni_card, build_qwen_omni_program
from v2.recipes.wan21 import build_wan21_card, build_wan_t2v_program
from v2.recipes.wan_causal import build_wan_causal_card, build_wan_causal_program

__all__ = [
    "build_wan21_card",
    "build_wan_t2v_program",
    "build_ltx2_card",
    "build_ltx2_program",
    "build_ltx2_av_program",
    "build_wan_causal_card",
    "build_wan_causal_program",
    "build_cosmos3_card",
    "build_cosmos3_program",
    "build_bagel_card",
    "build_bagel_program",
    "build_qwen_omni_card",
    "build_qwen_omni_program",
    "build_default_engine",
    "build_omni_engine",
]

_BUILDERS = [
    (build_wan21_card, build_wan_t2v_program),
    (build_ltx2_card, build_ltx2_program),
    (build_wan_causal_card, build_wan_causal_program),
]

# Omni cards: MoT shared-weight (Cosmos3 / BAGEL) + the cascaded thinker->talker->vocoder
# (Qwen-Omni, three disjoint experts / three loop types in one request).
_OMNI_BUILDERS = [
    (build_cosmos3_card, build_cosmos3_program),
    (build_bagel_card, build_bagel_program),
    (build_qwen_omni_card, build_qwen_omni_program),
]


def build_default_engine(engine: Any = None) -> Any:
    """Register Wan2.1, LTX2.3, and Wan-causal onto one engine (one resident instance each)."""
    from v2.runtime.cache import CacheManager
    from v2.core.card import load_card
    from v2.runtime import Engine
    eng = engine if engine is not None else Engine()
    for build_card, build_program in _BUILDERS:
        card = build_card()
        inst = load_card(card, cache_manager=CacheManager.from_card(card))
        eng.register(card.model_id, inst, build_program())
    return eng


def build_omni_engine(engine: Any = None) -> Any:
    """Register the omni cards (Cosmos3 + BAGEL + Qwen-Omni) onto one engine.

    Each MoT card is ONE resident instance whose ``transformer`` runs BOTH an ar_decode loop and a
    diffusion_denoise loop (shared weights) — true omni/MoT serving.
    """
    from v2.runtime.cache import CacheManager
    from v2.core.card import load_card
    from v2.runtime import Engine
    eng = engine if engine is not None else Engine()
    for build_card, build_program in _OMNI_BUILDERS:
        card = build_card()
        inst = load_card(card, cache_manager=CacheManager.from_card(card))
        eng.register(card.model_id, inst, build_program())
    return eng
