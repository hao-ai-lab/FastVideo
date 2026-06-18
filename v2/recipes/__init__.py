"""Model definitions — concrete (recipe, runtime) cards (design_v3 §4).

Phase 1: Wan2.1-1.3B (T2V), LTX2.3 (2-stage distilled), Wan-causal (self-forcing student).
``build_default_engine`` loads all three onto one engine (one resident instance per card).
"""
from __future__ import annotations

from typing import Any

from v2.recipes.adaptive import build_adaptive_card
from v2.recipes.adapters import build_adapter_card, build_adapter_program
from v2.recipes.bagel import build_bagel_card, build_bagel_program
from v2.recipes.cosmos3 import build_cosmos3_card, build_cosmos3_program
from v2.recipes.image_video import (
    build_flux_t2i_card,
    build_flux_t2i_program,
    build_t2i_i2v_extend_workflow,
    build_t2i_then_i2v_workflow,
    build_wan_i2v_card,
    build_wan_i2v_program,
)
from v2.recipes.ltx2 import build_ltx2_av_program, build_ltx2_card, build_ltx2_program
from v2.recipes.qwen_omni import build_qwen_omni_card, build_qwen_omni_program
from v2.recipes.reward import build_reward_card
from v2.recipes.speculative import build_speculative_card, build_speculative_program
from v2.recipes.tiled import build_tiled_card, build_tiled_program
from v2.recipes.unified import build_unified_card, build_unified_program
from v2.recipes.wan21 import build_wan21_card, build_wan_t2v_program
from v2.recipes.wan_causal import build_wan_causal_card, build_wan_causal_program

__all__ = [
    "build_wan21_card", "build_wan_t2v_program",
    "build_ltx2_card", "build_ltx2_program", "build_ltx2_av_program",
    "build_wan_causal_card", "build_wan_causal_program",
    "build_cosmos3_card", "build_cosmos3_program",
    "build_bagel_card", "build_bagel_program",
    "build_qwen_omni_card", "build_qwen_omni_program",
    "build_reward_card",
    "build_speculative_card", "build_speculative_program",
    "build_unified_card", "build_unified_program",
    "build_flux_t2i_card", "build_flux_t2i_program", "build_wan_i2v_card", "build_wan_i2v_program",
    "build_t2i_then_i2v_workflow", "build_t2i_i2v_extend_workflow", "register_workflows",
    "build_tiled_card", "build_tiled_program", "build_adaptive_card",
    "build_adapter_card", "build_adapter_program",
    "build_default_engine", "build_omni_engine", "build_unified_engine", "build_image_video_engine",
    "build_tiled_engine",
]

_BUILDERS = [
    (build_wan21_card, build_wan_t2v_program),
    (build_ltx2_card, build_ltx2_program),
    (build_wan_causal_card, build_wan_causal_program),
]

# Phase-2 omni cards: MoT shared-weight (Cosmos3/BAGEL) + the cascaded thinker→talker→vocoder
# (Qwen-Omni, three disjoint experts / three loop types in one request).
_OMNI_BUILDERS = [
    (build_cosmos3_card, build_cosmos3_program),
    (build_bagel_card, build_bagel_program),
    (build_qwen_omni_card, build_qwen_omni_program),
]


def build_default_engine(engine: Any = None) -> Any:
    """Register Wan2.1, LTX2.3, and Wan-causal onto one engine (one resident instance each)."""
    from v2.cache import CacheManager
    from v2.card import load_card
    from v2.runtime import Engine
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
    from v2.cache import CacheManager
    from v2.card import load_card
    from v2.runtime import Engine
    eng = engine if engine is not None else Engine()
    for build_card, build_program in _OMNI_BUILDERS:
        card = build_card()
        inst = load_card(card, cache_manager=CacheManager.from_card(card))
        eng.register(card.model_id, inst, build_program())
    return eng


# Declarative catalog of cross-model workflows: workflow_id -> (builder, required model cards).
# Adding a custom pipeline is one line here (the cross-model analog of _BUILDERS / _OMNI_BUILDERS;
# cf. vllm-omni's pipeline_registry). ``register_workflows`` registers each whose cards are present.
_WORKFLOWS: dict[str, tuple] = {
    "image_video.t2i_i2v": (build_t2i_then_i2v_workflow,
                            [(build_flux_t2i_card, build_flux_t2i_program),
                             (build_wan_i2v_card, build_wan_i2v_program)]),
}


def register_workflows(engine: Any, *, only: list[str] | None = None) -> Any:
    """Register catalog workflows (and the cards they require) onto ``engine``. ``only`` selects a
    subset by workflow_id; default registers all whose cards aren't yet present."""
    from v2.cache import CacheManager
    from v2.card import load_card
    names = only if only is not None else list(_WORKFLOWS)
    for wf_id in names:
        build_workflow, card_builders = _WORKFLOWS[wf_id]
        for build_card, build_program in card_builders:
            card = build_card()
            if not engine.serves(card.model_id):
                inst = load_card(card, cache_manager=CacheManager.from_card(card))
                engine.register(card.model_id, inst, build_program())
        engine.register_workflow(build_workflow())
    return engine


def build_image_video_engine(engine: Any = None) -> Any:
    """Register the T2I (``flux-t2i``) and I2V (``wan-i2v``) cards plus the ``image_video.t2i_i2v``
    workflow on one engine — two *separate* models chained by a cross-model workflow, addressable by
    its workflow_id like any servable (design_v3 §13)."""
    from v2.runtime import Engine
    eng = engine if engine is not None else Engine()
    return register_workflows(eng, only=["image_video.t2i_i2v"])


def build_tiled_engine(engine: Any = None) -> Any:
    """Register the tiled-decode card (``wan-tiled``, whose VAE decode emits ``VAE_TILE`` work units)
    alongside plain Wan2.1 — so an interleaved batch mixes ``VAE_TILE`` and ``DIFFUSION_STEP`` units
    through one admission/scheduler budget (the §17 heterogeneous-co-scheduling probe)."""
    from v2.cache import CacheManager
    from v2.card import load_card
    from v2.runtime import Engine
    eng = engine if engine is not None else Engine()
    for build_card, build_program in [(build_tiled_card, build_tiled_program),
                                      (build_wan21_card, build_wan_t2v_program)]:
        card = build_card()
        inst = load_card(card, cache_manager=CacheManager.from_card(card))
        eng.register(card.model_id, inst, build_program())
    return eng


def build_unified_engine(engine: Any = None) -> Any:
    """Register the unified LM+generator card (UniRL/PromptRL): a prompt-refiner ``llm`` expert + a
    flow ``transformer`` expert on the SAME engine, each driving its own loop. Two separate experts,
    one request, both trainable under one RL reward (the §10 joint-RL stress test)."""
    from v2.cache import CacheManager
    from v2.card import load_card
    from v2.runtime import Engine
    eng = engine if engine is not None else Engine()
    card = build_unified_card()
    inst = load_card(card, cache_manager=CacheManager.from_card(card))
    eng.register(card.model_id, inst, build_unified_program())
    return eng
