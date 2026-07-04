"""Model definitions — concrete (recipe, runtime) cards.

The kept v2 recipes: Wan2.1 (T2V) + Wan-causal (self-forcing student) + LTX-2 (distilled / base /
2.3 joint A/V), the omni MoT/cascade cards (Cosmos3 / BAGEL / Qwen-Omni), and the bucket-C ports
flux2 + matrixgame2 (resolved through ``v2.registry``). ``build_default_engine`` loads the core
diffusion cards onto one engine; ``build_omni_engine`` the omni cards.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from v2.recipes.bagel import build_bagel_card, build_bagel_program
from v2.recipes.cosmos3 import build_cosmos3_card, build_cosmos3_program
from v2.recipes.flux2 import build_flux2_card, build_flux2_program
from v2.recipes.ltx2 import build_ltx2_av_program, build_ltx2_card, build_ltx2_program
from v2.recipes.qwen_omni import build_qwen_omni_card, build_qwen_omni_program
from v2.recipes.wan21 import build_wan21_card, build_wan_t2v_program
from v2.recipes.wan21.i2v import build_wan21_i2v_card, build_wan21_i2v_program
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
    "build_image_video_engine",
    "build_t2i_then_i2v_workflow",
    "register_workflows",
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

_IMAGE_VIDEO_BUILDERS = [
    (lambda: build_flux2_card("flux-t2i"), build_flux2_program),
    (lambda: build_wan21_i2v_card("wan-i2v"), build_wan21_i2v_program),
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


def _register_cards(engine: Any, builders: list[tuple[Any, Any]]) -> Any:
    from v2.core.card import load_card
    from v2.runtime.cache import CacheManager

    for build_card, build_program in builders:
        card = build_card()
        if hasattr(engine, "serves") and engine.serves(card.model_id):
            continue
        inst = load_card(card, cache_manager=CacheManager.from_card(card))
        engine.register(card.model_id, inst, build_program())
    return engine


def _workflow_diffusion(state: dict[str, Any], *, seed: int, num_steps: int, num_frames: int) -> Any:
    from v2.core.request import DiffusionParams

    request = state.get("request")
    base = request.diffusion if request is not None else DiffusionParams()
    return replace(base, seed=seed, num_steps=num_steps, num_frames=num_frames)


def _image_pixels(artifact: Any) -> Any:
    import numpy as np

    pixels = getattr(artifact, "tensor", None)
    if pixels is None:
        pixels = getattr(artifact, "frames", None)
    arr = np.asarray(pixels, dtype="float32")
    if arr.ndim == 4 and arr.shape[0] in (1, 3, 4):
        arr = arr[:, 0]
    if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
        arr = arr[0]
    return arr


def build_t2i_then_i2v_workflow(*, workflow_id: str = "image_video.t2i_i2v", num_steps: int = 4) -> Any:
    """Build the toy FLUX-style T2I -> Wan-style I2V workflow used by ``v2_examples/workflows``."""
    from v2.core.program import Workflow, WorkflowStage
    from v2.core.request import ImagePart, TaskType, TextPart, make_request

    def t2i_request(state: dict[str, Any]) -> Any:
        seed = int(state.get("seed", 0))
        return make_request(
            TaskType.T2I,
            "flux-t2i",
            str(state.get("prompt", "")),
            diffusion=_workflow_diffusion(state, seed=seed, num_steps=num_steps, num_frames=1),
        )

    def i2v_request(state: dict[str, Any]) -> Any:
        seed = int(state.get("seed", 0)) + 1
        image = _image_pixels(state["t2i:image"])
        return make_request(
            TaskType.I2V,
            "wan-i2v",
            inputs=(TextPart(str(state.get("prompt", ""))), ImagePart(pixels=image)),
            diffusion=_workflow_diffusion(state, seed=seed, num_steps=num_steps, num_frames=81),
        )

    return Workflow(workflow_id, [
        WorkflowStage("flux-t2i", t2i_request, label="t2i"),
        WorkflowStage("wan-i2v", i2v_request, label="i2v"),
    ])


def build_image_video_engine(engine: Any = None) -> Any:
    """Register the toy T2I and I2V cards plus their cross-model workflow."""
    from v2.runtime import Engine

    eng = engine if engine is not None else Engine()
    _register_cards(eng, _IMAGE_VIDEO_BUILDERS)
    if not eng.serves("image_video.t2i_i2v"):
        eng.register_workflow(build_t2i_then_i2v_workflow())
    return eng


def register_workflows(engine: Any, *, only: list[str] | tuple[str, ...] | None = None) -> Any:
    """Declarative workflow catalog used by examples and tests."""
    wanted = set(only or ["image_video.t2i_i2v"])
    unknown = wanted - {"image_video.t2i_i2v"}
    if unknown:
        raise KeyError(f"unknown workflow ids: {sorted(unknown)}")
    if "image_video.t2i_i2v" in wanted:
        build_image_video_engine(engine)
    return engine
