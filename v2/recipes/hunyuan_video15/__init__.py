"""HunyuanVideo 1.5 (T2V) — rectified-flow video DiT ported into the v2 substrate.

Self-contained recipe package: the card declares its torch adapters via ``ComponentSpec.adapter``
(``HunyuanVideo15DiT``/``HunyuanVideo15VAE``/``HunyuanVideo15QwenEncoder``/``HunyuanVideo15ByT5Encoder`` in
``v2/recipes/hunyuan_video15/adapter.py``) plus a ``HunyuanVideo15DenoiseLoop`` (a thin
``WanDenoiseLoop`` subclass with z=32/16×/4× geometry). The flow-match loop math is unchanged from Wan; the
arch deltas (two text embeds, an image-embed list, the 33-channel i2v cond concat, the scalar-scaling VAE)
live in the adapters. Registered in ``v2/registry.py`` (by the orchestrator) for
``hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v``.
"""
from __future__ import annotations

from v2.recipes.hunyuan_video15.card import (
    build_hunyuan_video15_720p_card,
    build_hunyuan_video15_card,
)
from v2.recipes.hunyuan_video15.loop import HunyuanVideo15DenoiseLoop
from v2.recipes.hunyuan_video15.program import build_hunyuan_video15_program

__all__ = [
    "build_hunyuan_video15_card",
    "build_hunyuan_video15_720p_card",
    "build_hunyuan_video15_program",
    "HunyuanVideo15DenoiseLoop",
]
