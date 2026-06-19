"""Kandinsky-5.0-T2V-Lite — text→video ported into the v2 substrate.

Self-contained recipe package (the bucket-C pattern): the card declares its torch adapters via
``ComponentSpec.adapter`` (``Kandinsky5DiT`` / ``Kandinsky5QwenEncoder`` / ``Kandinsky5ClipEncoder`` /
``Kandinsky5VAE`` in ``v2/recipes/kandinsky5/adapter.py``) and a new ``Kandinsky5DenoiseLoop``
(flow-match Euler over a CHANNELS-LAST latent with a dual Qwen+CLIP conditioning stream). It reuses the
shared flow-match sampler (``FlowShiftPolicy.build_schedule`` + ``FLOW_MATCH_STEP``) and ``ClassicCFG``.
Registered in ``v2/registry.py`` by the orchestrator.
"""
from __future__ import annotations

from v2.recipes.kandinsky5.card import KANDINSKY5_NEG, build_kandinsky5_card, stamp_kandinsky5_checkpoints
from v2.recipes.kandinsky5.loop import Kandinsky5DenoiseLoop
from v2.recipes.kandinsky5.program import build_kandinsky5_program

__all__ = [
    "build_kandinsky5_card", "build_kandinsky5_program", "Kandinsky5DenoiseLoop", "stamp_kandinsky5_checkpoints",
    "KANDINSKY5_NEG"
]
