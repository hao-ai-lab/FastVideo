"""LongCat-Video (T2V) — flow-match denoiser with CFG-zero guidance, ported into the v2 substrate.

Self-contained recipe package (the bucket-C pattern): the card declares its torch adapter via
``ComponentSpec.adapter`` (``LongCatDiT`` in ``v2/platform/backends/torch_longcat.py``) and a new
``LongCatDenoiseLoop`` (explicit ``linspace(1.0, 0.001, num_steps)`` sigma schedule + CFG-zero
optimized-scale combine), reusing the Wan ``WanVAE`` + ``T5Encoder`` adapters and ``stamp_wan21_checkpoints``.
The DiT adapter negates the velocity to fold in the fastvideo stage's ``noise_pred = -noise_pred``.
Registered in ``v2/registry.py`` by the orchestrator (transformer ``LongCatTransformer3DModel``).
"""
from __future__ import annotations

from v2.recipes.longcat.card import build_longcat_card
from v2.recipes.longcat.loop import CFGZeroPolicy, LongCatDenoiseLoop
from v2.recipes.longcat.program import build_longcat_program

__all__ = ["build_longcat_card", "build_longcat_program", "LongCatDenoiseLoop", "CFGZeroPolicy"]
