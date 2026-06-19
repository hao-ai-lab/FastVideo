"""Stable Diffusion 3.5 Medium (text→image) — a FLOW-MATCH MMDiT ported into the v2 substrate.

Self-contained recipe package: the card declares its torch adapters via ``ComponentSpec.adapter``
(``SD3DiT``/``SD3VAE``/``SD3ClipEncoder``/``SD3T5Encoder`` in ``v2/platform/backends/torch_sd35.py``)
and ``SD3DenoiseLoop`` (flow-match Euler over 4D image latents with dual text conditioning: the
triple-encoder joint embed + the dual-CLIP pooled vector). Reuses ``FLOW_MATCH_STEP`` + ``FlowShiftPolicy``
+ ``ClassicCFG``. Registered in ``v2/registry.py`` (transformer ``_class_name`` = ``SD3Transformer2DModel``
for the arch fallback).
"""
from __future__ import annotations

from v2.recipes.sd35.card import build_sd35_card
from v2.recipes.sd35.loop import SD3DenoiseLoop
from v2.recipes.sd35.program import build_sd35_program

__all__ = ["build_sd35_card", "build_sd35_program", "SD3DenoiseLoop"]
