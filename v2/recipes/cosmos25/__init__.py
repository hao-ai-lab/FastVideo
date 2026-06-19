"""Cosmos-Predict2.5 (text->video) flow-match denoiser recipe.

Self-contained recipe: the card declares torch adapters (``Cosmos25DiT`` / ``Cosmos25WanVAE`` /
``Cosmos25Reason1Encoder`` in ``v2/platform/backends/torch_cosmos25.py``) plus ``Cosmos25DenoiseLoop``,
which reuses Wan's flow-shift sigma schedule + ``FLOW_MATCH_STEP`` + ``ClassicCFG`` but feeds the DiT the
plain per-frame sigma timestep ``[B,T]`` Cosmos2.5 needs (not Wan's ``sigma*1000``). Reuses
``stamp_wan21_checkpoints``. Size-agnostic (2B/14B resolve from the checkpoint config).
"""
from __future__ import annotations

from v2.recipes.cosmos25.card import build_cosmos25_card
from v2.recipes.cosmos25.loop import Cosmos25DenoiseLoop
from v2.recipes.cosmos25.program import build_cosmos25_program

__all__ = ["build_cosmos25_card", "build_cosmos25_program", "Cosmos25DenoiseLoop"]
