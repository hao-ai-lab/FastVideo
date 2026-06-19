"""Cosmos-Predict2 (Video2World) EDM-Karras denoiser recipe.

Self-contained recipe: the card declares torch adapters (``CosmosDiT``/``CosmosT5Encoder`` in
``v2/platform/backends/torch_cosmos.py``) plus ``CosmosDenoiseLoop`` (EDM preconditioning folded into
flow-match Euler), reusing the Wan VAE adapter + T5 + ``stamp_wan21_checkpoints``.
"""
from __future__ import annotations

from v2.recipes.cosmos2.card import build_cosmos2_card
from v2.recipes.cosmos2.loop import CosmosDenoiseLoop
from v2.recipes.cosmos2.program import build_cosmos2_program

__all__ = ["build_cosmos2_card", "build_cosmos2_program", "CosmosDenoiseLoop"]
