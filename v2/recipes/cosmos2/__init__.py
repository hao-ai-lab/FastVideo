"""Cosmos-Predict2 (Video2World) — EDM-Karras denoiser ported into the v2 substrate.

Self-contained recipe package (the bucket-C pattern): card declares its torch adapters via
``ComponentSpec.adapter`` (``CosmosDiT``/``CosmosT5Encoder`` in ``v2/platform/backends/torch_cosmos.py``)
and a new ``CosmosDenoiseLoop`` (EDM preconditioning folded into flow-match Euler), reusing the Wan VAE
adapter + T5 + ``stamp_wan21_checkpoints``. Registered in ``v2/registry.py``.
"""
from __future__ import annotations

from v2.recipes.cosmos2.card import build_cosmos2_card
from v2.recipes.cosmos2.loop import CosmosDenoiseLoop
from v2.recipes.cosmos2.program import build_cosmos2_program

__all__ = ["build_cosmos2_card", "build_cosmos2_program", "CosmosDenoiseLoop"]
