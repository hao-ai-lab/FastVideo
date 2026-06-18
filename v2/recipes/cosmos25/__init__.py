"""Cosmos-Predict2.5 (text->video) — a flow-match denoiser ported into the v2 substrate.

Self-contained recipe package (the bucket-C pattern): the card declares its torch adapters via
``ComponentSpec.adapter`` (``Cosmos25DiT`` / ``Cosmos25WanVAE`` / ``Cosmos25Reason1Encoder`` in
``v2/platform/backends/torch_cosmos25.py``) and a new ``Cosmos25DenoiseLoop`` (Wan's flow-shift sigma
schedule + ``FLOW_MATCH_STEP`` solver + ``ClassicCFG``, but the PLAIN per-frame sigma timestep ``[B,T]``
Cosmos2.5 needs — NOT Wan's ``sigma*1000``), reusing ``stamp_wan21_checkpoints``. Covers both the 2B and
14B sizes (the card is size-agnostic — size resolves from the checkpoint config). Registered in
``v2/registry.py`` by the orchestrator from the returned metadata.
"""
from __future__ import annotations

from v2.recipes.cosmos25.card import build_cosmos25_card
from v2.recipes.cosmos25.loop import Cosmos25DenoiseLoop
from v2.recipes.cosmos25.program import build_cosmos25_program

__all__ = ["build_cosmos25_card", "build_cosmos25_program", "Cosmos25DenoiseLoop"]
