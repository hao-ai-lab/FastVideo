"""LingBot-World-Base-Cam (camera-conditioned dual-guidance MoE i2v) ported into the v2 substrate.

Self-contained recipe package (the bucket-C pattern): the card declares its torch adapter via
``ComponentSpec.adapter`` (``LingBotWorldDiT`` in ``v2/platform/backends/torch_lingbotworld.py``) and a
new ``LingBotWorldDenoiseLoop`` (Wan i2v flow-match + boundary-routed MoE + DUAL guidance + camera/Plucker
conditioning), reusing the Wan VAE / T5 / CLIP adapters + ``stamp_wan21_checkpoints``. The program adds a
``camera_encode`` node (Plucker tensor; BRINGUP — needs a per-request camera input). Registered in
``v2/registry.py`` by the orchestrator from the returned metadata.
"""
from __future__ import annotations

from v2.recipes.lingbotworld.card import build_lingbotworld_card
from v2.recipes.lingbotworld.loop import LingBotWorldDenoiseLoop
from v2.recipes.lingbotworld.program import build_lingbotworld_program

__all__ = ["build_lingbotworld_card", "build_lingbotworld_program", "LingBotWorldDenoiseLoop"]
