"""HunyuanGameCraft (interactive camera/action-conditioned i2v) ported into the v2 substrate.

Self-contained recipe package: the card declares its torch adapters via ``ComponentSpec.adapter``
(``GameCraftDiT``/``GameCraftVAE``/``GameCraftLlamaEncoder``/``GameCraftClipEncoder`` in
``v2/platform/backends/torch_hunyuangamecraft.py``) plus a ``GameCraftDenoiseLoop`` (flow-match Euler +
33ch concat + per-step clean-ref injection). The registered path is the t2v/degenerate denoise; the
camera/action (CameraNet Plücker) conditioning is BRINGUP — it needs a request-API camera-input channel v2
lacks today. Registered in ``v2/registry.py`` by the orchestrator from this package's returned metadata.
"""
from __future__ import annotations

from v2.recipes.hunyuangamecraft.card import build_hunyuangamecraft_card
from v2.recipes.hunyuangamecraft.loop import GameCraftDenoiseLoop
from v2.recipes.hunyuangamecraft.program import build_hunyuangamecraft_program

__all__ = ["build_hunyuangamecraft_card", "build_hunyuangamecraft_program", "GameCraftDenoiseLoop"]
