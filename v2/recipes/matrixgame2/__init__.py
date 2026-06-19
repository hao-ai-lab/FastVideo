"""Matrix-Game 2.0 (Base, distilled) — interactive mouse/keyboard world model, as a self-contained v2
recipe package (bucket-C pattern).

The card declares its torch adapters via ``ComponentSpec.adapter`` (``MatrixGame2CausalDiT`` /
``MatrixGame2CLIPImageEncoder`` in ``v2/recipes/matrixgame2/adapter.py``) plus
``MatrixGame2CausalDMDLoop`` (few-step DMD: epsilon->x0 via the FlowUniPC sigma table + re-add_noise, causal
block-autoregressive with a sliding-window KV cache, action-conditioned). Reuses the Wan VAE adapter +
``stamp_wan21_checkpoints``. Registered in ``v2/registry.py``.

The registered path is the i2v world-rollout (first-frame + CLIP context); live mouse/keyboard action
routing is BRINGUP (needs a request-API extension — the loop + program already thread the action slots).
"""
from __future__ import annotations

from v2.recipes.matrixgame2.card import build_matrixgame2_card
from v2.recipes.matrixgame2.loop import MatrixGame2CausalDMDLoop
from v2.recipes.matrixgame2.program import build_matrixgame2_program

__all__ = ["build_matrixgame2_card", "build_matrixgame2_program", "MatrixGame2CausalDMDLoop"]
