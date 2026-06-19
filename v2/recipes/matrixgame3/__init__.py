"""Matrix-Game 3.0 (Base-Distilled) — autoregressive multi-clip world model, as a self-contained v2 recipe.

Bucket-C pattern: the card declares its torch adapter via ``ComponentSpec.adapter`` (``MatrixGame3DiT`` in
``v2/recipes/matrixgame3/adapter.py``) plus ``MatrixGame3DenoiseLoop`` (autoregressive multi-clip
loop: per-token timesteps + first-frame pinning; action/camera/KV-memory are BRINGUP), reusing the Wan VAE
adapter + T5 + ``stamp_wan21_checkpoints``. Registered in ``v2/registry.py`` (the ``MatrixGame3WanModel``
transformer fallback).
"""
from __future__ import annotations

from v2.recipes.matrixgame3.card import build_matrixgame3_card
from v2.recipes.matrixgame3.loop import MatrixGame3DenoiseLoop
from v2.recipes.matrixgame3.program import build_matrixgame3_program

__all__ = ["build_matrixgame3_card", "build_matrixgame3_program", "MatrixGame3DenoiseLoop"]
