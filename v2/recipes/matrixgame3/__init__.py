"""Matrix-Game 3.0 (Base-Distilled) — autoregressive multi-clip world model ported into the v2 substrate.

Self-contained recipe package (the bucket-C pattern): the card declares its torch adapter via
``ComponentSpec.adapter`` (``MatrixGame3DiT`` in ``v2/platform/backends/torch_matrixgame3.py``) and a new
``MatrixGame3DenoiseLoop`` (the autoregressive multi-clip loop: per-token timesteps + first-frame pinning;
action/camera/KV-memory are BRINGUP), reusing the Wan VAE adapter + T5 + ``stamp_wan21_checkpoints``. The
orchestrator registers it in ``v2/registry.py`` (the ``MatrixGame3WanModel`` transformer fallback).
"""
from __future__ import annotations

from v2.recipes.matrixgame3.card import build_matrixgame3_card
from v2.recipes.matrixgame3.loop import MatrixGame3DenoiseLoop
from v2.recipes.matrixgame3.program import build_matrixgame3_program

__all__ = ["build_matrixgame3_card", "build_matrixgame3_program", "MatrixGame3DenoiseLoop"]
