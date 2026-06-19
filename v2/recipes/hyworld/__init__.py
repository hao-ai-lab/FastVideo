"""HY-WorldPlay-Bidirectional (interactive world model) — ported into the v2 substrate.

Self-contained recipe package (the bucket-C pattern): the card declares its torch adapters via
``ComponentSpec.adapter`` (``HYWorldDiT`` / ``HYWorldVAE`` / ``HYWorldQwenEncoder`` /
``HYWorldByT5Encoder`` / ``HYWorldSiglipEncoder`` in ``v2/recipes/hyworld/adapter.py``) and a new
``HYWorldDenoiseLoop`` (chunk-rollout flow-match: per-chunk sweep + camera-aligned frozen context for
chunk>0). It reuses the flow-match ``FLOW_MATCH_STEP`` kernel + ``ClassicCFG`` policy and the Wan
``stamp_wan21_checkpoints`` superset (which covers ``text_encoder_2`` + ``image_encoder``).

The registered preset is the bidirectional t2v / degenerate (no action/camera) path that CPU-verifies;
the full action/camera/memory-retrieval conditioning is BRINGUP (needs a request-API extension to carry
the pose string + first-frame image). Registered in ``v2/registry.py``.
"""
from __future__ import annotations

from v2.recipes.hyworld.card import build_hyworld_card
from v2.recipes.hyworld.loop import HYWorldDenoiseLoop
from v2.recipes.hyworld.program import build_hyworld_program

__all__ = ["build_hyworld_card", "build_hyworld_program", "HYWorldDenoiseLoop"]
