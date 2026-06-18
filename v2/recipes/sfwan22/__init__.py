"""Self-Forcing Wan2.2-A14B — CAUSAL (chunk rollout + slab-KV) Wan2.2 MoE (two experts + boundary).

Self-contained recipe package: combines the v2 wan_causal chunk-rollout pattern with Wan2.2
BoundaryTimestepRouting and the few-step DMD math of fastvideo's ``CausalDMDDenosingStage``. Two HF ids:
  * ``FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers`` -> ``build_sfwan22_i2v_a14b_card`` (i2v);
  * ``rand0nmr/SFWan2.2-T2V-A14B-Diffusers``          -> ``build_sfwan22_t2v_a14b_card`` (t2v).
Both reuse the GPU CausalWan/Wan torch path via ``load_id`` (pure Wan arch -> no custom adapter).
"""
from __future__ import annotations

from v2.recipes.sfwan22.card import (
    SFWAN22_NEG_CN,
    build_sfwan22_card,
    build_sfwan22_i2v_a14b_card,
    build_sfwan22_t2v_a14b_card,
)
from v2.recipes.sfwan22.loop import SFWan22ChunkRolloutLoop
from v2.recipes.sfwan22.program import (
    build_sfwan22_i2v_program,
    build_sfwan22_program,
    build_sfwan22_t2v_program,
)

__all__ = [
    "build_sfwan22_card",
    "build_sfwan22_i2v_a14b_card",
    "build_sfwan22_t2v_a14b_card",
    "build_sfwan22_program",
    "build_sfwan22_i2v_program",
    "build_sfwan22_t2v_program",
    "SFWan22ChunkRolloutLoop",
    "SFWAN22_NEG_CN",
]
