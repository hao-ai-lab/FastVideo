"""Wan2.1-1.3B — bidirectional video diffusion (T2V)."""
from __future__ import annotations

from v2.recipes.wan21.card import (
    build_fastwan_qad_fp8_card,
    build_wan21_card,
    build_wan22_a14b_card,
    build_wan22_ti2v_card,
    build_wan_t2v_14b_card,
    stamp_wan21_checkpoints,
)
from v2.recipes.wan21.loop import WanDMDLoop, WanDenoiseLoop, latent_shape
from v2.recipes.wan21.program import build_wan_t2v_program

__all__ = [
    "build_wan21_card", "build_wan22_ti2v_card", "build_wan22_a14b_card", "build_wan_t2v_14b_card",
    "build_fastwan_qad_fp8_card", "stamp_wan21_checkpoints", "build_wan_t2v_program", "WanDenoiseLoop", "WanDMDLoop",
    "latent_shape"
]
