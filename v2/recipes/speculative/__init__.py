"""Speculative (draft-verify) decoding: a draft + target AR pair, exact + lower-latency (design_v3 §9.16)."""
from __future__ import annotations

from v2.recipes.speculative.card import build_speculative_card
from v2.recipes.speculative.loop import SpeculativeARLoop
from v2.recipes.speculative.program import build_speculative_program

__all__ = ["build_speculative_card", "build_speculative_program", "SpeculativeARLoop"]
