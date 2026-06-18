"""BAGEL/lance — canonical vllm-omni MoT (AR generate_text + diffusion generate_image). Phase 2."""
from __future__ import annotations

from v2.models.bagel.card import build_bagel_card
from v2.models.bagel.program import build_bagel_program

__all__ = ["build_bagel_card", "build_bagel_program"]
