"""Unified LM+generator model (UniRL/PromptRL): a prompt-refiner LM expert + a flow generator expert,
both trainable under one RL reward. Stress-tests that the Card/Loop/Program split holds for
joint multi-expert RL."""
from __future__ import annotations

from v2.recipes.unified.card import N_REFINE_ACTIONS, build_unified_card
from v2.recipes.unified.program import apply_refinement_node, build_unified_program

__all__ = ["build_unified_card", "build_unified_program", "apply_refinement_node", "N_REFINE_ACTIONS"]
