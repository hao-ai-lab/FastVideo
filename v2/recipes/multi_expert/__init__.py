"""Multi-expert card: N refiner LMs + one generator (N+1 trainable experts, N+1 loops) — the
substrate for N-way joint RL (design_v3 §4, §10). Generalizes the two-expert ``unified`` card."""
from __future__ import annotations

from v2.recipes.multi_expert.card import N_REFINE_ACTIONS, build_multi_expert_card, refiner_ids

__all__ = ["build_multi_expert_card", "refiner_ids", "N_REFINE_ACTIONS"]
