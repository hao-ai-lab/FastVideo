"""Reward-model card: a served scorer/verifier emitting REWARD_BATCH work units (design_v3 §10)."""
from __future__ import annotations

from .card import build_reward_card
from .loop import RewardLoop

__all__ = ["build_reward_card", "RewardLoop"]
