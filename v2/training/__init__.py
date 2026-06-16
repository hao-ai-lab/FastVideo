"""Training plane — rollout records behavior on the same loops it serves (design_v3 §10).

Dependency rule (enforced): ``training`` imports ``card``/``loop``/``runtime``/``models``; the
engine never imports ``training``. The rollout forward IS the serve forward plus capture.
"""
from __future__ import annotations

from .behavior import BehaviorRecord, TrajectoryStatus
from .methods import (
    DiffusionNFTMethod,
    DMD2Method,
    FineTuneMethod,
    SelfForcingMethod,
    TrainingMethod,
    UnifiedRLMethod,
    build_diffusion_nft,
    build_dmd2,
    build_finetune,
    build_self_forcing,
    build_unified_rl,
)
from .rewards import MultiRewardScorer, build_multi_reward_scorer
from .rollout import RolloutContext, Trajectory, TrajectoryBuffer, rollout_loop
from .weight_sync import WeightRole, WeightSyncPlan

__all__ = [
    "TrainingMethod", "FineTuneMethod", "DMD2Method", "DiffusionNFTMethod", "SelfForcingMethod",
    "UnifiedRLMethod",
    "build_finetune", "build_dmd2", "build_diffusion_nft", "build_self_forcing", "build_unified_rl",
    "BehaviorRecord", "TrajectoryStatus", "Trajectory", "TrajectoryBuffer", "RolloutContext",
    "rollout_loop", "WeightSyncPlan", "WeightRole", "MultiRewardScorer", "build_multi_reward_scorer",
]
