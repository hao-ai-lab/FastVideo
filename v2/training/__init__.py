"""Training plane — rollout records behavior on the same loops it serves (design_v3 §10).

Dependency rule (enforced): ``training`` imports ``card``/``loop``/``runtime``/``models``; the
engine never imports ``training``. The rollout forward IS the serve forward plus capture.
"""
from __future__ import annotations

from v2.training.behavior import BehaviorRecord, TrajectoryStatus
from v2.training.flywheel import derive_card, run_flywheel
from v2.training.methods import (
    DiffusionNFTMethod,
    DMD2Method,
    FineTuneMethod,
    JointMultiExpertRL,
    SelfForcingMethod,
    TrainingMethod,
    UnifiedRLMethod,
    WorkflowRLMethod,
    build_diffusion_nft,
    build_dmd2,
    build_finetune,
    build_joint_multi_rl,
    build_self_forcing,
    build_unified_rl,
    build_workflow_rl,
)
from v2.training.rewards import MultiRewardScorer, ServedRewardScorer, build_multi_reward_scorer
from v2.training.rollout import RolloutContext, Trajectory, TrajectoryBuffer, rollout_loop
from v2.training.weight_sync import WeightRole, WeightSyncController, WeightSyncPlan

__all__ = [
    "TrainingMethod", "FineTuneMethod", "DMD2Method", "DiffusionNFTMethod", "SelfForcingMethod",
    "UnifiedRLMethod", "JointMultiExpertRL", "WorkflowRLMethod",
    "build_finetune", "build_dmd2", "build_diffusion_nft", "build_self_forcing", "build_unified_rl",
    "build_joint_multi_rl", "build_workflow_rl", "run_flywheel", "derive_card",
    "BehaviorRecord", "TrajectoryStatus", "Trajectory", "TrajectoryBuffer", "RolloutContext",
    "rollout_loop", "WeightSyncPlan", "WeightSyncController", "WeightRole",
    "MultiRewardScorer", "ServedRewardScorer", "build_multi_reward_scorer",
]
