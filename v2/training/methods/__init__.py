"""Training methods — all on Wan2.1-1.3B, all over the SHARED loops."""
from __future__ import annotations

from v2.training.methods.base import TrainingMethod, new_instance, predict_x0
from v2.training.methods.diffusion_nft import DiffusionNFTMethod, build_diffusion_nft
from v2.training.methods.dmd2 import DMD2Method, build_dmd2
from v2.training.methods.finetune import FineTuneMethod, build_finetune
from v2.training.methods.joint_multi_rl import JointMultiExpertRL, build_joint_multi_rl
from v2.training.methods.self_forcing import SelfForcingMethod, build_self_forcing
from v2.training.methods.unified_rl import UnifiedRLMethod, build_unified_rl
from v2.training.methods.workflow_rl import WorkflowRLMethod, build_workflow_rl

__all__ = [
    "TrainingMethod",
    "new_instance",
    "predict_x0",
    "FineTuneMethod",
    "build_finetune",
    "DMD2Method",
    "build_dmd2",
    "DiffusionNFTMethod",
    "build_diffusion_nft",
    "SelfForcingMethod",
    "build_self_forcing",
    "UnifiedRLMethod",
    "build_unified_rl",
    "JointMultiExpertRL",
    "build_joint_multi_rl",
    "WorkflowRLMethod",
    "build_workflow_rl",
]
