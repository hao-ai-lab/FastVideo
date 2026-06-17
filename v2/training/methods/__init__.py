"""Training methods — all on Wan2.1-1.3B, all over the SHARED loops (design_v3 §10)."""
from __future__ import annotations

from .base import TrainingMethod, new_instance, predict_x0
from .diffusion_nft import DiffusionNFTMethod, build_diffusion_nft
from .dmd2 import DMD2Method, build_dmd2
from .finetune import FineTuneMethod, build_finetune
from .joint_multi_rl import JointMultiExpertRL, build_joint_multi_rl
from .self_forcing import SelfForcingMethod, build_self_forcing
from .unified_rl import UnifiedRLMethod, build_unified_rl

__all__ = [
    "TrainingMethod", "new_instance", "predict_x0",
    "FineTuneMethod", "build_finetune",
    "DMD2Method", "build_dmd2",
    "DiffusionNFTMethod", "build_diffusion_nft",
    "SelfForcingMethod", "build_self_forcing",
    "UnifiedRLMethod", "build_unified_rl",
    "JointMultiExpertRL", "build_joint_multi_rl",
]
