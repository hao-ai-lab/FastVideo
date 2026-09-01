# SPDX-License-Identifier: Apache-2.0
"""RL training methods."""

from fastvideo.train.methods.rl.diffusion_nft import DiffusionNFTMethod
from fastvideo.train.methods.rl.rvm import RVMMethod
from fastvideo.train.methods.rl.rvm_faithful import RVMFaithfulMethod
from fastvideo.train.methods.rl.rvm_local_metrics import RVMWithLocalMetricsMethod

__all__ = [
    "DiffusionNFTMethod",
    "RVMFaithfulMethod",
    "RVMMethod",
    "RVMWithLocalMetricsMethod",
]
