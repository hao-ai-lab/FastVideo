# SPDX-License-Identifier: Apache-2.0
"""RL training methods."""

from fastvideo.train.methods.rl.diffusion_nft import DiffusionNFTMethod
from fastvideo.train.methods.rl.rvm import RVMMethod

__all__ = ["DiffusionNFTMethod", "RVMMethod"]
