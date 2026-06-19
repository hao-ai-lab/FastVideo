# SPDX-License-Identifier: Apache-2.0
"""RL training methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastvideo.train.methods.rl.diffusion_nft import DiffusionNFTMethod
    from fastvideo.train.methods.rl.interleave_thinker import (
        InterleaveThinkerRLMethod, )

__all__ = [
    "DiffusionNFTMethod",
    "InterleaveThinkerRLMethod",
]


def __getattr__(name: str) -> object:
    if name == "DiffusionNFTMethod":
        from fastvideo.train.methods.rl.diffusion_nft import DiffusionNFTMethod

        return DiffusionNFTMethod
    if name == "InterleaveThinkerRLMethod":
        from fastvideo.train.methods.rl.interleave_thinker import (
            InterleaveThinkerRLMethod, )

        return InterleaveThinkerRLMethod
    raise AttributeError(name)
