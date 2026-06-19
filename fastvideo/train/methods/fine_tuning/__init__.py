# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastvideo.train.methods.fine_tuning.dfsft import DiffusionForcingSFTMethod
    from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod
    from fastvideo.train.methods.fine_tuning.interleave_thinker_sft import (
        InterleaveThinkerSFTMethod, )

__all__ = [
    "DiffusionForcingSFTMethod",
    "FineTuneMethod",
    "InterleaveThinkerSFTMethod",
]


def __getattr__(name: str) -> object:
    # Lazy import to avoid circular imports during registry bring-up.
    if name == "DiffusionForcingSFTMethod":
        from fastvideo.train.methods.fine_tuning.dfsft import (
            DiffusionForcingSFTMethod, )

        return DiffusionForcingSFTMethod
    if name == "FineTuneMethod":
        from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod

        return FineTuneMethod
    if name == "InterleaveThinkerSFTMethod":
        from fastvideo.train.methods.fine_tuning.interleave_thinker_sft import (
            InterleaveThinkerSFTMethod, )

        return InterleaveThinkerSFTMethod
    raise AttributeError(name)
