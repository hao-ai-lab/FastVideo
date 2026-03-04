# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastvideo.distillation.methods.fine_tuning.dfsft import DiffusionForcingSFTMethod
    from fastvideo.distillation.methods.fine_tuning.finetune import FineTuneMethod

__all__ = [
    "DiffusionForcingSFTMethod",
    "FineTuneMethod",
]


def __getattr__(name: str) -> object:
    # Lazy import to avoid circular imports during registry bring-up.
    if name == "DiffusionForcingSFTMethod":
        from fastvideo.distillation.methods.fine_tuning.dfsft import (
            DiffusionForcingSFTMethod,
        )

        return DiffusionForcingSFTMethod
    if name == "FineTuneMethod":
        from fastvideo.distillation.methods.fine_tuning.finetune import FineTuneMethod

        return FineTuneMethod
    raise AttributeError(name)
