# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from fastvideo.distillation.methods.base import DistillMethod

if TYPE_CHECKING:
    from fastvideo.distillation.methods.distribution_matching.dmd2 import DMD2Method

__all__ = [
    "DistillMethod",
    "DMD2Method",
]


def __getattr__(name: str) -> object:
    # Lazy import to avoid circular imports during registry bring-up.
    if name == "DMD2Method":
        from fastvideo.distillation.methods.distribution_matching.dmd2 import DMD2Method

        return DMD2Method
    raise AttributeError(name)
