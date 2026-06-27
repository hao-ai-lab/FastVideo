# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastvideo.train.methods.consistency_model.causal_cd import (
        CausalConsistencyDistillationMethod, )

__all__ = [
    "CausalConsistencyDistillationMethod",
]


def __getattr__(name: str) -> object:
    if name == "CausalConsistencyDistillationMethod":
        from fastvideo.train.methods.consistency_model.causal_cd import (
            CausalConsistencyDistillationMethod, )

        return CausalConsistencyDistillationMethod
    raise AttributeError(name)
