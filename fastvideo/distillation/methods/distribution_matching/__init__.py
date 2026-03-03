# SPDX-License-Identifier: Apache-2.0

from fastvideo.distillation.methods.distribution_matching.dmd2 import DMD2Method
from fastvideo.distillation.methods.distribution_matching.self_forcing import (
    SelfForcingMethod,
)

__all__ = [
    "DMD2Method",
    "SelfForcingMethod",
]
