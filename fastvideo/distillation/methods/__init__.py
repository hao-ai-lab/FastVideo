# SPDX-License-Identifier: Apache-2.0

from fastvideo.distillation.methods.base import DistillMethod
from fastvideo.distillation.methods.distribution_matching import DMD2Method

__all__ = [
    "DistillMethod",
    "DMD2Method",
]
