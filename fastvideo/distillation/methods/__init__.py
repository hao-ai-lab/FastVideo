# SPDX-License-Identifier: Apache-2.0

from fastvideo.distillation.methods.base import DistillMethod

__all__ = [
    "DistillMethod",
    "DMD2Method",
    "FineTuneMethod",
    "SelfForcingMethod",
    "DiffusionForcingSFTMethod",
]


def __getattr__(name: str) -> object:
    if name == "DMD2Method":
        from fastvideo.distillation.methods.distribution_matching.dmd2 import DMD2Method
        return DMD2Method
    if name == "FineTuneMethod":
        from fastvideo.distillation.methods.fine_tuning.finetune import FineTuneMethod
        return FineTuneMethod
    if name == "SelfForcingMethod":
        from fastvideo.distillation.methods.distribution_matching.self_forcing import SelfForcingMethod
        return SelfForcingMethod
    if name == "DiffusionForcingSFTMethod":
        from fastvideo.distillation.methods.fine_tuning.dfsft import DiffusionForcingSFTMethod
        return DiffusionForcingSFTMethod
    raise AttributeError(name)
