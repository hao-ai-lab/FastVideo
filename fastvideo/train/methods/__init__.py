# SPDX-License-Identifier: Apache-2.0

from fastvideo.train.methods.base import TrainingMethod

__all__ = [
    "TrainingMethod",
    "DMD2Method",
    "FineTuneMethod",
    "KDMethod",
    "SelfForcingMethod",
    "DiffusionForcingSFTMethod",
]


def __getattr__(name: str) -> object:
    if name == "DMD2Method":
        from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method
        return DMD2Method
    if name == "FineTuneMethod":
        from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod
        return FineTuneMethod
    if name == "KDMethod":
        from fastvideo.train.methods.knowledge_distillation.kd import KDMethod
        return KDMethod
    if name == "SelfForcingMethod":
        from fastvideo.train.methods.distribution_matching.self_forcing import SelfForcingMethod
        return SelfForcingMethod
    if name == "DiffusionForcingSFTMethod":
        from fastvideo.train.methods.fine_tuning.dfsft import DiffusionForcingSFTMethod
        return DiffusionForcingSFTMethod
    raise AttributeError(name)
