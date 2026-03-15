# SPDX-License-Identifier: Apache-2.0

from fastvideo.train.methods.base import TrainingMethod

__all__ = [
    "TrainingMethod",
    "DMD2Method",
    "FineTuneMethod",
    "SelfForcingMethod",
    "DiffusionForcingSFTMethod",
    "TeacherForcingSFTMethod",
]


def __getattr__(name: str) -> object:
    if name == "DMD2Method":
        from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method
        return DMD2Method
    if name == "FineTuneMethod":
        from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod
        return FineTuneMethod
    if name == "SelfForcingMethod":
        from fastvideo.train.methods.distribution_matching.self_forcing import SelfForcingMethod
        return SelfForcingMethod
    if name == "DiffusionForcingSFTMethod":
        from fastvideo.train.methods.fine_tuning.dfsft import DiffusionForcingSFTMethod
        return DiffusionForcingSFTMethod
    if name == "TeacherForcingSFTMethod":
        from fastvideo.train.methods.fine_tuning.tfsft import TeacherForcingSFTMethod
        return TeacherForcingSFTMethod
    raise AttributeError(name)
