# SPDX-License-Identifier: Apache-2.0

from fastvideo.distillation.validators.base import DistillValidator, ValidationRequest
from fastvideo.distillation.validators.wan import WanValidator

__all__ = [
    "DistillValidator",
    "ValidationRequest",
    "WanValidator",
]
