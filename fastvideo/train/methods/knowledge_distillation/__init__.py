# SPDX-License-Identifier: Apache-2.0

from fastvideo.train.methods.knowledge_distillation.h3_rest import H3RESTMethod
from fastvideo.train.methods.knowledge_distillation.kd import (
    KDCausalMethod,
    KDMethod,
)

__all__ = ["H3RESTMethod", "KDCausalMethod", "KDMethod"]
