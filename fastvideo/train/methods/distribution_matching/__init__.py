# SPDX-License-Identifier: Apache-2.0

from fastvideo.train.methods.distribution_matching.anyflow import AnyFlowMethod
from fastvideo.train.methods.distribution_matching.anyflow_pretrain import (
    AnyFlowPretrainMethod, )
from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method
from fastvideo.train.methods.distribution_matching.self_forcing import (
    SelfForcingMethod, )
from fastvideo.train.methods.distribution_matching.streaming_long_tuning import (
    StreamingLongTuningMethod, )
from fastvideo.train.methods.distribution_matching.tdm import TDMMethod

__all__ = [
    "AnyFlowMethod",
    "AnyFlowPretrainMethod",
    "DMD2Method",
    "SelfForcingMethod",
    "StreamingLongTuningMethod",
    "TDMMethod",
]
