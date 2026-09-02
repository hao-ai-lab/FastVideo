# SPDX-License-Identifier: Apache-2.0
"""Expose the MiniMax H3 plugins for YAML ``_target_`` resolution."""

from fastvideo.train.models.minimax_h3.minimax_h3 import (
    MiniMaxH3Model as MiniMaxH3Model, )
from fastvideo.train.models.minimax_h3.minimax_h3_dmd import (
    MiniMaxH3DMDModel as MiniMaxH3DMDModel, )
from fastvideo.train.models.minimax_h3.minimax_h3_rest import (
    MiniMaxH3RESTModel as MiniMaxH3RESTModel,
    MiniMaxH3RESTTeacherModel as MiniMaxH3RESTTeacherModel,
)
from fastvideo.train.models.minimax_h3.minimax_h3_rvm import (
    MiniMaxH3RVMModel as MiniMaxH3RVMModel, )
