# SPDX-License-Identifier: Apache-2.0
"""Expose the MiniMax H3 plugin for YAML ``_target_`` resolution."""

from fastvideo.train.models.minimax_h3.minimax_h3 import (
    MiniMaxH3Model as MiniMaxH3Model, )
from fastvideo.train.models.minimax_h3.minimax_h3_dmd import (
    MiniMaxH3DMDLatentLayout as MiniMaxH3DMDLatentLayout,
    MiniMaxH3DMDModel as MiniMaxH3DMDModel,
    MiniMaxH3LoraDMDModel as MiniMaxH3LoraDMDModel,
)
from fastvideo.train.models.minimax_h3.minimax_h3_pdd import (
    MiniMaxH3PDDModel as MiniMaxH3PDDModel, )
from fastvideo.train.models.minimax_h3.minimax_h3_ref2v_dmd import (
    MiniMaxH3Ref2VDMDModel as MiniMaxH3Ref2VDMDModel, )
from fastvideo.train.models.minimax_h3.minimax_h3_ref2v_pdd import (
    MiniMaxH3Ref2VPDDModel as MiniMaxH3Ref2VPDDModel, )
