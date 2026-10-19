# SPDX-License-Identifier: Apache-2.0
"""OmniRef PDD: fixed ordered reference conditions, target-only trajectories.

Only the student's output projections widen. Reference inputs, teacher weights,
packing, augmentation seeds and P2 multi-region VSA use the Ref2VA adapter.
"""

from fastvideo.train.models.minimax_h3.minimax_h3_pdd import MiniMaxH3PDDMixin
from fastvideo.train.models.minimax_h3.minimax_h3_ref2v_dmd import MiniMaxH3Ref2VDMDModel


class MiniMaxH3Ref2VPDDModel(MiniMaxH3PDDMixin, MiniMaxH3Ref2VDMDModel):
    """Reference-conditioned PDD student on the top-level transformer_ref."""


__all__ = ["MiniMaxH3Ref2VPDDModel"]
