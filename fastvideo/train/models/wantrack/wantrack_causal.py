# SPDX-License-Identifier: Apache-2.0
"""Causal WanTrack training model."""

from fastvideo.train.models.wan.wan_causal import WanCausalModel
from fastvideo.train.models.wantrack.wantrack import WanTrackModel


class WanTrackCausalModel(WanTrackModel, WanCausalModel):
    """WanTrack with block-causal attention and streaming cache support."""

    _transformer_cls_name = "CausalTrackWanTransformer3DModel"
