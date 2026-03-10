# SPDX-License-Identifier: Apache-2.0
"""Reward functions for RL-based video generation training."""

from fastvideo.train.methods.rl.reward.hpsv3 import (
    hpsv3_general_score,
    hpsv3_percentile_score,
)
from fastvideo.train.methods.rl.reward.ocr import (
    video_ocr_score,
)
from fastvideo.train.methods.rl.reward.videoalign import (
    videoalign_mq_score,
    videoalign_ta_score,
)

__all__ = [
    "hpsv3_general_score",
    "hpsv3_percentile_score",
    "video_ocr_score",
    "videoalign_mq_score",
    "videoalign_ta_score",
]
