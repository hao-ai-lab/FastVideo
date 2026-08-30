# SPDX-License-Identifier: Apache-2.0
"""Reusable RL training primitives."""

from fastvideo.train.methods.rl.common.minimax_h3_rvm import (
    H3RVMSamplingConfig,
    H3RVMSamplingResult,
    MiniMaxH3RVMSampler,
)
from fastvideo.train.methods.rl.common.prompt_sampling import (
    KRepeatSample,
    distributed_k_repeat_indices,
)
from fastvideo.train.methods.rl.common.rvm_utils import (
    detached_rvm_surrogate,
    five_percent_interval,
    partition_indices,
    standardize_group_rewards,
    visual_text_from_h3_prompt,
)
from fastvideo.train.methods.rl.common.sampling import (
    DiffusionSampler,
    SamplingConfig,
    SamplingResult,
)
from fastvideo.train.methods.rl.common.validation import (
    RLValidationConfig,
    media_to_video_array,
    validation_caption,
    validation_shard_indices,
)

__all__ = [
    "DiffusionSampler",
    "H3RVMSamplingConfig",
    "H3RVMSamplingResult",
    "KRepeatSample",
    "MiniMaxH3RVMSampler",
    "RLValidationConfig",
    "SamplingConfig",
    "SamplingResult",
    "detached_rvm_surrogate",
    "distributed_k_repeat_indices",
    "five_percent_interval",
    "media_to_video_array",
    "partition_indices",
    "standardize_group_rewards",
    "validation_caption",
    "validation_shard_indices",
    "visual_text_from_h3_prompt",
]
