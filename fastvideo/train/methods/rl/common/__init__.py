# SPDX-License-Identifier: Apache-2.0
"""Reusable RL training primitives."""

from fastvideo.train.methods.rl.common.sampling import (
    DiffusionSampler,
    SamplingConfig,
    SamplingResult,
)
from fastvideo.train.methods.rl.common.prompt_sampling import (
    KRepeatSample,
    distributed_k_repeat_indices,
)
from fastvideo.train.methods.rl.common.grpo import (
    GRPOLossResult,
    compute_grpo_loss,
)
from fastvideo.train.methods.rl.common.validation import (
    RLValidationConfig,
    media_to_video_array,
    validation_caption,
    validation_shard_indices,
)

__all__ = [
    "DiffusionSampler",
    "GRPOLossResult",
    "KRepeatSample",
    "RLValidationConfig",
    "SamplingConfig",
    "SamplingResult",
    "compute_grpo_loss",
    "distributed_k_repeat_indices",
    "media_to_video_array",
    "validation_caption",
    "validation_shard_indices",
]
