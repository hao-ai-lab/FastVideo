# SPDX-License-Identifier: Apache-2.0
"""PromptRL: prompt-refinement RL for Wan video generation."""

from fastvideo.train.methods.rl.promptrl.advantages import (
    group_relative_advantages,
    route_generator_advantages,
    route_refiner_advantages,
)
from fastvideo.train.methods.rl.promptrl.config import (
    PromptRLMethodConfig,
    RefinerSamplingConfig,
    RewardServiceConfig,
    RoleOptimizerConfig,
    RolloutConfig,
)
from fastvideo.train.methods.rl.promptrl.bundle import (
    BundleManifest,
    export_promptrl_bundle,
    extract_generator_lora,
    load_bundle_manifest,
)
from fastvideo.train.methods.rl.promptrl.inference import (
    PromptRefiner,
    RefinementResult,
)
from fastvideo.train.methods.rl.promptrl.method import PromptRLMethod
from fastvideo.train.methods.rl.promptrl.prompts import (
    GroupAssignment,
    ParsedCompletion,
    PromptDataset,
    PromptRecord,
    group_assignments,
    parse_answer_tag,
    render_refinement_prompt,
)
from fastvideo.train.methods.rl.promptrl.rewards import (
    HttpRewardProvider,
    RewardProvider,
    RewardResult,
    RewardSample,
    RewardServiceError,
)

__all__ = [
    "BundleManifest",
    "GroupAssignment",
    "HttpRewardProvider",
    "ParsedCompletion",
    "PromptDataset",
    "PromptRLMethod",
    "PromptRLMethodConfig",
    "PromptRecord",
    "PromptRefiner",
    "RefinementResult",
    "RefinerSamplingConfig",
    "RewardProvider",
    "RewardResult",
    "RewardSample",
    "RewardServiceConfig",
    "RewardServiceError",
    "RoleOptimizerConfig",
    "RolloutConfig",
    "export_promptrl_bundle",
    "extract_generator_lora",
    "group_assignments",
    "group_relative_advantages",
    "load_bundle_manifest",
    "parse_answer_tag",
    "render_refinement_prompt",
    "route_generator_advantages",
    "route_refiner_advantages",
]
