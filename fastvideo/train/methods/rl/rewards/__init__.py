# SPDX-License-Identifier: Apache-2.0
"""Reusable reward models for training methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastvideo.train.methods.rl.rewards.interleave_thinker import (
    EditScoreProvider,
    InterleavePlannerRewardResult,
    InterleavePlannerRewardScorer,
    InterleaveThinkerAnswer,
    InterleaveThinkerEditRequest,
    InterleaveThinkerEditScore,
    InterleaveThinkerRewardResult,
    InterleaveThinkerRewardScorer,
    extract_interleave_answer,
    extract_interleave_plan_payload,
    interleave_format_reward,
    interleave_planner_format_reward,
    interleave_judge_accuracy_reward,
    normalize_interleave_response,
    score_interleave_planner_rewards,
    score_interleave_thinker_rewards,
)

if TYPE_CHECKING:
    from fastvideo.train.methods.rl.rewards.frame_rewards import (
        ClipScoreScorer,
        PickScoreScorer,
    )
    from fastvideo.train.methods.rl.rewards.interleave_api import (
        ConstantInterleaveEditScorer,
        GeminiInterleaveImageScorer,
        GeminiNanoBananaEditScorer,
    )
    from fastvideo.train.methods.rl.rewards.media import (
        MultiRewardScorer,
        RewardScorer,
    )


def build_multi_reward_scorer(
    reward_weights,
    *,
    device="cuda",
    scorers: dict[str, RewardScorer] | None = None,
) -> MultiRewardScorer:
    from fastvideo.train.methods.rl.rewards.frame_rewards import (
        ClipScoreScorer,
        PickScoreScorer,
    )
    from fastvideo.train.methods.rl.rewards.media import MultiRewardScorer

    available: dict[str, RewardScorer] = dict(scorers or {})
    if not available:
        available = {
            "pickscore": PickScoreScorer(device=device),
            "clipscore": ClipScoreScorer(device=device),
        }
    return MultiRewardScorer(reward_weights, scorers=available)


def __getattr__(name: str) -> object:
    if name in {"ClipScoreScorer", "PickScoreScorer"}:
        from fastvideo.train.methods.rl.rewards.frame_rewards import (
            ClipScoreScorer,
            PickScoreScorer,
        )

        return {
            "ClipScoreScorer": ClipScoreScorer,
            "PickScoreScorer": PickScoreScorer,
        }[name]
    if name in {
            "ConstantInterleaveEditScorer",
            "GeminiInterleaveImageScorer",
            "GeminiNanoBananaEditScorer",
    }:
        from fastvideo.train.methods.rl.rewards.interleave_api import (
            ConstantInterleaveEditScorer,
            GeminiInterleaveImageScorer,
            GeminiNanoBananaEditScorer,
        )

        return {
            "ConstantInterleaveEditScorer": ConstantInterleaveEditScorer,
            "GeminiInterleaveImageScorer": GeminiInterleaveImageScorer,
            "GeminiNanoBananaEditScorer": GeminiNanoBananaEditScorer,
        }[name]
    if name in {"MultiRewardScorer", "RewardScorer", "select_first_frame"}:
        from fastvideo.train.methods.rl.rewards.media import (
            MultiRewardScorer,
            RewardScorer,
            select_first_frame,
        )

        return {
            "MultiRewardScorer": MultiRewardScorer,
            "RewardScorer": RewardScorer,
            "select_first_frame": select_first_frame,
        }[name]
    raise AttributeError(name)


__all__ = [
    "ClipScoreScorer",
    "ConstantInterleaveEditScorer",
    "EditScoreProvider",
    "GeminiInterleaveImageScorer",
    "GeminiNanoBananaEditScorer",
    "InterleavePlannerRewardResult",
    "InterleavePlannerRewardScorer",
    "InterleaveThinkerAnswer",
    "InterleaveThinkerEditRequest",
    "InterleaveThinkerEditScore",
    "InterleaveThinkerRewardResult",
    "InterleaveThinkerRewardScorer",
    "MultiRewardScorer",
    "PickScoreScorer",
    "RewardScorer",
    "build_multi_reward_scorer",
    "extract_interleave_answer",
    "extract_interleave_plan_payload",
    "interleave_format_reward",
    "interleave_planner_format_reward",
    "interleave_judge_accuracy_reward",
    "normalize_interleave_response",
    "score_interleave_planner_rewards",
    "score_interleave_thinker_rewards",
    "select_first_frame",
]
