# SPDX-License-Identifier: Apache-2.0
"""Reusable reward models for training methods."""

from fastvideo.train.methods.rl.rewards.frame_rewards import (
    ClipScoreScorer,
    PickScoreScorer,
)
from fastvideo.train.methods.rl.rewards.interleave_api import (
    ConstantInterleaveEditScorer,
    GeminiInterleaveImageScorer,
    GeminiNanoBananaEditScorer,
)
from fastvideo.train.methods.rl.rewards.interleave_thinker import (
    EditScoreProvider,
    InterleaveThinkerAnswer,
    InterleaveThinkerEditRequest,
    InterleaveThinkerEditScore,
    InterleaveThinkerRewardResult,
    InterleaveThinkerRewardScorer,
    extract_interleave_answer,
    interleave_format_reward,
    interleave_judge_accuracy_reward,
    normalize_interleave_response,
    score_interleave_thinker_rewards,
)
from fastvideo.train.methods.rl.rewards.media import (
    MultiRewardScorer,
    RewardScorer,
    select_first_frame,
)


def build_multi_reward_scorer(
    reward_weights,
    *,
    device="cuda",
    scorers: dict[str, RewardScorer] | None = None,
) -> MultiRewardScorer:
    available: dict[str, RewardScorer] = dict(scorers or {})
    if not available:
        available = {
            "pickscore": PickScoreScorer(device=device),
            "clipscore": ClipScoreScorer(device=device),
        }
    return MultiRewardScorer(reward_weights, scorers=available)


__all__ = [
    "ClipScoreScorer",
    "ConstantInterleaveEditScorer",
    "EditScoreProvider",
    "GeminiInterleaveImageScorer",
    "GeminiNanoBananaEditScorer",
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
    "interleave_format_reward",
    "interleave_judge_accuracy_reward",
    "normalize_interleave_response",
    "score_interleave_thinker_rewards",
    "select_first_frame",
]
