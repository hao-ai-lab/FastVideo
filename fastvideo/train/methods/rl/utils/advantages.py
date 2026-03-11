# SPDX-License-Identifier: Apache-2.0
"""Advantage computation for RL training."""

from __future__ import annotations

from typing import Any

import numpy as np

from fastvideo.logger import init_logger
from fastvideo.train.methods.rl.utils.stat_tracking import (
    EPSILON,
    PerPromptStatTracker,
)

logger = init_logger(__name__)


def _normalize_rewards(
    rewards: np.ndarray,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """Normalize rewards to zero mean and unit variance."""
    return (rewards - rewards.mean()) / (
        rewards.std() + epsilon
    )


def _compute_kl_advantages(
    gathered_kl: np.ndarray,
    kl_stat_tracker: PerPromptStatTracker | None,
    prompts: list[str] | None,
    use_per_prompt: bool,
) -> np.ndarray:
    """Compute KL advantages (negative = penalty)."""
    if use_per_prompt and kl_stat_tracker is not None:
        return kl_stat_tracker.update(
            prompts, -gathered_kl
        )
    return _normalize_rewards(-gathered_kl)


def calculate_zero_std_ratio(
    prompts,
    gathered_rewards: dict[str, np.ndarray],
    reward_key: str = "avg",
) -> float:
    """Compute fraction of prompts with zero std."""
    prompts_arr = np.array(prompts)
    rewards = gathered_rewards.get(reward_key)
    if rewards is None:
        return 0.0
    unique = np.unique(prompts_arr)
    zero_count = 0
    for p in unique:
        r = rewards[prompts_arr == p]
        if np.std(r) < EPSILON:
            zero_count += 1
    return zero_count / max(len(unique), 1)


def compute_advantages(
    reward_fn_cfg: dict[str, float],
    weight_advantages: bool,
    per_prompt_stat_tracking: bool,
    kl_reward: float,
    samples: dict[str, Any],
    gathered_rewards: dict[str, np.ndarray],
    gathered_kl: np.ndarray,
    prompts: list[str] | None,
    stat_tracker: PerPromptStatTracker | None,
    reward_stat_trackers: (
        dict[str, PerPromptStatTracker] | None
    ),
    kl_stat_tracker: PerPromptStatTracker | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute advantages from gathered rewards and KL.

    Supports two modes:
    - Mode 1 (default): Weight rewards, then advantages.
    - Mode 2 (weight_advantages=True): Per-reward
      advantages, then weight.

    Returns:
        (advantages, log_dict)
    """
    log_dict: dict[str, Any] = {}

    if weight_advantages:
        if per_prompt_stat_tracking:
            if reward_stat_trackers is None:
                msg = (
                    "reward_stat_trackers required when "
                    "weight_advantages=True and "
                    "per_prompt_stat_tracking=True"
                )
                raise ValueError(msg)

            weighted_list = []
            for reward_name in reward_fn_cfg:
                raw_key = f"{reward_name}_raw"
                adv = reward_stat_trackers[
                    reward_name
                ].update(
                    prompts, gathered_rewards[raw_key]
                )
                weight = reward_fn_cfg[reward_name]
                weighted_list.append(adv * weight)

            if kl_reward > 0:
                if kl_stat_tracker is None:
                    msg = (
                        "kl_stat_tracker required when "
                        "weight_advantages=True and "
                        "kl_reward > 0"
                    )
                    raise ValueError(msg)
                kl_adv = _compute_kl_advantages(
                    gathered_kl,
                    kl_stat_tracker,
                    prompts,
                    use_per_prompt=True,
                )
                weighted_list.append(
                    kl_adv * kl_reward
                )

            advantages = sum(weighted_list)

            first_name = next(iter(reward_fn_cfg))
            group_size, trained_num = (
                reward_stat_trackers[
                    first_name
                ].get_stats()
            )
            zero_std_ratios = {}
            for rn in reward_fn_cfg:
                raw_key = f"{rn}_raw"
                zero_std_ratios[
                    f"zero_std_ratio_{rn}"
                ] = calculate_zero_std_ratio(
                    prompts,
                    gathered_rewards,
                    reward_key=f"ori_{raw_key}",
                )
            log_dict = {
                "group_size": group_size,
                "trained_prompt_num": trained_num,
                **zero_std_ratios,
            }
            for t in reward_stat_trackers.values():
                t.clear()
            if kl_stat_tracker is not None:
                kl_stat_tracker.clear()
        else:
            weighted_list = []
            for reward_name in reward_fn_cfg:
                raw_key = f"{reward_name}_raw"
                raw = gathered_rewards[raw_key]
                adv = _normalize_rewards(raw)
                weight = reward_fn_cfg[reward_name]
                weighted_list.append(adv * weight)

            if kl_reward > 0:
                kl_adv = _compute_kl_advantages(
                    gathered_kl,
                    None,
                    None,
                    use_per_prompt=False,
                )
                weighted_list.append(
                    kl_adv * kl_reward
                )

            advantages = sum(weighted_list)

    elif per_prompt_stat_tracking:
        if stat_tracker is None:
            msg = (
                "stat_tracker required when "
                "per_prompt_stat_tracking=True"
            )
            raise ValueError(msg)
        advantages = stat_tracker.update(
            prompts, gathered_rewards["avg"]
        )
        group_size, trained_num = (
            stat_tracker.get_stats()
        )
        zero_std = calculate_zero_std_ratio(
            prompts, gathered_rewards
        )
        log_dict = {
            "group_size": group_size,
            "trained_prompt_num": trained_num,
            "zero_std_ratio": zero_std,
        }
        stat_tracker.clear()
    else:
        advantages = _normalize_rewards(
            gathered_rewards["avg"]
        )

    return advantages, log_dict
