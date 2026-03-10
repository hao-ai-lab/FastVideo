# SPDX-License-Identifier: Apache-2.0
"""Per-prompt statistics tracking for advantage computation."""

from __future__ import annotations

import numpy as np
import torch

EPSILON = 1e-4


class PerPromptStatTracker:
    """Track per-prompt reward history and compute advantages."""

    def __init__(
        self,
        use_global_std: bool = False,
        max_group_std: bool = False,
    ):
        self.use_global_std = use_global_std
        self.max_group_std = max_group_std
        self.stats: dict[str, np.ndarray] = {}
        self.history_prompts: set[int] = set()

    def update(
        self,
        prompts,
        rewards,
        mode: str = "grpo",
    ) -> np.ndarray:
        """Update stats and compute advantages.

        Args:
            prompts: Iterable of prompt strings.
            rewards: Array-like rewards aligned with prompts.
            mode: Advantage mode: grpo|rwr|sft|dpo.

        Returns:
            Advantages array aligned with prompts.
        """
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards) * 0.0

        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))
            self.stats[prompt] = np.stack(
                self.stats[prompt]
            )

        max_std = None
        if self.max_group_std and len(unique) > 0:
            prompt_stds = []
            for prompt in unique:
                prompt_std = (
                    np.std(
                        self.stats[prompt],
                        axis=0,
                        keepdims=True,
                    )
                    + EPSILON
                )
                prompt_stds.append(prompt_std)
            max_std_value = max(
                np.max(std) for std in prompt_stds
            )
            max_std = np.full_like(
                prompt_stds[0], max_std_value
            )

        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            mean = np.mean(
                self.stats[prompt], axis=0, keepdims=True
            )
            if self.use_global_std:
                std = (
                    np.std(rewards, axis=0, keepdims=True)
                    + EPSILON
                )
            elif self.max_group_std:
                std = max_std
            else:
                std = (
                    np.std(
                        self.stats[prompt],
                        axis=0,
                        keepdims=True,
                    )
                    + EPSILON
                )
            if mode == "grpo":
                advantages[prompts == prompt] = (
                    prompt_rewards - mean
                ) / std
            elif mode == "rwr":
                advantages[prompts == prompt] = (
                    prompt_rewards
                )
            elif mode == "sft":
                advantages[prompts == prompt] = (
                    (
                        torch.tensor(prompt_rewards)
                        == torch.max(
                            torch.tensor(prompt_rewards)
                        )
                    )
                    .float()
                    .numpy()
                )
            elif mode == "dpo":
                pa = torch.tensor(prompt_rewards)
                max_idx = torch.argmax(pa)
                min_idx = torch.argmin(pa)
                if max_idx == min_idx:
                    min_idx = 0
                    max_idx = 1
                result = torch.zeros_like(pa).float()
                result[max_idx] = 1.0
                result[min_idx] = -1.0
                advantages[prompts == prompt] = (
                    result.numpy()
                )
        return advantages

    def get_stats(self) -> tuple[float, int]:
        """Return (avg_group_size, num_unique_prompts)."""
        avg = (
            sum(len(v) for v in self.stats.values())
            / len(self.stats)
            if self.stats
            else 0
        )
        return avg, len(self.history_prompts)

    def clear(self):
        """Clear stored statistics."""
        self.stats = {}
