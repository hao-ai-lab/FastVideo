# SPDX-License-Identifier: Apache-2.0
"""
Per-prompt statistics tracking for GRPO training.

This module ports the PerPromptStatTracker from FlowGRPO to FastVideo.
For GRPO, mean/std are computed from the current batch only; the pipeline
calls clear() after each compute_advantages so the next step gets fresh stats.

Ported from: flow_grpo/flow_grpo/stat_tracking.py
"""

import numpy as np
from typing import Union

import torch

from fastvideo.logger import init_logger

logger = init_logger(__name__)


class PerPromptStatTracker:
    """
    Tracks reward statistics per unique prompt for advantage normalization.
    
    This class maintains running statistics (mean, std) for each unique prompt
    and computes normalized advantages using either per-prompt or global statistics.
    
    Used in GRPO training to normalize advantages within groups of samples
    generated from the same prompt, which helps stabilize training when different
    prompts have different reward scales.
    """

    def __init__(self, global_std: bool = False):
        """
        Initialize the per-prompt stat tracker.
        
        Args:
            global_std: If True, use global std across all rewards for normalization.
                       If False, use per-prompt std (default, recommended for GRPO).
        """
        self.global_std = global_std
        self.stats: dict[str, list] = {}  # Maps prompt -> list of rewards
        self.history_prompts: set[int] = set()  # Set of hashed prompts seen

    def update(
        self,
        prompts: Union[list[str], np.ndarray],
        rewards: Union[list[float], np.ndarray, torch.Tensor],
        type: str = 'grpo'
    ) -> np.ndarray:
        """
        Compute normalized advantages from rewards (GRPO: per-prompt mean/std).

        Caller should call clear() after each step so mean/std use the current batch only.
        """
        prompts = np.array(prompts)
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.detach().cpu().numpy()
        rewards = np.array(rewards, dtype=np.float64)

        if rewards.ndim > 1:
            rewards = rewards.reshape(len(prompts), -1).mean(axis=1)
        assert len(prompts) == len(rewards), \
            f"Prompts ({len(prompts)}) and rewards ({len(rewards)}) must have same length"

        unique_prompts = np.unique(prompts)
        advantages = np.zeros_like(rewards, dtype=np.float64)

        # First pass: store current-batch rewards per prompt (used for mean/std)
        for prompt in unique_prompts:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards.tolist())
            self.history_prompts.add(hash(prompt))

        # Second pass: GRPO (reward - mean) / std per prompt
        for prompt in unique_prompts:
            prompt_mask = prompts == prompt
            prompt_rewards = rewards[prompt_mask]
            all_prompt_rewards = np.array(self.stats[prompt])
            mean = np.mean(all_prompt_rewards, axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4
            else:
                std = np.std(all_prompt_rewards, axis=0, keepdims=True) + 1e-4
            advantages[prompt_mask] = (prompt_rewards - mean) / std

        return advantages

    def get_stats(self) -> tuple[float, int]:
        """
        Get statistics about tracked prompts.
        
        Returns:
            avg_group_size: Average number of samples per unique prompt
            history_prompts: Number of unique prompts seen (across all updates)
        """
        if not self.stats:
            avg_group_size = 0.0
        else:
            total_samples = sum(len(v) for v in self.stats.values())
            avg_group_size = total_samples / len(self.stats)
        
        history_prompts = len(self.history_prompts)
        
        return avg_group_size, history_prompts

    def clear(self) -> None:
        """
        Clear stats so the next update() uses current-batch-only mean/std.
        Called by the pipeline after each compute_advantages (GRPO).
        """
        self.stats = {}
        logger.debug("Cleared per-prompt statistics (kept %d unique prompts in history)", 
                    len(self.history_prompts))
