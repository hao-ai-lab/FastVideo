# SPDX-License-Identifier: Apache-2.0
"""
Per-prompt statistics tracking for GRPO training.

This module ports the PerPromptStatTracker from FlowGRPO to FastVideo.
It tracks reward statistics per unique prompt and computes normalized advantages.

Ported from:
- flow_grpo/flow_grpo/stat_tracking.py

Key adaptations:
1. Uses FastVideo's logging instead of print statements
2. Works with single GPU (no distributed logic)
3. Supports numpy arrays and torch tensors
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
        Update statistics and compute normalized advantages.
        
        Args:
            prompts: List or array of prompt strings (one per sample)
            rewards: Array or tensor of reward values (one per sample)
            type: Advantage computation type:
                - 'grpo': Normalize by (reward - mean) / std (default)
                - 'rwr': Return rewards as-is (reward-weighted regression)
                - 'sft': Binary advantages (1 for max, 0 otherwise)
                - 'dpo': DPO-style advantages (1 for max, -1 for min)
        
        Returns:
            advantages: Normalized advantages array [num_samples] or [num_samples, ...]
                       Shape matches rewards shape
        """
        # Convert to numpy arrays
        prompts = np.array(prompts)
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.detach().cpu().numpy()
        rewards = np.array(rewards, dtype=np.float64)
        
        # Ensure rewards are 1D (one reward per sample)
        # FlowGRPO expects rewards to be aggregated per sample
        if rewards.ndim > 1:
            # If multi-dimensional, flatten or take mean
            # For [B, num_steps] shape, we typically want one reward per sample
            # So we take the mean across timesteps
            if rewards.ndim == 2:
                # Assume shape is [B, num_steps] - take mean across timesteps
                rewards = rewards.mean(axis=1)
            else:
                # Flatten and take mean for higher dimensions
                rewards = rewards.reshape(len(prompts), -1).mean(axis=1)
        
        # Ensure prompts and rewards have matching lengths
        assert len(prompts) == len(rewards), \
            f"Prompts ({len(prompts)}) and rewards ({len(rewards)}) must have same length"
        
        unique_prompts = np.unique(prompts)
        advantages = np.zeros_like(rewards, dtype=np.float64)
        
        # First pass: collect rewards for each prompt
        for prompt in unique_prompts:
            prompt_mask = prompts == prompt
            prompt_rewards = rewards[prompt_mask]
            
            # Store rewards in stats
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards.tolist())
            self.history_prompts.add(hash(prompt))
        
        # Second pass: compute statistics and advantages
        for prompt in unique_prompts:
            prompt_mask = prompts == prompt
            prompt_rewards = rewards[prompt_mask]
            
            # Stack all historical rewards for this prompt
            if len(self.stats[prompt]) > 0:
                all_prompt_rewards = np.array(self.stats[prompt])
            else:
                all_prompt_rewards = prompt_rewards
            
            # Compute mean and std
            mean = np.mean(all_prompt_rewards, axis=0, keepdims=True)
            
            if self.global_std:
                # Use global std across all rewards
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4
            else:
                # Use per-prompt std
                std = np.std(all_prompt_rewards, axis=0, keepdims=True) + 1e-4
            
            # Compute advantages based on type
            if type == 'grpo':
                # GRPO: normalize by (reward - mean) / std
                advantages[prompt_mask] = (prompt_rewards - mean) / std
            elif type == 'rwr':
                # Reward-weighted regression: use rewards as-is
                advantages[prompt_mask] = prompt_rewards
            elif type == 'sft':
                # Supervised fine-tuning: binary (1 for max, 0 otherwise)
                max_reward = np.max(prompt_rewards)
                advantages[prompt_mask] = (prompt_rewards == max_reward).astype(np.float64)
            elif type == 'dpo':
                # DPO-style: 1 for max, -1 for min
                prompt_rewards_tensor = torch.tensor(prompt_rewards)
                max_idx = torch.argmax(prompt_rewards_tensor)
                min_idx = torch.argmin(prompt_rewards_tensor)
                
                # If all rewards are the same, use first two indices
                if max_idx == min_idx:
                    min_idx = torch.tensor(0)
                    max_idx = torch.tensor(1) if len(prompt_rewards_tensor) > 1 else torch.tensor(0)
                
                result = torch.zeros_like(prompt_rewards_tensor, dtype=torch.float64)
                result[max_idx] = 1.0
                result[min_idx] = -1.0
                advantages[prompt_mask] = result.numpy()
            else:
                raise ValueError(f"Unknown advantage type: {type}. Must be one of: 'grpo', 'rwr', 'sft', 'dpo'")
        
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
        Clear all statistics (but keep history_prompts for tracking).
        
        This is typically called after each epoch to reset per-epoch statistics
        while maintaining a record of all prompts seen during training.
        """
        self.stats = {}
        logger.debug("Cleared per-prompt statistics (kept %d unique prompts in history)", 
                    len(self.history_prompts))
