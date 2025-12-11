# SPDX-License-Identifier: Apache-2.0
"""
GRPO (Group Relative Policy Optimization) algorithm implementation.

This module implements GRPO and Flow-GRPO for video generation models.
It includes GRPO-Guard safety mechanisms for stable training.

References:
    - Flow-GRPO: https://github.com/yifan123/flow_grpo
    - GRPO: DeepSeek-R1 paper
"""

from typing import Any

import torch
import torch.nn.functional as F

from fastvideo.logger import init_logger
from .base import BaseRLAlgorithm

logger = init_logger(__name__)


class GRPOAlgorithm(BaseRLAlgorithm):
    """
    GRPO (Group Relative Policy Optimization) algorithm.

    GRPO is a variant of PPO that uses group-relative advantages and
    additional safety mechanisms (GRPO-Guard) for stable training of
    generative models.

    Key features:
    - PPO-style clipped surrogate objective
    - RatioNorm correction for importance sampling bias
    - Gradient reweighting across denoising steps
    - Optional value function for variance reduction
    """

    @property
    def name(self) -> str:
        return "grpo"

    @property
    def requires_value_model(self) -> bool:
        # GRPO can work with or without a value model
        # When using single-step (Flow-GRPO-Fast), value model is optional
        return self.config.rl_use_value_model

    @property
    def requires_reference_model(self) -> bool:
        # GRPO doesn't require a separate reference model
        # KL is computed implicitly through the importance ratio
        return False

    def _validate_config(self) -> None:
        """Validate GRPO-specific configuration."""
        config = self.config

        if config.rl_policy_clip_range <= 0:
            raise ValueError(
                f"rl_policy_clip_range must be positive, got {config.rl_policy_clip_range}"
            )

        if config.rl_max_importance_ratio <= 1.0:
            raise ValueError(
                f"rl_max_importance_ratio must be > 1.0, got {config.rl_max_importance_ratio}"
            )

        logger.info(
            "GRPO config validated: clip_range=%.2f, max_ratio=%.1f, "
            "use_guard=%s, ratio_norm=%s",
            config.rl_policy_clip_range,
            config.rl_max_importance_ratio,
            config.rl_use_grpo_guard,
            config.rl_ratio_norm_correction
        )

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using GAE-lambda.

        Args:
            rewards: Rewards [B] or [B, T]
            values: Value predictions [B] or [B, T]
            next_values: Next value predictions [B] or [B, T]
            dones: Episode termination flags [B] or [B, T]

        Returns:
            advantages: GAE advantages
            returns: TD(lambda) returns
        """
        if dones is None:
            dones = torch.zeros_like(rewards)

        gamma = self.config.rl_gamma
        lambda_ = self.config.rl_lambda

        # Compute TD residuals: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        deltas = rewards + gamma * next_values * (1.0 - dones) - values

        # If single step (no time dimension), return directly
        if deltas.dim() == 1:
            advantages = deltas
            returns = advantages + values
            return self._normalize_advantages(advantages), returns

        # Multi-step: compute GAE recursively
        batch_size, num_steps = deltas.shape
        advantages = torch.zeros_like(deltas)
        gae = torch.zeros(batch_size, device=deltas.device)

        # Backward pass to compute GAE
        for t in reversed(range(num_steps)):
            gae = deltas[:, t] + gamma * lambda_ * (1.0 - dones[:, t]) * gae
            advantages[:, t] = gae

        # Returns are advantages + values
        returns = advantages + values

        return self._normalize_advantages(advantages), returns

    def _normalize_advantages(
        self,
        advantages: torch.Tensor,
        epsilon: float = 1e-8
    ) -> torch.Tensor:
        """Normalize advantages to have zero mean and unit variance."""
        if self.config.rl_normalize_advantages:
            mean = advantages.mean()
            std = advantages.std()
            return (advantages - mean) / (std + epsilon)
        return advantages

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Compute GRPO policy loss with importance sampling and clipping.

        Implements the core GRPO objective with GRPO-Guard safety mechanisms:
        - Importance ratio clipping (PPO-style)
        - RatioNorm correction (GRPO-Guard)
        - Ratio clamping for extreme values

        Args:
            log_probs: Log probabilities from current policy [B]
            old_log_probs: Log probabilities from old policy [B]
            advantages: Advantages [B]

        Returns:
            loss: Policy loss (scalar)
            info: Dictionary with diagnostic information
        """
        clip_range = self.config.rl_policy_clip_range
        use_ratio_norm = self.config.rl_ratio_norm_correction
        max_importance_ratio = self.config.rl_max_importance_ratio

        # Compute importance ratio: r_t = pi_new(a|s) / pi_old(a|s)
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        # Clamp extreme ratios for numerical stability
        ratio = torch.clamp(ratio, 1.0 / max_importance_ratio, max_importance_ratio)

        # RatioNorm correction (GRPO-Guard)
        # Corrects bias in importance sampling when ratio >> 1
        if use_ratio_norm and self.config.rl_use_grpo_guard:
            ratio_mean = ratio.mean()
            ratio = ratio / (ratio_mean + 1e-8)

        # Clipped surrogate objective
        ratio_clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        surrogate1 = ratio * advantages
        surrogate2 = ratio_clipped * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # Compute diagnostics
        with torch.no_grad():
            # Clip fraction: how often ratios were clipped
            clip_fraction = (
                (ratio < 1.0 - clip_range) | (ratio > 1.0 + clip_range)
            ).float().mean()

            # KL divergence (approximate)
            kl_div = log_ratio.mean()

            # Importance ratio stats
            importance_ratio_mean = ratio.mean()
            importance_ratio_std = ratio.std()

        info = {
            "policy_loss": policy_loss.item(),
            "clip_fraction": clip_fraction.item(),
            "kl_divergence": kl_div.item(),
            "importance_ratio_mean": importance_ratio_mean.item(),
            "importance_ratio_std": importance_ratio_std.item(),
        }

        return policy_loss, info

    def compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Compute value function loss with optional clipping.

        Args:
            values: Value predictions from current model [B]
            returns: Target returns [B]
            old_values: Value predictions from old model [B] (for clipping)

        Returns:
            loss: Value loss (scalar)
            info: Dictionary with diagnostic information
        """
        clip_range = self.config.rl_value_clip_range

        # Standard MSE loss
        value_loss_unclipped = F.mse_loss(values, returns, reduction="none")

        # Clipped value loss (PPO-style)
        if old_values is not None:
            values_clipped = old_values + torch.clamp(
                values - old_values,
                -clip_range,
                clip_range
            )
            value_loss_clipped = F.mse_loss(values_clipped, returns, reduction="none")
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
        else:
            value_loss = value_loss_unclipped.mean()

        # Compute diagnostics
        with torch.no_grad():
            explained_variance = 1.0 - (returns - values).var() / (returns.var() + 1e-8)

        info = {
            "value_loss": value_loss.item(),
            "explained_variance": explained_variance.item(),
            "value_mean": values.mean().item(),
            "value_std": values.std().item(),
        }

        return value_loss, info

    def apply_gradient_reweighting(
        self,
        gradients: torch.Tensor,
        timesteps: torch.Tensor,
        num_train_timesteps: int = 1000
    ) -> torch.Tensor:
        """
        Apply GRPO-Guard gradient reweighting across denoising steps.

        This reweights gradients based on the timestep to balance learning
        across different noise levels.

        Args:
            gradients: Gradients to reweight [B, ...]
            timesteps: Timesteps at which gradients were computed [B]
            num_train_timesteps: Total number of training timesteps

        Returns:
            reweighted_gradients: Reweighted gradients [B, ...]
        """
        if not self.config.rl_gradient_reweighting or not self.config.rl_use_grpo_guard:
            return gradients

        # Compute timestep weights (higher weight for later timesteps)
        timestep_weights = 1.0 + (timesteps.float() / num_train_timesteps)
        timestep_weights = timestep_weights.view(-1, *([1] * (gradients.dim() - 1)))

        return gradients * timestep_weights
