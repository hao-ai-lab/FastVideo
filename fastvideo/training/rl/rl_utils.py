# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for RL/GRPO training.
"""

from typing import Any

import torch
import torch.nn.functional as F

from fastvideo.logger import init_logger

logger = init_logger(__name__)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor | None = None,
    gamma: float = 0.99,
    lambda_: float = 0.95
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE-lambda).

    GAE reduces variance in advantage estimation while allowing some bias.
    This is a key component of modern policy gradient methods like PPO and GRPO.

    Args:
        rewards: Rewards at each step [B, T] or [B]
        values: Value predictions at each step [B, T] or [B]
        next_values: Value predictions at next step [B, T] or [B]
        dones: Episode termination flags [B, T] or [B] (1 if done, 0 otherwise)
        gamma: Discount factor
        lambda_: GAE lambda parameter (0=TD(0), 1=Monte Carlo)

    Returns:
        advantages: GAE advantages [B, T] or [B]
        returns: TD(lambda) returns [B, T] or [B]

    Reference:
        Schulman et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
        https://arxiv.org/abs/1506.02438
    """
    if dones is None:
        dones = torch.zeros_like(rewards)

    # Compute TD residuals: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    deltas = rewards + gamma * next_values * (1.0 - dones) - values

    # If single step (no time dimension), return directly
    if deltas.dim() == 1:
        advantages = deltas
        returns = advantages + values
        return advantages, returns

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

    return advantages, returns


def normalize_advantages(
    advantages: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Normalize advantages to have zero mean and unit variance.

    This is a common practice in PPO and GRPO to stabilize training.

    Args:
        advantages: Raw advantages [B, ...]
        epsilon: Small constant for numerical stability

    Returns:
        normalized_advantages: Normalized advantages [B, ...]
    """
    mean = advantages.mean()
    std = advantages.std()
    return (advantages - mean) / (std + epsilon)

#TODO(jiali): refactor into algorithm
def compute_grpo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2,
    use_ratio_norm: bool = True,
    max_importance_ratio: float = 10.0
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute GRPO policy loss with importance sampling and clipping.

    This implements the core GRPO objective with safety mechanisms from GRPO-Guard:
    - Importance ratio clipping (PPO-style)
    - RatioNorm correction (GRPO-Guard)
    - Ratio clamping for extreme values

    Args:
        log_probs: Log probabilities from current policy [B]
        old_log_probs: Log probabilities from old policy [B]
        advantages: Advantages [B]
        clip_range: Clipping range for importance ratios
        use_ratio_norm: Apply RatioNorm correction (GRPO-Guard)
        max_importance_ratio: Maximum importance ratio before clamping

    Returns:
        loss: Policy loss (scalar)
        info: Dictionary with diagnostic information

    Reference:
        - PPO: Schulman et al. "Proximal Policy Optimization Algorithms"
        - GRPO-Guard: RatioNorm and gradient reweighting
    """
    # Compute importance ratio: r_t = pi_new(a|s) / pi_old(a|s)
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # Clamp extreme ratios for numerical stability
    ratio = torch.clamp(ratio, 1.0 / max_importance_ratio, max_importance_ratio)

    # RatioNorm correction (GRPO-Guard)
    # Corrects bias in importance sampling when ratio >> 1
    if use_ratio_norm:
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
        clip_fraction = ((ratio < 1.0 - clip_range) | (ratio > 1.0 + clip_range)).float().mean()

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
    values: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor | None = None,
    clip_range: float = 0.2,
    use_clipping: bool = True
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute value function loss with optional clipping.

    Args:
        values: Value predictions from current model [B]
        returns: Target returns (from GAE) [B]
        old_values: Value predictions from old model [B] (for clipping)
        clip_range: Clipping range for value updates
        use_clipping: Whether to use clipped value loss (PPO-style)

    Returns:
        loss: Value loss (scalar)
        info: Dictionary with diagnostic information
    """
    # Standard MSE loss
    value_loss_unclipped = F.mse_loss(values, returns, reduction="none")

    # Clipped value loss (PPO-style)
    if use_clipping and old_values is not None:
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


def compute_policy_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute policy entropy for exploration bonus.

    Args:
        log_probs: Log probabilities [B]

    Returns:
        entropy: Mean entropy across batch (scalar)
    """
    # For continuous actions: H = -log_prob (assuming Gaussian)
    # For discrete: H = -sum(p * log(p))
    # Here we use a simple approximation
    entropy = -log_probs.mean()
    return entropy


def apply_gradient_reweighting(
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
    # Compute timestep weights (higher weight for later timesteps)
    # This is a simple linear weighting, can be made more sophisticated
    timestep_weights = 1.0 + (timesteps.float() / num_train_timesteps)
    timestep_weights = timestep_weights.view(-1, *([1] * (gradients.dim() - 1)))

    return gradients * timestep_weights


def sample_random_timesteps(
    batch_size: int,
    min_timestep: int,
    max_timestep: int,
    device: torch.device,
    generator: torch.Generator | None = None
) -> torch.Tensor:
    """
    Sample random timesteps for noise injection (Flow-GRPO-Fast).

    Args:
        batch_size: Number of samples
        min_timestep: Minimum timestep
        max_timestep: Maximum timestep
        device: Device for tensor
        generator: Random generator for reproducibility

    Returns:
        timesteps: Random timesteps [B]
    """
    if generator is not None:
        timesteps = torch.randint(
            min_timestep,
            max_timestep + 1,
            (batch_size,),
            device=device,
            generator=generator
        )
    else:
        timesteps = torch.randint(
            min_timestep,
            max_timestep + 1,
            (batch_size,),
            device=device
        )

    return timesteps


def compute_reward_statistics(
    rewards: torch.Tensor
) -> dict[str, float]:
    """
    Compute statistics for reward distribution.

    Args:
        rewards: Reward values [B]

    Returns:
        stats: Dictionary with mean, std, min, max
    """
    return {
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item(),
        "reward_min": rewards.min().item(),
        "reward_max": rewards.max().item(),
    }


def check_early_stopping(
    kl_divergence: float,
    target_kl: float
) -> bool:
    """
    Check if training should stop early based on KL divergence.

    Args:
        kl_divergence: Current KL divergence
        target_kl: Target KL threshold

    Returns:
        should_stop: True if KL exceeds target
    """
    if kl_divergence > target_kl:
        logger.warning(
            "Early stopping triggered: KL divergence %.4f > target %.4f",
            kl_divergence,
            target_kl
        )
        return True
    return False


def compute_log_probs_from_model_output(
    model_output: torch.Tensor,
    target: torch.Tensor,
    noise_level: float = 0.1
) -> torch.Tensor:
    """
    Compute log probabilities from model predictions.

    For diffusion models, we approximate log probabilities using the
    negative squared error (assuming Gaussian likelihood).

    Args:
        model_output: Model predictions [B, C, T, H, W]
        target: Target values [B, C, T, H, W]
        noise_level: Assumed noise level (std) for Gaussian likelihood

    Returns:
        log_probs: Log probabilities [B]
    """
    # Compute mean squared error per sample
    mse = ((model_output - target) ** 2).flatten(1).mean(dim=1)

    # Log probability under Gaussian: log p(x) = -0.5 * (x - mu)^2 / sigma^2 + const
    log_probs = -0.5 * mse / (noise_level ** 2)

    return log_probs


def check_for_nan_inf(tensor: torch.Tensor, name: str) -> None:
    """
    Check tensor for NaN or Inf values and raise error if found.

    Args:
        tensor: Tensor to check
        name: Name for error message
    """
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values")
    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Inf values")
