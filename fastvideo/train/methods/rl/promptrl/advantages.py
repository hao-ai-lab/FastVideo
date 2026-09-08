# SPDX-License-Identifier: Apache-2.0
"""Group-relative advantage computation for PromptRL.

Composite rewards (format + VideoScore2) are normalized within each
``(group_id, reward_tag)`` group.  Groups with effectively zero reward
variance produce zero advantages so they contribute no policy gradient.

Advantage routing:

* The refiner GRPO loss consumes only the *refined* samples'
  advantages; retained-original ranks receive zero refiner advantage
  so every rank still runs compatible refiner forward/backward paths
  and distributed collectives stay aligned.
* Wan's clipped flow-policy loss consumes *all* samples' advantages.

All advantage tensors are detached before either policy loss so no
gradient crosses between the refiner and the generator.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

#: Group reward standard deviations below this are treated as zero.
ZERO_VARIANCE_EPS = 1e-8


def group_relative_advantages(
    rewards: torch.Tensor,
    group_keys: Sequence[str],
    *,
    zero_variance_eps: float = ZERO_VARIANCE_EPS,
) -> torch.Tensor:
    """Normalize *rewards* within each group.

    Args:
        rewards: 1-D float tensor, one scalar per sample.
        group_keys: group label per sample; samples sharing a label are
            normalized together.  Labels typically combine the original
            prompt id and the reward tag so different reward tags never
            share a normalization basin.
        zero_variance_eps: groups whose (population) standard deviation
            is below this threshold produce all-zero advantages.

    Returns:
        Detached float32 advantages with the same shape as *rewards*.
    """
    rewards = rewards.detach().float()
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1-D, got shape {tuple(rewards.shape)}")
    if len(group_keys) != int(rewards.shape[0]):
        raise ValueError(f"group_keys length {len(group_keys)} does not match "
                         f"rewards shape {tuple(rewards.shape)}")

    advantages = torch.zeros_like(rewards)
    unique_keys = sorted(set(group_keys))
    for key in unique_keys:
        indices = [i for i, k in enumerate(group_keys) if k == key]
        index_tensor = torch.tensor(indices, dtype=torch.long, device=rewards.device)
        group = rewards[index_tensor]
        std = group.std(unbiased=False)
        if not torch.isfinite(std) or float(std) < zero_variance_eps:
            continue  # effectively zero variance -> zero advantages
        advantages[index_tensor] = (group - group.mean()) / std
    return advantages.detach()


def route_refiner_advantages(
    advantages: torch.Tensor,
    refiner_participation: Sequence[bool],
) -> torch.Tensor:
    """Zero out advantages for samples that skip the refiner loss.

    Retained-original ranks get exactly zero refiner advantage while
    still executing the refiner forward/backward path (required to keep
    DDP/replicated gradient collectives aligned).
    """
    advantages = advantages.detach().float()
    if len(refiner_participation) != int(advantages.shape[0]):
        raise ValueError("refiner_participation length does not match advantages")
    mask = torch.tensor([1.0 if flag else 0.0 for flag in refiner_participation],
                        dtype=advantages.dtype,
                        device=advantages.device)
    return (advantages * mask).detach()


def route_generator_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """Advantages consumed by the Wan flow-policy loss: every sample."""
    return advantages.detach().float()
