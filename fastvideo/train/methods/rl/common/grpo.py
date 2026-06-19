# SPDX-License-Identifier: Apache-2.0
"""Shared GRPO/PPO-ratio loss helpers for RL methods."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class GRPOLossResult:
    """Token-masked GRPO objective and scalar diagnostics."""

    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    approx_kl: torch.Tensor
    clipped_fraction: torch.Tensor
    mean_ratio: torch.Tensor
    token_count: torch.Tensor


def compute_grpo_loss(
    *,
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_range: float = 0.2,
    reference_logprobs: torch.Tensor | None = None,
    kl_coef: float = 0.0,
) -> GRPOLossResult:
    """Compute a masked GRPO/PPO-ratio loss.

    Shapes:
      - ``current_logprobs`` and ``old_logprobs``: ``[B, T]``.
      - ``advantages``: ``[B]``.
      - ``response_mask``: ``[B, T]`` with non-zero values for trainable
        response tokens.
    """
    current_logprobs = _require_2d("current_logprobs", current_logprobs)
    old_logprobs = _require_2d("old_logprobs", old_logprobs).to(current_logprobs.device)
    if current_logprobs.shape != old_logprobs.shape:
        raise ValueError("current_logprobs and old_logprobs must have the same shape")

    mask = _require_2d("response_mask", response_mask).to(
        device=current_logprobs.device,
        dtype=current_logprobs.dtype,
    )
    if mask.shape != current_logprobs.shape:
        raise ValueError("response_mask must have the same shape as logprobs")

    advantages = advantages.to(device=current_logprobs.device, dtype=current_logprobs.dtype)
    if advantages.ndim != 1 or int(advantages.shape[0]) != int(current_logprobs.shape[0]):
        raise ValueError("advantages must have shape [B] matching logprobs")

    token_count = mask.sum()
    if float(token_count.detach().cpu()) <= 0.0:
        raise ValueError("GRPO loss requires at least one response token")

    clip = float(clip_range)
    if clip < 0.0:
        raise ValueError("clip_range must be non-negative")

    log_ratio = current_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    expanded_advantages = advantages[:, None]
    surrogate = torch.minimum(
        ratio * expanded_advantages,
        clipped_ratio * expanded_advantages,
    )
    policy_loss = -_masked_mean(surrogate, mask)

    old_policy_kl = _masked_mean((ratio - 1.0) - log_ratio, mask)
    clipped_fraction = _masked_mean((ratio - clipped_ratio).abs().gt(1.0e-6).to(mask.dtype), mask)
    mean_ratio = _masked_mean(ratio, mask)

    if reference_logprobs is None or float(kl_coef) == 0.0:
        kl_loss = torch.zeros((), device=current_logprobs.device, dtype=current_logprobs.dtype)
    else:
        reference_logprobs = _require_2d("reference_logprobs", reference_logprobs).to(current_logprobs.device)
        if reference_logprobs.shape != current_logprobs.shape:
            raise ValueError("reference_logprobs must have the same shape as current_logprobs")
        ref_delta = reference_logprobs - current_logprobs
        kl_loss = _masked_mean(torch.exp(ref_delta) - ref_delta - 1.0, mask)

    total_loss = policy_loss + float(kl_coef) * kl_loss
    return GRPOLossResult(
        total_loss=total_loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        approx_kl=old_policy_kl,
        clipped_fraction=clipped_fraction,
        mean_ratio=mean_ratio,
        token_count=token_count.detach(),
    )


def _require_2d(
    name: str,
    value: torch.Tensor,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [B, T], got {tuple(value.shape)}")
    return value


def _masked_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denom


__all__ = ["GRPOLossResult", "compute_grpo_loss"]
