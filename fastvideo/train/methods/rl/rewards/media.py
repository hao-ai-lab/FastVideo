# SPDX-License-Identifier: Apache-2.0
"""Generic media reward composition utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch

RewardScorer = Callable[[torch.Tensor, Sequence[str]], torch.Tensor]


def select_first_frame(media: torch.Tensor) -> torch.Tensor:
    """Return first-frame media as ``[B, C, H, W]``.

    This is a helper for reward models that are intrinsically frame-based
    (for example PickScore and CLIPScore). Video-aware rewards should inspect
    the full ``[B, C, T, H, W]`` tensor themselves.
    """
    if not torch.is_tensor(media):
        raise TypeError(f"media must be a torch.Tensor, got {type(media).__name__}")
    if media.ndim == 5:
        return media[:, :, 0]
    if media.ndim == 4:
        return media
    raise ValueError("media must have shape [B, C, H, W] or [B, C, T, H, W], "
                     f"got {tuple(media.shape)}")


class MultiRewardScorer:
    """Weighted sum of reusable media reward scorers.

    Mirrors DiffusionNFT's ``flow_grpo/rewards.py::multi_score`` behavior,
    while leaving frame selection to each concrete reward.
    """

    def __init__(
        self,
        reward_weights: Mapping[str, float],
        *,
        scorers: Mapping[str, RewardScorer],
    ) -> None:
        self.reward_weights = {str(k): float(v) for k, v in reward_weights.items()}
        if not self.reward_weights:
            raise ValueError("reward_weights must contain at least one reward")

        self.scorers = dict(scorers)
        unsupported = sorted(set(self.reward_weights) - set(self.scorers))
        if unsupported:
            raise ValueError(f"Unsupported reward(s): {unsupported}. "
                             f"Available rewards: {sorted(self.scorers)}")

    @torch.no_grad()
    def __call__(
        self,
        media: torch.Tensor,
        prompts: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        prompt_count = len(prompts)
        if media.shape[0] != prompt_count:
            raise ValueError(f"media batch size ({media.shape[0]}) must match prompt count ({prompt_count})")
        total: torch.Tensor | None = None
        details: dict[str, torch.Tensor] = {}
        for name, weight in self.reward_weights.items():
            scores = self.scorers[name](media, prompts).detach().float()
            if scores.ndim != 1 or int(scores.shape[0]) != prompt_count:
                raise ValueError(f"Reward {name!r} must return shape [{prompt_count}], got {tuple(scores.shape)}")
            details[name] = scores
            weighted = scores * float(weight)
            total = weighted if total is None else total.to(weighted.device) + weighted
        assert total is not None
        details["avg"] = total
        return details
