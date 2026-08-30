# SPDX-License-Identifier: Apache-2.0
"""RAFT dynamic-tracking reward from the RVM video experiments."""

from __future__ import annotations

import math
from typing import Any

import torch

_RAFT_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


def _get_raft(device: torch.device, variant: str) -> tuple[Any, Any]:
    key = (str(device), variant)
    if key in _RAFT_CACHE:
        return _RAFT_CACHE[key]
    try:
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )
    except ImportError as exc:
        raise ImportError("DynamicTrackingScorer requires torchvision optical-flow models.") from exc
    if variant == "large":
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=True)
    elif variant == "small":
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=True)
    else:
        raise ValueError("RAFT variant must be 'large' or 'small'")
    model = model.eval().requires_grad_(False).to(device)
    transforms = weights.transforms()
    _RAFT_CACHE[key] = (model, transforms)
    return model, transforms


def _pair_indices(num_frames: int, num_pairs: int) -> list[tuple[int, int]]:
    if num_frames < 2:
        raise ValueError("Dynamic tracking requires at least two video frames")
    pairs = min(int(num_pairs), num_frames - 1)
    if pairs <= 0:
        raise ValueError("num_pairs must be positive")
    indices = torch.linspace(0, num_frames - 1, pairs + 1).round().long().tolist()
    return list(zip(indices[:-1], indices[1:], strict=True))


class DynamicTrackingScorer:
    """Resolution-aware RVM DT reward in ``[0, 1]``.

    For every equally spaced frame pair, RAFT estimates optical flow. The pair
    score is the mean magnitude of the fastest 5% of pixels, divided by
    ``tau = 6 * min(H, W) / 256`` and clipped at one. Scores are averaged over
    pairs exactly as specified in Appendix C.3 of the RVM paper.
    """

    def __init__(
        self,
        *,
        device: torch.device | str = "cuda",
        num_pairs: int = 8,
        top_fraction: float = 0.05,
        variant: str = "large",
        pair_batch_size: int = 4,
    ) -> None:
        self.device = torch.device(device)
        self.num_pairs = int(num_pairs)
        self.top_fraction = float(top_fraction)
        self.variant = str(variant).strip().lower()
        self.pair_batch_size = int(pair_batch_size)
        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1]")
        if self.pair_batch_size <= 0:
            raise ValueError("pair_batch_size must be positive")

    @torch.no_grad()
    def __call__(self, media: torch.Tensor, prompts) -> torch.Tensor:
        del prompts
        if media.ndim != 5 or media.shape[1] not in (1, 3):
            raise ValueError(f"media must be [B,C,T,H,W], got {tuple(media.shape)}")
        videos = media.detach().float().clamp(0, 1).to(self.device)
        if videos.shape[1] == 1:
            videos = videos.repeat(1, 3, 1, 1, 1)
        batch_size, _, num_frames, height, width = videos.shape
        pairs = _pair_indices(num_frames, self.num_pairs)
        first: list[torch.Tensor] = []
        second: list[torch.Tensor] = []
        owners: list[int] = []
        for sample_index in range(batch_size):
            for first_index, second_index in pairs:
                first.append(videos[sample_index, :, first_index])
                second.append(videos[sample_index, :, second_index])
                owners.append(sample_index)

        model, transforms = _get_raft(self.device, self.variant)
        pair_scores: list[torch.Tensor] = []
        threshold = 6.0 * min(height, width) / 256.0
        for start in range(0, len(first), self.pair_batch_size):
            image1 = torch.stack(first[start : start + self.pair_batch_size])
            image2 = torch.stack(second[start : start + self.pair_batch_size])
            image1, image2 = transforms(image1, image2)
            flow = model(image1, image2)[-1].float()
            magnitude = torch.linalg.vector_norm(flow, dim=1).flatten(1)
            top_k = max(1, int(math.floor(self.top_fraction * magnitude.shape[1])))
            fastest = magnitude.topk(top_k, dim=1, sorted=False).values.mean(dim=1)
            pair_scores.extend((fastest / threshold).clamp(max=1.0).unbind(0))

        totals = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        counts = torch.zeros_like(totals)
        for owner, score in zip(owners, pair_scores, strict=True):
            totals[owner] += score
            counts[owner] += 1.0
        return totals / counts.clamp_min(1.0)


__all__ = ["DynamicTrackingScorer"]
