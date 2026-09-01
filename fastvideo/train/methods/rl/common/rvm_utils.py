# SPDX-License-Identifier: Apache-2.0
"""Pure utilities shared by reward-based velocity matching methods."""

from __future__ import annotations

import math

import torch


def visual_text_from_h3_prompt(prompt: str) -> str:
    """Extract the visual description from an H3 multimodal prompt document.

    H3 prompt rows commonly contain three labelled fields. Video-only reward
    models should see the visual field, not the audio soundscape or music text.
    Unlabelled prompts are returned unchanged.
    """
    text = str(prompt).strip()
    marker = "integrated_multimodal_description:"
    if marker not in text:
        return text
    visual = text.split(marker, 1)[1]
    for end_marker in ("overall_soundscape:", "non_diegetic_music:"):
        if end_marker in visual:
            visual = visual.split(end_marker, 1)[0]
    return visual.strip()


def standardize_group_rewards(
    rewards: torch.Tensor,
    *,
    scale: float = 0.1,
    eps: float = 1e-4,
    clip: float | None = 5.0,
    positive_only: bool = False,
    normalization_std: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one prompt's reward vector into RVM advantages.

    The numerator is always centered within the prompt group. By default the
    denominator is that group's population standard deviation, preserving the
    legacy behavior. Passing ``normalization_std`` implements the published RVM
    video recipe: subtract each prompt-group mean but divide every sample in the
    rollout batch by one globally computed reward standard deviation.

    The returned second value is the prompt group's own population standard
    deviation, which remains useful for zero-variance diagnostics.
    """
    if rewards.ndim != 1:
        raise ValueError(
            f"rewards must be one-dimensional, got {tuple(rewards.shape)}"
        )
    if rewards.numel() < 2:
        raise ValueError(
            "RVM group normalization requires at least two samples"
        )
    values = rewards.detach().float()
    group_std = values.std(unbiased=False)
    denominator = group_std
    if normalization_std is not None:
        denominator = torch.as_tensor(
            normalization_std,
            device=values.device,
            dtype=values.dtype,
        )
        if denominator.numel() != 1:
            raise ValueError("normalization_std must be scalar")
        if float(denominator) < 0.0:
            raise ValueError("normalization_std must be nonnegative")
    advantages = (
        values - values.mean()
    ) / (denominator + float(eps))
    if clip is not None:
        advantages = advantages.clamp(
            -float(clip),
            float(clip),
        )
    if positive_only:
        advantages = advantages.clamp_min(0.0)
    return advantages * float(scale), group_std


def detached_rvm_surrogate(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    coefficient: torch.Tensor | float,
    reference: torch.Tensor | None = None,
    anchor_beta: float = 0.0,
) -> torch.Tensor:
    """Return a nonnegative surrogate with the exact signed RVM gradient.

    A literal signed MSE has an unbounded scalar objective for negative
    advantages. This function constructs a detached regression target so the
    gradient is

        coefficient * (prediction - target)
        + anchor_beta * (prediction - reference)

    under the same elementwise-mean normalization as ordinary flow matching.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have identical shapes"
        )
    pred_detached = prediction.detach()
    coeff = torch.as_tensor(
        coefficient,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    while coeff.ndim < prediction.ndim:
        coeff = coeff.unsqueeze(-1)
    gradient = coeff * (
        pred_detached - target.detach()
    )
    if float(anchor_beta) != 0.0:
        if reference is None:
            raise ValueError(
                "reference is required when anchor_beta is nonzero"
            )
        if reference.shape != prediction.shape:
            raise ValueError(
                "reference and prediction must have identical shapes"
            )
        gradient = gradient + float(anchor_beta) * (
            pred_detached - reference.detach()
        )
    surrogate_target = (
        pred_detached - gradient
    ).detach()
    return 0.5 * torch.mean(
        (prediction - surrogate_target) ** 2
    )


def five_percent_interval(max_optimizer_steps: int) -> int:
    """Return the integer interval corresponding to 5% of a run."""
    steps = int(max_optimizer_steps)
    if steps <= 0:
        raise ValueError(
            "max_optimizer_steps must be positive"
        )
    return max(
        1,
        int(math.ceil(0.05 * steps)),
    )


def partition_indices(
    num_items: int,
    num_partitions: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> list[torch.Tensor]:
    """Shuffle indices once and split them into near-equal nonempty chunks."""
    count = int(num_items)
    parts = int(num_partitions)
    if count <= 0:
        raise ValueError("num_items must be positive")
    if parts <= 0 or parts > count:
        raise ValueError(
            "num_partitions must be in [1, num_items]"
        )
    order = torch.randperm(
        count,
        generator=generator,
        device=device,
    )
    return [
        chunk
        for chunk in torch.tensor_split(order, parts)
        if chunk.numel()
    ]
