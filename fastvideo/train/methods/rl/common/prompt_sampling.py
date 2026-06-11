# SPDX-License-Identifier: Apache-2.0
"""Prompt-row sampling helpers for online RL methods.

This module chooses and repeats dataset prompt rows across ranks for RL training
batches. Here, "sampling" means selection, not generator sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class KRepeatSample:
    """Local prompt indices for one distributed K-repeat sampling batch."""

    local_indices: list[int]
    unique_prompt_count: int


def distributed_k_repeat_indices(
    *,
    dataset_length: int,
    batch_size: int,
    repeats_per_prompt: int,
    world_size: int,
    rank: int,
    seed: int,
) -> KRepeatSample:
    """Mirror DiffusionNFT's distributed K-repeat prompt sampler.

    Adapted from DiffusionNFT's
    ``scripts/train_nft_sd3.py::DistributedKRepeatSampler``.
    """
    dataset_length = int(dataset_length)
    batch_size = int(batch_size)
    repeats_per_prompt = int(repeats_per_prompt)
    world_size = int(world_size)
    rank = int(rank)
    if dataset_length <= 0:
        raise ValueError("dataset_length must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if repeats_per_prompt <= 0:
        raise ValueError("repeats_per_prompt must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    total_samples = world_size * batch_size
    if total_samples % repeats_per_prompt != 0:
        raise ValueError("world_size * batch_size must be divisible by repeats_per_prompt "
                         f"({world_size} * {batch_size} vs {repeats_per_prompt})")
    unique_prompt_count = total_samples // repeats_per_prompt
    if unique_prompt_count > dataset_length:
        raise ValueError("K-repeat sampling needs at least as many rows as unique prompts "
                         f"per sampling batch ({dataset_length} < {unique_prompt_count})")

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    indices = torch.randperm(dataset_length, generator=generator)[:unique_prompt_count].tolist()
    repeated_indices = [idx for idx in indices for _ in range(repeats_per_prompt)]
    shuffled_order = torch.randperm(len(repeated_indices), generator=generator).tolist()
    shuffled_samples = [int(repeated_indices[idx]) for idx in shuffled_order]

    start = rank * batch_size
    end = start + batch_size
    return KRepeatSample(
        local_indices=shuffled_samples[start:end],
        unique_prompt_count=unique_prompt_count,
    )
