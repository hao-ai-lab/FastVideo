# SPDX-License-Identifier: Apache-2.0
"""Distributed helpers for PromptRL.

One original prompt is replicated across the training ranks of a group;
each rank produces one independently seeded candidate.  After scoring,
rewards are gathered across ranks and validated identically on *every*
rank: any timeout, duplicate/missing sample, non-finite score, or
cardinality mismatch fails the step consistently everywhere instead of
substituting a zero reward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist

from fastvideo.train.methods.rl.promptrl.rewards.provider import RewardResult


class RewardConsistencyError(RuntimeError):
    """Raised on every rank when group reward collection is invalid."""


@dataclass(frozen=True, slots=True)
class RewardFailure:
    """Picklable sentinel for a rank-local reward collection failure."""

    rank: int
    error_type: str
    message: str


def dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_rank() -> int:
    return int(dist.get_rank()) if dist_ready() else 0


def world_size() -> int:
    return int(dist.get_world_size()) if dist_ready() else 1


def all_gather_objects(obj: Any) -> list[Any]:
    """Gather a picklable object from every rank (rank order)."""
    if not dist_ready():
        return [obj]
    gathered: list[Any] = [None] * world_size()
    dist.all_gather_object(gathered, obj)
    return gathered


def gather_group_reward_results(
    local_result: RewardResult,
    *,
    group_id: str,
    expected_group_size: int,
) -> list[RewardResult]:
    """Gather one result per rank and validate identically everywhere."""
    gathered: list[RewardResult] = all_gather_objects(local_result)
    validate_group_reward_results(
        gathered,
        group_id=group_id,
        expected_group_size=expected_group_size,
    )
    return gathered


def validate_group_reward_results(
    gathered: list[RewardResult | RewardFailure],
    *,
    group_id: str,
    expected_group_size: int,
) -> None:
    """Validate a full group's results; raise consistently on any problem.

    Every rank runs this on the identical gathered list, so a timeout,
    duplicate/missing sample, non-finite score, or cardinality mismatch
    fails the step on *all* ranks instead of substituting a zero reward.
    """
    problems: list[str] = []
    if len(gathered) != expected_group_size:
        problems.append(f"cardinality: expected {expected_group_size} results, "
                        f"gathered {len(gathered)}")
    failures = [r for r in gathered if isinstance(r, RewardFailure)]
    for failure in failures:
        problems.append(
            f"rank {failure.rank} reward failure: {failure.error_type}: {failure.message}")
    reward_results = [r for r in gathered if isinstance(r, RewardResult)]
    sample_ids = [r.sample_id for r in reward_results]
    if len(set(sample_ids)) != len(sample_ids):
        problems.append(f"duplicate sample ids: {sorted(sample_ids)}")
    for result in gathered:
        if isinstance(result, RewardFailure):
            continue
        if not isinstance(result, RewardResult):
            problems.append(f"unexpected gathered object: {type(result).__name__}")
            continue
        if not math.isfinite(float(result.score)):
            problems.append(f"non-finite score for sample {result.sample_id!r}")
        for key, value in result.details.items():
            if not math.isfinite(float(value)):
                problems.append(f"non-finite detail {key!r} for sample "
                                f"{result.sample_id!r}")
    if problems:
        raise RewardConsistencyError(f"PromptRL group {group_id!r} reward validation failed: "
                                     + "; ".join(problems))
