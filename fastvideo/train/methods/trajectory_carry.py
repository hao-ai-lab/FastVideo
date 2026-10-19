# SPDX-License-Identifier: Apache-2.0
"""Trajectory carry for carried-rollout distillation methods.

A carried method walks the student's own sampling grid one rung per
``single_train_step`` call and keeps the trajectory in memory across calls
(FastGen's backward simulation): exactly one generation forward per call
instead of one full-grid walk. One independent stream exists per
gradient-accumulation round, selected round-robin because the trainer never
passes the round index, and the streams are staggered across the grid so
student updates, which revisit a stream's rung modulo ``gcd(num_steps,
update interval)``, see every rung. Carries are transient and never
checkpointed (FastGen's ``CarryCallback`` contract): after a resume every
stream restarts from fresh noise.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from fastvideo.train.utils.config import get_optional_int


class TrajectoryCarry:
    """Per-slot carried trajectory state."""

    def __init__(self, num_slots: int) -> None:
        if isinstance(num_slots, bool) or not isinstance(num_slots, int) or num_slots <= 0:
            raise ValueError(f"num_slots must be a positive int, got {num_slots!r}")
        self.num_slots = num_slots
        self.slots: list[dict[str, Any] | None] = [None] * num_slots
        self.seeded: list[bool] = [False] * num_slots
        self._calls = 0

    def next_slot(self) -> int:
        slot = self._calls % self.num_slots
        self._calls += 1
        return slot

    def set(self, slot: int, *, state: torch.Tensor, rung: int, raw_batch: dict[str, Any]) -> None:
        self.slots[slot] = {"state": state.detach(), "rung": int(rung), "raw_batch": raw_batch}

    def clear(self, slot: int) -> None:
        self.slots[slot] = None


def snapshot_raw_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Adopt the incoming loader batch as a trajectory's conditioning.

    A trajectory keeps the prompt it set out with for its whole walk, so the
    raw dict is snapshotted (tensors detached and kept on the student device)
    and mid-walk calls rebuild the training batch from it via
    ``prepare_batch``; H3's ``prepare_batch`` reads the dict without mutating
    it and rebuilds the packed layout and VSA metadata deterministically.
    """

    def snapshot(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            result = value.detach().to(device)
            # Moving an already-local tensor can preserve loader-owned storage.
            return result.clone() if result.data_ptr() == value.data_ptr() else result
        if isinstance(value, dict):
            return {key: snapshot(item) for key, item in value.items()}
        if isinstance(value, list):
            return [snapshot(item) for item in value]
        if isinstance(value, tuple):
            return tuple(snapshot(item) for item in value)
        return value

    return {key: snapshot(value) for key, value in batch.items()}


def carry_rank_world(training_config: Any) -> tuple[int, int]:
    """Stagger rank and stream-group count for the carried rollout.

    Mirrors ``TrainingMethod.on_train_start``'s RNG grouping: ranks inside one
    sequence-parallel group shard the same document and must walk one shared
    trajectory, so they share a stagger rank (with ``sp_size=1`` this is
    exactly the global rank). Falls back to a single group when the
    distributed world is not initialized (CPU tests, single-process runs).
    """
    try:
        from fastvideo.distributed import get_world_group
        world_group = get_world_group()
        global_rank = int(world_group.rank)
        world_size = int(world_group.world_size)
    except (AssertionError, ImportError, RuntimeError):
        global_rank, world_size = 0, 1
    sp_size = max(1, int(getattr(getattr(training_config, "distributed", None), "sp_size", 1) or 1))
    return global_rank // sp_size, max(1, world_size // sp_size)


def stagger_rank_and_groups(
    training_config: Any,
    rank_world: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Stagger rank and group count, rank-independent for native-shape data.

    Exact-shape batches must remain shape-synchronous across every rank:
    rank-staggered clears would let one rank adopt the next loader bucket
    while its peers still carry the previous geometry. Native-shape data
    therefore keeps only the slot staggering (one stream group).
    """
    rank, groups = carry_rank_world(training_config) if rank_world is None else rank_world
    if bool(getattr(getattr(training_config, "data", None), "native_shape_bucketing", False)):
        return 0, 1
    return rank, groups


def stagger_offset(*, stagger_rank: int, num_slots: int, slot: int, num_steps: int) -> int:
    """First rung of a (rank, slot) stream: streams spread evenly over the grid."""
    return (stagger_rank * num_slots + slot) % num_steps


def validate_carry_coverage(
    *,
    streams: int,
    num_steps: int,
    interval: int,
    owner: str,
    grid_label: str = "num_steps",
    interval_label: str = "update interval",
) -> None:
    """Reject configurations that leave carried rungs untrained.

    Student updates revisit a stream's rung modulo ``gcd(num_steps,
    interval)``, so every residue class must be represented by a (rank, slot)
    stream; the consecutive stagger offsets cover all classes exactly when
    there are at least ``gcd`` streams.
    """
    phase_classes = math.gcd(num_steps, interval)
    if streams < phase_classes:
        raise ValueError(f"{owner} cannot cover every student rung with {grid_label}={num_steps}, "
                         f"{interval_label}={interval}, and {streams} trajectory stream(s) "
                         "(stagger groups x slots). Student updates preserve the rung modulo "
                         f"gcd(grid, interval)={phase_classes}, but only {streams} stream phase(s) are "
                         "present. Use at least that many rank-slot streams or choose a coprime update "
                         "interval.")


def parse_carry_slots(
    method_config: dict[str, Any],
    loop_config: Any,
    *,
    key: str = "rollout_carry_slots",
) -> int:
    """Number of carried streams per rank; must equal the gradient-accumulation rounds.

    The trainer calls ``single_train_step`` once per accumulation round
    without passing the round index; the round-robin slot selection only
    matches the trainer's cadence when the counts agree.
    """
    where = f"method.{key}"
    grad_accum = max(1, int(getattr(loop_config, "gradient_accumulation_steps", 1) or 1))
    slots_raw = get_optional_int(method_config, key, where=where)
    slots = grad_accum if slots_raw is None else int(slots_raw)
    if slots <= 0:
        raise ValueError(f"{where} must be positive, got {slots}")
    if slots != grad_accum:
        raise ValueError(f"{where} must equal training.loop.gradient_accumulation_steps, got "
                         f"slots={slots} vs gradient_accumulation_steps={grad_accum}")
    return slots


__all__ = [
    "TrajectoryCarry",
    "carry_rank_world",
    "parse_carry_slots",
    "snapshot_raw_batch",
    "stagger_offset",
    "stagger_rank_and_groups",
    "validate_carry_coverage",
]
