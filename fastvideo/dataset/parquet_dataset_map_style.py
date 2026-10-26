# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import os
import pickle
import random
import time
from collections import Counter, defaultdict, deque
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from numbers import Integral
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
# Torch in general
import torch
import tqdm
# Dataset
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from fastvideo.platforms import current_platform

from fastvideo.dataset.sequence_capacity import (
    select_sequence_capacity,
    validate_fixed_sequence_capacities,
    validate_sequence_capacity_buckets,
)
from fastvideo.dataset.shape_bucket import parse_video_shape_bucket_id
from fastvideo.dataset.utils import collate_rows_from_parquet_schema
from fastvideo.distributed import (get_sp_world_size, get_world_group, get_world_rank, get_world_size)
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class DP_SP_BatchSampler(Sampler[list[int]]):
    """
    A simple sequential batch sampler that yields batches of indices.
    """

    def __init__(
        self,
        batch_size: int,
        dataset_size: int,
        num_sp_groups: int,
        sp_world_size: int,
        global_rank: int,
        drop_last: bool = True,
        drop_first_row: bool = False,
        seed: int = 0,
        sample_bucket_ids: Sequence[str] | None = None,
    ):
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.drop_last = drop_last
        self.seed = seed
        self.num_sp_groups = num_sp_groups
        self.global_rank = global_rank
        self.sp_world_size = sp_world_size

        # ── epoch-level RNG ────────────────────────────────────────────────
        if batch_size <= 0 or num_sp_groups <= 0 or sp_world_size <= 0:
            raise ValueError("batch_size, num_sp_groups, and sp_world_size must be positive")

        rng = torch.Generator().manual_seed(self.seed)
        if sample_bucket_ids is None:
            # Legacy behavior: one permutation over the complete dataset.
            global_indices = torch.randperm(self.dataset_size, generator=rng)
            if drop_first_row:
                global_indices = global_indices[global_indices != 0]
                self.dataset_size -= 1

            if self.drop_last:
                num_batches = self.dataset_size // self.batch_size
                num_global_batches = num_batches // self.num_sp_groups
                global_indices = global_indices[:num_global_batches * self.num_sp_groups * self.batch_size]
            elif self.dataset_size % (self.num_sp_groups * self.batch_size) != 0:
                padding_size = self.num_sp_groups * self.batch_size - (self.dataset_size %
                                                                       (self.num_sp_groups * self.batch_size))
                logger.info("Padding the dataset from %d to %d", self.dataset_size, self.dataset_size + padding_size)
                global_indices = torch.cat([global_indices, global_indices[:padding_size]])
            self.bucket_schedule: tuple[str, ...] | None = None
            self.bucket_padding: dict[str, int] | None = None
            self.num_padded_samples = 0
        else:
            if len(sample_bucket_ids) != self.dataset_size:
                raise ValueError("sample_bucket_ids must contain one identifier per dataset row, got "
                                 f"{len(sample_bucket_ids)} for dataset_size={self.dataset_size}")
            global_indices, self.bucket_schedule = self._bucketed_global_schedule(
                sample_bucket_ids,
                rng=rng,
                drop_first_row=drop_first_row,
            )
            if drop_first_row:
                self.dataset_size -= 1

        # shard the indices to each sp group
        ith_sp_group = self.global_rank // self.sp_world_size
        sp_group_local_indices = global_indices[ith_sp_group::self.num_sp_groups]
        self.sp_group_local_indices = sp_group_local_indices
        logger.info("Dataset size for each sp group: %d", len(sp_group_local_indices))

    def _bucketed_global_schedule(
        self,
        sample_bucket_ids: Sequence[str],
        *,
        rng: torch.Generator,
        drop_first_row: bool,
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        """Build same-shape rounds shared by every data-parallel group.

        One round contains ``num_sp_groups * batch_size`` rows from exactly
        one bucket. Strided DP sharding below gives every group a distinct
        local batch while preserving the same bucket at that microstep. SP
        ranks map to the same group and therefore receive identical indices.
        """
        by_bucket: dict[str, list[int]] = defaultdict(list)
        for index, bucket_id in enumerate(sample_bucket_ids):
            if drop_first_row and index == 0:
                continue
            parse_video_shape_bucket_id(bucket_id)
            by_bucket[bucket_id].append(index)

        samples_per_round = self.num_sp_groups * self.batch_size
        rounds: list[torch.Tensor] = []
        round_bucket_ids: list[str] = []
        bucket_padding: dict[str, int] = {}
        padded = 0
        for bucket_id in sorted(by_bucket):
            bucket_indices = torch.tensor(by_bucket[bucket_id], dtype=torch.long)
            bucket_indices = bucket_indices[torch.randperm(len(bucket_indices), generator=rng)]
            remainder = len(bucket_indices) % samples_per_round
            if remainder:
                # Native bucketing must retain every frozen row, including a
                # rare bucket smaller than one global microbatch. Repeat only
                # within that bucket; legacy drop_last behavior remains in the
                # unbucketed branch above.
                padding_size = samples_per_round - remainder
                repeats = (padding_size + len(bucket_indices) - 1) // len(bucket_indices)
                padding = bucket_indices.repeat(repeats)[:padding_size]
                bucket_indices = torch.cat((bucket_indices, padding))
                padded += padding_size
                bucket_padding[bucket_id] = padding_size
                logger.info(
                    "Exact-shape bucket %s has %d row(s); repeated %d row(s) to fill global microbatches of %d",
                    bucket_id,
                    len(by_bucket[bucket_id]),
                    padding_size,
                    samples_per_round,
                )
            bucket_rounds = list(bucket_indices.reshape(-1, samples_per_round).unbind(0))
            rounds.extend(bucket_rounds)
            round_bucket_ids.extend([bucket_id] * len(bucket_rounds))

        if not rounds:
            raise ValueError("Exact-shape bucketing requires at least one dataset row")
        round_order = torch.randperm(len(rounds), generator=rng).tolist()
        self.bucket_padding = bucket_padding
        self.num_padded_samples = padded
        if padded:
            logger.info("Exact-shape bucketing repeated %d row(s) to fill bucket-local global microbatches", padded)
        return (
            torch.cat([rounds[index] for index in round_order]),
            tuple(round_bucket_ids[index] for index in round_order),
        )

    def __iter__(self):
        indices = self.sp_group_local_indices
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield batch_indices.tolist()

    def __len__(self):
        return len(self.sp_group_local_indices) // self.batch_size


class PackedBatchIndices(list):
    """One local pack plus synchronized-window weighting metadata.

    ``DataLoader`` forwards the object itself to ``Dataset.__getitems__``.
    Keeping metadata on the index list avoids ambiguous tuple lookups when a
    padded pack is repeated in a different accumulation window.
    """

    def __init__(
        self,
        indices: Sequence[int],
        *,
        local_joint_token_count: int,
        window_document_count: int,
        window_joint_token_count: int,
        window_document_counts: Sequence[int],
        local_window_document_counts: Sequence[int],
        accumulation_index: int,
        num_dp_groups: int,
        gradient_accumulation_steps: int,
        local_effective_document_count: float | None = None,
        window_effective_document_count: float | None = None,
        window_effective_document_counts: Sequence[float] | None = None,
        local_window_effective_document_counts: Sequence[float] | None = None,
        local_window_compatibility_ids: Sequence[str] | None = None,
        schedule_fingerprint: str = "",
    ) -> None:
        super().__init__(int(index) for index in indices)
        self.local_document_count = len(self)
        self.local_joint_token_count = int(local_joint_token_count)
        self.window_document_count = int(window_document_count)
        self.window_joint_token_count = int(window_joint_token_count)
        self.window_document_counts = tuple(int(count) for count in window_document_counts)
        self.local_window_document_counts = tuple(int(count) for count in local_window_document_counts)
        self.accumulation_index = int(accumulation_index)
        self.num_dp_groups = int(num_dp_groups)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.local_effective_document_count = float(
            self.local_document_count
            if local_effective_document_count is None else local_effective_document_count)
        self.window_effective_document_count = float(
            self.window_document_count
            if window_effective_document_count is None else window_effective_document_count)
        self.window_effective_document_counts = tuple(
            self.window_document_counts
            if window_effective_document_counts is None else
            (float(count) for count in window_effective_document_counts))
        self.local_window_effective_document_counts = tuple(
            self.local_window_document_counts
            if local_window_effective_document_counts is None else
            (float(count) for count in local_window_effective_document_counts))
        self.local_window_compatibility_ids = tuple(str(value) for value in (local_window_compatibility_ids or ()))
        self.schedule_fingerprint = str(schedule_fingerprint)
        if self.window_effective_document_count <= 0:
            raise ValueError("A packed accumulation window must contain at least one effective document")
        if self.local_effective_document_count < 0:
            raise ValueError("local_effective_document_count must be non-negative")
        self.loss_scale = (self.num_dp_groups * self.gradient_accumulation_steps *
                           self.local_effective_document_count / self.window_effective_document_count)


@dataclass(frozen=True, slots=True)
class _PackedScheduleEntry:
    indices: tuple[int, ...]
    local_joint_token_count: int
    window_document_count: int
    window_joint_token_count: int
    window_document_counts: tuple[int, ...]
    local_window_document_counts: tuple[int, ...]
    local_effective_document_count: float
    window_effective_document_count: float
    window_effective_document_counts: tuple[float, ...]
    local_window_effective_document_counts: tuple[float, ...]
    local_window_compatibility_ids: tuple[str, ...]
    accumulation_index: int


@dataclass(frozen=True, slots=True)
class _PackedGlobalEntry:
    indices: tuple[int, ...]
    compatibility_id: str
    effective_document_count: float


class _PackedScheduleIterator(Iterator[PackedBatchIndices]):
    """Stateful cursor with an exact schedule/config resume guard."""

    def __init__(
        self,
        schedule: Sequence[_PackedScheduleEntry],
        *,
        schedule_fingerprint: str,
        num_dp_groups: int,
        gradient_accumulation_steps: int,
    ) -> None:
        self._schedule = schedule
        self._schedule_fingerprint = schedule_fingerprint
        self._num_dp_groups = int(num_dp_groups)
        self._gradient_accumulation_steps = int(gradient_accumulation_steps)
        self._cursor = 0

    def __iter__(self) -> "_PackedScheduleIterator":
        return self

    def __next__(self) -> PackedBatchIndices:
        if self._cursor >= len(self._schedule):
            raise StopIteration
        entry = self._schedule[self._cursor]
        self._cursor += 1
        return PackedBatchIndices(
            entry.indices,
            local_joint_token_count=entry.local_joint_token_count,
            window_document_count=entry.window_document_count,
            window_joint_token_count=entry.window_joint_token_count,
            window_document_counts=entry.window_document_counts,
            local_window_document_counts=entry.local_window_document_counts,
            local_effective_document_count=entry.local_effective_document_count,
            window_effective_document_count=entry.window_effective_document_count,
            window_effective_document_counts=entry.window_effective_document_counts,
            local_window_effective_document_counts=entry.local_window_effective_document_counts,
            local_window_compatibility_ids=entry.local_window_compatibility_ids,
            accumulation_index=entry.accumulation_index,
            num_dp_groups=self._num_dp_groups,
            gradient_accumulation_steps=self._gradient_accumulation_steps,
            schedule_fingerprint=self._schedule_fingerprint,
        )

    def state_dict(self) -> dict[str, Any]:
        next_accumulation_index = (
            self._schedule[self._cursor].accumulation_index
            if self._cursor < len(self._schedule) else 0)
        return {
            "version": 1,
            "schedule_fingerprint": self._schedule_fingerprint,
            "cursor": self._cursor,
            "next_accumulation_index": next_accumulation_index,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if int(state_dict.get("version", -1)) != 1:
            raise ValueError("Unsupported packed sampler checkpoint state version")
        saved_fingerprint = str(state_dict.get("schedule_fingerprint", ""))
        if saved_fingerprint != self._schedule_fingerprint:
            raise ValueError(
                "Packed sampler schedule fingerprint changed across resume: "
                f"checkpoint={saved_fingerprint!r}, runtime={self._schedule_fingerprint!r}. "
                "Keep the data, packing limits, fixed sequence capacities, seed, DP/SP topology, "
                "and gradient accumulation unchanged.")
        cursor = int(state_dict.get("cursor", -1))
        if cursor < 0 or cursor > len(self._schedule):
            raise ValueError(f"Packed sampler checkpoint cursor {cursor} is outside [0, {len(self._schedule)}]")
        expected_accumulation_index = self._schedule[cursor].accumulation_index if cursor < len(self._schedule) else 0
        saved_accumulation_index = int(state_dict.get("next_accumulation_index", -1))
        if saved_accumulation_index != expected_accumulation_index:
            raise ValueError(
                "Packed sampler checkpoint accumulation index does not match its schedule cursor: "
                f"saved={saved_accumulation_index}, expected={expected_accumulation_index}")
        self._cursor = cursor


class DP_SP_PackedBatchSampler(Sampler[list[int]]):
    """Pack variable-shape documents under a joint-token capacity.

    Every SP rank in a group receives the same pack. Distinct DP groups
    receive distinct packs, while every accumulation window has the same
    number of rounds on every rank. Packs never cross a compatibility class
    (currently text-only versus latent-bearing H3 rows).
    """

    def __init__(
        self,
        *,
        sample_joint_token_counts: Sequence[int],
        sample_compatibility_ids: Sequence[str],
        max_joint_tokens: int,
        max_samples_per_pack: int | None,
        num_sp_groups: int,
        sp_world_size: int,
        global_rank: int,
        gradient_accumulation_steps: int,
        drop_first_row: bool = False,
        seed: int = 0,
        data_identity: Sequence[tuple[str, int, int, int]] | None = None,
        fixed_sequence_capacities: Sequence[int] | None = None,
    ) -> None:
        capacities = validate_fixed_sequence_capacities(fixed_sequence_capacities)
        dataset_size = len(sample_joint_token_counts)
        if len(sample_compatibility_ids) != dataset_size:
            raise ValueError("sample_compatibility_ids must contain one identifier per dataset row")
        if isinstance(max_joint_tokens, bool) or not isinstance(max_joint_tokens, Integral) or max_joint_tokens <= 0:
            raise ValueError("max_joint_tokens must be a positive integer")
        if capacities and max_joint_tokens > capacities[-1]:
            raise ValueError("max_joint_tokens exceeds the largest fixed_sequence_capacities value")
        if (max_samples_per_pack is not None
                and (isinstance(max_samples_per_pack, bool) or not isinstance(max_samples_per_pack, Integral)
                     or max_samples_per_pack <= 0)):
            raise ValueError("max_samples_per_pack must be None or a positive integer")
        if num_sp_groups <= 0 or sp_world_size <= 0 or gradient_accumulation_steps <= 0:
            raise ValueError("num_sp_groups, sp_world_size, and gradient_accumulation_steps must be positive")
        if global_rank < 0 or global_rank >= num_sp_groups * sp_world_size:
            raise ValueError("global_rank is outside the declared DP/SP topology")

        counts: list[int] = []
        for index, count in enumerate(sample_joint_token_counts):
            if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
                raise ValueError(f"Joint-token count for row {index} must be a positive integer, got {count!r}")
            if count > max_joint_tokens:
                raise ValueError(f"Row {index} needs {count} joint tokens, exceeding max_joint_tokens="
                                 f"{max_joint_tokens}")
            counts.append(int(count))

        eligible_indices = list(range(dataset_size))
        if drop_first_row:
            eligible_indices = eligible_indices[1:]
        if not eligible_indices:
            raise ValueError("Cross-sample sequence packing requires at least one eligible dataset row")

        self.dataset_size = len(eligible_indices)
        self.max_joint_tokens = int(max_joint_tokens)
        self.max_samples_per_pack = (None if max_samples_per_pack is None else int(max_samples_per_pack))
        self.num_sp_groups = int(num_sp_groups)
        self.sp_world_size = int(sp_world_size)
        self.global_rank = int(global_rank)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.seed = int(seed)
        self.num_padded_packs = 0

        fingerprint_payload: dict[str, Any] = {
            "version": 1,
            "algorithm": "h3-cross-sample-pack-v2-fractional-padding",
            "sample_joint_token_counts": counts,
            "sample_compatibility_ids": list(sample_compatibility_ids),
            "max_joint_tokens": self.max_joint_tokens,
            "max_samples_per_pack": self.max_samples_per_pack,
            "num_sp_groups": self.num_sp_groups,
            "sp_world_size": self.sp_world_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "drop_first_row": bool(drop_first_row),
            "seed": self.seed,
            "data_identity": list(data_identity or ()),
        }
        # Preserve pre-capacity checkpoint fingerprints when the option is off.
        if capacities:
            fingerprint_payload["fixed_sequence_capacities"] = capacities

        rng = torch.Generator().manual_seed(self.seed)
        by_compatibility: dict[str, list[int]] = defaultdict(list)
        for index in eligible_indices:
            compatibility_id = sample_compatibility_ids[index]
            if not isinstance(compatibility_id, str) or not compatibility_id:
                raise ValueError(f"Compatibility identifier for row {index} must be a non-empty string")
            by_compatibility[compatibility_id].append(index)

        # One global microstep must use a single compatibility class so every
        # DP rank takes the same data-forcing/control-flow branch. Pad only to
        # a DP round here, then mix those rounds across accumulation windows.
        # Repeated entries are fractionally reweighted below, so small
        # compatibility classes are not overrepresented in the objective.
        rounds: list[tuple[str, tuple[_PackedGlobalEntry, ...]]] = []
        for compatibility_id in sorted(by_compatibility):
            packs = self._pack_compatibility_group(
                by_compatibility[compatibility_id],
                counts=counts,
                rng=rng,
            )
            pack_order = torch.randperm(len(packs), generator=rng).tolist()
            packs = [packs[index] for index in pack_order]
            for start in range(0, len(packs), self.num_sp_groups):
                round_packs = [
                    _PackedGlobalEntry(
                        indices=pack,
                        compatibility_id=compatibility_id,
                        effective_document_count=len(pack),
                    ) for pack in packs[start:start + self.num_sp_groups]
                ]
                if len(round_packs) < self.num_sp_groups:
                    padding_size = self.num_sp_groups - len(round_packs)
                    source_packs = tuple(round_packs)
                    round_packs.extend(
                        _PackedGlobalEntry(
                            indices=source_packs[index % len(source_packs)].indices,
                            compatibility_id=compatibility_id,
                            effective_document_count=0.0,
                        ) for index in range(padding_size))
                    self.num_padded_packs += padding_size
                rounds.append((compatibility_id, tuple(round_packs)))

        round_order = torch.randperm(len(rounds), generator=rng).tolist()
        rounds = [rounds[index] for index in round_order]
        remainder = len(rounds) % self.gradient_accumulation_steps
        if remainder:
            padding_rounds = self.gradient_accumulation_steps - remainder
            source_rounds = tuple(rounds)
            for index in range(padding_rounds):
                compatibility_id, source_entries = source_rounds[index % len(source_rounds)]
                rounds.append((
                    compatibility_id,
                    tuple(
                        _PackedGlobalEntry(
                            indices=entry.indices,
                            compatibility_id=entry.compatibility_id,
                            effective_document_count=0.0,
                        ) for entry in source_entries),
                ))
                self.num_padded_packs += self.num_sp_groups

        # Every original pack is disjoint in dataset-row identity. If padding
        # repeats one, split its document weight equally over all physical
        # occurrences. This preserves exactly one unit of weight per source
        # document while allowing every repeated rank to train on an
        # independent stochastic trajectory instead of wasting its backward.
        multiplicities = Counter(entry.indices for _, round_entries in rounds for entry in round_entries)
        rounds = [
            (
                compatibility_id,
                tuple(
                    _PackedGlobalEntry(
                        indices=entry.indices,
                        compatibility_id=entry.compatibility_id,
                        effective_document_count=len(entry.indices) / multiplicities[entry.indices],
                    ) for entry in round_entries),
            ) for compatibility_id, round_entries in rounds
        ]
        fingerprint_payload["global_round_schedule"] = [
            {
                "compatibility_id": compatibility_id,
                "entries": [{
                    "indices": entry.indices,
                    "effective_document_count": entry.effective_document_count,
                } for entry in round_entries],
            } for compatibility_id, round_entries in rounds
        ]
        self.schedule_fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

        windows = [
            rounds[start:start + self.gradient_accumulation_steps]
            for start in range(0, len(rounds), self.gradient_accumulation_steps)
        ]
        self.compatibility_schedule = tuple(compatibility_id for compatibility_id, _ in rounds)
        self.global_pack_schedule = tuple(
            entry.indices for _, round_entries in rounds for entry in round_entries)
        self.global_pack_effective_document_counts = tuple(
            entry.effective_document_count for _, round_entries in rounds for entry in round_entries)

        ith_sp_group = self.global_rank // self.sp_world_size
        local_schedule: list[_PackedScheduleEntry] = []
        for window in windows:
            flat_window = tuple(entry for _, round_entries in window for entry in round_entries)
            window_document_counts = tuple(len(entry.indices) for entry in flat_window)
            window_effective_document_counts = tuple(entry.effective_document_count for entry in flat_window)
            window_document_count = sum(window_document_counts)
            window_effective_document_count = sum(window_effective_document_counts)
            if window_effective_document_count <= 0:
                raise AssertionError("Packed schedule construction emitted an all-padding accumulation window")
            window_joint_token_count = sum(
                counts[index] for entry in flat_window for index in entry.indices)
            local_entries = tuple(round_entries[ith_sp_group] for _, round_entries in window)
            local_window_document_counts = tuple(len(entry.indices) for entry in local_entries)
            local_window_effective_document_counts = tuple(
                entry.effective_document_count for entry in local_entries)
            local_window_compatibility_ids = tuple(entry.compatibility_id for entry in local_entries)
            for accumulation_round, entry in enumerate(local_entries):
                local_schedule.append(
                    _PackedScheduleEntry(
                        indices=entry.indices,
                        local_joint_token_count=sum(counts[index] for index in entry.indices),
                        window_document_count=window_document_count,
                        window_joint_token_count=window_joint_token_count,
                        window_document_counts=window_document_counts,
                        local_window_document_counts=local_window_document_counts,
                        local_effective_document_count=entry.effective_document_count,
                        window_effective_document_count=window_effective_document_count,
                        window_effective_document_counts=window_effective_document_counts,
                        local_window_effective_document_counts=local_window_effective_document_counts,
                        local_window_compatibility_ids=local_window_compatibility_ids,
                        accumulation_index=accumulation_round,
                    ))
        self._local_schedule = tuple(local_schedule)
        logger.info(
            "Packed %d rows into %d global packs (%d padding packs); each SP group receives %d packs",
            self.dataset_size,
            len(self.global_pack_schedule),
            self.num_padded_packs,
            len(self._local_schedule),
        )

    def _pack_compatibility_group(
        self,
        indices: Sequence[int],
        *,
        counts: Sequence[int],
        rng: torch.Generator,
    ) -> list[tuple[int, ...]]:
        """Deterministic large-with-small packing with linear memory use."""
        shuffled = torch.randperm(len(indices), generator=rng).tolist()
        ordered = [indices[position] for position in shuffled]
        # Python's sort is stable, so the seeded shuffle randomizes equal-cost
        # ties without changing the capacity-first ordering.
        ordered.sort(key=lambda index: counts[index])
        packs: list[tuple[int, ...]] = []
        left = 0
        right = len(ordered) - 1
        while left <= right:
            largest = ordered[right]
            right -= 1
            pack = [largest]
            total = counts[largest]
            while left <= right and (self.max_samples_per_pack is None or len(pack) < self.max_samples_per_pack):
                smallest = ordered[left]
                if total + counts[smallest] > self.max_joint_tokens:
                    break
                pack.append(smallest)
                total += counts[smallest]
                left += 1
            packs.append(tuple(pack))
        return packs

    def __iter__(self) -> Iterator[PackedBatchIndices]:
        return _PackedScheduleIterator(
            self._local_schedule,
            schedule_fingerprint=self.schedule_fingerprint,
            num_dp_groups=self.num_sp_groups,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )

    def __len__(self) -> int:
        return len(self._local_schedule)


def _shape_bucket_id_from_parquet_path(file_path: str) -> str:
    """Return and validate the sole ``bucket=...`` ancestor of a parquet."""
    matches = [part for part in Path(file_path).parts if part.startswith("bucket=")]
    if len(matches) != 1:
        raise ValueError(
            "Native-shape parquet paths must have exactly one ancestor named "
            "'bucket=<width>x<height>-<num_frames>f', got "
            f"{file_path!r} with bucket ancestors {matches}"
        )
    bucket_id = matches[0]
    parse_video_shape_bucket_id(bucket_id)
    return bucket_id


def shape_bucket_ids_from_parquet_files(
    parquet_files: Sequence[str],
    lengths: Sequence[int],
) -> list[str]:
    """Expand canonical path bucket IDs to one identifier per dataset row."""
    if len(parquet_files) != len(lengths):
        raise ValueError("parquet_files and lengths must have matching lengths")
    sample_bucket_ids: list[str] = []
    for file_path, length in zip(parquet_files, lengths, strict=True):
        if int(length) < 0:
            raise ValueError(f"Parquet row counts must be non-negative, got {length}")
        bucket_id = _shape_bucket_id_from_parquet_path(str(file_path))
        sample_bucket_ids.extend([bucket_id] * int(length))
    return sample_bucket_ids


@dataclass(frozen=True, slots=True)
class H3SequencePackingMetadata:
    """Per-row scheduling metadata loaded without tensor payload columns."""

    joint_token_counts: tuple[int, ...]
    compatibility_ids: tuple[str, ...]


def _shape_has_elements(shape: Any) -> bool:
    if shape is None or shape == []:
        return False
    if not isinstance(shape, list | tuple) or not shape:
        raise ValueError(f"A latent shape must be a list of positive integers or empty, got {shape!r}")
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension <= 0:
            raise ValueError(f"A latent shape must contain positive integers, got {shape!r}")
    return True


def _scan_h3_sequence_packing_metadata_serial(
    parquet_files: Sequence[str],
    lengths: Sequence[int],
    sample_bucket_ids: Sequence[str],
    parquet_schema: pa.Schema,
) -> H3SequencePackingMetadata:
    """Read H3 joint-token costs and row compatibility without latent bytes.

    The target visual/audio geometry comes from the canonical bucket path;
    only ``*_shape`` metadata is projected from parquet. For a T2VA runtime
    schema, empty latent shapes identify data-free rows and non-empty paired
    shapes identify data-forcing rows. Ref2VA additionally stores already
    patchified reference video/audio rows: each row is one joint token (the
    reference audio shape already includes both stereo channels).
    """
    if len(parquet_files) != len(lengths):
        raise ValueError("parquet_files and lengths must have matching lengths")
    expected_rows = sum(int(length) for length in lengths)
    if len(sample_bucket_ids) != expected_rows:
        raise ValueError("sample_bucket_ids must contain one identifier per dataset row")

    runtime_fields = set(parquet_schema.names)
    runtime_video_latents = "vae_latent_bytes" in runtime_fields
    runtime_audio_latents = "audio_latent_bytes" in runtime_fields
    if runtime_video_latents != runtime_audio_latents:
        raise ValueError("H3 sequence packing supports only text-only or paired video-audio runtime schemas")

    reference_shapes = {
        "reference_video_latent_rows_shape": 96,
        "reference_audio_latent_rows_shape": 32,
    }
    runtime_references = any(name in runtime_fields for name in reference_shapes)
    if runtime_references and not all(name in runtime_fields for name in reference_shapes):
        raise ValueError("H3 Ref2VA metadata requires both reference video and audio row shapes")

    # Lazy import keeps the generic parquet module free of MiniMax pipeline
    # imports unless this explicitly H3-only feature is enabled, while using
    # the model's canonical frame conversion functions as the source of truth.
    from fastvideo.pipelines.basic.minimax_h3.packing import (
        audio_latent_num_frames,
        video_latent_num_frames,
    )

    file_cache: dict[str, tuple[tuple[int, ...], tuple[str, ...]]] = {}
    all_costs: list[int] = []
    all_compatibility_ids: list[str] = []
    sample_offset = 0
    for file_path, declared_length in zip(parquet_files, lengths, strict=True):
        declared_length = int(declared_length)
        bucket_id = _shape_bucket_id_from_parquet_path(str(file_path))
        bucket = parse_video_shape_bucket_id(bucket_id)
        if bucket.width % 32 or bucket.height % 32:
            raise ValueError(f"H3 sequence packing requires bucket dimensions divisible by 32, got {bucket_id}")
        bucket_ids = sample_bucket_ids[sample_offset:sample_offset + declared_length]
        if any(row_bucket_id != bucket_id for row_bucket_id in bucket_ids):
            raise ValueError(f"Expanded bucket metadata disagrees with parquet path {file_path!r}")
        sample_offset += declared_length

        cached = file_cache.get(str(file_path))
        if cached is None:
            parquet_file = pq.ParquetFile(file_path)
            physical_fields = set(parquet_file.schema_arrow.names)
            if "text_embedding_shape" not in physical_fields:
                raise ValueError(f"H3 sequence packing requires text_embedding_shape in {file_path!r}")
            projected_columns = ["text_embedding_shape"]
            physical_video_shape = "vae_latent_shape" in physical_fields
            physical_audio_shape = "audio_latent_shape" in physical_fields
            if runtime_video_latents:
                if physical_video_shape != physical_audio_shape:
                    raise ValueError(f"H3 data-forcing metadata in {file_path!r} must contain both video and audio "
                                     "latent shapes or neither")
                if physical_video_shape:
                    projected_columns.extend(("vae_latent_shape", "audio_latent_shape"))
            if runtime_references:
                missing = set(reference_shapes) - physical_fields
                if missing:
                    raise ValueError(f"H3 Ref2VA metadata in {file_path!r} is missing reference shape(s): "
                                     f"{sorted(missing)}")
                projected_columns.extend(reference_shapes)
            if any(column.endswith("_bytes") for column in projected_columns):
                raise AssertionError("Sequence-packing metadata scans must never project tensor bytes")

            metadata_columns = parquet_file.read(columns=projected_columns).to_pydict()
            text_shapes = metadata_columns["text_embedding_shape"]
            if len(text_shapes) != declared_length:
                raise ValueError(f"Parquet metadata row count changed for {file_path!r}: expected {declared_length}, "
                                 f"found {len(text_shapes)}")

            video_rows = video_latent_num_frames(bucket.num_frames) * (bucket.height // 32) * (bucket.width // 32)
            audio_rows = 2 * audio_latent_num_frames(bucket.num_frames)
            file_costs: list[int] = []
            file_compatibility_ids: list[str] = []
            for row_index, text_shape in enumerate(text_shapes):
                if (not isinstance(text_shape, list | tuple) or len(text_shape) != 2
                        or any(isinstance(dimension, bool) or not isinstance(dimension, Integral)
                               or int(dimension) <= 0 for dimension in text_shape)):
                    raise ValueError(f"H3 row {row_index} in {file_path!r} has invalid text_embedding_shape "
                                     f"{text_shape!r}")
                text_tokens = int(text_shape[0])
                reference_rows = 0
                if runtime_references:
                    for name, feature_dim in reference_shapes.items():
                        shape = metadata_columns[name][row_index]
                        if (not isinstance(shape, list | tuple) or len(shape) != 2
                                or any(isinstance(dim, bool) or not isinstance(dim, Integral) for dim in shape)
                                or shape[0] < 0 or shape[1] != feature_dim):
                            raise ValueError(f"H3 Ref2VA row {row_index} in {file_path!r} has invalid {name} "
                                             f"{shape!r}; expected [non-negative rows, {feature_dim}]")
                        reference_rows += int(shape[0])
                file_costs.append(video_rows + text_tokens + audio_rows + reference_rows)

                if runtime_video_latents and physical_video_shape:
                    has_video = _shape_has_elements(metadata_columns["vae_latent_shape"][row_index])
                    has_audio = _shape_has_elements(metadata_columns["audio_latent_shape"][row_index])
                    if has_video != has_audio:
                        raise ValueError(f"H3 row {row_index} in {file_path!r} must carry both video and audio "
                                         "latents or neither")
                    file_compatibility_ids.append("latent-bearing" if has_video else "text-only")
                else:
                    file_compatibility_ids.append("text-only")
            cached = (tuple(file_costs), tuple(file_compatibility_ids))
            file_cache[str(file_path)] = cached
        elif len(cached[0]) != declared_length:
            raise ValueError(f"Repeated parquet path {file_path!r} has inconsistent declared lengths")
        all_costs.extend(cached[0])
        all_compatibility_ids.extend(cached[1])

    return H3SequencePackingMetadata(
        joint_token_counts=tuple(all_costs),
        compatibility_ids=tuple(all_compatibility_ids),
    )


def scan_h3_sequence_packing_metadata(
    parquet_files: Sequence[str],
    lengths: Sequence[int],
    sample_bucket_ids: Sequence[str],
    parquet_schema: pa.Schema,
    *,
    num_workers: int = 1,
) -> H3SequencePackingMetadata:
    """Preserve serial row order/validation while bounding parallel metadata I/O.

    Only rank zero calls the parallel path during distributed initialization.
    Each distinct file is read once by the unchanged serial scanner; repeated
    paths are expanded afterward in their original order. Neither the published
    parquet index nor any tensor payload is rewritten. At most twice the worker
    count is submitted, rather than one Future per file in a large publication.
    """
    if type(num_workers) is not int or not 1 <= num_workers <= 32:
        raise ValueError("H3 metadata num_workers must be an integer between 1 and 32")
    if num_workers == 1 or len(parquet_files) == 0:
        return _scan_h3_sequence_packing_metadata_serial(
            parquet_files, lengths, sample_bucket_ids, parquet_schema)
    if len(parquet_files) != len(lengths):
        raise ValueError("parquet_files and lengths must have matching lengths")
    if len(sample_bucket_ids) != sum(int(length) for length in lengths):
        raise ValueError("sample_bucket_ids must contain one identifier per dataset row")

    jobs: dict[str, tuple[int, tuple[str, ...]]] = {}
    offset = 0
    for path, length in zip(parquet_files, lengths, strict=True):
        path, length = str(path), int(length)
        if length < 0:
            raise ValueError(f"Parquet row counts must be non-negative, got {length}")
        bucket = _shape_bucket_id_from_parquet_path(path)
        buckets = tuple(sample_bucket_ids[offset:offset + length])
        offset += length
        if any(value != bucket for value in buckets):
            raise ValueError(f"Expanded bucket metadata disagrees with parquet path {path!r}")
        if path in jobs and jobs[path][0] != length:
            raise ValueError(f"Repeated parquet path {path!r} has inconsistent declared lengths")
        jobs.setdefault(path, (length, buckets))

    def scan_one(path: str) -> H3SequencePackingMetadata:
        length, buckets = jobs[path]
        return _scan_h3_sequence_packing_metadata_serial([path], [length], buckets, parquet_schema)

    start = time.monotonic()
    logger.info("Scanning H3 scheduling metadata: %d unique parquets with %d CPU readers", len(jobs), num_workers)
    completed: dict[str, H3SequencePackingMetadata] = {}
    paths = iter(jobs)
    with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="h3-metadata") as pool:
        pending = deque((path, pool.submit(scan_one, path)) for path in islice(paths, 2 * num_workers))
        while pending:
            path, future = pending.popleft()
            completed[path] = future.result()
            if len(completed) % 10000 == 0 or len(completed) == len(jobs):
                logger.info("H3 scheduling metadata: %d/%d files in %.1fs", len(completed), len(jobs),
                            time.monotonic() - start)
            if (path := next(paths, None)) is not None:
                pending.append((path, pool.submit(scan_one, path)))
    return H3SequencePackingMetadata(
        joint_token_counts=tuple(value for path in parquet_files for value in completed[str(path)].joint_token_counts),
        compatibility_ids=tuple(value for path in parquet_files for value in completed[str(path)].compatibility_ids),
    )


def load_h3_sequence_packing_metadata(
    parquet_files: Sequence[str],
    lengths: Sequence[int],
    sample_bucket_ids: Sequence[str],
    parquet_schema: pa.Schema,
) -> H3SequencePackingMetadata:
    """Scan once on rank zero, then broadcast lightweight metadata."""
    metadata, error = None, None
    if get_world_rank() == 0:
        try:
            metadata = scan_h3_sequence_packing_metadata(
                parquet_files, lengths, sample_bucket_ids, parquet_schema, num_workers=32)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    metadata, error = get_world_group().broadcast_object((metadata, error), src=0)
    if error is not None:
        raise RuntimeError(f"Rank zero failed to collect H3 sequence-packing metadata: {error}")
    if not isinstance(metadata, H3SequencePackingMetadata):
        raise RuntimeError("Rank zero did not publish valid H3 sequence-packing metadata")
    return metadata


def _parquet_data_identity(path: str, declared_length: int) -> tuple[str, int, int, int]:
    stat = os.stat(path)
    return os.path.realpath(path), int(declared_length), int(stat.st_size), int(stat.st_mtime_ns)


def scan_parquet_data_identity(
    parquet_files: Sequence[str],
    lengths: Sequence[int],
    *,
    num_workers: int = 1,
) -> list[tuple[str, int, int, int]]:
    """Return (realpath, declared rows, size, mtime_ns) per listed file, in input order.

    Per-file metadata latency dominates on shared filesystems, so the parallel
    path overlaps it with a bounded submission window. Each distinct entry is
    stat'ed once and repeats are expanded afterward in their original order.
    """
    if type(num_workers) is not int or not 1 <= num_workers <= 32:
        raise ValueError("Parquet identity num_workers must be an integer between 1 and 32")
    entries = [(str(path), int(length)) for path, length in zip(parquet_files, lengths, strict=True)]
    if num_workers == 1:
        return [_parquet_data_identity(path, length) for path, length in entries]
    completed: dict[tuple[str, int], tuple[str, int, int, int]] = {}
    jobs = iter(dict.fromkeys(entries))
    with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="parquet-identity") as pool:
        pending = deque((job, pool.submit(_parquet_data_identity, *job)) for job in islice(jobs, 2 * num_workers))
        while pending:
            job, future = pending.popleft()
            completed[job] = future.result()
            if (job := next(jobs, None)) is not None:
                pending.append((job, pool.submit(_parquet_data_identity, *job)))
    return [completed[entry] for entry in entries]


def load_parquet_data_identity(parquet_files: Sequence[str],
                               lengths: Sequence[int]) -> list[tuple[str, int, int, int]]:
    """Stat once on rank zero, then broadcast; every rank would compute the same rows."""
    identity, error = None, None
    if get_world_rank() == 0:
        start = time.monotonic()
        try:
            identity = scan_parquet_data_identity(parquet_files, lengths, num_workers=32)
            logger.info("Parquet data identity: %d files in %.1fs", len(identity), time.monotonic() - start)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    # Publish failures too, so peers never wait on a rank that already raised.
    identity, error = get_world_group().broadcast_object((identity, error), src=0)
    if error is not None:
        raise RuntimeError(f"Rank zero failed to collect parquet data identity: {error}")
    if not isinstance(identity, list) or len(identity) != len(parquet_files):
        raise RuntimeError("Rank zero did not publish valid parquet data identity")
    return identity


def _parse_data_path_specs(path: str | Sequence[str] | dict[str, int]) -> list[tuple[str, int]]:
    """Parse one or more dataset roots with old-framework repeat counts."""
    if isinstance(path, dict):
        return [(str(root), int(repeat)) for root, repeat in path.items() if int(repeat) > 0]
    if isinstance(path, Sequence) and not isinstance(path, str):
        return [(str(root), 1) for root in path]

    specs: list[tuple[str, int]] = []
    for part in str(path).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            dir_path, count_str = part.rsplit(":", 1)
            count = int(count_str)
        else:
            dir_path, count = part, 1
        if count > 0:
            specs.append((dir_path.strip(), count))
    return specs


def _resolve_cached_parquet_file(file_path: str) -> tuple[str, bool]:
    resolved = os.path.realpath(file_path, strict=True)
    return resolved, os.path.isfile(resolved)


def _resolve_cached_parquet_files(file_names: Sequence[str], *, num_workers: int) -> Iterator[tuple[str, bool]]:
    """Yield (realpath, is_file) in input order; a bounded window overlaps per-file filesystem latency."""
    names = iter(file_names)
    with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="parquet-index") as pool:
        pending = deque(pool.submit(_resolve_cached_parquet_file, name) for name in islice(names, 2 * num_workers))
        while pending:
            result = pending.popleft().result()
            if (name := next(names, None)) is not None:
                pending.append(pool.submit(_resolve_cached_parquet_file, name))
            yield result


def _read_required_parquet_cache(dataset_root: str, cache_file: str):
    """Validate the published index without walking, repairing or reordering it."""
    resolved_cache = os.path.realpath(cache_file, strict=True)
    if os.path.commonpath([dataset_root, resolved_cache]) != dataset_root:
        raise ValueError(f"Parquet cache points outside dataset root: {cache_file}")
    with open(cache_file, "rb") as f:
        cached = pickle.load(f)
        if f.read(1):
            raise ValueError("Parquet cache contains trailing data")
    if not isinstance(cached, (list, tuple)) or len(cached) != 2:
        raise ValueError("Parquet cache must contain file names and row counts")
    file_names, lengths = cached
    if not isinstance(file_names, (list, tuple)) or not isinstance(lengths, (list, tuple)):
        raise ValueError("Parquet cache file names and row counts must be sequences")
    if not file_names or len(file_names) != len(lengths):
        raise ValueError("Parquet cache is empty or file count and length count do not match")
    for file_path, length in zip(file_names, lengths, strict=True):
        if not isinstance(file_path, str) or not file_path:
            raise ValueError(f"Invalid cached parquet path: {file_path!r}")
        if isinstance(length, bool) or not isinstance(length, Integral) or length < 0:
            raise ValueError(f"Invalid cached parquet row count for {file_path!r}: {length!r}")
    declared_lengths: dict[str, int] = {}
    resolved_files = _resolve_cached_parquet_files(file_names, num_workers=32)
    for file_path, length, (resolved, is_file) in zip(file_names, lengths, resolved_files, strict=True):
        if os.path.commonpath([dataset_root, resolved]) != dataset_root:
            raise ValueError(f"Cached parquet path points outside dataset root: {file_path}")
        if not is_file:
            raise ValueError(f"Cached parquet path is not a file: {file_path}")
        if resolved in declared_lengths and declared_lengths[resolved] != length:
            raise ValueError(f"Repeated cached parquet path has inconsistent row counts: {file_path}")
        declared_lengths[resolved] = length
    if not any(lengths):
        raise ValueError("Parquet cache contains no rows")
    # Preserve the exact cached path spellings, order, repeats and row counts.
    return file_names, lengths


def get_parquet_files_and_length(path: str | Sequence[str] | dict[str, int], *,
                                  require_existing_parquet_cache: bool = False):
    if type(require_existing_parquet_cache) is not bool:
        raise ValueError("require_existing_parquet_cache must be a boolean")
    specs = _parse_data_path_specs(path)
    if len(specs) != 1 or specs[0][1] != 1:
        all_file_names: list[str] = []
        all_lengths: list[int] = []
        for root, repeat in specs:
            file_names, lengths = get_parquet_files_and_length(
                str(root), require_existing_parquet_cache=require_existing_parquet_cache)
            for _ in range(repeat):
                all_file_names.extend(file_names)
                all_lengths.extend(lengths)
        if not all_file_names:
            raise FileNotFoundError("No parquet files found under dataset paths: "
                                    f"{path}. "
                                    "Please verify these paths point to preprocessed parquet data.")
        file_lengths = sorted(
            zip(all_file_names, all_lengths, strict=True),
            key=lambda x: x[0],
        )
        file_names_sorted, lengths_sorted = zip(*file_lengths, strict=True)
        return file_names_sorted, lengths_sorted

    dataset_root = os.path.realpath(os.path.expanduser(specs[0][0]))
    # Check if cached info exists
    cache_dir = os.path.join(dataset_root, "map_style_cache")
    cache_file = os.path.join(cache_dir, "file_info.pkl")

    if require_existing_parquet_cache:
        cached, error = None, None
        if get_world_rank() == 0:
            try:
                cached = _read_required_parquet_cache(dataset_root, cache_file)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        # Publish both success and failure; no peer waits at the fallback
        # barrier or rereads a potentially changed index after rank zero.
        cached, error = get_world_group().broadcast_object((cached, error), src=0)
        if error is not None:
            raise RuntimeError(f"Required parquet cache failed validation at {cache_file}: {error}")
        if cached is None:
            raise RuntimeError(f"Rank zero did not publish required parquet cache: {cache_file}")
        return cached

    # Only rank 0 checks for cache and scans files if needed
    if get_world_rank() == 0:
        cache_loaded = False
        file_names_sorted = None
        lengths_sorted = None

        # First try to load existing cache
        if os.path.exists(cache_file):
            logger.info("Loading cached file info from %s", cache_file)
            try:
                with open(cache_file, "rb") as f:
                    file_names_sorted, lengths_sorted = pickle.load(f)
                file_names_sorted = tuple(
                    os.path.realpath(os.path.join(os.getcwd(), p) if not os.path.isabs(p) else p)
                    for p in file_names_sorted)
                files_outside_dataset_root = [
                    file_path for file_path in file_names_sorted
                    if os.path.commonpath([dataset_root, file_path]) != dataset_root
                ]
                missing_files = [file_path for file_path in file_names_sorted if not os.path.exists(file_path)]
                if files_outside_dataset_root:
                    logger.warning(
                        "Cached parquet file list points outside dataset root "
                        "(%s). Cache will be rebuilt. First out-of-root file: %s",
                        dataset_root,
                        files_outside_dataset_root[0],
                    )
                    cache_loaded = False
                elif missing_files:
                    logger.warning(
                        "Cached parquet file list contains %d missing files. "
                        "Cache will be rebuilt. First missing file: %s",
                        len(missing_files),
                        missing_files[0],
                    )
                    cache_loaded = False
                else:
                    cache_loaded = True
                    logger.info("Successfully loaded cached file info")
            except Exception as e:
                logger.error("Error loading cached file info: %s", str(e))
                logger.info("Falling back to scanning files")
                cache_loaded = False

        # If cache not loaded (either doesn't exist or failed to load), scan files
        if not cache_loaded:
            logger.info("Scanning parquet files to get lengths")
            lengths = []
            file_names = []
            for root, _, files in os.walk(dataset_root):
                for file in sorted(files):
                    if file.endswith('.parquet'):
                        file_path = os.path.realpath(os.path.join(root, file))
                        file_names.append(file_path)
            if len(file_names) == 0:
                raise FileNotFoundError("No parquet files found under dataset path: "
                                        f"{path}. "
                                        "Please verify this path points to preprocessed parquet "
                                        "data.")
            for file_path in tqdm.tqdm(file_names, desc="Reading parquet files to get lengths"):
                num_rows = pq.ParquetFile(file_path).metadata.num_rows
                lengths.append(num_rows)
            # sort according to file name to ensure all rank has the same order
            file_names_sorted, lengths_sorted = zip(*sorted(zip(file_names, lengths, strict=True), key=lambda x: x[0]),
                                                    strict=True)
            # Save the cache
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump((file_names_sorted, lengths_sorted), f)
            logger.info("Saved file info to %s", cache_file)

    # Wait for rank 0 to finish creating/loading cache
    world_group = get_world_group()
    world_group.barrier()

    # Now all ranks load the cache (it should exist and be valid now)
    logger.info("Loading cached file info from %s after barrier", cache_file)
    with open(cache_file, "rb") as f:
        file_names_sorted, lengths_sorted = pickle.load(f)
    if len(file_names_sorted) == 0:
        raise RuntimeError("Cached parquet metadata is empty after synchronization at "
                           f"{cache_file}. "
                           "Please verify the dataset path and regenerate cache.")
    if len(file_names_sorted) != len(lengths_sorted):
        raise RuntimeError("Cached parquet metadata is corrupted at "
                           f"{cache_file}: file count and length count do not match.")

    return file_names_sorted, lengths_sorted


def read_row_from_parquet_file(
    parquet_files: list[str],
    global_row_idx: int,
    lengths: list[int],
    columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    '''
    Read a row from a parquet file.
    Args:
        parquet_files: List[str]
        global_row_idx: int
        lengths: List[int]
    Returns:
    '''
    # find the parquet file and local row index
    cumulative = 0
    file_index = 0
    local_row_idx = 0

    for file_index in range(len(lengths)):
        if cumulative + lengths[file_index] > global_row_idx:
            local_row_idx = global_row_idx - cumulative
            break
        cumulative += lengths[file_index]
    else:
        # If we reach here, global_row_idx is out of bounds
        raise IndexError(f"global_row_idx {global_row_idx} is out of bounds for dataset")

    parquet_file = pq.ParquetFile(parquet_files[file_index])

    # Calculate the row group to read into memory and the local idx
    # This way we can avoid reading in the entire parquet file
    cumulative = 0
    row_group_index = 0
    local_index = 0

    for i in range(parquet_file.num_row_groups):
        num_rows = parquet_file.metadata.row_group(i).num_rows
        if cumulative + num_rows > local_row_idx:
            row_group_index = i
            local_index = local_row_idx - cumulative
            break
        cumulative += num_rows
    else:
        # If we reach here, local_row_idx is out of bounds for this parquet file
        raise IndexError(f"local_row_idx {local_row_idx} is out of bounds for parquet file {parquet_files[file_index]}")

    # Project at the Parquet reader boundary. This is especially important for
    # data-free training over a T2VA superset: the text-only schema must not
    # pull hundreds of MiB of unused video/audio latent bytes into host memory.
    row_group = parquet_file.read_row_group(row_group_index, columns=columns).to_pydict()
    row_dict = {k: v[local_index] for k, v in row_group.items()}
    del row_group

    return row_dict


# ────────────────────────────────────────────────────────────────────────────
# 2.  Dataset with batched __getitems__
# ────────────────────────────────────────────────────────────────────────────
class LatentsParquetMapStyleDataset(Dataset):
    """
    Return latents[B,C,T,H,W] and embeddings[B,L,D] in pinned CPU memory.
    Note: 
    Using parquet for map style dataset is not efficient, we mainly keep it for backward compatibility and debugging.
    """

    def __init__(
        self,
        path: str | Sequence[str] | dict[str, int],
        batch_size: int,
        parquet_schema: pa.Schema,
        cfg_rate: float = 0.0,
        seed: int = 42,
        drop_last: bool = True,
        drop_first_row: bool = False,
        text_padding_length: int | None = 512,
        max_presentation_tokens: int | None = None,
        native_shape_bucketing: bool = False,
        sequence_packing_enabled: bool = False,
        sequence_packing_max_joint_tokens: int | None = None,
        sequence_packing_max_samples_per_pack: int | None = None,
        gradient_accumulation_steps: int = 1,
        fixed_sequence_capacities: Sequence[int] | None = None,
        require_existing_parquet_cache: bool = False,
    ):
        super().__init__()
        self.fixed_sequence_capacities = validate_fixed_sequence_capacities(fixed_sequence_capacities)
        if self.fixed_sequence_capacities:
            if not native_shape_bucketing:
                raise ValueError("fixed_sequence_capacities requires native_shape_bucketing=True")
            if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size != 1:
                raise ValueError("fixed_sequence_capacities requires batch_size=1")
            # A full-document cost must not describe a truncated presentation.
            text_padding_length = None
        self.path = path
        self.cfg_rate = cfg_rate
        self.parquet_schema = parquet_schema
        self.seed = seed
        # Create a seeded random generator for deterministic CFG
        self.rng = random.Random(seed)
        logger.info("Initializing LatentsParquetMapStyleDataset with path: %s", path)
        self.parquet_files, self.lengths = get_parquet_files_and_length(
            path, require_existing_parquet_cache=require_existing_parquet_cache)
        self.batch = batch_size
        self.text_padding_length = text_padding_length
        self.max_presentation_tokens = max_presentation_tokens
        self.sequence_packing_enabled = sequence_packing_enabled
        self.sample_bucket_ids = (
            shape_bucket_ids_from_parquet_files(self.parquet_files, self.lengths) if native_shape_bucketing else None
        )
        self.sample_joint_token_counts: tuple[int, ...] | None = None
        self.sample_compatibility_ids: tuple[str, ...] | None = None
        self.sequence_packing_schedule_fingerprint: str | None = None
        num_sp_groups = get_world_size() // get_sp_world_size()
        data_identity: list[tuple[str, int, int, int]] = []
        if sequence_packing_enabled or self.fixed_sequence_capacities:
            if batch_size != 1:
                raise ValueError("Cross-sample sequence packing requires batch_size=1")
            if self.sample_bucket_ids is None:
                raise ValueError("Cross-sample sequence packing requires native_shape_bucketing=True")
            if sequence_packing_enabled and sequence_packing_max_joint_tokens is None:
                raise ValueError("Cross-sample sequence packing requires sequence_packing_max_joint_tokens")
            packing_metadata = load_h3_sequence_packing_metadata(
                self.parquet_files,
                self.lengths,
                self.sample_bucket_ids,
                self.parquet_schema,
            )
            self.sample_joint_token_counts = packing_metadata.joint_token_counts
            self.sample_compatibility_ids = packing_metadata.compatibility_ids
            if self.fixed_sequence_capacities:
                # Validate the complete scan, rather than stopping at the first
                # overflowing row in the sampler or a worker's first batch.
                validate_sequence_capacity_buckets(
                    self.sample_joint_token_counts, self.sample_bucket_ids, self.fixed_sequence_capacities)
            data_identity = load_parquet_data_identity(self.parquet_files, self.lengths)
        if sequence_packing_enabled:
            self.sampler = DP_SP_PackedBatchSampler(
                sample_joint_token_counts=self.sample_joint_token_counts,
                sample_compatibility_ids=self.sample_compatibility_ids,
                max_joint_tokens=sequence_packing_max_joint_tokens,
                max_samples_per_pack=sequence_packing_max_samples_per_pack,
                num_sp_groups=num_sp_groups,
                sp_world_size=get_sp_world_size(),
                global_rank=get_world_rank(),
                gradient_accumulation_steps=gradient_accumulation_steps,
                drop_first_row=drop_first_row,
                seed=seed,
                data_identity=data_identity,
                fixed_sequence_capacities=self.fixed_sequence_capacities or None,
            )
            self.sequence_packing_schedule_fingerprint = self.sampler.schedule_fingerprint
        else:
            self.sampler = DP_SP_BatchSampler(
                batch_size=batch_size,
                dataset_size=sum(self.lengths),
                num_sp_groups=num_sp_groups,
                sp_world_size=get_sp_world_size(),
                global_rank=get_world_rank(),
                drop_last=drop_last,
                drop_first_row=drop_first_row,
                seed=seed,
                sample_bucket_ids=self.sample_bucket_ids,
            )
        # Native-shape sampling keeps its existing schedule and cursor restore
        # path. Guard its capacity contract in the dataset checkpoint state;
        # the packed sampler also includes capacities in its own fingerprint.
        self.sequence_capacity_fingerprint: str | None = None
        if self.fixed_sequence_capacities:
            fingerprint_payload = {
                "fixed_sequence_capacities": self.fixed_sequence_capacities,
                "sample_joint_token_counts": self.sample_joint_token_counts,
                "sample_bucket_ids": self.sample_bucket_ids,
                "data_identity": data_identity,
            }
            self.sequence_capacity_fingerprint = hashlib.sha256(
                json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        logger.info("Dataset initialized with %d parquet files and %d rows", len(self.parquet_files), sum(self.lengths))

    def state_dict(self) -> dict[str, Any]:
        return {"sequence_capacity_fingerprint": self.sequence_capacity_fingerprint}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("sequence_capacity_fingerprint") != self.sequence_capacity_fingerprint:
            raise ValueError("Dataset sequence capacity fingerprint changed across resume; keep the data and "
                             "fixed_sequence_capacities unchanged")

    def _with_sequence_capacity(self, batch: dict[str, Any], indices: Sequence[int]) -> dict[str, Any]:
        capacities = getattr(self, "fixed_sequence_capacities", ())
        if capacities:
            if self.sample_joint_token_counts is None:
                raise RuntimeError("Fixed sequence capacity metadata was not initialized")
            real_tokens = sum(self.sample_joint_token_counts[index] for index in indices)
            capacity = select_sequence_capacity(real_tokens, capacities)
            batch["_fixed_sequence_capacity"] = capacity
            batch["_packed_joint_token_count"] = real_tokens
            # Token padding is not sample-repeat padding or a loss weight.
            batch["_sequence_padding_ratio"] = capacity / real_tokens
        return batch

    def get_validation_negative_prompt(self) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get the negative prompt for validation. 
        This method ensures the negative prompt is loaded and cached properly.
        Returns the processed negative prompt data (latents, embeddings, masks, info).
        """

        # Read first row from first parquet file
        file_path = self.parquet_files[0]
        row_idx = 0
        # Read the negative prompt data
        row_dict = read_row_from_parquet_file(
            [file_path],
            row_idx,
            [self.lengths[0]],
            columns=self.parquet_schema.names,
        )

        batch = collate_rows_from_parquet_schema([row_dict],
                                                 self.parquet_schema,
                                                 self.text_padding_length,
                                                 cfg_rate=0.0,
                                                 rng=self.rng,
                                                 max_presentation_tokens=getattr(self, "max_presentation_tokens",
                                                                                 None))
        negative_prompt = batch['info_list'][0]['prompt']
        negative_prompt_embedding = batch['text_embedding']
        negative_prompt_attention_mask = batch['text_attention_mask']
        if len(negative_prompt_embedding.shape) == 2:
            negative_prompt_embedding = negative_prompt_embedding.unsqueeze(0)
        if len(negative_prompt_attention_mask.shape) == 1:
            negative_prompt_attention_mask = negative_prompt_attention_mask.unsqueeze(0).unsqueeze(0)

        return negative_prompt_embedding, negative_prompt_attention_mask, negative_prompt

    # PyTorch calls this ONLY because the batch_sampler yields a list
    def __getitems__(self, indices: list[int]) -> dict[str, Any]:
        """
        Batch fetch using read_row_from_parquet_file for each index.
        """
        rows = [
            read_row_from_parquet_file(
                self.parquet_files,
                idx,
                self.lengths,
                columns=self.parquet_schema.names,
            ) for idx in indices
        ]

        # Inject sample indices for deterministic CFG dropout
        # that is reproducible across checkpoint resume.
        for row, idx in zip(rows, indices):
            row["_sample_index"] = idx

        if getattr(self, "sequence_packing_enabled", False):
            if not isinstance(indices, PackedBatchIndices):
                raise RuntimeError("The packed dataset requires PackedBatchIndices metadata from its sampler")
            expected_fingerprint = getattr(self, "sequence_packing_schedule_fingerprint", None)
            if expected_fingerprint is not None and indices.schedule_fingerprint != expected_fingerprint:
                raise RuntimeError("Packed batch indices do not belong to this dataset's packing schedule")
            if self.sample_bucket_ids is None or self.sample_joint_token_counts is None:
                raise RuntimeError("Packed dataset metadata was not initialized")
            compatibility_ids = {
                self.sample_compatibility_ids[index] for index in indices
            } if self.sample_compatibility_ids is not None else set()
            if len(compatibility_ids) != 1:
                raise RuntimeError(f"Packed sampler emitted incompatible rows: {sorted(compatibility_ids)}")

            packed_samples: list[dict[str, Any]] = []
            packed_bucket_ids: list[str] = []
            for row, index in zip(rows, indices, strict=True):
                sample = collate_rows_from_parquet_schema(
                    [row],
                    self.parquet_schema,
                    self.text_padding_length,
                    cfg_rate=self.cfg_rate,
                    seed=self.seed,
                    max_presentation_tokens=getattr(self, "max_presentation_tokens", None),
                )
                bucket_id = self.sample_bucket_ids[index]
                sample["_shape_bucket_id"] = bucket_id
                packed_samples.append(sample)
                packed_bucket_ids.append(bucket_id)

            packed_token_counts = tuple(self.sample_joint_token_counts[index] for index in indices)
            if sum(packed_token_counts) != indices.local_joint_token_count:
                raise RuntimeError("Packed sampler and dataset joint-token counts disagree")
            batch = {
                "_packed_samples": packed_samples,
                "_packed_sample_indices": tuple(int(index) for index in indices),
                "_packed_shape_bucket_ids": tuple(packed_bucket_ids),
                "_packed_sample_joint_token_counts": packed_token_counts,
                "_packed_joint_token_count": indices.local_joint_token_count,
                "_packed_compatibility_id": compatibility_ids.pop(),
                "_sequence_packing_local_document_count": indices.local_document_count,
                "_sequence_packing_window_document_count": indices.window_document_count,
                "_sequence_packing_window_document_counts": indices.window_document_counts,
                "_sequence_packing_local_window_document_counts": indices.local_window_document_counts,
                "_sequence_packing_local_effective_document_count": indices.local_effective_document_count,
                "_sequence_packing_window_effective_document_count": indices.window_effective_document_count,
                "_sequence_packing_window_effective_document_counts": indices.window_effective_document_counts,
                "_sequence_packing_local_window_effective_document_counts":
                indices.local_window_effective_document_counts,
                "_sequence_packing_local_window_compatibility_ids": indices.local_window_compatibility_ids,
                "_sequence_packing_local_joint_token_count": indices.local_joint_token_count,
                "_sequence_packing_window_joint_token_count": indices.window_joint_token_count,
                "_sequence_packing_accumulation_index": indices.accumulation_index,
                "_sequence_packing_num_dp_groups": indices.num_dp_groups,
                "_sequence_packing_gradient_accumulation_steps": indices.gradient_accumulation_steps,
                "_sequence_packing_loss_scale": indices.loss_scale,
                "_sequence_packing_schedule_fingerprint": indices.schedule_fingerprint,
            }
            return self._with_sequence_capacity(batch, indices)

        batch = collate_rows_from_parquet_schema(rows,
                                                 self.parquet_schema,
                                                 self.text_padding_length,
                                                 cfg_rate=self.cfg_rate,
                                                 seed=self.seed,
                                                 max_presentation_tokens=getattr(self, "max_presentation_tokens",
                                                                                 None))
        if self.sample_bucket_ids is not None:
            bucket_ids = {self.sample_bucket_ids[index] for index in indices}
            if len(bucket_ids) != 1:
                raise RuntimeError(f"Exact-shape sampler emitted a mixed bucket batch: {sorted(bucket_ids)}")
            batch["_shape_bucket_id"] = bucket_ids.pop()
        return self._with_sequence_capacity(batch, indices)

    def __len__(self):
        return sum(self.lengths)


# ────────────────────────────────────────────────────────────────────────────
# 3.  Loader helper – everything else stays just like your original trainer
# ────────────────────────────────────────────────────────────────────────────
def passthrough(batch):
    return batch


def build_parquet_map_style_dataloader(path,
                                       batch_size,
                                       num_data_workers,
                                       parquet_schema,
                                       cfg_rate=0.0,
                                       drop_last=True,
                                       drop_first_row=False,
                                       text_padding_length=512,
                                       max_presentation_tokens=None,
                                       seed=42,
                                       native_shape_bucketing=False,
                                       sequence_packing_enabled=False,
                                       sequence_packing_max_joint_tokens=None,
                                       sequence_packing_max_samples_per_pack=None,
                                       gradient_accumulation_steps=1,
                                       fixed_sequence_capacities=None,
                                       require_existing_parquet_cache=False) -> tuple[LatentsParquetMapStyleDataset,
                                                                              StatefulDataLoader]:
    dataset = LatentsParquetMapStyleDataset(path,
                                            batch_size,
                                            cfg_rate=cfg_rate,
                                            drop_last=drop_last,
                                            drop_first_row=drop_first_row,
                                            text_padding_length=text_padding_length,
                                            max_presentation_tokens=max_presentation_tokens,
                                            parquet_schema=parquet_schema,
                                            seed=seed,
                                            native_shape_bucketing=native_shape_bucketing,
                                            sequence_packing_enabled=sequence_packing_enabled,
                                            sequence_packing_max_joint_tokens=sequence_packing_max_joint_tokens,
                                            sequence_packing_max_samples_per_pack=
                                            sequence_packing_max_samples_per_pack,
                                            gradient_accumulation_steps=gradient_accumulation_steps,
                                            fixed_sequence_capacities=fixed_sequence_capacities,
                                            require_existing_parquet_cache=require_existing_parquet_cache)

    loader = StatefulDataLoader(
        dataset,
        batch_sampler=dataset.sampler,
        collate_fn=passthrough,
        num_workers=num_data_workers,
        pin_memory=True,
        persistent_workers=num_data_workers > 0,
    )
    return dataset, loader
