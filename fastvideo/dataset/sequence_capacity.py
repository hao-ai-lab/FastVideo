# SPDX-License-Identifier: Apache-2.0
"""Fixed H3 joint-sequence capacities (not latent tensor element counts)."""

from collections.abc import Sequence
from numbers import Integral
from typing import Any


def validate_fixed_sequence_capacities(values: Sequence[int] | None) -> tuple[int, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("fixed_sequence_capacities must be a non-empty list of positive integer multiples of 256")
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value <= 0 or value % 256
           for value in values):
        raise ValueError("fixed_sequence_capacities must contain positive integer multiples of 256")
    return tuple(sorted(set(int(value) for value in values)))


def select_sequence_capacity(real_tokens: int, capacities: Sequence[int]) -> int:
    """Choose from validated capacities; the caller's ordering is immaterial."""
    if isinstance(real_tokens, bool) or not isinstance(real_tokens, Integral) or real_tokens <= 0:
        raise ValueError(f"Real joint-token count must be a positive integer, got {real_tokens!r}")
    fitting = [capacity for capacity in capacities if capacity >= real_tokens]
    if fitting:
        return min(fitting)
    raise ValueError(f"Sequence has {real_tokens} real tokens, exceeding fixed_sequence_capacities={list(capacities)}")


def sequence_capacity_metrics(batch: Any) -> dict[str, float]:
    """Log the active (possibly carried) document, not the incoming loader row."""
    capacity = int(getattr(batch, "sequence_padded_num_tokens", 0) or 0)
    if not capacity:
        return {}
    real_tokens = int(getattr(batch, "sequence_packing_num_tokens", 0) or 0)
    if not 0 < real_tokens <= capacity:
        raise ValueError("Sequence capacity metrics require 0 < real tokens <= padded tokens")
    return {
        "sequence_padding_ratio": capacity / real_tokens,
        "sequence_padded_tokens": float(capacity),
        "sequence_real_tokens": float(real_tokens),
    }


def validate_sequence_capacity_buckets(
    counts: Sequence[int], bucket_ids: Sequence[str], capacities: Sequence[int],
) -> None:
    """Report every oversized bucket before a training iterator is created."""
    if not capacities:
        raise ValueError("fixed_sequence_capacities must not be empty for bucket validation")
    largest = max(capacities)
    offending: dict[str, int] = {}
    for count, bucket in zip(counts, bucket_ids, strict=True):
        if count > largest:
            offending[bucket] = max(count, offending.get(bucket, 0))
    if offending:
        raise ValueError(f"fixed_sequence_capacities largest capacity {largest} is exceeded by bucket(s): "
                         + ", ".join(f"{bucket}: {count} tokens" for bucket, count in sorted(offending.items())))
