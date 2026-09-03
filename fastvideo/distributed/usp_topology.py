# SPDX-License-Identifier: Apache-2.0
"""USP (Unified Sequence Parallelism) topology construction.

USP splits each sequence-parallel group into a 2D mesh of
``ring_size x ulysses_size == sp_size`` ranks:
  - Ring group: ``ring_size`` ranks that run pure Ring Attention with each
    other, exchanging K/V shards peer-to-peer around the ring.
  - Ulysses group: ``ulysses_size`` ranks that run a head<->sequence
    all-to-all with each other before/after the Ring step.
``ring_size == 1`` degenerates to pure Ulysses (the Ulysses group is the full
SP group, no Ring group exists). ``ring_size == sp_size`` degenerates to pure
Ring (the Ring group is the full SP group, no Ulysses group exists). In both
degenerate cases the corresponding group is simply the SP group itself, so no
extra process group is created.

This module owns the *policy* of deriving ring/ulysses sizes and deciding
which subgroups to build; ``fastvideo.distributed.parallel_state`` stays a
plain process-group registry that calls into this module once (inside
``initialize_model_parallel``) and stores the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastvideo.distributed.parallel_state import GroupCoordinator


@dataclass(frozen=True)
class USPTopology:
    """The Ring x Ulysses (USP) 2D sub-mesh built inside one SP group."""
    ring_size: int
    ulysses_size: int
    ring_group: GroupCoordinator | None
    ulysses_group: GroupCoordinator | None


def _build_usp_subgroup_ranks(
    sp_group_ranks: list[list[int]],
    ring_size: int,
    ulysses_size: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Split each SP replica's ranks into a 2D (ring x ulysses) mesh.

    Ranks within one SP replica are laid out row-major with ``ring_size`` as
    the outer (slow) dimension and ``ulysses_size`` as the inner (fast)
    dimension, i.e. ``rank_in_group = ring_idx * ulysses_size + ulysses_idx``.
    Ulysses groups are therefore contiguous blocks of ranks (favoring the
    high-bandwidth, low-latency intra-node all-to-all), while Ring groups are
    strided across those blocks (favoring Ring's point-to-point, latency-
    tolerant communication pattern, which tolerates spanning nodes).

    This also matches how the sequence itself is sharded across the SP group
    (contiguous chunks in rank order): each Ulysses group's ranks hold
    contiguous sequence shards that concatenate into one contiguous
    "ring chunk", and ranks that share a Ring group hold the same head
    subset after the Ulysses all-to-all.

    Returns ``(ulysses_group_ranks, ring_group_ranks)``: the flattened list of
    rank-lists for every Ulysses group and every Ring group, across all SP
    replicas.
    """
    ulysses_group_ranks: list[list[int]] = []
    ring_group_ranks: list[list[int]] = []
    for sp_ranks in sp_group_ranks:
        assert len(sp_ranks) == ring_size * ulysses_size
        for ring_idx in range(ring_size):
            ulysses_group_ranks.append(sp_ranks[ring_idx * ulysses_size:(ring_idx + 1) * ulysses_size])
        for ulysses_idx in range(ulysses_size):
            ring_group_ranks.append([sp_ranks[ring_idx * ulysses_size + ulysses_idx] for ring_idx in range(ring_size)])
    return ulysses_group_ranks, ring_group_ranks


def build_usp_topology(
    sp_group: GroupCoordinator,
    sp_group_ranks: list[list[int]],
    ring_size: int,
    local_rank: int,
    backend: str,
) -> USPTopology:
    """Derive ring/ulysses sizes and build the USP subgroups within ``sp_group``.

    The pure-Ring and pure-Ulysses degenerate cases reuse ``sp_group`` itself
    rather than creating a redundant, identical process group.
    """
    # Lazy import: avoids a module-load-time cycle. parallel_state.py calls
    # into this module from initialize_model_parallel(), so a module-level
    # import back into parallel_state.py here would try to bind a name that
    # doesn't exist yet while parallel_state.py is still executing its own
    # top level. parallel_state.py already uses this same lazy-import
    # pattern elsewhere (e.g. `current_platform` inside GroupCoordinator).
    from fastvideo.distributed.parallel_state import init_model_parallel_group

    sequence_model_parallel_size = sp_group.world_size
    assert sequence_model_parallel_size % ring_size == 0, (
        f"sequence_model_parallel_size ({sequence_model_parallel_size}) must be divisible by "
        f"ring_size ({ring_size})")
    ulysses_size = sequence_model_parallel_size // ring_size

    if ring_size == 1:
        return USPTopology(ring_size=ring_size, ulysses_size=ulysses_size, ring_group=None, ulysses_group=sp_group)
    if ulysses_size == 1:
        return USPTopology(ring_size=ring_size, ulysses_size=ulysses_size, ring_group=sp_group, ulysses_group=None)

    ulysses_group_ranks, ring_group_ranks = _build_usp_subgroup_ranks(sp_group_ranks, ring_size, ulysses_size)
    ulysses_group = init_model_parallel_group(ulysses_group_ranks, local_rank, backend, group_name="ulysses")
    ring_group = init_model_parallel_group(ring_group_ranks, local_rank, backend, group_name="ring")
    return USPTopology(ring_size=ring_size,
                       ulysses_size=ulysses_size,
                       ring_group=ring_group,
                       ulysses_group=ulysses_group)
