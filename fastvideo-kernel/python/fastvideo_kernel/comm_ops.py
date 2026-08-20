# SPDX-License-Identifier: Apache-2.0
"""Ulysses sequence-parallel all-to-all ops.

Thin wrappers over csrc/comm/ulysses_all_to_all.cu. The kernel stores directly
into peers' memory through NCCL's device API, so the caller supplies an
ncclComm_t and NCCL owns the window, the topology and the barrier.
"""

import torch

try:
    from fastvideo_kernel._C import fastvideo_kernel_ops as _ops
except ImportError:  # pragma: no cover - no compiled extension in this install
    _ops = None

_SUPPORTED_WORLD_SIZES = (2, 4, 6, 8)


def is_available() -> bool:
    """Whether this wheel was built with the Ulysses all-to-all kernel."""
    return _ops is not None and hasattr(_ops, "ulysses_a2a")


def _require() -> None:
    if not is_available():
        raise RuntimeError(
            "the Ulysses all-to-all kernel is not present in this fastvideo-kernel build; "
            "rebuild with ./build.sh or install a wheel that includes csrc/comm/")


def lsa_covers_group(comm_ptr: int, world_size: int) -> bool:
    """Whether every rank in the group is load-store accessible to every other.

    This is the whole topology check: NCCL worked it out at ncclCommInitRank and
    reports it as ncclTeamLsa, so there is nothing to probe.
    """
    _require()
    return bool(_ops.ulysses_lsa_covers_group(int(comm_ptr), int(world_size)))


def init(comm_ptr: int, nbytes: int, rank: int, world_size: int) -> int:
    """Register a symmetric window and create the device comm. Collective.

    ``comm_ptr`` is an ncclComm_t for the group. Returns an opaque handle, freed
    with :func:`dispose`.
    """
    _require()
    if world_size not in _SUPPORTED_WORLD_SIZES:
        raise ValueError(f"ulysses a2a supports world sizes {_SUPPORTED_WORLD_SIZES}, "
                         f"got {world_size}")
    return int(_ops.init_ulysses_a2a(int(comm_ptr), int(nbytes), int(rank), int(world_size)))


def dispose(handle: int) -> None:
    """Release a handle from :func:`init`. It is dangling afterwards."""
    _require()
    _ops.dispose_ulysses_a2a(int(handle))


def all_to_all(handle: int, inp: torch.Tensor, out: torch.Tensor, B: int, S_local: int, H: int,
               D: int, mode: int) -> None:
    """Run one fused all-to-all on the current stream, writing into ``out``.

    ``mode == 0``: ``[B, S_local, H, D] -> [B, S_global, H_local, D]``
    ``mode == 1``: ``[B, S_global, H_local, D] -> [B, S_local, H, D]``

    ``H`` is the global head count. Every rank must call with consistent
    geometry in the same order.
    """
    _require()
    _ops.ulysses_a2a(int(handle), inp, out, int(B), int(S_local), int(H), int(D), int(mode))
