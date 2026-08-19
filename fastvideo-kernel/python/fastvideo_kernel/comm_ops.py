# SPDX-License-Identifier: Apache-2.0
"""Ulysses sequence-parallel all-to-all ops.

Thin wrappers over the compiled kernel in ``csrc/comm/ulysses_all_to_all.cu``.
These are deliberately unopinionated: they validate nothing beyond what the C++
side already enforces, and they know nothing about process groups or IPC. The
orchestration -- allocating the IPC-shared staging buffers, exchanging handles,
probing the topology, and agreeing on a backend across ranks -- lives in
``fastvideo/distributed/device_communicators/``, because it needs
``torch.distributed`` and this package deliberately does not depend on it.

``is_available()`` is the supported way to ask whether the kernel was built into
this wheel; every entry point raises a clear error rather than an AttributeError
when it was not.
"""

from typing import List

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


def signal_size() -> int:
    """Bytes each rank must allocate for its IPC-shared signal buffer."""
    _require()
    return int(_ops.ulysses_signal_size())


def init(out_ipc_ptrs: List[int], signal_ipc_ptrs: List[int], rank: int, world_size: int,
         full_nvlink: bool) -> int:
    """Create an all-to-all context over already-opened IPC pointers.

    ``out_ipc_ptrs[j]`` and ``signal_ipc_ptrs[j]`` must be pointers valid in
    *this* process that address rank ``j``'s buffers -- i.e. the result of
    ``cudaIpcOpenMemHandle`` on the handle rank ``j`` exported. Building that
    table is the caller's job.

    ``full_nvlink`` must be True: the kernel pushes over all-pairs NVLink P2P and
    has no non-P2P path, so the caller has to have verified the topology.

    Returns an opaque handle (a host-side ``UlyssesA2A*`` as an int, not a device
    pointer). Free it with :func:`dispose`.

    The C++ side zeroes this rank's signal buffer with an async memset, so the
    caller must synchronize the device and then issue a process-group barrier
    before the first :func:`all_to_all`.
    """
    _require()
    if world_size not in _SUPPORTED_WORLD_SIZES:
        raise ValueError(f"ulysses a2a supports world sizes {_SUPPORTED_WORLD_SIZES}, "
                         f"got {world_size}")
    if not full_nvlink:
        raise ValueError("full_nvlink=False is not supported: the fused kernel pushes over "
                         "all-pairs NVLink P2P and has no non-P2P path")
    return int(_ops.init_ulysses_a2a(list(out_ipc_ptrs), list(signal_ipc_ptrs), int(rank),
                                     int(world_size), bool(full_nvlink)))


def dispose(handle: int) -> None:
    """Release a handle from :func:`init`. It is dangling afterwards."""
    _require()
    _ops.dispose_ulysses_a2a(int(handle))


def all_to_all(handle: int, inp: torch.Tensor, out: torch.Tensor, B: int, S_local: int, H: int,
               D: int, mode: int) -> None:
    """Run one fused all-to-all, writing the result into ``out``.

    ``mode == 0`` (scatter heads): ``[B, S_local, H, D] -> [B, S_global, H_local, D]``
    ``mode == 1`` (gather heads):  ``[B, S_global, H_local, D] -> [B, S_local, H, D]``

    ``H`` is the *global* head count; ``H_local = H // world_size`` and
    ``S_global = S_local * world_size``. Runs on the current CUDA stream. Every
    rank must call with consistent geometry in the same order -- a mismatch is a
    collective failure, as with any collective library.
    """
    _require()
    _ops.ulysses_a2a(int(handle), inp, out, int(B), int(S_local), int(H), int(D), int(mode))
