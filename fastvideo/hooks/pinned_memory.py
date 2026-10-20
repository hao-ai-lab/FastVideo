# SPDX-License-Identifier: Apache-2.0
"""Exact-size pinned host storage for inference offload."""

import weakref
from collections.abc import Iterable

import torch

from fastvideo.logger import init_logger

logger = init_logger(__name__)
_ALIGNMENT = 256
_PAGE_ALIGNMENT = 4096


def _unregister(buffer: torch.Tensor, device: int) -> None:
    # The allocation must outlive any outstanding nonblocking H2D copies.
    try:
        with torch.cuda.device(device):
            torch.cuda.synchronize()
            error = torch.cuda.cudart().cudaHostUnregister(buffer.data_ptr())
        if error != 0:
            logger.warning("cudaHostUnregister failed: %s", error)
    except Exception as exc:
        # CUDA may already be unavailable during interpreter shutdown.
        logger.warning("Could not unregister pinned host arena: %s", exc)


class PinnedTensorArena:
    """Pack tensors into one CUDA-registered CPU allocation, aligned to 256 bytes.

    PyTorch's pinned allocator rounds large allocations to powers of two. Registering
    ordinary host storage avoids that overhead. Typed views retain this owner, so
    registration survives even if the module or offload state is dropped first.
    If registration is unavailable, allocate conventional pinned tensors instead.
    Call ``close`` only after all views and pending copies have been released.
    """

    def __init__(self, tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
        self.offsets: dict[str, tuple[int, int]] = {}
        size = 0
        for name, tensor in tensors:
            size = (size + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
            length = tensor.numel() * tensor.element_size()
            self.offsets[name] = (size, length)
            size += length
        self.nbytes = size
        self.buffer: torch.Tensor | None = None
        self._finalizer: weakref.finalize | None = None
        if not size:
            return
        # Register dedicated pages: small malloc allocations can otherwise share a
        # registered page with another arena. The extra space is bounded by 8 KiB.
        span = (size + _PAGE_ALIGNMENT - 1) // _PAGE_ALIGNMENT * _PAGE_ALIGNMENT
        allocation = torch.empty(span + _PAGE_ALIGNMENT - 1, dtype=torch.uint8, device="cpu")
        start = (-allocation.data_ptr()) % _PAGE_ALIGNMENT
        # Give the aligned region its own storage base. Tensor.is_pinned() queries
        # the storage pointer, which would precede the registered region for a
        # plain narrow() view. The memoryview retains the original allocation.
        buffer = torch.frombuffer(memoryview(allocation.numpy())[start:start + span], dtype=torch.uint8)
        device = torch.cuda.current_device()
        try:
            error = torch.cuda.cudart().cudaHostRegister(buffer.data_ptr(), span, 0)
            if error != 0:
                raise RuntimeError(f"cudaHostRegister returned {error}")
        except Exception as exc:
            logger.warning("Exact-size host registration failed; using the pinned allocator: %s", exc)
            return
        self.buffer = buffer
        self._finalizer = weakref.finalize(self, _unregister, buffer, device)

    def empty_like(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Return a contiguous host view with the source's dtype and shape."""
        if self.buffer is None:
            return torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True)
        offset, length = self.offsets[name]
        host = self.buffer.narrow(0, offset, length).view(tensor.dtype).reshape(tensor.shape)
        host._pinned_arena = self
        return host

    def close(self) -> None:
        """Unregister before releasing storage; safe to call more than once."""
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None
        self.buffer = None
