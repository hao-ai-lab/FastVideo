# SPDX-License-Identifier: Apache-2.0
"""Tensor collectives with explicit input ownership until their consumers wait."""

from dataclasses import dataclass
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass
class PendingTensor:
    output: torch.Tensor
    source: torch.Tensor
    work: dist.Work
    logical_shape: torch.Size | None = None

    def wait(self) -> torch.Tensor:
        """Order consumers on the current CUDA stream after the collective."""
        self.work.wait()
        if self.logical_shape is None:
            return self.output
        peer_elements = math.prod(self.logical_shape) // self.output.shape[0]
        return self.output[:, :peer_elements].contiguous().view(self.logical_shape)


def all_to_all_tensor_async(source: torch.Tensor, group: dist.ProcessGroup, *, align_bytes: int = 1) -> PendingTensor:
    """Exchange equal peer slices, optionally aligning their NCCL buffer offsets.

    Small FP32 scale slices with an odd head count can otherwise start eight
    bytes off a 16-byte boundary and defeat NCCL's vectorized copies. Padding
    only the transport buffer retains the caller's logical tensor layout.
    """
    logical_shape = None
    if align_bytes > 1:
        if align_bytes % source.element_size():
            raise ValueError("Collective alignment must be a multiple of the element size")
        world = dist.get_world_size(group)
        if source.numel() % world:
            raise ValueError("Equal all-to-all slices require a tensor divisible by the group size")
        elements = source.numel() // world
        padding = -elements % (align_bytes // source.element_size())
        if padding:
            logical_shape = source.shape
            source = F.pad(source.reshape(world, elements), (0, padding))
    output = torch.empty_like(source)
    work = dist.all_to_all_single(output, source, group=group, async_op=True)
    assert work is not None
    return PendingTensor(output, source, work, logical_shape)


def all_gather_tensor_async(source: torch.Tensor, group: dist.ProcessGroup, world: int) -> PendingTensor:
    output = torch.empty((world, *source.shape), device=source.device, dtype=source.dtype)
    work = dist.all_gather_into_tensor(output, source, group=group, async_op=True)
    assert work is not None
    return PendingTensor(output, source, work)


@dataclass
class PeerSlices:
    """Two-rank receive whose local slice stays in the producer's storage."""
    local: torch.Tensor
    remote: torch.Tensor
    template: torch.Tensor
    rank: int

    def materialize(self) -> torch.Tensor:
        """Restore source-rank order for consumers without a two-pointer kernel."""
        slices = (self.local, self.remote) if self.rank == 0 else (self.remote, self.local)
        return torch.stack([part.view(torch.uint8) for part in slices]).view(self.template.dtype)


@dataclass
class PendingPeerSlices:
    source: torch.Tensor
    remote: torch.Tensor
    rank: int
    works: list[dist.Work]

    def wait(self) -> PeerSlices:
        for work in self.works:
            work.wait()
        return PeerSlices(self.source[self.rank], self.remote.view(self.source.dtype), self.source, self.rank)


def exchange_peer_tensor_async(source: torch.Tensor, group: dist.ProcessGroup) -> PendingPeerSlices:
    """Exchange only the remote half of a contiguous two-rank payload.

    Every rank must enqueue calls in the same order. The producer is retained
    through completion and consumption, including the local slice that bypasses
    NCCL. Receivers index the two slices directly to avoid a concatenation copy.
    """
    if dist.get_world_size(group) != 2 or source.shape[0] != 2 or not source.is_contiguous():
        raise ValueError("Peer-slice exchange requires two contiguous destination slices")
    rank = dist.get_rank(group)
    peer = dist.get_global_rank(group, 1 - rank)
    send = source[1 - rank].view(torch.uint8)
    remote = torch.empty_like(send)
    works = dist.batch_isend_irecv(
        [dist.P2POp(dist.irecv, remote, peer, group),
         dist.P2POp(dist.isend, send, peer, group)])
    return PendingPeerSlices(source, remote, rank, works)
