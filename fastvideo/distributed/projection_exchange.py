# SPDX-License-Identifier: Apache-2.0
"""Two-rank QKV traffic that can start before all projections are ready."""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from fastvideo.distributed.async_collectives import PeerSlices


class Completion:

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.works: list[dist.Work] = []
        self.waited_streams: set[int] = set()
        self.ready = torch.cuda.Event()
        self.producer = torch.cuda.current_stream(device).cuda_stream

    def wait(self) -> None:
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if stream not in self.waited_streams:
            if stream != self.producer:
                torch.cuda.current_stream(self.device).wait_event(self.ready)
            for work in self.works:
                work.wait()
            self.waited_streams.add(stream)


@dataclass
class PendingInput:
    source: torch.Tensor
    remote: torch.Tensor
    rank: int
    completion: Completion

    def wait(self) -> PeerSlices:
        self.completion.wait()
        return PeerSlices(self.source[self.rank], self.remote.view(self.source.dtype), self.source, self.rank)


@dataclass
class PendingScale:
    source: torch.Tensor
    output: torch.Tensor
    completion: Completion

    def wait(self) -> torch.Tensor:
        self.completion.wait()
        return self.output


class ProjectionExchange:
    """Exchange Q/K/V as each projection becomes ready, retaining their storage.

    Input is [head groups, destinations, QKV, rows, heads, 128]. Call start
    in Q/K/V order on the producing CUDA stream, then packets on that stream.
    Input payload and scales must remain unmodified until all consumers wait.
    The local payload aliases its producer; only the remote slice travels.
    """

    def __init__(self, payload: torch.Tensor, scale: torch.Tensor, group: dist.ProcessGroup) -> None:
        if (dist.get_world_size(group) != 2 or payload.ndim != 6 or payload.shape[1:3] != (2, 3)
                or payload.shape[-1] != 128 or not payload.is_cuda or not payload.is_contiguous()
                or payload.dtype != torch.float8_e4m3fn or scale.shape != payload.shape[:-1]
                or scale.device != payload.device or scale.dtype != torch.float32 or not scale.is_contiguous()):
            raise ValueError("Projection exchange requires contiguous two-rank FP8 QKV and FP32 scales")
        self.payload, self.scale, self.group = payload, scale, group
        self.rank = dist.get_rank(group)
        self.peer = dist.get_global_rank(group, 1 - self.rank)
        self.remote = torch.empty((payload.shape[0], *payload.shape[2:]), device=payload.device, dtype=torch.uint8)
        self.received_scale = torch.empty_like(scale)
        self.completion = Completion(payload.device)
        self.next_projection = 0

    def start(self, projection: int) -> None:
        if projection != self.next_projection or projection not in (0, 1, 2):
            raise ValueError("Every rank must start projections in Q, K, V order")
        self.next_projection += 1
        ops: list[dist.P2POp] = []
        for chunk in range(self.payload.shape[0]):
            send = self.payload[chunk, 1 - self.rank, projection].view(torch.uint8)
            remote = self.remote[chunk, projection]
            send_scale = self.scale[chunk, 1 - self.rank, projection]
            remote_scale = self.received_scale[chunk, 1 - self.rank, projection]
            ops.extend((dist.P2POp(dist.irecv, remote, self.peer,
                                   self.group), dist.P2POp(dist.isend, send, self.peer, self.group),
                        dist.P2POp(dist.irecv, remote_scale, self.peer,
                                   self.group), dist.P2POp(dist.isend, send_scale, self.peer, self.group)))
        self.completion.works.extend(dist.batch_isend_irecv(ops))

    def packets(self) -> list[tuple[PendingInput, PendingScale]]:
        """Return consumers after all projections have been queued on this stream."""
        if self.next_projection != 3:
            raise RuntimeError("All three projection sends must start before consumption")
        self.received_scale[:, self.rank].copy_(self.scale[:, self.rank])
        self.completion.ready.record()
        return [(PendingInput(self.payload[c], self.remote[c], self.rank,
                              self.completion), PendingScale(self.scale[c], self.received_scale[c], self.completion))
                for c in range(self.payload.shape[0])]
