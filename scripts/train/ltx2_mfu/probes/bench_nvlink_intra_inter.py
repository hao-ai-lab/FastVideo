#!/usr/bin/env python3
"""Compare two-rank NCCL all-reduce within and across GB200 trays."""

from __future__ import annotations

import json
import os
import socket
import time

import torch
import torch.distributed as dist


def benchmark(label: str, ranks: list[int], group: dist.ProcessGroup) -> None:
    rank = dist.get_rank()
    if rank in ranks:
        sizes_and_iterations = [
            (1 << 20, 200),
            (16 << 20, 100),
            (256 << 20, 40),
            (1 << 30, 20),
        ]
        for size_bytes, iterations in sizes_and_iterations:
            tensor = torch.ones(size_bytes // 4, device="cuda", dtype=torch.float32)
            for _ in range(20):
                dist.all_reduce(tensor, group=group)
            torch.cuda.synchronize()
            dist.barrier(group=group)

            start = time.perf_counter()
            for _ in range(iterations):
                dist.all_reduce(tensor, group=group)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / iterations

            slowest = torch.tensor(elapsed, device="cuda", dtype=torch.float64)
            dist.all_reduce(slowest, op=dist.ReduceOp.MAX, group=group)
            if rank == ranks[0]:
                seconds = float(slowest.item())
                print(
                    "NVLINK_RESULT "
                    + json.dumps(
                        {
                            "algorithmic_gbps": size_bytes / seconds / 1e9,
                            "label": label,
                            "latency_us": seconds * 1e6,
                            "ranks": ranks,
                            "size_bytes": size_bytes,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            del tensor, slowest
    dist.barrier()


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    if dist.get_world_size() != 8:
        raise RuntimeError("expected exactly eight ranks across two four-GPU trays")

    intra = dist.new_group([0, 1], backend="nccl")
    inter = dist.new_group([0, 4], backend="nccl")
    if dist.get_rank() in (0, 4):
        print(
            "NVLINK_TOPOLOGY "
            + json.dumps(
                {
                    "hostname": socket.gethostname(),
                    "local_rank": local_rank,
                    "rank": dist.get_rank(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()

    benchmark("intra_tray", [0, 1], intra)
    benchmark("inter_tray", [0, 4], inter)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
