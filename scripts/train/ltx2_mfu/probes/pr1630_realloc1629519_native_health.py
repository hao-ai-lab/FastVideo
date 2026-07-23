#!/usr/bin/env python3
import json
import os
import statistics
from datetime import timedelta

import torch
import torch.distributed as dist


local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", timeout=timedelta(seconds=180))

scalar = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
dist.all_reduce(scalar)
torch.cuda.synchronize(local_rank)
assert scalar.item() == 36.0
print(
    "HEALTH_GATE "
    + json.dumps(
        {
            "phase": "scalar",
            "rank": rank,
            "local_rank": local_rank,
            "world_size": dist.get_world_size(),
            "sum": scalar.item(),
            "host": os.uname().nodename,
        },
        sort_keys=True,
    ),
    flush=True,
)

world_size = dist.get_world_size()
payload = torch.ones(128 * 1024 * 1024, dtype=torch.bfloat16, device=f"cuda:{local_rank}")
for _ in range(8):
    dist.all_reduce(payload)
    payload.div_(world_size)
torch.cuda.synchronize(local_rank)
dist.barrier()

latencies_ms = []
for _ in range(128):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    dist.all_reduce(payload)
    end.record()
    end.synchronize()
    latencies_ms.append(start.elapsed_time(end))
    payload.div_(world_size)

torch.cuda.synchronize(local_rank)
assert payload[0].item() == 1.0
sorted_ms = sorted(latencies_ms)
mean_ms = statistics.fmean(latencies_ms)
max_mean = torch.tensor(mean_ms, device=f"cuda:{local_rank}")
dist.all_reduce(max_mean, op=dist.ReduceOp.MAX)
payload_gb = payload.numel() * payload.element_size() / 1e9
print(
    "HEALTH_GATE "
    + json.dumps(
        {
            "phase": "sustained_all_reduce",
            "rank": rank,
            "local_rank": local_rank,
            "host": os.uname().nodename,
            "iterations": len(latencies_ms),
            "payload_bytes": payload.numel() * payload.element_size(),
            "mean_ms": mean_ms,
            "p50_ms": statistics.median(sorted_ms),
            "p95_ms": sorted_ms[int(0.95 * (len(sorted_ms) - 1))],
            "max_ms": max(sorted_ms),
            "slowest_rank_mean_ms": max_mean.item(),
            "slowest_rank_bus_gbps": payload_gb * (2 * (world_size - 1) / world_size) / (max_mean.item() / 1000),
            "value": payload[0].item(),
        },
        sort_keys=True,
    ),
    flush=True,
)
dist.barrier()
dist.destroy_process_group()
