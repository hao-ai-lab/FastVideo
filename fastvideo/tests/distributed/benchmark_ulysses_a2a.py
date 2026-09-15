# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun on one NVLink host; compare revisions using identical argv.

Reports host-to-completion latency (including agreement), p50/p95 of rank-max
samples, for bf16 Wan2.1-14B and MiniMax-H3 5s/720p attention operands. No model
weights are required. Set FASTVIDEO_ULYSSES_A2A=off for the NCCL baseline.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-fused", action="store_true")
    args = parser.parse_args()

    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
    from fastvideo.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase
    from fastvideo.distributed.parallel_state import get_sp_group

    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(2026 + rank)
    maybe_init_distributed_environment_and_model_parallel(1, world)
    comm = get_sp_group().device_communicator
    records = []
    try:
        for name, sequence, heads in [("Wan2.1-14B", 75600, 40), ("MiniMax-H3", 37296, 56)]:
            assert sequence % world == 0
            scatter = torch.randn(3, sequence // world, heads, 128, dtype=torch.bfloat16, device=device)
            gather = torch.randn(1, sequence, heads // world, 128, dtype=torch.bfloat16, device=device)
            for operand, dims in [(scatter, (2, 1)), (gather, (1, 2))]:
                actual = comm.all_to_all_4D(operand, *dims)
                expected = DeviceCommunicatorBase.all_to_all_4D(comm, operand, *dims)
                assert torch.equal(actual, expected), f"{name} {dims}: parity failed"
                del actual, expected
            armed = comm.ulysses_a2a is not None and comm.ulysses_a2a._handle is not None
            if args.expect_fused:
                assert armed, "benchmark requires all ranks to engage the fused kernel"
            for repeat in range(args.rounds):
                for operation in ("scatter", "gather", "layer"):
                    def run():
                        if operation in ("scatter", "layer"):
                            comm.all_to_all_4D(scatter, 2, 1)
                        if operation in ("gather", "layer"):
                            comm.all_to_all_4D(gather, 1, 2)

                    for _ in range(args.warmup):
                        run()
                    torch.cuda.synchronize()
                    samples = []
                    for _ in range(args.iters):
                        dist.barrier(group=get_sp_group().cpu_group)
                        start = time.perf_counter_ns()
                        run()
                        torch.cuda.synchronize()
                        samples.append((time.perf_counter_ns() - start) / 1000)
                    samples_tensor = torch.tensor(samples, dtype=torch.float64, device=device)
                    dist.all_reduce(samples_tensor, op=dist.ReduceOp.MAX)
                    maximums = samples_tensor.cpu().tolist()
                    record = dict(tag=args.tag, model=name, operation=operation, round=repeat,
                                  world_size=world, fused=armed, p50_us=statistics.median(maximums),
                                  p95_us=sorted(maximums)[int(0.95 * (len(maximums) - 1))],
                                  rank_max_samples_us=maximums)
                    records.append(record)
                    if rank == 0:
                        print(json.dumps({k: v for k, v in record.items() if k != "rank_max_samples_us"}), flush=True)
            del scatter, gather
        if rank == 0:
            args.output.write_text(json.dumps(records, indent=2) + "\n")
    finally:
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
