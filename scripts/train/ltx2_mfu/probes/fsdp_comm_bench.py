import argparse
import statistics

import torch
import torch.distributed as dist


def timed(fn, stream: torch.cuda.Stream, warmup: int = 5, iterations: int = 20) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record()
            fn()
            end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), statistics.mean(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symm", action="store_true")
    parser.add_argument("--pg-alloc", action="store_true")
    parser.add_argument("--elements", type=int, default=67_500_000)
    args = parser.parse_args()

    rank = int(__import__("os").environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    group_name = dist.group.WORLD.group_name
    world_size = dist.get_world_size()
    stream = torch.cuda.Stream(priority=-1)

    if args.symm and args.pg_alloc:
        raise ValueError("choose only one allocator")
    if args.symm:
        import torch.distributed._symmetric_memory as symm_mem

        symm_mem.set_backend("NCCL")
        pool = symm_mem.get_mem_pool(torch.device("cuda", rank))
        with torch.cuda.use_mem_pool(pool):
            ag_out = torch.empty(args.elements * world_size, dtype=torch.bfloat16, device=rank)
            rs_in = torch.empty(args.elements * world_size, dtype=torch.bfloat16, device=rank)
            rs_out = torch.empty(args.elements, dtype=torch.bfloat16, device=rank)
        symm_mem.rendezvous(ag_out, group=group_name)
        symm_mem.rendezvous(rs_in, group=group_name)
        symm_mem.rendezvous(rs_out, group=group_name)
    elif args.pg_alloc:
        backend = dist.group.WORLD._get_backend(torch.device("cuda", rank))
        if not backend.supports_tensor_alloc(torch.device("cuda", rank)):
            raise RuntimeError("ProcessGroupNCCL tensor allocator is unavailable")
        ag_out = backend.allocate_tensor(
            args.elements * world_size, dtype=torch.bfloat16, device=torch.device("cuda", rank)
        )
        rs_in = backend.allocate_tensor(
            args.elements * world_size, dtype=torch.bfloat16, device=torch.device("cuda", rank)
        )
        rs_out = backend.allocate_tensor(
            args.elements, dtype=torch.bfloat16, device=torch.device("cuda", rank)
        )
    else:
        ag_out = torch.empty(args.elements * world_size, dtype=torch.bfloat16, device=rank)
        rs_in = torch.empty(args.elements * world_size, dtype=torch.bfloat16, device=rank)
        rs_out = torch.empty(args.elements, dtype=torch.bfloat16, device=rank)

    ag_in = ag_out.narrow(0, rank * args.elements, args.elements)
    ag_in.fill_(rank + 1)
    rs_in.fill_(1)

    def ag() -> None:
        dist.all_gather_into_tensor(ag_out, ag_in)

    def rs_bf16() -> None:
        dist.reduce_scatter_tensor(rs_out, rs_in, op=dist.ReduceOp.SUM)

    ag_median, ag_mean = timed(ag, stream)
    rs_median, rs_mean = timed(rs_bf16, stream)

    if not args.symm:
        rs32_in = torch.empty(args.elements * world_size, dtype=torch.float32, device=rank)
        rs32_out = torch.empty(args.elements, dtype=torch.float32, device=rank)
        rs32_in.fill_(1)

        def rs_fp32() -> None:
            dist.reduce_scatter_tensor(rs32_out, rs32_in, op=dist.ReduceOp.SUM)

        rs32_median, rs32_mean = timed(rs_fp32, stream)
    else:
        rs32_median = rs32_mean = float("nan")

    if rank == 0:
        mode = "symm_ce" if args.symm else "pg_alloc" if args.pg_alloc else "default"
        print(f"mode={mode} elements_per_rank={args.elements} world={world_size}")
        print(f"ag_bf16_ms median={ag_median:.3f} mean={ag_mean:.3f}")
        print(f"rs_bf16_ms median={rs_median:.3f} mean={rs_mean:.3f}")
        if not args.symm:
            print(f"rs_fp32_ms median={rs32_median:.3f} mean={rs32_mean:.3f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
