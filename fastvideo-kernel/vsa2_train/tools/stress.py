"""Repeat the VSA2 forward and backward at the student's shapes on poisoned outputs, to catch silently dropped work.

A CTA that never runs its work item leaves its rows as they were. Every call here first fills the outputs with NaN,
so a dropped item shows as NaN rows among the valid ones. The forward has no atomics, so every repeat must also match
the first bit for bit. This is how the VSA-128 kernels' B300 bug showed: a hand-decoded cluster-launch-control
response misread successful cancels on sm_103a, and the cancelled tiles' work was dropped.

    python tools/stress.py --solution solution --build-dir /tmp/vsa2-build --workload s8 s20 --seconds 300

Under torchrun, ``--comm-mib`` keeps an all-gather of that many MiB per rank in flight on a side stream throughout, as
FSDP's prefetch does in training: VSA-128's race showed only under such concurrency. Every rank then runs the same
number of calls per workload (the collectives must pair up across ranks) and rank 0 prints the totals.

    torchrun --nproc_per_node 8 tools/stress.py --build-dir /tmp/vsa2-build --comm-mib 512 --seconds 600

Exits 1 when any call leaves NaN on a valid row or a forward repeat differs from the first.
"""

import argparse
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loader  # noqa: E402
import spec  # noqa: E402


def valid_rows(workload, device):
    thw, grid, p, nv, n = spec.geometry(workload)
    _, video_valid = spec._tile_positions(thw, grid, device)
    text = spec.WORKLOADS[workload]["text"]
    return torch.cat([torch.arange(p * spec.ROWS, device=device) < text, video_valid])


class SideTraffic:
    """An all-gather kept in flight on a side stream and a stop flag agreed by every rank, under torchrun."""

    def __init__(self, comm_mib, device):
        import torch.distributed as dist

        self.dist = dist
        dist.init_process_group("nccl", device_id=device)
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        # Its own communicator, so the side traffic overlaps the stop-flag all-reduce on the default group.
        self.group = dist.new_group(backend="nccl")
        self.stream = torch.cuda.Stream(device)
        self.payload = torch.randn(comm_mib * 2**20 // 2, device=device, dtype=torch.bfloat16)
        self.gathered = torch.empty(self.world * self.payload.numel(), device=device, dtype=torch.bfloat16)
        self.device = device

    def launch(self):
        with torch.cuda.stream(self.stream):
            self.dist.all_gather_into_tensor(self.gathered, self.payload, group=self.group, async_op=True)

    def stop(self, local):
        flag = torch.tensor([int(local)], device=self.device, dtype=torch.int32)
        self.dist.all_reduce(flag, op=self.dist.ReduceOp.MAX)
        return bool(flag.item())

    def total(self, values):
        totals = torch.tensor(values, device=self.device, dtype=torch.int64)
        self.dist.all_reduce(totals)
        return totals.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--solution", default="solution")
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--workload", nargs="+", default=["s8", "s20"])
    parser.add_argument("--seconds", type=float, default=300.0, help="per workload")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--comm-mib", type=int, default=0, help="side-stream all-gather per rank, under torchrun")
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank if args.comm_mib else 0)
    torch.cuda.set_device(device)
    side = SideTraffic(args.comm_mib, device) if args.comm_mib else None
    rank = side.rank if side else 0
    build_dir = os.path.join(args.build_dir, f"rank{local_rank}") if side else args.build_dir
    module = loader.load(os.path.abspath(args.solution), build_dir=build_dir, chdir=False).module
    if rank == 0:
        print(f"{torch.cuda.get_device_name()} (sm_{''.join(map(str, torch.cuda.get_device_capability()))}), "
              f"solution {os.path.abspath(args.solution)}"
              f"{f', {side.world} ranks with a {args.comm_mib} MiB side-stream all-gather' if side else ''}", flush=True)
    failed = False
    for workload in args.workload:
        q, k, v, gate, dout, t, h, w, text, topk = spec.make_inputs(workload, args.seed + rank, device)
        out, selection, lse, dq, dk, dv, dgate = spec.make_outputs([q, k, v, gate, dout, t, h, w, text, topk])
        valid = valid_rows(workload, device)
        first = None
        calls = nan_fwd = nan_bwd = mismatched = worst = 0
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            if side is None and elapsed >= args.seconds:
                break
            if side is not None and calls % 16 == 0 and side.stop(elapsed >= args.seconds):
                break
            if side is not None and calls % 4 == 0:
                side.launch()
            for x in (out, lse, selection):
                x.fill_(math.nan)
            module.forward(q, k, v, gate, t, h, w, text, topk, out, selection, lse)
            for x in (dq, dk, dv, dgate):
                x.fill_(math.nan)
            module.backward(q, k, v, gate, out, lse, selection, dout, t, h, w, text, dq, dk, dv, dgate)
            torch.cuda.synchronize(device)
            calls += 1
            bad_rows = (out[0, :, valid].isnan().any(-1) | lse[0, :, valid].isnan()).sum().item()
            nan_fwd += bad_rows > 0
            grads_bad = sum(g[0, :, valid].isnan().any(-1).sum().item() for g in (dq, dk, dv, dgate))
            nan_bwd += grads_bad > 0
            if first is None:
                first = out.clone()
            else:
                differ = (out[0, :, valid] != first[0, :, valid]).any(-1).sum().item()
                mismatched += differ > 0
                worst = max(worst, differ)
        rate = calls / max(time.monotonic() - start, 1e-9)
        if side is not None:
            torch.cuda.synchronize(device)
            calls, nan_fwd, nan_bwd, mismatched = side.total([calls, nan_fwd, nan_bwd, mismatched])
            rate *= side.world
        ok = nan_fwd == 0 and nan_bwd == 0 and mismatched == 0
        failed |= not ok
        if rank == 0:
            print(f"{workload}: {calls} calls ({rate:.1f}/s); forward calls with NaN rows {nan_fwd}, backward calls with "
                  f"NaN gradient rows {nan_bwd}, forward repeats differing from the first {mismatched}"
                  f"{'' if side else f' (worst {worst} rows)'}: {'ok' if ok else 'FAILED'}", flush=True)
    if side is not None:
        side.dist.destroy_process_group()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
