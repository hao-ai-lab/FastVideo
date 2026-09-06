# SPDX-License-Identifier: Apache-2.0
"""Exploratory GB200/SP4 study; synthetic operands and transformer blocks.

Prepared mode is a benchmark-only upper bound: every rank follows this script's
identical fixed contract and warms the largest window before timing. It is not
a production replacement for the helper's dynamic collective fallback protocol.
Run using torchrun; results include all rank-max samples and parity checks.
"""

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--section', choices=['collectives', 'blocks'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=5)
    args = parser.parse_args()
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
    from fastvideo.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase
    from fastvideo.distributed.device_communicators.ulysses_a2a import _FusedUlyssesA2A
    from fastvideo.distributed.parallel_state import get_sp_group

    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.manual_seed(20260906 + rank)
    maybe_init_distributed_environment_and_model_parallel(1, world)
    group = get_sp_group()
    comm = group.device_communicator
    helper = comm.ulysses_a2a
    assert helper is not None
    records = []

    def emit(record):
        records.append(record)
        if rank == 0:
            args.output.write_text(json.dumps(records, indent=2) + '\n')
            print(json.dumps({k: v for k, v in record.items() if 'samples' not in k}), flush=True)

    def measure(fn, metadata, *, iterations=None, repeats=1):
        count = iterations or args.iters
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        for repeat in range(args.rounds):
            wall, gpu = [], []
            for _ in range(count):
                dist.barrier(group=group.cpu_group)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin = time.perf_counter_ns()
                start.record()
                for _ in range(repeats):
                    fn()
                end.record()
                torch.cuda.synchronize()
                wall.append((time.perf_counter_ns() - begin) / 1000 / repeats)
                gpu.append(start.elapsed_time(end) * 1000 / repeats)
            values = torch.tensor([wall, gpu], dtype=torch.float64, device='cuda')
            dist.all_reduce(values, op=dist.ReduceOp.MAX)
            wall, gpu = values.cpu().tolist()
            emit(dict(**metadata, round=repeat, repeats=repeats, world=world,
                      wall_p50_us=statistics.median(wall), gpu_p50_us=statistics.median(gpu),
                      wall_rank_max_samples_us=wall, gpu_rank_max_samples_us=gpu))

    def a2a(x, mode, route):
        dims = (2, 1) if mode == 0 else (1, 2)
        if route == 'nccl':
            return DeviceCommunicatorBase.all_to_all_4D(comm, x, *dims)
        if route == 'safe':
            return comm.all_to_all_4D(x, *dims)
        assert route == 'prepared'
        return _FusedUlyssesA2A.apply(helper, x, mode)

    def prepare(operands):
        # Use the full existing protocol outside the measured region.
        for x, mode in sorted(operands, key=lambda item: item[0].numel(), reverse=True):
            actual = a2a(x, mode, 'safe')
            expected = a2a(x, mode, 'nccl')
            assert torch.equal(actual, expected)
        assert helper._handle is not None
        torch.cuda.synchronize()

    try:
        workloads = [('small', 8192, 40), ('Wan', 75600, 40), ('H3', 37296, 56)]
        if args.section == 'collectives':
            for model, sequence, heads in workloads:
                x = torch.randn(3, sequence // world, heads, 128, device='cuda', dtype=torch.bfloat16)
                y = torch.randn(1, sequence, heads // world, 128, device='cuda', dtype=torch.bfloat16)
                prepare([(x, 0), (y, 1)])
                signature, _ = helper._call_signature(x, 2, 1)
                measure(lambda: helper._agree_call(signature), dict(model=model, operation='agreement', route='safe'))
                for mode, operand in [(0, x), (1, y)]:
                    operation = 'scatter' if mode == 0 else 'gather'
                    for route in ('nccl', 'safe', 'prepared'):
                        measure(lambda: a2a(operand, mode, route), dict(model=model, operation=operation, route=route))
                    dst = torch.empty_like(operand)
                    measure(lambda: dst.copy_(operand), dict(model=model, operation=operation + '_copy', route='copy'))
                for route in ('nccl', 'safe', 'prepared'):
                    def pair():
                        a2a(x, 0, route)
                        a2a(y, 1, route)
                    measure(pair, dict(model=model, operation='pair_streamed', route=route),
                            iterations=max(3, args.iters // 4), repeats=20)
                    x.requires_grad_(True)
                    y.requires_grad_(True)
                    dx = torch.randn(3, sequence, heads // world, 128, device='cuda', dtype=torch.bfloat16)
                    dy = torch.randn(1, sequence // world, heads, 128, device='cuda', dtype=torch.bfloat16)
                    def training_pair():
                        ox = a2a(x, 0, route)
                        oy = a2a(y, 1, route)
                        torch.autograd.grad((ox, oy), (x, y), (dx, dy))
                    measure(training_pair, dict(model=model, operation='pair_fwd_bwd', route=route),
                            iterations=max(3, args.iters // 4), repeats=5)
                    x.requires_grad_(False)
                    y.requires_grad_(False)
                    del dx, dy
                # All ranks collectively capture the same already-prepared calls.
                capture_stream = torch.cuda.Stream()
                capture_stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    captured_x = a2a(x, 0, 'prepared')
                    captured_y = a2a(y, 1, 'prepared')
                torch.cuda.current_stream().wait_stream(capture_stream)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(captured_x, a2a(x, 0, 'nccl'))
                assert torch.equal(captured_y, a2a(y, 1, 'nccl'))
                measure(graph.replay, dict(model=model, operation='pair_graph', route='prepared'),
                        iterations=max(3, args.iters // 4), repeats=20)
                if model != 'small':
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for route in ('nccl', 'safe', 'prepared'):
                            with torch.profiler.record_function(route):
                                a2a(x, 0, route)
                                a2a(y, 1, route)
                                torch.cuda.synchronize()
                    if rank == 0:
                        prof.export_chrome_trace(str(args.output.with_name(model + '-collectives-trace.json')))
                del x, y, dst, captured_x, captured_y, graph
                gc.collect()
                torch.cuda.empty_cache()
        else:
            from fastvideo.attention.utils.flash_attn_default import fa_version, flash_attn_func_compilable
            for model, sequence, heads in workloads:
                hidden = heads * 128
                x = torch.randn(1, sequence // world, hidden, device='cuda', dtype=torch.bfloat16,
                                requires_grad=True)
                weights = [torch.nn.Parameter(torch.randn(out_dim, in_dim, device='cuda', dtype=torch.bfloat16)
                                              * in_dim**-0.5)
                           for in_dim, out_dim in [(hidden, 3 * hidden), (hidden, hidden),
                                                   (hidden, 4 * hidden), (4 * hidden, hidden)]]
                dy = torch.randn_like(x)
                probe = torch.empty(3, sequence // world, heads, 128, device='cuda', dtype=torch.bfloat16)
                # Initialize before parity; otherwise uninitialized NaNs can fail equality.
                probe.zero_()
                prepare([(probe, 0)])
                del probe

                def block(route):
                    projected = F.linear(x, weights[0]).reshape(1, sequence // world, 3, heads, 128)
                    qkv = torch.cat(projected.unbind(dim=2), dim=0)
                    q, k, v = a2a(qkv, 0, route).chunk(3, dim=0)
                    attended = flash_attn_func_compilable(q, k, v, causal=False)
                    local = a2a(attended.contiguous(), 1, route).flatten(2)
                    residual = x + F.linear(local, weights[1])
                    return residual + F.linear(F.gelu(F.linear(residual, weights[2]), approximate='tanh'), weights[3])

                # Compare outputs and all input/weight gradients under the identical compute recipe.
                reference = block('nccl')
                reference_grads = torch.autograd.grad(reference, [x, *weights], dy)
                for route in ('safe', 'prepared'):
                    result = block(route)
                    gradients = torch.autograd.grad(result, [x, *weights], dy)
                    torch.testing.assert_close(result, reference, rtol=0, atol=0)
                    # FA backward uses atomic reductions: allow its bf16 summation variation.
                    for actual, expected in zip(gradients, reference_grads):
                        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
                    del result, gradients
                del reference, reference_grads
                for training in (False, True):
                    for route in ('nccl', 'safe', 'prepared'):
                        def step():
                            with torch.set_grad_enabled(training):
                                out = block(route)
                                if training:
                                    torch.autograd.grad(out, [x, *weights], dy)
                        measure(step, dict(model=model, operation='block_train' if training else 'block_infer',
                                           route=route, flash_attention=fa_version, sequence=sequence, heads=heads,
                                           note='synthetic dense block; no FSDP, optimizer, norms, or checkpointing'),
                                iterations=max(3, args.iters // 4))
                del x, weights, dy
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        cleanup_dist_env_and_memory()


if __name__ == '__main__':
    main()
