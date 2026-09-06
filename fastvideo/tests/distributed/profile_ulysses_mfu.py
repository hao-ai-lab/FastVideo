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
    parser.add_argument('--probe-dir', type=Path)
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--sparse-probe-dir', type=Path)
    parser.add_argument('--models', nargs='+', default=['small', 'Wan', 'H3'])
    parser.add_argument('--paired', action='store_true')
    parser.add_argument('--h3-lengths', nargs='+', type=int)
    parser.add_argument('--probe-window-gib', type=int, default=1)
    parser.add_argument('--routes', nargs='+')
    parser.add_argument('--chunked', action='store_true')
    parser.add_argument('--packed-count', type=int, default=3)
    args = parser.parse_args()
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
    from fastvideo.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase
    from fastvideo.distributed.device_communicators.ulysses_a2a import _FusedUlyssesA2A, UlyssesA2AHelper
    import fastvideo.distributed.device_communicators.ulysses_a2a as ulysses_module
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
    engagement = {}
    probe_helpers = {}
    if args.probe_dir:
        import sys
        sys.path.insert(0, str(args.probe_dir))
        import ulysses_launch_probe as probe_ops

        class ProbeHelper(UlyssesA2AHelper):
            def _call_signature(self, x, scatter_dim, gather_dim):
                # Experimental per-helper cap; this script has one host thread.
                original = ulysses_module.MAX_WINDOW_BYTES
                ulysses_module.MAX_WINDOW_BYTES = args.probe_window_gib * 1024**3
                try:
                    return super()._call_signature(x, scatter_dim, gather_dim)
                finally:
                    ulysses_module.MAX_WINDOW_BYTES = original

            def _allocate(self, nbytes):
                return probe_ops.allocate_ulysses_a2a(nbytes, rank, world, torch.cuda.current_device())

            def _register_window(self, handle):
                probe_ops.register_ulysses_a2a_window(handle, self._comm_ptr())

            def _create_dev_comm(self, handle):
                probe_ops.create_ulysses_a2a_dev_comm(handle)

            def _dispose(self, handle, *, synchronize):
                if synchronize:
                    torch.cuda.synchronize()
                probe_ops.dispose_ulysses_a2a(handle)

            def run_armed(self, x, mode):
                b, s, h, d = x.shape
                local_s, global_h = (s, h) if mode == 0 else (s // world, h * world)
                shape = (b, s * world, h // world, d) if mode == 0 else (b, s // world, h * world, d)
                out = torch.empty(shape, device=x.device, dtype=x.dtype)
                probe_ops.ulysses_a2a(self._handle, x, out, b, local_s, global_h, d, mode,
                                      self.probe_blocks, 512, True)
                return out

        for blocks in (36, 72, 144):
            candidate = ProbeHelper(helper.cpu_group, helper.device_group, world, helper.device, helper.pynccl_comm)
            candidate.probe_blocks = blocks
            probe_helpers[f'probe{blocks}'] = candidate
        if args.chunked:
            candidate = ProbeHelper(helper.cpu_group, helper.device_group, world, helper.device, helper.pynccl_comm)
            candidate.probe_blocks = 144
            probe_helpers['chunk144'] = candidate

            class ChunkIntoHelper(ProbeHelper):
                """Benchmark-only chunked protocol with one owned final output.

                Full-call agreement fixes the chunk count/order on all ranks;
                inherited setup votes protect allocation/registration. The real
                registered capacity is one batch plane and is advertised as such.
                """

                def try_all_to_all_4D(self, x, scatter_dim, gather_dim):
                    if self._disabled_reason is not None or torch.compiler.is_compiling():
                        return None
                    signature, reason = self._call_signature(x, scatter_dim, gather_dim)
                    fused, permanent, lifecycle = self._agree_call(signature)
                    if not fused:
                        if not lifecycle:
                            self.close()
                            self._disable('inconsistent chunk-window lifecycle')
                        if permanent:
                            self._disable(reason or 'peer declined chunked operation')
                        return None
                    slot_bytes = signature[-2] // x.shape[0]
                    assert slot_bytes <= 1024**3, 'one batch plane must fit the bounded window'
                    if self._handle is None:
                        if not self._build(slot_bytes):
                            return None
                    elif slot_bytes > self._nbytes:
                        if not self.close() or not self._build(slot_bytes):
                            return None
                    return _FusedUlyssesA2A.apply(self, x, signature[2])

                def run_armed(self, x, mode):
                    b, s, h, d = x.shape
                    local_s, global_h = (s, h) if mode == 0 else (s // world, h * world)
                    shape = (b, s * world, h // world, d) if mode == 0 else (b, s // world, h * world, d)
                    out = torch.empty(shape, device=x.device, dtype=x.dtype)
                    for plane in range(b):
                        probe_ops.ulysses_a2a(self._handle, x[plane:plane + 1], out[plane:plane + 1],
                                              1, local_s, global_h, d, mode, 144, 512, True)
                    return out

            candidate = ChunkIntoHelper(helper.cpu_group, helper.device_group, world, helper.device, helper.pynccl_comm)
            candidate.probe_blocks = 144
            probe_helpers['chunk_into144'] = candidate
    routes = ['nccl', 'safe', 'prepared', *probe_helpers]
    if args.routes:
        assert set(args.routes) <= set(routes)
        routes = args.routes
    if args.h3_lengths:
        assert 'prepared' not in routes, 'legacy prepared control cannot cover the 1 GiB cap transition'

    def emit(record):
        record.update(sequence=sequence, heads=heads, packed_count=args.packed_count,
                      probe_window_gib=args.probe_window_gib)
        records.append(record)
        if rank == 0:
            args.output.write_text(json.dumps(records, indent=2) + '\n')
            print(json.dumps({k: v for k, v in record.items() if 'samples' not in k}), flush=True)

    def measure(fn, metadata, *, iterations=None, repeats=1, round_index=None):
        count = iterations or args.iters
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        for repeat in (range(args.rounds) if round_index is None else [round_index]):
            wall, gpu = [], []
            before = dict(engagement)
            torch.cuda.reset_peak_memory_stats()
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
                      forward_engagement={k: v - before.get(k, 0) for k, v in engagement.items() if v > before.get(k, 0)},
                      torch_peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      registered_windows_gib=sum(h._nbytes for h in [helper, *probe_helpers.values()]) / 1024**3,
                      wall_rank_max_samples_us=wall, gpu_rank_max_samples_us=gpu))

    def a2a(x, mode, route):
        dims = (2, 1) if mode == 0 else (1, 2)
        if route == 'nccl':
            key = f'{route}:{mode}:nccl'
            engagement[key] = engagement.get(key, 0) + 1
            return DeviceCommunicatorBase.all_to_all_4D(comm, x, *dims)
        if route == 'safe':
            result = helper.try_all_to_all_4D(x, *dims)
            key = f'{route}:{mode}:{"fused" if result is not None else "nccl"}'
            engagement[key] = engagement.get(key, 0) + 1
            return result if result is not None else DeviceCommunicatorBase.all_to_all_4D(comm, x, *dims)
        if route == 'chunk144':
            candidate = probe_helpers[route]
            # Agree the whole contract before ranks enter a variable number of chunks.
            signature, _ = candidate._call_signature(x, *dims)
            if not candidate._agree_call(signature)[0]:
                return DeviceCommunicatorBase.all_to_all_4D(comm, x, *dims)
            results = []
            for chunk in x.split(1, dim=0):
                assert chunk.numel() * chunk.element_size() <= 1024**3
                result = candidate.try_all_to_all_4D(chunk, *dims)
                key = f'{route}:{mode}:{"fused" if result is not None else "nccl"}'
                engagement[key] = engagement.get(key, 0) + 1
                results.append(result if result is not None else DeviceCommunicatorBase.all_to_all_4D(comm, chunk, *dims))
            return results[0] if len(results) == 1 else torch.cat(results, dim=0)
        if route in probe_helpers:
            result = probe_helpers[route].try_all_to_all_4D(x, *dims)
            assert result is not None, f'{route} unexpectedly declined'
            key = f'{route}:{mode}:fused'
            engagement[key] = engagement.get(key, 0) + 1
            return result
        assert route == 'prepared'
        return _FusedUlyssesA2A.apply(helper, x, mode)

    def prepare(operands):
        # Use the full existing protocol outside the measured region.
        for x, mode in sorted(operands, key=lambda item: item[0].numel(), reverse=True):
            actual = a2a(x, mode, 'safe')
            expected = a2a(x, mode, 'nccl')
            assert torch.equal(actual, expected)
            for route in [r for r in routes if r in probe_helpers]:
                assert torch.equal(a2a(x, mode, route), expected)
        if 'prepared' in routes:
            assert helper._handle is not None
        torch.cuda.synchronize()

    try:
        workloads = [('small', 8192, 40), ('Wan', 75600, 40), ('H3', 37296, 56)]
        workloads = [w for w in workloads if w[0] in args.models]
        if args.h3_lengths:
            workloads = [(f'H3-{length}', length, 56) for length in args.h3_lengths]
        if args.section == 'collectives':
            for model, sequence, heads in workloads:
                x = torch.randn(args.packed_count, sequence // world, heads, 128, device='cuda', dtype=torch.bfloat16)
                y = torch.randn(1, sequence, heads // world, 128, device='cuda', dtype=torch.bfloat16)
                prepare([(x, 0), (y, 1)])
                signature, _ = helper._call_signature(x, 2, 1)
                measure(lambda: helper._agree_call(signature), dict(model=model, operation='agreement', route='safe'))
                for mode, operand in [(0, x), (1, y)]:
                    operation = 'scatter' if mode == 0 else 'gather'
                    for route in routes:
                        measure(lambda: a2a(operand, mode, route), dict(model=model, operation=operation, route=route))
                    dst = torch.empty_like(operand)
                    measure(lambda: dst.copy_(operand), dict(model=model, operation=operation + '_copy', route='copy'))
                for route in routes:
                    def pair():
                        a2a(x, 0, route)
                        a2a(y, 1, route)
                    measure(pair, dict(model=model, operation='pair_streamed', route=route),
                            iterations=max(3, args.iters // 4), repeats=20)
                    x.requires_grad_(True)
                    y.requires_grad_(True)
                    dx = torch.randn(args.packed_count, sequence, heads // world, 128, device='cuda', dtype=torch.bfloat16)
                    dy = torch.randn(1, sequence // world, heads, 128, device='cuda', dtype=torch.bfloat16)
                    def training_pair():
                        ox = a2a(x, 0, route)
                        oy = a2a(y, 1, route)
                        return torch.autograd.grad((ox, oy), (x, y), (dx, dy))
                    if args.h3_lengths:
                        expected_grads = torch.autograd.grad((a2a(x, 0, 'nccl'), a2a(y, 1, 'nccl')),
                                                             (x, y), (dx, dy))
                        actual_grads = training_pair()
                        assert all(torch.equal(a, b) for a, b in zip(actual_grads, expected_grads))
                        del expected_grads, actual_grads
                    measure(training_pair, dict(model=model, operation='pair_fwd_bwd', route=route),
                            iterations=max(3, args.iters // 4), repeats=5)
                    x.requires_grad_(False)
                    y.requires_grad_(False)
                    del dx, dy
                if 'prepared' not in routes:
                    del x, y, dst
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
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
                inner = heads * 128
                hidden = 5376 if args.h3_lengths else inner
                ffn = 14336 if args.h3_lengths else 4 * hidden
                x = torch.randn(1, sequence // world, hidden, device='cuda', dtype=torch.bfloat16,
                                requires_grad=True)
                weights = [torch.nn.Parameter(torch.randn(out_dim, in_dim, device='cuda', dtype=torch.bfloat16)
                                              * in_dim**-0.5)
                           for in_dim, out_dim in [(hidden, 3 * inner), (inner, hidden),
                                                   (hidden, 2 * ffn if args.h3_lengths else ffn), (ffn, hidden)]]
                dy = torch.randn_like(x)
                probe = torch.empty(3, sequence // world, heads, 128, device='cuda', dtype=torch.bfloat16)
                # Initialize before parity; otherwise uninitialized NaNs can fail equality.
                probe.zero_()
                prepare([(probe, 0)])
                del probe
                if args.sparse:
                    from fastvideo_kernel.block_sparse_attn import block_sparse_attn_sm100a_op
                    if args.sparse_probe_dir:
                        import sys
                        sys.path.insert(0, str(args.sparse_probe_dir))
                        import ulysses_sparse_probe
                        import fastvideo_kernel.block_sparse_attn_sm100a as sparse_module
                        sparse_module._FWD_BY_BLOCK[64] = ulysses_sparse_probe.block_sparse_sm100a_fwd
                    padded_sequence = ((sequence + 127) // 128) * 128
                    block_count = padded_sequence // 64
                    topk = max(1, block_count // 10)
                    ids = ((torch.arange(block_count, device='cuda')[:, None]
                            + torch.arange(topk, device='cuda')[None, :]) % block_count).to(torch.int32)
                    sparse_ids = ids[None, None].expand(1, heads // world, -1, -1).contiguous()
                    sparse_num = torch.full((1, heads // world, block_count), topk, device='cuda', dtype=torch.int32)
                    vbs = (sequence - torch.arange(block_count, device='cuda') * 64).clamp(0, 64).to(torch.int32)

                def block(route):
                    projected = F.linear(x, weights[0]).reshape(1, sequence // world, 3, heads, 128)
                    qkv = torch.cat(projected.unbind(dim=2), dim=0)
                    q, k, v = a2a(qkv, 0, route).chunk(3, dim=0)
                    if args.sparse:
                        padded = [F.pad(t.permute(0, 2, 1, 3), (0, 0, 0, padded_sequence - sequence)).contiguous()
                                  for t in (q, k, v)]
                        attended = block_sparse_attn_sm100a_op(*padded, sparse_ids, sparse_num, vbs)[0]
                        attended = attended[:, :, :sequence].permute(0, 2, 1, 3)
                    else:
                        attended = flash_attn_func_compilable(q, k, v, causal=False)
                    local = a2a(attended.contiguous(), 1, route).flatten(2)
                    residual = x + F.linear(local, weights[1])
                    activated = F.linear(residual, weights[2])
                    if args.h3_lengths:
                        value, gate = activated.chunk(2, dim=-1)
                        activated = value * F.silu(gate)
                    else:
                        activated = F.gelu(activated, approximate='tanh')
                    return residual + F.linear(activated, weights[3])

                # Compare outputs and all input/weight gradients under the identical compute recipe.
                reference = block('nccl')
                reference_grads = torch.autograd.grad(reference, [x, *weights], dy)
                for route in [r for r in routes if r != 'nccl']:
                    result = block(route)
                    gradients = torch.autograd.grad(result, [x, *weights], dy)
                    torch.testing.assert_close(result, reference, rtol=0, atol=0)
                    # FA backward uses atomic reductions: allow its bf16 summation variation.
                    for actual, expected in zip(gradients, reference_grads):
                        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
                    del result, gradients
                del reference, reference_grads
                for training in (False, True):
                    for round_index in (range(args.rounds) if args.paired else [None]):
                        ordered = routes if round_index is None else routes[round_index % len(routes):] + routes[:round_index % len(routes)]
                        for route in ordered:
                            def step():
                                with torch.set_grad_enabled(training):
                                    out = block(route)
                                    if training:
                                        torch.autograd.grad(out, [x, *weights], dy)
                            measure(step, dict(model=model, operation='block_train' if training else 'block_infer',
                                               route=route, flash_attention='sm100a64+triton_bwd' if args.sparse else fa_version,
                                               sparse=args.sparse, sequence=sequence, heads=heads,
                                               note='synthetic block; no FSDP, optimizer, norms, checkpointing, or sparse routing'),
                                    iterations=max(3, args.iters // 4), round_index=round_index)
                del x, weights, dy
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        for candidate in probe_helpers.values():
            candidate.close()
        cleanup_dist_env_and_memory()


if __name__ == '__main__':
    main()
