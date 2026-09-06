# SPDX-License-Identifier: Apache-2.0
"""Build and measure an isolated launch-configuration variant of the real kernel.

Only the CTA/thread counts and optional copy-out suppression differ. Suppressing
copy-out measures the movement kernel, not a usable tensor-returning operation.
The installed kernel and production dispatch are never replaced.
"""

import argparse
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist


def build(directory):
    from torch.utils.cpp_extension import load
    repo = Path(__file__).resolve().parents[3]
    header = (repo / 'fastvideo-kernel/include/comm/ulysses_all_to_all.cuh').read_text()
    source = (repo / 'fastvideo-kernel/csrc/comm/ulysses_all_to_all.cu').read_text()
    header = header.replace('constexpr int kMaxBlocks = 36;', 'constexpr int kMaxBlocks = 144;')
    source = source.replace('int64_t H, int64_t D, int64_t mode) {',
                            'int64_t H, int64_t D, int64_t mode, int probe_blocks, int probe_threads, bool copy_out) {')
    source = source.replace('std::min<int64_t>(fi::kMaxBlocks, num_rows)', 'std::min<int64_t>(probe_blocks, num_rows)')
    source = source.replace('const int threads = fi::kUlyssesThreads;',
                            'TORCH_CHECK(probe_blocks > 0 && probe_blocks <= fi::kMaxBlocks, "bad block count");\n'
                            '  TORCH_CHECK(probe_threads == 128 || probe_threads == 256 || probe_threads == 512, '
                            '"bad thread count");\n  const int threads = probe_threads;')
    source = source.replace('// Copy this rank\'s completed result out of the window.',
                            'if (!copy_out) return;\n  // Copy this rank\'s completed result out of the window.')
    source += '\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { register_ulysses_a2a(m); }\n'
    (directory / 'comm').mkdir(parents=True, exist_ok=True)
    (directory / 'comm/ulysses_all_to_all.cuh').write_text(header)
    (directory / 'probe.cu').write_text(source)
    nccl = Path(next(iter(importlib.util.find_spec('nvidia.nccl').submodule_search_locations)))
    os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'
    os.environ['MAX_JOBS'] = '2'
    return load(name='ulysses_launch_probe', sources=[str(directory / 'probe.cu')],
                extra_include_paths=[str(directory), str(nccl / 'include')],
                extra_cuda_cflags=['-O3', '-std=c++17'],
                extra_ldflags=[str(nccl / 'lib/libnccl.so.2')],
                build_directory=str(directory), verbose=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--sparse-build-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.build_dir.mkdir(parents=True, exist_ok=True)
    if args.sparse_build_only:
        from torch.utils.cpp_extension import load
        repo = Path(__file__).resolve().parents[3]
        source_dir = repo / 'fastvideo-kernel/csrc/attention'
        source = (source_dir / 'block_sparse_sm100a.cu').read_text()
        source += ('\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { '
                   'm.def("block_sparse_sm100a_fwd", &block_sparse_sm100a_fwd); }\n')
        generated = args.build_dir / 'sparse.cu'
        generated.write_text(source)
        os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'
        os.environ['MAX_JOBS'] = '2'
        load(name='ulysses_sparse_probe', sources=[str(generated)],
             extra_include_paths=[str(source_dir)],
             extra_cuda_cflags=['-O3', '-std=c++17', '-DVSA_BHSD=true'],
             extra_ldflags=['-L/usr/local/cuda/lib64/stubs', '-lcuda'],
             build_directory=str(args.build_dir), verbose=True)
        return
    if args.build_only:
        build(args.build_dir)
        return
    import sys
    sys.path.insert(0, str(args.build_dir))
    import ulysses_launch_probe as probe
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
    from fastvideo.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase
    from fastvideo.distributed.parallel_state import get_sp_group
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.manual_seed(20260906 + rank)
    maybe_init_distributed_environment_and_model_parallel(1, world)
    comm = get_sp_group().device_communicator
    records = []
    try:
        for model, sequence, heads in [('small', 8192, 40), ('Wan', 75600, 40), ('H3', 37296, 56)]:
            handle = probe.allocate_ulysses_a2a(3 * sequence // world * heads * 128 * 2,
                                                rank, world, torch.cuda.current_device())
            probe.register_ulysses_a2a_window(handle, comm.ulysses_a2a._comm_ptr())
            probe.create_ulysses_a2a_dev_comm(handle)
            for mode in (0, 1):
                shape = (3, sequence // world, heads, 128) if mode == 0 else (1, sequence, heads // world, 128)
                x = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
                dims = (2, 1) if mode == 0 else (1, 2)
                expected = DeviceCommunicatorBase.all_to_all_4D(comm, x, *dims)
                out = torch.empty_like(expected)
                for blocks in (12, 18, 36, 72, 144):
                    for threads in (128, 256, 512):
                        def run(copy_out=True):
                            probe.ulysses_a2a(handle, x, out, shape[0], sequence // world,
                                              heads, 128, mode, blocks, threads, copy_out)
                        run()
                        torch.cuda.synchronize()
                        assert torch.equal(out, expected), (model, mode, blocks, threads)
                        for copy_out in (True, False):
                            for _ in range(5):
                                run(copy_out)
                            torch.cuda.synchronize()
                            samples = []
                            for _ in range(3):
                                dist.barrier(group=get_sp_group().cpu_group)
                                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                start.record()
                                for _ in range(20):
                                    run(copy_out)
                                end.record()
                                torch.cuda.synchronize()
                                samples.append(start.elapsed_time(end) * 1000 / 20)
                            tensor = torch.tensor(samples, device='cuda', dtype=torch.float64)
                            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
                            samples = tensor.cpu().tolist()
                            record = dict(model=model, mode=mode, blocks=blocks, threads=threads,
                                          copy_out=copy_out, p50_us=statistics.median(samples), samples_us=samples,
                                          bytes=x.numel() * x.element_size(), parity=True)
                            records.append(record)
                            if rank == 0:
                                print(json.dumps(record), flush=True)
                                args.output.write_text(json.dumps(records, indent=2) + '\n')
                del x, out, expected
            torch.cuda.synchronize()
            probe.dispose_ulysses_a2a(handle)
    finally:
        cleanup_dist_env_and_memory()


if __name__ == '__main__':
    main()
