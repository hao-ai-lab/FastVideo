# SPDX-License-Identifier: Apache-2.0
"""A/B benchmark: independent-read vs rank-0-broadcast weight loading.

Compares two strategies:
  - "before" (independent): every rank reads safetensors from disk to GPU
  - "after"  (broadcast):   rank 0 reads from disk, broadcasts to other ranks

Usage:
    # 1 GPU
    torchrun --nproc_per_node=1 fastvideo/models/loader/benchmarks/benchmark_weight_loading_comparison.py \
        --model-path /path/to/model --subfolder transformer

    # 2 GPUs
    torchrun --nproc_per_node=2 fastvideo/models/loader/benchmarks/benchmark_weight_loading_comparison.py \
        --model-path /path/to/model --subfolder transformer

    # 4 GPUs
    torchrun --nproc_per_node=4 fastvideo/models/loader/benchmarks/benchmark_weight_loading_comparison.py \
        --model-path /path/to/model --subfolder transformer
"""

import argparse
import glob
import os
import time

import torch
import torch.distributed as dist
from safetensors.torch import safe_open
from tqdm.auto import tqdm

from fastvideo.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    get_node_group,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.models.loader.weight_utils import (
    SAFETENSORS_TO_TORCH_DTYPE,
    filter_duplicate_safetensors_files,
)

SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"


def resolve_safetensors_files(model_path: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(
            f"No .safetensors files found in {model_path}")
    index_file = os.path.join(model_path, SAFE_WEIGHTS_INDEX_NAME)
    if os.path.exists(index_file):
        files = filter_duplicate_safetensors_files(
            files, model_path, SAFE_WEIGHTS_INDEX_NAME)
    return files


def load_independent(files: list[str], device: str):
    """Before-PR behavior: every rank reads every tensor from disk to GPU."""
    for st_file in files:
        with safe_open(st_file, framework="pt", device=device) as f:
            for name in f.keys():
                param = f.get_tensor(name)
                yield name, param


def load_broadcast(files: list[str], device: str, node_group,
                   async_op: bool = False):
    """After-PR behavior: rank 0 reads from disk, broadcasts to other ranks."""
    local_rank = node_group.local_rank
    handles = []
    for st_file in files:
        with safe_open(st_file, framework="pt", device=device) as f:
            for name in f.keys():
                if local_rank == 0:
                    param = f.get_tensor(name)
                else:
                    sl = f.get_slice(name)
                    shape = sl.get_shape()
                    dtype = SAFETENSORS_TO_TORCH_DTYPE[sl.get_dtype()]
                    param = torch.empty(shape, device=device, dtype=dtype)
                if node_group.world_size > 1:
                    group = node_group.device_group
                    if async_op:
                        handle = dist.broadcast(
                            param,
                            src=dist.get_global_rank(group, 0),
                            async_op=True,
                            group=group,
                        )
                        handles.append(handle)
                    else:
                        dist.broadcast(
                            param,
                            src=dist.get_global_rank(group, 0),
                            group=group,
                        )
                yield name, param
        if async_op:
            for handle in handles:
                handle.wait()
            handles.clear()


def run_benchmark(label, loader_fn, files, warmup, repeats):
    rank = dist.get_rank()

    # Count tensors/bytes on first pass
    total_params = 0
    total_bytes = 0
    for name, tensor in loader_fn(files):
        total_params += 1
        total_bytes += tensor.nelement() * tensor.element_size()
    dist.barrier()

    # Warmup
    for _ in range(warmup):
        for _ in loader_fn(files):
            pass
        dist.barrier()

    # Timed runs
    times = []
    for _ in range(repeats):
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in loader_fn(files):
            pass
        torch.cuda.synchronize()
        dist.barrier()
        times.append(time.perf_counter() - t0)

    if rank == 0:
        avg = sum(times) / len(times)
        best = min(times)
        throughput = total_bytes / avg / 1e9
        print(
            f"  {label:30s}  "
            f"avg {avg:.3f}s  best {best:.3f}s  "
            f"throughput {throughput:.2f} GB/s  "
            f"({total_params} tensors, {total_bytes/1e9:.2f} GB)"
        )
        return avg, best, throughput, total_bytes
    return None, None, None, None


def main():
    parser = argparse.ArgumentParser(
        description="A/B benchmark: independent vs broadcast weight loading")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--subfolder", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    load_path = args.model_path
    if args.subfolder:
        load_path = os.path.join(load_path, args.subfolder)

    files = resolve_safetensors_files(load_path)

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    node_group = get_node_group()
    local_rank = node_group.local_rank
    device = f"cuda:{local_rank}"

    if rank == 0:
        print(f"\n{'='*72}")
        print(f"Weight Loading Benchmark: {load_path}")
        print(f"GPUs: {world_size}  |  safetensors files: {len(files)}  "
              f"|  warmup: {args.warmup}  |  repeats: {args.repeats}")
        print(f"{'='*72}")

    # --- Before PR: every rank reads independently ---
    before_avg, before_best, before_tp, total_bytes = run_benchmark(
        "before (independent read)",
        lambda f: load_independent(f, device),
        files, args.warmup, args.repeats,
    )

    # Clear GPU memory between benchmarks
    torch.cuda.empty_cache()
    dist.barrier()

    # --- After PR: rank 0 reads, sync broadcasts ---
    after_sync_avg, after_sync_best, after_sync_tp, _ = run_benchmark(
        "after  (sync broadcast)",
        lambda f: load_broadcast(f, device, node_group, async_op=False),
        files, args.warmup, args.repeats,
    )

    # Clear GPU memory between benchmarks
    torch.cuda.empty_cache()
    dist.barrier()

    # --- After PR: rank 0 reads, async broadcasts ---
    after_async_avg, after_async_best, after_async_tp, _ = run_benchmark(
        "after  (async broadcast)",
        lambda f: load_broadcast(f, device, node_group, async_op=True),
        files, args.warmup, args.repeats,
    )

    if rank == 0:
        print(f"{'─'*72}")
        if before_avg and after_sync_avg:
            sync_speedup = before_avg / after_sync_avg
            print(f"  sync  speedup: {sync_speedup:.2f}x  "
                  f"(saved {before_avg - after_sync_avg:.3f}s)")
        if before_avg and after_async_avg:
            async_speedup = before_avg / after_async_avg
            print(f"  async speedup: {async_speedup:.2f}x  "
                  f"(saved {before_avg - after_async_avg:.3f}s)")
        print(f"{'='*72}\n")

    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
