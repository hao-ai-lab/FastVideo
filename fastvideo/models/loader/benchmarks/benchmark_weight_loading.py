# SPDX-License-Identifier: Apache-2.0
"""Benchmark for model weight loading speed.

Measures the time to load model weights from safetensors files using
different strategies (CPU vs GPU, broadcast vs independent).

Usage (single GPU):
    python fastvideo/models/loader/benchmarks/benchmark_weight_loading.py \
        --model-path /path/to/model

Usage (multi-GPU, e.g. 4 GPUs):
    torchrun --nproc_per_node=4 \
        fastvideo/models/loader/benchmarks/benchmark_weight_loading.py \
        --model-path /path/to/model
"""

import argparse
import glob
import os
import time

import torch
import torch.distributed as dist

from fastvideo.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    get_node_group,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.logger import init_logger
from fastvideo.models.loader.weight_utils import (
    filter_duplicate_safetensors_files,
    safetensors_weights_iterator,
)

SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"

logger = init_logger(__name__)


def resolve_safetensors_files(model_path: str) -> list[str]:
    """Discover safetensors files in a model directory."""
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(
            f"No .safetensors files found in {model_path}")
    index_file = os.path.join(model_path, SAFE_WEIGHTS_INDEX_NAME)
    if os.path.exists(index_file):
        files = filter_duplicate_safetensors_files(
            files, model_path, SAFE_WEIGHTS_INDEX_NAME)
    return files


def benchmark_loading(
    files: list[str],
    to_cpu: bool,
    warmup: int,
    repeats: int,
    label: str,
) -> None:
    """Run the weight loading benchmark and print results."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    node_group = get_node_group()

    # Count total params and bytes on first pass
    total_params = 0
    total_bytes = 0
    for name, tensor in safetensors_weights_iterator(files, to_cpu=to_cpu):
        total_params += 1
        total_bytes += tensor.nelement() * tensor.element_size()

    if rank == 0:
        logger.info("[%s] %d tensors, %.2f GB total",
                    label, total_params, total_bytes / 1e9)

    # Warmup
    for _ in range(warmup):
        for _ in safetensors_weights_iterator(files, to_cpu=to_cpu):
            pass
        if dist.is_initialized():
            dist.barrier()

    # Timed runs
    times = []
    for i in range(repeats):
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        t0 = time.perf_counter()
        for _ in safetensors_weights_iterator(files, to_cpu=to_cpu):
            pass
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        if dist.is_initialized():
            dist.barrier()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    if rank == 0:
        avg = sum(times) / len(times)
        best = min(times)
        throughput = total_bytes / avg / 1e9
        logger.info(
            "[%s] avg %.3fs | best %.3fs | throughput %.2f GB/s "
            "(over %d runs, %d warmup, %d GPU(s), node_size=%d)",
            label, avg, best, throughput,
            repeats, warmup, node_group.world_size, node_group.world_size,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark model weight loading speed")
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to a local model directory containing .safetensors files")
    parser.add_argument(
        "--subfolder", type=str, default=None,
        help="Subfolder within model-path to load (e.g. 'transformer', "
             "'text_encoder'). If not set, loads from model-path directly.")
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Number of warmup iterations (default: 1)")
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Number of timed iterations (default: 3)")
    args = parser.parse_args()

    load_path = args.model_path
    if args.subfolder:
        load_path = os.path.join(load_path, args.subfolder)

    files = resolve_safetensors_files(load_path)

    # Set default env vars for single-GPU standalone mode (no torchrun)
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if rank == 0:
        logger.info("Model path: %s", load_path)
        logger.info("Safetensors files: %d", len(files))
        logger.info("World size: %d", world_size)

    # Benchmark: load to CPU (no broadcast, each rank reads independently)
    benchmark_loading(files, to_cpu=True, warmup=args.warmup,
                      repeats=args.repeats, label="to_cpu=True")

    # Benchmark: load to GPU (rank 0 reads, broadcasts to others)
    if torch.cuda.is_available():
        benchmark_loading(files, to_cpu=False, warmup=args.warmup,
                          repeats=args.repeats, label="to_cpu=False (broadcast)")

    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
