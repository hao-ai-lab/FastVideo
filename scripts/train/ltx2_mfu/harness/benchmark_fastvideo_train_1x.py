#!/usr/bin/env python3
"""Run the shared timing harness and report single-GPU peak allocation."""

import runpy

import torch
import torch.distributed as dist


runpy.run_path("/mnt/benchmark_fastvideo_train.py", run_name="__main__")
if dist.get_world_size() != 1:
    raise RuntimeError(f"expected one rank, got {dist.get_world_size()}")
print(f"BF16_PEAK_GIB {torch.cuda.max_memory_allocated() / 2**30:.6f}", flush=True)
