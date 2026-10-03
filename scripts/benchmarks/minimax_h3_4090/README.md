# FastH3 on a single RTX 4090

Use the private `FastVideo/FastH3-Pruned-8Step-FP8-ckpt300` checkpoint.
Keep its `fastvideo_inference.json`: nine sigma-grid points produce eight
DMD forwards. Preserve VSA sparsity 0.8 and tile size 64.

## Setup and validation

The October 3, 2026 pod has one RTX 4090 (24,564 MiB), driver 580.126.20,
a 99,999,997,952-byte host cgroup limit, and 150 GB disk. Its runtime is
PyTorch 2.12.0+cu126, CUDA toolkit 12.6, and FlashInfer 0.7.1rc2.

Exact-size host arenas replace the pinned allocator for layerwise blocks
and H3 module swaps. Each arena packs typed views at 256-byte offsets into
dedicated CUDA-registered pages. Live views retain their registration owner.
Mutation and hook detachment unregister old arenas. Registration failures
fall back to the ordinary pinned allocator with a warning.

Validation on this pod: all 12 tests passed with the command below.
The tests cover repeated offloaded forwards, BF16, mixed-dtype exact copies,
owner lifetime, registration fallback, large-buffer mutation, detachment
after prefetch, and persistent H3 swaps with changing buffers.

```bash
source /workspace/env.sh
source /workspace/venv/bin/activate
cd /workspace/fastvideo
python -P -m pytest fastvideo/tests/hooks/test_pinned_memory.py \
  fastvideo/tests/hooks/test_layerwise_offload.py -q
```

The supplied `handoff_4090/kernel_microbench/pinned_memory.py` measured
5.06 GiB extra cgroup usage for 2.87 GiB through `pin_memory()`, versus
2.87 GiB using direct host registration. Pinned H2D measured 25.9 GB/s;
pageable H2D measured 10.2 GB/s. These are microbenchmarks, not clip timings.

The supplied `handoff_4090/kernel_microbench/t_fp8.py` measured:

| K → N, M = 38,976 | Fused quantization | FP8 GEMM + scale epilogue |
| --- | --- | --- |
| 5376 → 5376 | 0.68 ms | 11.27 ms |
| 5376 → 28672 | 0.68 ms | 40.39 ms |
| 14336 → 5376 | 2.10 ms | 27.96 ms |

Commands on the pod, run before clip benchmarks:

```bash
cd /workspace
python -P kernel_microbench/pinned_memory.py
python -P kernel_microbench/t_fp8.py
```

## End-to-end baseline

Run one warmup and at least two timed requests. The benchmark saves clips,
the exact Python command, runtime environment, sampling geometry, config,
source commit, each wall time, and the median to `results.json`. Stage logs
include GPU allocation peaks and conditioning, denoise, and decode timings.
Per-run host samples report both total cgroup memory and anonymous memory;
total usage includes checkpoint file cache. Both are sampled every 100 ms
and include all processes in the pod's cgroup.

```bash
cd /workspace
export FASTVIDEO_SOURCE_COMMIT=ce877b200
export FASTVIDEO_H3_PARK_MODULES=vae,audio_vae
export MAX_JOBS=4
python -P /workspace/fastvideo/scripts/benchmarks/minimax_h3_4090/bench_pod.py \
  baseline-480p /workspace/vol/pruned_fp8_300 fp8 \
  --offload-buffers --lazy --no-vae-compile \
  --height 480 --width 832 --frames 243 --timed 2
```

For the full-resolution baseline, change the name to `baseline-768p`,
height to 768, and width to 1344; retain 243 frames. Layerwise offload,
lazy component loading, and eager VAE decode are the starting configuration.
Do not compare these numbers with a different frame count or decoder.

After a baseline works, measure FFN chunk sizes 16,384 and 8,192, then
increase resident DiT blocks within the measured GPU budget. Kernel or
decoder changes also need same-seed visual and auditory comparison.
