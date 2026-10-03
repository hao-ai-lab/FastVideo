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

Completed baseline at source commit `ce877b200`, checkpoint revision
`f2ef54f9ff2091762ab8689b6514dcab5bc1d383`:

| Configuration | Median e2e | Denoise stage | Video decode stage | Peak GPU allocated | Peak host anon | Peak total cgroup |
| --- | --- | --- | --- | --- | --- | --- |
| FP8, layerwise, lazy, eager H3 VAE, 832×480, 243 frames | 163.34 s | 91.72 s | 34.54 s | 17.47 GiB | 28.91 GiB | 76.95 GiB |

Two timed requests took 163.67 s and 163.00 s after one warmup.
Stage times are medians and include deferred loading. Memory columns are
maxima across the timed requests. Total cgroup usage includes file cache.
The GPU peak occurs during conditioning. The saved clip contains 243 frames
at 24 fps (10.125 seconds) and an AAC audio track. A contact-sheet inspection
confirms a coherent pottery scene; speech and same-seed reference parity
still need review before claiming quality equivalence.

Summarize a completed run while excluding warmup:

```bash
python -P /workspace/fastvideo/scripts/benchmarks/minimax_h3_4090/summarize.py \
  /workspace/outputs/baseline-480p/results.json /workspace/baseline-480p.log
```

After a baseline works, measure FFN chunk sizes 16,384 and 8,192, then
increase resident DiT blocks within the measured GPU budget. Kernel or
decoder changes also need same-seed visual and auditory comparison.

## Tile-first attention and full-resolution profiling

Commit `c5d9f8132` shares compatible FP8 Q/K/V activation quantization and
releases dead block activations before residual modulation. The opt-in
`FASTVIDEO_H3_VSA_TILE_FIRST=1` scatters the attention input before projection,
then uses the existing BF16 VSA kernel. It retains tile-64 selection, partial
key validity and the learned compression branch. It supports eager,
single-rank inference; grad, compile, and multi-rank requests use the generic
path. Ten CUDA tests passed on the 4090, including mixed partial tiles,
active/zero gates, fused/unfused RoPE and FP8/nonquantized projections.

The original 1344×768 baseline failed with a GPU OOM in post-attention
modulation before the FFN. No successful 768p timing is established yet.
A profiling run was launched with the following command; retrieve its
results when SSH access is restored. The server recognizes the public key; the
local passphrase-protected private key needs its agent/keychain identity loaded. Profiling/capture timings are
for diagnosis and must not be used as the final speed claim.

```bash
FASTVIDEO_SOURCE_COMMIT=c5d9f8132 \
FASTVIDEO_H3_PARK_MODULES=vae,audio_vae \
FASTVIDEO_H3_FFN_CHUNK_TOKENS=16384 \
FASTVIDEO_H3_VSA_TILE_FIRST=1 \
FASTVIDEO_H3_CAPTURE_QKV=/workspace/qkv-768p MAX_JOBS=4 \
python -P /workspace/fastvideo/scripts/benchmarks/minimax_h3_4090/bench_pod.py \
  tile-first-768p-profile /workspace/vol/pruned_fp8_300 fp8 \
  --offload-buffers --lazy --no-vae-compile \
  --height 768 --width 1344 --frames 243 --timed 2 --profile
```

`FASTVIDEO_H3_CAPTURE_QKV` saves the first inputs from layers 0, 20 and 41,
two heads each, with full real sequences, masks, valid tile sizes and packed
row indices. Disable both capture and profiling for final clip timings.

The separate `minimax_h3_sparse_int8.py` prototype uses INT8 QK and FP8 PV,
FP32 accumulators and the original 64-token mask. It has no automatic pipeline
route. Offline compilation with Triton 3.8.0 for sm89 passed all four entry points
and emitted native INT8 and FP8 MMA instructions (20,480 bytes of shared
memory for the attention kernel). This is compilation evidence only. It must
pass CUDA tests and real-QKV/clip checks before integration.
Run its microbenchmark on an idle GPU:

```bash
python -P /workspace/fastvideo/scripts/benchmarks/minimax_h3_4090/bench_sparse_qkv.py \
  /workspace/qkv-768p --output /workspace/sparse-qkv-results.json
```

SpargeAttn at `ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a` built on the pod
with CUDA 12.8, `TORCH_CUDA_ARCH_LIST=8.9`, and `MAX_JOBS=4`. The upstream
`-Xcompiler -include,cassert` workaround was removed from `setup.py` to
avoid GCC 13 duplicate standard-library definitions. It is not selected by
the pipeline: its public 128-query/64-key adapter also needs correct masking
of partial H3 tiles before a meaningful parity comparison.
