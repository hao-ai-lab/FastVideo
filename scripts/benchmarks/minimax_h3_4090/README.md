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
modulation before the FFN. The tile-first/FFN-16384 profiling run subsequently completed all three
768p clips without OOM: diagnostic median 349.76 s, denoise stage 244.07 s,
video decode stage 65.94 s, peak GPU allocated 17.55 GiB. A run without
profiling is still required for a release speed claim.
The profiling run used the following command. For SSH on macOS,
`-o UseKeychain=yes` retrieves the stored passphrase when the agent has no
loaded identities. Profiling/capture timings are
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
route. Initial real-QKV tests found approximately 1.9× fine-kernel speedup
but 4.3–13.4% relative L2 error and a strict partial-tile elementwise test
failure. Do not select it for shipping. The microbenchmark now includes
BF16 QK/PV ablations to isolate that error. Offline compilation with Triton 3.8.0 for sm89 passed all four entry points
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

## Cached-component 480p result

At `4d9846573`, retain components between requests (omit `--lazy`), and set
`FASTVIDEO_H3_VSA_TILE_FIRST=1` and `FASTVIDEO_H3_FFN_CHUNK_TOKENS=16384`.
Keep all other baseline settings, including the eager light VAE and original
BF16 attention kernel. After one warmup the two timed requests took 115.87 s
and 115.03 s, median **115.45 s** (29.3% less wall time than the lazy baseline).
Conditioning/denoise/video-decode stage medians were 10.85/72.62/24.98 s.
Peak GPU allocation was 19.02 GiB, host anon 43.96 GiB, total cgroup 92.72 GiB
including file cache. This recipe requires more host RAM than the 32 GB target;
its minimum RAM has not been tested under a smaller host limit.

Decoded raw video and PCM audio SHA256 hashes match the baseline exactly for
both ceramics and harbor at seed 20260929. This establishes output identity
for these two prompts; it does not establish the checkpoint's BF16-reference
quality on other prompts. Raw results, clips and hash evidence are saved in
`output/fasth3-4090-20261003/` beside the workspace.

## sm89 kernel precision choices

`FASTVIDEO_H3_VSA_SM89_KERNEL=bf16` opts into the new entirely BF16 tile-64
fine kernel. `int8` uses per-token INT8 QK with BF16 PV. `original` is the
unchanged default. Resolution happens when the backend is constructed;
unsupported devices, grad and compile requests retain the original route.
Both preserve the original tile selection, partial key masks and gated
compression. The rejected FP8-PV experiment is only exposed in the diagnostic
microbenchmark, never the pipeline route.

Two-head real-QKV captures at 1344×768, layers 0/20/41, measured:

| QK / PV | Fine-kernel speedup including input quantization | Relative L2 vs original BF16 |
| --- | --- | --- |
| BF16 / BF16 | 1.23× | 0.005–0.008% |
| INT8 / BF16 | 1.59× | 0.58–0.62% |
| INT8 / FP8 (rejected) | 1.91× | 4.3–13.4% |

These are fine-kernel microbenchmarks, not end-to-end clip speedups. Same-seed
clip checks are required for the INT8 route. All 44 targeted CUDA/CPU checks
passed for native/tile-first routing, partial tiles, learned compression,
shared FP8 projections and sequential component restoration.

At `9a8465ac4`, CPU-offloaded VAEs also remain on the host during denoising;
the encode/decode stages move them on demand. This frees room for resident
DiT blocks without changing any model arithmetic.

## Six resident blocks and encoder priorities

At `6d3c4cda5`, the opt-in BF16 fine kernel with six resident DiT blocks,
cached components and the settings below measured **111.27 s** median for
832×480, 243 frames. Timed requests were 111.61/110.92 s after one warmup.
Conditioning, denoise and video decode medians were 11.20/67.56/25.49 s.
Peak GPU allocation was 21.63 GiB, host anonymous memory 42.10 GiB and
total cgroup usage 90.94 GiB, including file cache.

```bash
FASTVIDEO_SOURCE_COMMIT=6d3c4cda5 \
FASTVIDEO_H3_PARK_MODULES=vae,audio_vae \
FASTVIDEO_H3_FFN_CHUNK_TOKENS=16384 \
FASTVIDEO_H3_VSA_TILE_FIRST=1 FASTVIDEO_H3_VSA_SM89_KERNEL=bf16 \
FASTVIDEO_LAYERWISE_RESIDENT_BLOCKS=6 MAX_JOBS=4 \
python -P /workspace/fastvideo/scripts/benchmarks/minimax_h3_4090/bench_pod.py \
  sm89-bf16-480p-resident6 /workspace/vol/pruned_fp8_300 fp8 \
  --offload-buffers --no-vae-compile \
  --height 480 --width 832 --frames 243 --timed 2
```

The BF16 kernel changes floating-point reductions: the ceramics clip is
visually coherent in the sampled contact sheet but is not identical to the
original-kernel clip (decoded-video SSIM 0.597653). This is a speed candidate,
not proof of quality equivalence. The original kernel remains the default.

The same cached/six-resident-block BF16 recipe at `26390848b`, with name
`sm89-bf16-480p-5s-resident6` and `--frames 124`, measured **65.27 s** median.
The legal frame count represents 5.167 s at 24 fps. Timed requests were
65.08/65.46 s after a 110.34 s warmup. Conditioning/denoise/video-decode
medians were 11.74/34.22/13.44 s. Peak GPU allocation was 21.62 GiB,
host anon 41.14 GiB and total cgroup 88.45 GiB. These generation wall times
include decode/export and exclude initial generator construction. No
profiling or QKV capture was enabled.

At `26390848b`, per-key-tile V scaling reduced experimental INT8-QK/FP8-PV
real-tensor error to 0.84–1.54%, with 1.77× fine-kernel speedup. Dynamic
per-query, per-key-block P scaling also preserves contributions that would
underflow with the fixed P scale; it measured 0.80–1.52% error and 1.71×
speedup. All 30 focused CUDA kernel/routing tests passed. These FP8-PV routes
remain microbenchmark-only and need clip validation.

The current text encoder is the trimmed 50-layer Qwen3-VL with serialized
NVFP4 weights, dequantized to BF16 per linear on sm89. The existing serialized
blockwise FP8 encoder requires sm100+ and FlashInfer's Blackwell GEMM; it
cannot run on the 4090 as written. An Ada FP8 implementation would also need
encoder streaming because its weights are larger. Reading the current
checkpoint tensor shapes gives 15.33 GiB total encoder weights, including
11.35 GiB packed values and 1.42 GiB block scales. Replacing those packed
values with FP8 while retaining the other tensors projects about 25.3 GiB
before activations (the FP8 block-scale overhead is small). This is a storage
estimate, not a measured FP8 encoder. First try fused NVFP4
dequantization and avoid per-linear GPU scalar synchronization; then compare
a native sm89 FP8 encoder at equal prompts. The later streamed/fused conditioning stage is about 0.55 s, so a new
encoder export must be measured against that implementation.

Remaining speed experiments include VAE compilation and tile-batch tuning,
INT8 decoder epilogue fusion, direct strided fine-attention reads, and fused
norm/activation quantization. See the completed smaller-VRAM results below.
The cached recipe's host peak does not establish a 32 GB system-RAM minimum.


## Streamed encoder and smaller VRAM caps

At `753e560f6`, `FASTVIDEO_H3_ENCODER_LAYERWISE=1` streams the language
layers separately from DiT residency, retaining token embeddings and unused
vision modules on the CPU. This route currently supports text-only T2VA;
visual references fail explicitly. The pipeline preserves the streamed
placement. Twenty-eight offload/encoder/stage tests passed, including exact
repeated BF16 and NVFP4 parity.

At `78540b635`, `FASTVIDEO_H3_ENCODER_FUSED_DEQUANT=1` expands packed NVFP4
weights with one Triton pass. Fifteen strict tests passed across FP32, FP16,
BF16, swizzled scales and repeated encoder forwards. On three large stress
matrices the expansion was 25.3–25.9× faster and used 8× less temporary GPU
memory than Torch expansion. This is a dequantization microbenchmark; the
whole conditioning stage measured 0.55–0.62 seconds in the clip runs below.
The current encoder remains NVFP4 storage with BF16 GEMMs on Ada.

All rows use 832×480, 243 frames, eight DMD forwards, sparsity 0.8, tile 64,
cached components, INT8 QK/BF16 PV and the eager light H3 VAE. Each median
has one warmup and two timed requests. Source is `78540b635` except the
16 GiB row (`753e560f6`, before fused dequantization).

| 4090 configuration | Median e2e | Timed requests | Denoise | Video decode | Peak GPU allocated | Peak host anon |
| --- | --- | --- | --- | --- | --- | --- |
| 16 GiB cap, 6 resident | 104.03 s | 100.10 / 107.96 s | 72.42 s | 25.54 s | 11.28 GiB | 39.86 GiB |
| 12 GiB cap, 0 resident | 107.27 s | 111.02 / 103.52 s | 77.38 s | 25.48 s | 8.68 GiB | 42.35 GiB |
| Uncapped, 30 resident | 98.97 s | 98.88 / 99.05 s | 69.12 s | 25.47 s | 21.69 GiB | 29.44 GiB |

Set `FASTVIDEO_CUDA_MEMORY_CAP_GIB=12` for the 12 GiB recipe; unset it for
the full card. Set resident blocks to the table value. Both fused rows use:

```bash
FASTVIDEO_SOURCE_COMMIT=78540b635 \
FASTVIDEO_H3_PARK_MODULES=vae,audio_vae \
FASTVIDEO_H3_ENCODER_LAYERWISE=1 FASTVIDEO_H3_ENCODER_FUSED_DEQUANT=1 \
FASTVIDEO_H3_VSA_TILE_FIRST=1 FASTVIDEO_H3_VSA_SM89_KERNEL=int8 \
FASTVIDEO_LAYERWISE_RESIDENT_BLOCKS=30 FASTVIDEO_H3_FFN_CHUNK_TOKENS=16384 \
FASTVIDEO_H3_VAE_TILE_BATCH=28 MAX_JOBS=4 \
python -P /workspace/fastvideo/scripts/benchmarks/minimax_h3_4090/bench_pod.py \
  sm89-int8-480p-resident30-fused /workspace/vol/pruned_fp8_300 fp8 \
  --offload-buffers --no-vae-compile --height 480 --width 832 --frames 243 --timed 2
```

The 12 GiB and 30-resident ceramics clips have identical decoded-video and
PCM audio hashes to the six-resident INT8 clip. These placement and dequant
changes preserve the candidate's output; quality equivalence of INT8
attention to the original BF16 checkpoint still requires motion/speech review.
Allocator caps emulate available VRAM on a 4090, not another card's speed.
Host peaks include all pod processes; actual 32 GB host-limit support has
not been established. The 30-resident warmup reached 30.26 GiB anonymous
memory and the timed runs reached 77.50 GiB total cgroup usage including cache.

The 8 GiB cap at `fcdba37fc` completed its warmup but OOMed on the timed
harbor prompt in fine attention. Do not report it as supported. A 34-resident
experiment completed denoising but OOMed during VAE INT8 epilogue allocation.
Both failures motivate subsequent memory work rather than speed claims.


## Consumer kernel memory work on the release core

The consumer branch is rebased onto release core `a97d23f09` (fork PR #45).
Historical measured commits remain reachable through tag
`h3-consumer-fp8-measured-20261003`; rebase changes their branch commit IDs.

`7a0d7d33b` lets the INT8-QK/BF16-PV fine kernel read BSHD-backed views
without retaining three full BF16 layout copies. Q/K quantization writes
contiguous INT8 arrays and the output stays BHSD; the K-mean reduction keeps
the reference's contiguous reduction order. Four CUDA tests require exact
output equality for partial tiles, empty selections, multiple batches and
partner padding, together with a lower peak allocation.

`3c0668f6c` adds the opt-in eager
`FASTVIDEO_H3_VAE_INT8_FUSED_DEQUANT=1`. One Triton pass applies the INT32
GEMM's row scale, output-channel scale and optional bias, then casts the
result. FP32 operations retain separate rounding steps (FP fusion disabled).
Strict tests cover zero rows, small input batches, FP32/FP16/BF16, bias,
large INT32 accumulators and tiny scales. Compiled and grad paths retain
the reference implementation. Combine it with shared QKV and transpose
views using `FASTVIDEO_H3_VAE_INT8_SHARED_QKV=1` and
`FASTVIDEO_H3_VAE_INT8_TRANSPOSE_VIEW=1`.

After the core rebase, 120 focused tests passed (one unrelated GPU cudagraph
check excluded) and pre-commit passed. `87b22a5d8` adapts capture and tests
to packed-segment metadata while retaining the core's calibrated NVFP4
activation-scale guard. `fb92af176` adds optional NVML total-device-memory
samples every 100 ms, alongside host samples. Summaries distinguish sampled
total GPU usage from PyTorch's allocated peak. Allocator caps omit driver
and external CUDA memory, so the 8 GiB total-budget experiment uses a
7.25 GiB allocator cap and must also satisfy the observed NVML budget.
That trial completed its warmup and first timed request, but sampled total
GPU usage reached about 8.28 GiB. It therefore does **not** meet a strict
8 GiB device target. A tighter allocator cap still needs validation.
These are 4090 simulations; real lower-VRAM and 30-series performance still
needs those devices.


## Warmed release-core clip results

At `fb92af176`, after one warmup and two timed requests:

| Clip | Median e2e | Timed requests | Conditioning | Denoise | Video decode | Peak allocated | Sampled total GPU | Peak host anon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 832×480, 124 frames / 5.167 s | **41.75 s** | 42.16 / 41.33 s | 0.55 s | 32.69 s | 6.63 s | 22.01 GiB | 23.983 GiB | 24.78 GiB |
| 832×480, 243 frames / 10.125 s | **79.67 s** | 79.64 / 79.70 s | 0.55 s | 63.51 s | 13.23 s | 20.29 GiB | 23.985 GiB | 27.02 GiB |

Warmups took 84.40 / 125.16 s. These wall times include audio, frame export
and MP4 saving, and exclude initial generator construction. The 5 s recipe
keeps 34 DiT blocks resident; the 10 s recipe keeps 30. Both use the same
checkpoint revision and eight DMD forwards as the earlier rows. No profiling,
QKV capture, alternate decoder or sparsity increase is enabled.

```bash
source /workspace/env.sh
source /workspace/venv/bin/activate
cd /workspace
export PYTHONPATH="/workspace/fastvideo-core:${PYTHONPATH:-}"
export FASTVIDEO_SOURCE_COMMIT=fb92af176
export FASTVIDEO_H3_PARK_MODULES=vae,audio_vae
export FASTVIDEO_H3_ENCODER_LAYERWISE=1 FASTVIDEO_H3_ENCODER_FUSED_DEQUANT=1
export FASTVIDEO_H3_VSA_TILE_FIRST=1 FASTVIDEO_H3_VSA_SM89_KERNEL=int8
export FASTVIDEO_LAYERWISE_RESIDENT_BLOCKS=34 FASTVIDEO_H3_FFN_CHUNK_TOKENS=16384
export FASTVIDEO_H3_VAE_TILE_BATCH=28
export FASTVIDEO_H3_VAE_INT8_SHARED_QKV=1 FASTVIDEO_H3_VAE_INT8_TRANSPOSE_VIEW=1
export FASTVIDEO_H3_VAE_INT8_FUSED_DEQUANT=1 MAX_JOBS=4
unset FASTVIDEO_CUDA_MEMORY_CAP_GIB
python -P /workspace/fastvideo-core/scripts/benchmarks/minimax_h3_4090/bench_pod.py \
  sm89-int8-fast2-480p-5s /workspace/vol/pruned_fp8_300 fp8 \
  --offload-buffers --no-vae-compile --height 480 --width 832 --frames 124 --timed 2
```

For 10 s, set resident blocks to 30, name to `sm89-int8-fast2-480p-10s`,
and frames to 243. Both decoded video and PCM audio hash-identically to the
previous INT8 candidate for ceramics and harbor at 10 s. This validates these
memory/decode changes on those prompts, while the INT8 attention candidate
still differs from the original BF16 attention clips and needs full quality
review. The sampled 5 s contact sheet is coherent. Raw clips, hashes and
results live in `output/fasth3-4090-20261003/` beside the worktree.

The earlier unprofiled 1344×768, 243-frame run at `fcdba37fc` completed at
279.94 s median (289.06 / 270.82 s, 306.80 s warmup), with 12 resident blocks
and shared VAE QKV/transpose views, before direct-layout attention and fused
VAE epilogues. Its stage medians were 0.55 s conditioning, 227.21 s denoise,
44.82 s video decode and 0.48 s audio. Peak allocation was 21.19 GiB and host
anonymous memory 39.81 GiB. The updated 768p run was queued after the
7.25 GiB allocator-cap trial. SSH became unreachable before the final
results could be collected; updated 768p speed and strict 8 GiB support
remain unverified.

Track B is staged in [draft PR #46](https://github.com/aryan5v/FastVideo/pull/46),
stacked on the shared release core in #45. Historical timing sources are
preserved by the `h3-consumer-fp8-measured-20261003` tag.
