# SM120 BF16 tile-64 sparse attention

This opt-in forward targets RTX PRO 6000 Blackwell (compute capability 12.0),
BF16 inputs, head dimension 128 and 64-token tiles. It overlaps asynchronous
K/V shared-memory copies with QK, softmax and PV using Gluon `mma_v2`.
The VSA pooling, top-k selection, index ordering, compression gate and untiling
are unchanged. It needs neither checkpoint conversion nor training.

`FASTVIDEO_H3_VSA_SM120=1` selects the new kernel; the default is `0`.
Unsupported devices, layouts, tile sizes, gradients, DiT compilation and
conflicting fine-attention overrides fail explicitly. This is forward-only,
eager inference. The default backend and training path are unchanged.

## Environment and installation

Use the FastVideo checkout containing this change on Ubuntu, with an SM120 GPU,
PyTorch 2.12/CUDA 13 and Triton 3.7 (the original development environment).
This change adds a Python/Gluon JIT kernel, not a new compiled CUDA extension.
With PyTorch/CUDA and Triton already installed, run from `/root/code/FastVideo`:

```bash
python -m pip install -e .
export PYTHONPATH="$PWD/fastvideo-kernel/python${PYTHONPATH:+:$PYTHONPATH}"
```

This recipe disables FA4 and optional H3 fusions, so the `fasth3` extra is not
required. In particular, plain pip does not use the repository's uv-only FA4
source pin, and an unversioned FA4 dependency may fail to resolve when the
package index only offers prereleases. Do not install that extra for this test.
The base project dependencies already include pytest.

The source path is necessary when an installed fastvideo-kernel wheel does not
yet contain `block_sparse_attn_sm120.py`. Both benchmark modes use this same
source tree.

## Numerical tests and microbenchmark

```bash
pytest fastvideo-kernel/tests/test_vsa_sm120.py -v
python fastvideo-kernel/benchmarks/bench_vsa_sm120.py
```

The tests compare directly against the current native Triton forward and an
independent FP32 dense reference. They cover batch > 1, 28/56 heads, unequal
Q/K lengths, partial tiles, empty rows, zero-valid tiles, padding, compact
indices, int32/int64 metadata, repeatability, input preservation, contract
errors, and H3 dispatch with/without the learned compression gate. GPU tests
skip on non-SM120 hardware; a skip is not a validation result. The existing
kernel CI lane discovers this file via `fastvideo-kernel/tests/`.

The benchmark warms both kernels, validates output and log2-LSE, alternates
measurement order and reports CUDA-event samples, medians and extra allocated
memory. It excludes mask/index construction and layout conversion; synthetic
indices follow H3's dense prefix-query and exempt prefix-key policy. Run
additional shapes rather than treating one synthetic point as universal:

```bash
python fastvideo-kernel/benchmarks/bench_vsa_sm120.py --heads 28
python fastvideo-kernel/benchmarks/bench_vsa_sm120.py --tiles 512 --prefix-tiles 8 --sparsity 0.5
```

For trusted real-input captures, `--capture /path/to/layer.pt` accepts a tensor
dictionary containing contiguous BHSD `q`, `k`, `v`, boolean `mask` and KV valid
sizes `vbs`. The existing tile-first H3 capture hook writes this format. Capture
replay tests that input only; enabling tile-first is a separate model change
and must not be mixed into the paired end-to-end comparison below.

Microbenchmark JSON and progress logs default to
`/root/code/outputs/profiles/YYYYMMDD_HHMMSS/result.{json,log}` when the checkout
is `/root/code/FastVideo`. Use `--output /path/to/result.json` to override.

## Single-GPU model comparison

Use a complete BF16 checkpoint at your local
`/path/to/FastH3-8-Step-V2`. The scripts use the official online AdaLN cache
equally in both modes. Set `MODEL_PATH` to the actual
checkpoint directory before running; the script's default is a placeholder.

```bash
export MODEL_PATH=/path/to/FastH3-8-Step-V2
bash scripts/benchmarks/minimax_h3_sm120/run_1gpu.sh native
bash scripts/benchmarks/minimax_h3_sm120/run_1gpu.sh sm120
```

Each command runs one excluded warmup followed by three seed-42 generations,
1344x768, requested 360 frames, and the checkpoint's trained eight-forward
schedule. Only the sparse forward differs. DiT/region/VAE compilation, FP4,
tile-first and other optional H3 fusions are disabled for attribution.
The text-only Prompt Encoder uses the existing layerwise CPU offload path
(`FASTVIDEO_H3_ENCODER_LAYERWISE=1`), retaining CPU token embeddings and
streaming language layers to the GPU. VAEs are CPU-offloaded. Sequential
placement loads the DiT after the first text encode; on later requests,
`FASTVIDEO_H3_PARK_MODULES=vae,audio_vae` keeps the DiT on the GPU instead of
allocating another host copy before encoding. The encoder remains available
across requests. Both benchmark modes use this same placement, without
quantization. Host RAM must still accommodate the offloaded weights.
Lazy disk reload is disabled. Warmup and measured videos have separate names.
Both fine-attention routes release temporary contiguous Q/K/V copies before
allocating the BSHD output, reducing overlap between full-sequence buffers.
This changes tensor lifetimes only, not attention arithmetic or selection.
Additional example arguments, such as `--prompt`, `--seed`, or `--repeats`, may
be appended; use identical overrides for both modes.

Videos: `/root/code/outputs/fasth3_sm120/{native,sm120}/`.
Logs: `/root/code/outputs/profiles/YYYYMMDD_HHMMSS/{native,sm120}.log`.
The existing example prints individual and median DiT and E2E wall times.
E2E excludes initial generator construction, includes per-request offload,
decoding and output handling, and excludes the warmup from the summary.
Do not subtract transfers from E2E or compare against historical single-run
timings from a different checkout. Repeat in reversed order to check drift.

## Validation and reporting

**Numerical tolerance is not a promise of video equivalence.** The MMA layout
changes reduction order relative to native Triton. Small kernel differences can
accumulate over denoising steps. Kernel tests and microbenchmarks alone do not
establish model-level quality; validate generated videos against an unchanged
baseline with the same checkpoint, prompt, seed and execution settings.

For a PR, attach GPU/runtime versions, source revision and diff, numerical test
output, microbenchmark JSON, paired DiT/E2E measurements, and five fixed-prompt,
fixed-seed video comparisons with MS-SSIM. Retain the raw videos and scores;
do not lower a quality threshold or regenerate the baseline from candidate
code to make a failing comparison pass. Report single-GPU scope and any
quality regression explicitly. Testing 28 heads on one GPU is not distributed
validation. Keep the switch off by default and request maintainer review of
the numerical/quality tradeoff before proposing a default change.
