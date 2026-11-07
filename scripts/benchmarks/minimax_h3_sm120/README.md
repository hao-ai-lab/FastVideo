# Single-GPU MiniMax-H3 sparse-attention comparison

Compare native Triton with the opt-in Gluon BF16 forward on one RTX PRO 6000
Blackwell (SM120, 96 GB). Both modes use the same full BF16 checkpoint, online
AdaLN cache and CPU-offload settings. The Gluon kernel supports head dimension
128, 64-token tiles and eager inference only; it is not bit-exact with native
Triton. PyTorch 2.12.0+cu130 and Triton 3.7.0 were used for validation.

## Run

From the FastVideo checkout, with FastVideo and its CUDA dependencies installed:

```bash
export MODEL_PATH=/path/to/FastH3-8-Step-V2
bash scripts/benchmarks/minimax_h3_sm120/run_1gpu.sh native
bash scripts/benchmarks/minimax_h3_sm120/run_1gpu.sh sm120
```

`native` sets `FASTVIDEO_H3_VSA_SM120=0`; `sm120` sets it to `1`. The script
automatically uses the source kernel package, including
`triton_kernels/block_sparse_attn_gluon.py`. No additional CUDA extension
build or checkpoint conversion is required.

The default prompt is p02 (a red fox in a snowy forest), with 1344x768 output,
362 frames at 24 FPS and eight DiT forwards. Each mode runs one excluded
warmup at seed 999, then three measured generations at seed 42. Optional H3
fusions, FP4, FA4 and compilation are disabled. The text encoder uses layerwise
CPU offload, the DiT stays on GPU between requests, and the VAEs are
CPU-offloaded. Host RAM must accommodate the offloaded weights.

For a single generation without performance warmup:

```bash
bash scripts/benchmarks/minimax_h3_sm120/run_1gpu.sh sm120 --no-warmup --repeats 1
```

Additional example arguments can be appended. Use identical sampling and
execution overrides for both modes when comparing results.

For a checkout at `/root/code/FastVideo`, videos are saved to
`/root/code/outputs/fasth3_sm120/{native,sm120}/` and logs to
`/root/code/outputs/profiles/YYYYMMDD_HHMMSS/{native,sm120}.log`.
E2E includes per-request transfers, decoding and saving, but excludes initial
generator construction and warmup.

## Kernel and H3 tests

```bash
export PYTHONPATH="$PWD/fastvideo-kernel/python${PYTHONPATH:+:$PYTHONPATH}"
pytest fastvideo-kernel/tests/test_vsa_gluon_forward.py -v
pytest fastvideo/tests/attention/test_vsa_h3_gluon.py -v
python fastvideo-kernel/benchmarks/bench_vsa_gluon.py
```

The GPU tests skip on non-SM120 devices. Kernel numerical tolerances do not
establish video equivalence; report model-level comparisons separately. See
[Gluon sparse attention](../../../docs/inference/fasth3-distilled.md#optional-gluon-sparse-forward-on-sm120)
for the interface and validation scope.
