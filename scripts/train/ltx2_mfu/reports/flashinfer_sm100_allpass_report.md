# SM100 NVFP4 LTX-2 all-pass gate

Verdict: **reject this installed FlashInfer/CuTe-DSL primitive**. Even its
prequantized lower bound is 1.789x over the 2.5x gate. Counting delayed-scale
quantization is slower than BF16.

The direct PyTorch-vendored kernel sweep was also stopped without timing. Its
SM100 kernel compiled after compatibility aliases for the CUTLASS 4.6 enum
spellings, but TVM-FFI rejected the packed-FP4 runtime shape (logical
`[4290,4096]`, physical packed tensor `[4290,2048]`). This environment lacks
the optional `cutlass_api` shape adapter. The public wrapper below is therefore
the only executable deployable path in this environment, and it fails decisively.

## Reproduction

```bash
FV_JOBID=1619868 .agents/skills/run-fastvideo-dlcluster/dlrun --profile gb200 \
  'unset TORCH_LOGS; python /mnt/bench_flashinfer_sm100_allpass.py | tee /mnt/bench_flashinfer_sm100_allpass.jsonl'
```

- GPU: NVIDIA GB200, SM100 (`(10, 0)`)
- PyTorch: `2.12.0+cu130`; CUDA: `13.0`
- FlashInfer: `0.6.14`
- NVIDIA CUTLASS DSL: `4.6.0.dev0`
- Timing: CUDA events, 3 warmups, median of 5 samples, 3 launches/sample
- Seed: `20260721`; BF16 output and FP32 accumulator
- Kernel: FlashInfer `Sm100BlockScaledPersistentDenseGemmKernel`
- NVFP4: E2M1 values, E4M3 scale per 16 values, swizzled 128x4 SF layout
- `backend="cute-dsl"`, PDL enabled, FP4 global alpha applied in GEMM
- The wrapper autotunes SM100 tile, cluster, A/B swap, and prefetch tactics
  using cold-L2 and CUDA-graph timing. The wrapper did not expose the selected
  tactic in the result.

## Semantics

- `prequant`: GEMM only; an optimistic absolute lower bound.
- `delayed`: fixed/delayed global scale, but per-16 activation/dY scale and FP4
  conversion are recomputed. Fprop/dgrad reuse row/column weight caches; both
  caches are refreshed once per step and counted. Wgrad quantizes both inputs.
- `exact`: same as delayed, plus current-tensor FP32 amax for every dynamic
  tensor and every weight-cache refresh.
- The probe includes global postscale but no bias/epilogue or cache bookkeeping,
  so totals are optimistic. It does not claim delayed-scale training quality.

## Results (milliseconds per logical call)

| pack | phase | QMM shape MxKxN | BF16 | prequant | delayed | exact | weight refresh delayed | weight refresh exact | rel RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| self_qkv | fwd | 4290x4096x12288 | 0.2655 | 0.1230 | 0.1527 | 0.5410 | 0.0528 | 0.3396 | 0.13439 |
| self_qkv | dgrad | 4290x12288x4096 | 0.2677 | 0.1221 | 0.2205 | 0.8213 | 0.0500 | 0.3376 | 0.13440 |
| self_qkv | wgrad | 12288x4320x4096 | 0.2826 | 0.1583 | 0.2075 | 0.6206 | - | - | 0.13456 |
| video_dd (x3) | fwd | 4290x4096x4096 | 0.1045 | 0.1314 | 0.1914 | 0.5242 | 0.0681 | 0.2144 | 0.13441 |
| video_dd (x3) | dgrad | 4290x4096x4096 | 0.1050 | 0.1278 | 0.1957 | 0.5386 | 0.0703 | 0.2136 | 0.13442 |
| video_dd (x3) | wgrad | 4096x4320x4096 | 0.1111 | 0.1376 | 0.2530 | 0.5719 | - | - | 0.13433 |
| text_kv | fwd | 1024x4096x8192 | 0.0627 | 0.1292 | 0.1936 | 0.5104 | 0.0692 | 0.2451 | 0.13437 |
| text_kv | dgrad | 1024x8192x4096 | 0.0577 | 0.0879 | 0.1372 | 0.3633 | 0.0429 | 0.2388 | 0.13444 |
| text_kv | wgrad | 8192x1024x4096 | 0.0553 | 0.0881 | 0.1653 | 0.4073 | - | - | 0.13445 |
| ffn_up | fwd | 4290x4096x16384 | 0.3598 | 0.1665 | 0.2210 | 0.6604 | 0.0750 | 0.4404 | 0.13445 |
| ffn_up | dgrad | 4290x16384x4096 | 0.3509 | 0.1550 | 0.2695 | 1.0563 | 0.0638 | 0.4281 | 0.13444 |
| ffn_up | wgrad | 16384x4320x4096 | 0.3693 | 0.2076 | 0.2876 | 0.7495 | - | - | 0.13429 |
| ffn_down | fwd | 4290x16384x4096 | 0.3509 | 0.1639 | 0.2932 | 1.0623 | 0.0789 | 0.4395 | 0.13434 |
| ffn_down | dgrad | 4290x4096x16384 | 0.3588 | 0.1646 | 0.2185 | 0.6643 | 0.0721 | 0.4401 | 0.13451 |
| ffn_down | wgrad | 4096x4320x16384 | 0.3492 | 0.2011 | 0.2792 | 0.7620 | - | - | 0.13457 |

## Weighted 48-block gate

| tier | total ms | ms/block | speedup vs 198.405 ms separate BF16 | gate |
|---|---:|---:|---:|---:|
| same-run packed BF16 | 196.431 | 4.092 | 1.010x | baseline |
| prequant lower bound | 141.954 | 2.957 | 1.398x | fail |
| delayed + counted quant/cache refresh | 263.332 | 5.486 | 0.753x | fail |
| exact amax + counted quant/cache refresh | 831.173 | 17.316 | 0.239x | fail |

Required: <=79.362 ms total / <=1.653 ms per block (2.5x). Robust target:
<=73.483 ms / <=1.531 ms (2.7x). All outputs were finite; relative RMS was
0.13429--0.13457.

## Artifacts

- `/mnt/bench_flashinfer_sm100_allpass.py`:
  `d97164405a3aee1f9e716c85905e72973c096a08028a7ef2bd8d0a029f1d3611`
- `/mnt/bench_flashinfer_sm100_allpass.jsonl`:
  `18d6801c97a458f6ac8f31acc5277d22a8006507658157587ff1649a1041d2b0`
- `/mnt/bench_flashinfer_sm100_nvfp4.py`:
  `859086a066c1eda218857a2578dadb378ad91587d0633131e30a7ffc09ed84be`
- `/mnt/bench_flashinfer_sm100_nvfp4.jsonl`:
  `d089b9f8b3bc90f8c33fedc25eb1dcedc0e1e8c182213cff769c393a6f664a4a`
- `/mnt/bench_vendored_sm100_nvfp4.py` (compile-only blocked driver, eight
  `(tile, cluster)` configurations):
  `cf31e6e108b65387606b76a1ed5f3428ab074db1baba75e3f0f2b0722eeb84c2`
