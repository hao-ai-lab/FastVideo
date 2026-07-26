# LTX-2 packed FP8 raw-arithmetic gate on GB200

Verdict: **neither candidate reaches 50% MFU with the allowed 20--30 ms graph
win**. All-pass FP8 is closest, but its optimistic prequantized ceiling still
needs 31.893 ms of graph/runtime savings. NVFP4 forward plus FP8 backward needs
35.529 ms.

## Reproduction

```bash
FV_JOBID=1619868 .agents/skills/run-fastvideo-dlcluster/dlrun --profile gb200 \
  'unset TORCH_LOGS; CUDA_VISIBLE_DEVICES=0 python \
   /mnt/benchmark_scaled_mm_fp8_packed_ltx2.py | tee \
   /mnt/benchmark_scaled_mm_fp8_packed_ltx2.jsonl'
```

- NVIDIA GB200, SM100; PyTorch `2.12.0+cu130`; CUDA 13.0
- Raw `torch._scaled_mm`, E4M3 x E4M3, BF16 output
- `use_fast_accum=True` and `False` measured separately
- CUDA events; 5 warmups; median of 9 samples; 5 launches/sample
- Inputs are prequantized and scales are fixed at one. Quantization, amax,
  weight-cache refresh, bias/epilogue, and integration overhead are excluded.
- Wgrad M is zero-padded to 32 for the FP8 operand; the same padded operands are
  used by its BF16 control.

## Exact 15-case results (milliseconds)

| pack | phase | QMM MxKxN | BF16 | FP8 fast | FP8 accurate | rel RMS |
|---|---|---:|---:|---:|---:|---:|
| self QKV | fwd | 4290x4096x12288 | 0.26055 | 0.13806 | 0.13729 | 0.04248 |
| self QKV | dgrad | 4290x12288x4096 | 0.26508 | 0.13468 | 0.14084 | 0.06548 |
| self QKV | wgrad | 12288x4320x4096 | 0.29159 | 0.14506 | 0.14403 | 0.06248 |
| video 4096 (x3) | fwd | 4290x4096x4096 | 0.11565 | 0.06964 | 0.06891 | 0.04245 |
| video 4096 (x3) | dgrad | 4290x4096x4096 | 0.09596 | 0.06310 | 0.06190 | 0.06549 |
| video 4096 (x3) | wgrad | 4096x4320x4096 | 0.10033 | 0.06216 | 0.06186 | 0.06250 |
| text KV | fwd | 1024x4096x8192 | 0.05478 | 0.06217 | 0.06128 | 0.04251 |
| text KV | dgrad | 1024x8192x4096 | 0.05403 | 0.06116 | 0.05908 | 0.06548 |
| text KV | wgrad | 8192x1024x4096 | 0.05416 | 0.06138 | 0.06079 | 0.06249 |
| FFN up | fwd | 4290x4096x16384 | 0.34341 | 0.18270 | 0.17642 | 0.04246 |
| FFN up | dgrad | 4290x16384x4096 | 0.34303 | 0.16908 | 0.17049 | 0.06547 |
| FFN up | wgrad | 16384x4320x4096 | 0.35014 | 0.18934 | 0.18112 | 0.06248 |
| FFN down | fwd | 4290x16384x4096 | 0.33720 | 0.18710 | 0.18467 | 0.04246 |
| FFN down | dgrad | 4290x4096x16384 | 0.34957 | 0.17601 | 0.17059 | 0.06545 |
| FFN down | wgrad | 4096x4320x16384 | 0.33691 | 0.17840 | 0.18355 | 0.06249 |

The error column uses the fast-accumulation output; it measures input FP8
rounding plus accumulation error. This is an arithmetic check, not a training
quality gate.

## Weighted 48-block totals

| tier | fprop | dgrad | wgrad | total | speedup |
|---|---:|---:|---:|---:|---:|
| packed BF16 | 64.459 | 62.381 | 64.021 | 190.861 ms | 1.000x |
| FP8 fast accumulation | 37.391 | 35.051 | 36.512 | 108.953 ms | 1.752x |
| FP8 accurate accumulation | 36.786 | 34.882 | 36.243 | 107.911 ms | 1.769x |

Accurate accumulation is 1.043 ms faster in aggregate on these shapes, so it
is the best candidate.

## Fixed-arena composition

Projection savings use the same-run BF16 control:

`projected_step = 403.725 - 190.861 + candidate - graph_win`

For the hybrid, the most optimistic prior prequantized NVFP4 fprop measurement
(40.422 ms) is combined with this run's FP8 dgrad+wgrad. Cross-run use favors
the hybrid; normalizing clocks would make it slightly slower.

| candidate | raw total | arithmetic saving | +20 ms graph | +30 ms graph | graph win actually needed |
|---|---:|---:|---:|---:|---:|
| all FP8 accurate | 107.911 ms | 82.950 ms | 300.775 ms (48.02% MFU) | 290.775 ms (49.67%) | 31.893 ms |
| NVFP4 fwd + FP8 accurate bwd | 111.547 ms | 79.314 ms | 304.411 ms (47.45%) | 294.411 ms (49.06%) | 35.529 ms |

Target: 288.882 ms / 50% MFU. Because these are prequantized ceilings, real
quantization and cache costs can only widen the 1.893 ms and 5.529 ms gaps.
Do not integrate either path on the promise of a 20--30 ms graph win. Revisit
all-pass FP8 only after an independently measured graph/runtime saving exceeds
31.9 ms with enough margin for quantization.

## Artifacts

- `/mnt/benchmark_scaled_mm_fp8_packed_ltx2.py`:
  `99e8c70c550a44d058f33076a754899de0780527296e9e97b4e683b5494e378f`
- `/mnt/benchmark_scaled_mm_fp8_packed_ltx2.jsonl`:
  `a4320af4b689968bf9cf1840f77c0fb47b574fbca82662c5629c54aef8b7d918`
