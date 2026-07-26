# QuACK 0.5 complete LTX-2 projection gate

Scratch source only; FastVideo source remains pinned and untouched at
`7f139e2b28610063d2f30526ba8f0ccae5d88944` (PR #1630).

## Counted contract

- FP32 master weights stay resident; the timed operator starts from BF16
  working weights, activations, and output gradients. FP32 Adam moments and
  master-to-working synchronization remain in the surrounding training
  runtime, not this projection-only timer.
- Exact-current FP32 amax is computed once for each underlying `weight`, `x`,
  and `dY`. Its scalar is reused across orientations.
- Each step refreshes rowwise and transposed/columnwise NVFP4 weight caches.
- `x` and `dY` are each quantized rowwise and in the transposed layout needed
  by wgrad. The current QuACK helper requires BF16 transpose/pad
  materialization, which is included. The immutable text context's two packs
  are amortized once across all 48 blocks; other activations and every `dY`
  are layer-specific.
- QuACK config 5/6/7 exact-shape dispatch runs fprop, dgrad, and wgrad with
  FP32 accumulation and BF16 QMM outputs.
- Separate compiled epilogues apply the three global postscales, forward bias,
  and BF16 bias-gradient reduction. The paired gate denominator remains the
  three bare BF16 GEMMs used by the earlier 196.431 ms gate, so FP4 overhead
  cannot improve its own denominator.

## Next GB200 run

```bash
.agents/skills/run-fastvideo-dlcluster/dlpush \
  /private/tmp/bench_quack_sm100_nvfp4_complete_projection.py \
  /mnt/bench_quack_sm100_nvfp4_complete_projection.py

FV_JOBID=1622676 .agents/skills/run-fastvideo-dlcluster/dlrun --profile gb200 \
  'unset TORCH_LOGS; PYTHONPATH=/mnt/quack_kernels-0.5.0-py3-none-any.whl python /mnt/bench_quack_sm100_nvfp4_complete_projection.py --smoke-only --pack self_qkv | tee /mnt/bench_quack_sm100_nvfp4_complete_projection_smoke.jsonl'

FV_JOBID=1622676 .agents/skills/run-fastvideo-dlcluster/dlrun --profile gb200 \
  'unset TORCH_LOGS; PYTHONPATH=/mnt/quack_kernels-0.5.0-py3-none-any.whl python /mnt/bench_quack_sm100_nvfp4_complete_projection.py | tee /mnt/bench_quack_sm100_nvfp4_complete_projection.jsonl'
```

Run the smoke first. The full result is admissible only if every sampled
fprop/dgrad/wgrad output is finite and the aggregate
`ratio_normalized_complete_ms` is at most 63.463 ms; 59.524 ms is the margin
target.

## Hard feasibility concerns

1. The prequantized dispatch had only 3.255 ms normalized headroom. This probe
   adds six exact-scale/quantization paths per logical pack, two large BF16
   transposes, three output-scale passes, bias, dbias, and two weight-cache
   refreshes. Passing without a fused transpose-quantizer and QMM epilogue is
   unlikely; failure is still a useful stop signal. After amortizing the text
   context, the exact shapes contain 30.161 billion underlying BF16 elements.
   The current separate amax/row-quant/transpose/column-quant flow has an
   optimistic 312.5 GiB traffic floor (41.9 ms even at 8 TB/s), before scale
   swizzle, QMM, or epilogues. Even an ideal two-pass dual-orientation
   quantizer has a 144.0 GiB / 19.3 ms floor, or about 14.8 normalized ms,
   already above the 3.255 ms margin unless it overlaps QMM execution.
2. QuACK's existing GEMM writes BF16 before global postscale. The separate
   BF16 epilogue measures deployable traffic but is not bit-equivalent to
   postscaling/biasing the FP32 accumulator before one BF16 cast. A production
   operator needs an in-kernel alpha+bias epilogue and must repeat numerical
   gates.
3. This schedules the three backward products explicitly; it does not measure
   `torch.autograd.Function` dispatch, FSDP reduce-scatter overlap, fixed-arena
   lifetimes, or optimizer work. Do not extrapolate step MFU until the operator
   passes this stricter local gate and distributed overlap is measured.
4. Cross-block batched wgrad is intentionally absent: it changes activation
   lifetime and communication timing. Add it only if the sequential complete
   path narrowly misses and memory/overlap can be demonstrated end to end.
5. All-pass NVFP4 gradients replace PR #1630's FP4-forward/BF16-backward STE
   recipe. Even a latency pass therefore needs sampled gradient cosine/norm,
   finite-loss, and overfit-quality gates before it is a training candidate.
