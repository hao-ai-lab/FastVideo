# BF16 kernel research plan toward 50% MFU

Successor plan to the 2026-07-22 systems campaign. The feasibility budget in
`REPORT.md` fixes the constraint: with today's kernels a theoretical-max
systems endgame caps at 48.34% MFU at 4x/B2, so 50% under the standard BF16
contract requires raising combined GEMM+FA4 throughput from 56.7% of the
2,450 TFLOP/s convention to 58.9% (max-systems) or 64.7% (realistic-systems).
Every library-level lever is measured and closed; what remains is kernel work.
Experiments below are ordered by expected value per engineering day. All gates
use the A/X/B contract on one healthy allocation with the committed harness;
microgates use exact shapes and occurrence weighting like the CUTLASS gates.

## 1. FA4 forward-efficiency anomaly (days; expected up to +0.5-1.3 MFU pp)

Measured at B2: forward cute kernel 41.9 ms/step for 35.86 executed TFLOP
(856 TF/s, 34.9%) while backward runs 89.66 TFLOP in 79.2 ms (1,131 TF/s,
46.2%). Forward being 11 points less efficient than backward is inverted
relative to every flash-attention generation and suggests a schedule or
occupancy defect rather than an algorithmic floor, e.g. tile-tail masking at
S=4,290 (33.5 tiles of 128), a persistent-scheduler underfill at B*H=64, or a
cluster shape mismatch on the 152-SM part.

- Probe: extend `probes/benchmark_ltx2_video_attention_backends.py` with
  Nsight/Kineto per-kernel occupancy and a padded-shape variant (S=4,352 with
  `seqused`-style used-length arguments if the CuTe interface exposes them;
  otherwise explicit KV zero-padding plus output slicing, which is exact for
  non-causal attention with zero-padded KV only if the extra keys are masked —
  verify parity first; reject if masking is unsupported).
- Kill criteria: padded/tuned forward fails parity, or the best variant saves
  <5 ms/step at B2 (below one MFU tenth after drift).
- Ceiling if forward reaches backward's 46.2%: 41.9 -> 31.7 ms/step; combined
  kernel band 56.7% -> 57.9%. Meaningful only stacked with experiment 2.
- STATUS 2026-07-22: tail-tile sub-line KILLED by `probes/bench_fa4_tail_tile.py`
  — ragged S=4,290 matches clean 4,224/4,352 per-FLOP (-0.24% at B2, +2.56% at
  B3, noise), so padding/seqused buys nothing. The probe also reproduces the
  fwd<bwd inversion standalone (fwd 686-718 vs bwd 869-955 TF/s on a bare GPU,
  no FSDP/compile): the anomaly is intrinsic to the FA4 CuTe forward schedule
  at B*H=64-96, S~4.3k, D=128 non-causal. Remaining path is upstream FA4
  forward schedule/occupancy work (or CuTe tile/cluster config sweeps), not
  input-shape tricks.

## 2. cuBLASLt algo enumeration for the two worst GEMM bands (days; expected +0.5-1.5 pp)

The band decomposition shows video_dd (three 4,096x4,096 projections,
144 calls/step, 51-53% efficiency) and text_kv (45%) are the efficiency
holes; ffn/qkv run 59-68%. TunableOp is non-functional on this build, but a
direct `cublasLtMatmulAlgoGetHeuristic`/`cublasLtMatmulAlgoCheck` sweep per
exact shape, layout, and epilogue — outside torch, then pinned via a small
custom op for the winning (algo, workspace) pairs — has not been tried. The
prior CUTLASS/TorchInductor rejections do not cover native cuBLASLt algo
pinning.

- Probe: standalone C++/python ctypes sweep on the five packed shapes at B2/B3
  orientations with bias epilogues, occurrence-weighted; compare against the
  committed 416.5 ms single-GPU band.
- Integration only if the weighted band improves >=3%: a `torch.library`
  custom op wrapping `cublasLtMatmul` with pinned algos for the two worst
  shapes, behind an opt-in config; trainer A/X/B afterward.
- Kill criteria: heuristic sweep's best is within 2% of nvjet's current
  selection (would confirm nvjet is already near-optimal and close this line).
- STATUS 2026-07-22: sweep ran (`probes/bench_ltx2_cublaslt_algo_sweep.py`);
  kill criterion NOT met — weighted band -7.98% at B2 and B3, -9.4% best-of,
  bit-exact parity on all winners. A scratch end-to-end gate
  (`probes/lt_pinned_ops.cpp` + `harness/benchmark_fastvideo_train_lt_pinned.py`,
  336 projections, dgrad/wgrad only) already banks **-0.786% step / +0.234 MFU
  points** at 4x/B2 through the wrapper's own overheads (eager dbias, graph
  breaks, contiguity copies). Remaining upside sits in a fused integration:
  an Inductor-level mm lowering or ReplicatedLinear quant-method hook that
  pins algos inside compiled backward and keeps dbias fused; expected total
  +1.9 to +2.2 MFU points at 4x/B2. Then re-gate on a healthy allocation.

## 3. Whole-step wall-minus-union gap (days; expected +0.5-1.0 pp)

32 ms/step of non-busy wall remains at B2 (720.5 wall vs 688.6 busy union).
The prior CUDA-graph attempt was blocked by three successive
device-constant-creation capture failures, and the report already notes the
systematic fix: pre-register immutable device constants (patchifier grids,
scale factors) as buffers. That change is small, benefits eager latency too,
and re-opens regional graph capture for the densest launch bursts (the
pointwise chains between GEMMs).

- Gate order: constant pre-registration alone (may already shave host time),
  then capture of the norm/modulation chains only. Whole-step capture stays
  out of scope per the standing decision boundary.
- Kill criteria: <8 ms/step from the combination, or any interaction with
  FSDP2 symmetric memory.

## 4. CuTe-DSL persistent BF16 GEMM for ffn shapes (weeks; expected 0 to +2 pp, high risk)

Only if 1-3 land and 50% is still short. Hand-tiled CuTe-DSL GEMMs for the
M=8,580/12,870 ffn shapes with fused bias(+GELU prologue on up-projection)
would target the 59% ffn band. Prior evidence is unfavorable — the QuACK-era
paired-BF16 CuTe reference ran below nvjet, TorchInductor CUTLASS regressed,
and cuBLASLt fused GELU was 80% slower — so this line needs a fresh mechanism
(persistent kernels with programmatic dependent launch across the up/act/down
chain), not a retiling of closed attempts. Treat as research with a hard
one-week timebox and the standard >=3% weighted-band bar before any
integration work.

## Explicitly out of scope

NVFP4/FP8 quantization (deferred by the author), TF32/BF16 reduced-precision
state (violates the standard-training contract), and whole-transformer
mega-kernels (rejected in the main report). The 1,965 MHz rack-059/057 bins
remain fine for ratio microgates but are baseline-ineligible; final MFU rows
need a 2,062 MHz-class allocation.
