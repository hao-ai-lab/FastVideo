# LTX-2 50%-MFU optimization gate (4x/8x GB200)

> Tracker status (2026-07-22): PR #1630 is intentionally an ongoing cumulative experiment branch. The authoritative stopping-point MFU rows are 40.810314% on 4x GB200 at local batch 2, 43.623053% on a healthy 8x two-tray allocation at local batch 3, and 30.478319% for the same 8x recipe on a degraded allocation. Standard 1x full-parameter training has no valid MFU result because FP32 master weights and FP32 Adam state OOM at the first optimizer step.

> Precision correction: historical 1x W&B runs `5h2cy7m9`, `zj6wqtpg`, `nzh2k20t`, and `gbjiw7og` used `training.dit_precision=bf16`. Their 11.71-13.46% MFU figures used persistent BF16 parameters and BF16 Adam state, so they are capacity diagnostics only and are not comparable to the standard FP32-master contract below.


Date: 2026-07-21

FastVideo source: `7f139e2b28610063d2f30526ba8f0ccae5d88944` for the original research gates; measured optimization follow-ups through `20c36acef`, followed by validation fixes `0e60a0e9c` and `3f3f06541` (PR #1630)

Fresh allocations: Slurm `1622676` and `1624214`, 4x NVIDIA GB200; Slurm `1623561`, 8x NVIDIA GB200 across two collocated trays

Workload: `FastVideo/LTX2-Distilled-Diffusers`, standard mixed precision (FP32 master weights/optimizer state, BF16 working compute and reductions), 81x480x832, dense FA4, regional compile, no activation checkpointing, repeated/prefetched input, singleton uniform-timestep embedding at SP=1, and persistent projection packing where labeled. Local batch and gradient accumulation are 1 unless the batch-capacity or accumulation sections explicitly say otherwise.

## Precision contract

The native FSDP source runs intentionally load FP32 original parameters and use FSDP mixed precision to gather/cast BF16 working parameters for compute. Those FP32 originals are the optimizer masters; `default_dtype: torch.float32` is therefore expected and is not a benchmark mismatch. The fixed-arena prototype implements the same convention explicitly with a BF16 working parameter/gradient arena and sharded FP32 masters. A separately labeled Transformer Engine gate retains FP32 masters/moments but uses BF16 registered shards and gradients; it is a measured alternative precision layout, not part of the production baseline.

## Measurement and profiler stack

- End-to-end MFU is authoritative: FastVideo's inclusive `step_time_sec` is gathered across ranks, each step uses its slowest-rank time, and the reported result is the median after 10 warmup steps.
- The deferred-gradient-norm gate also records independent batch-fetch-to-batch-fetch wall intervals and closes the final interval with a CUDA synchronization. Its true-wall result is authoritative over the trainer's internal timer when checking for timer artifacts.
- PyTorch Profiler/Kineto traces provide operator and kernel attribution; exact-shape CUDA-event microbenchmarks gate individual attention, GEMM, quantization, and collective hypotheses. TorchInductor logs and controlled NCCL A/B/A runs cover compile and communication changes.
- W&B records training/validation observability but is not the source of truth for the slowest-rank MFU rows.
- `nsys-ai` was audited at upstream commit `d51180069d29439d56e7be2f2102f44886e00bb05` and is useful as an isolated offline reader for Nsight Systems SQLite exports. It is not a collector, the current GB200 host/container has no `nsys` binary, and its 2.250 PFLOP/s GB200 default conflicts with this report's accepted 2.450 PFLOP/s convention. Do not use its MFU output or add it as a FastVideo dependency; once a collector is available, analyze each rank separately and retain this harness as authoritative.

## One-GPU capacity boundary

The last one-rank audit, at `7f139e2b28610063d2f30526ba8f0ccae5d88944`, cannot complete the first optimizer step on one 184.31 GiB GB200 under the standard precision contract. Full activation checkpointing, `reshard_after_forward=true`, expandable allocator segments, explicit lazy CUDA module loading, and regional compile on/off all still OOM during FP32 Adam moment creation. A final dense-FA4 retry also unloaded the validation-only VAE before training and failed on a 64 MiB allocation with 182.83 GiB allocated and 3.50 MiB free. The measured optimization head was not re-gated on one GPU; optimizer offload or reduced-precision state would define a different baseline.

## Verdict

A whole-transformer mega-kernel is still **not** justified, but the remaining 50%-MFU gap is now kernel research rather than systems tuning.

- For unchanged vanilla BF16, a corrected measured-head fixed-arena prototype reaches 400.616610 ms / 36.054708% MFU and leaves 111.734 ms to the 288.882 ms 50%-MFU target. That is real architecture evidence, but it is benchmark-only: the 4x/B1/accumulation-1/LTX-specific scratch runtime adds 35.829 GiB peak allocation and omits DCP/export/EMA and generic lifecycle integration. Native FSDP2 symmetric-memory collectives, repeated/prefetched input, deferred gradient-norm materialization, non-final accumulation no-sync, opt-in between-microbatch parameter retention, singleton timestep embedding, persistent projection packing, and two-modules-per-group FSDP2 sharding are the production stack. At exact head `20c36acef`, grouping saves 7.969633 ms / 1.930% without checkpointing and 23.796931 ms / 3.063% with the committed full-checkpoint recipe. BF16 registered shards with TE FP32 masters save only 5.908 ms while adding memory and lifecycle complexity. Whole-step CUDA-graph capture remains blocked before replay.
- CUTLASS is closed for the current B2 shapes. The original B1 exact-shape gate found a 72.184321 ms arithmetic opportunity, but the unrestricted trainer gate was unsafe, an exact-name B2 trainer candidate regressed 2.143% while emitting 7,330 illegal-access warnings, and the final singleton-swizzle B2 A/X/B microgate was parity/safety clean but regressed 30.752773 ms / 7.8520%. No CUTLASS source or configuration is retained.
- For BF16-equivalent low-precision training, QuACK 0.5 clears only the prequantized arithmetic gate: exact-shape dispatch reaches 3.263x paired BF16 / 60.208 ms normalized. The complete sequential operator is a hard reject at 2,072.035 ms versus 250.557 ms paired BF16, or 1,624.429 ms normalized against the required <=63.463 ms.
- Quantization is not a small wrapper cost. After amortizing immutable text context, even an ideal fused dual-orientation quantizer has a 144.0 GiB / 19.3 ms traffic floor at an ideal 8 TB/s, already larger than the 3.255 ms prequantized margin unless production is overlapped under QMM execution or the arithmetic kernel gets faster.
- The complete smoke is also below a plausible quality gate: video/self-QKV/FFN forward and dgrad cosine is about 0.814-0.817 with 58.3-58.8% relative RMS; wgrad is about 0.991 / 13.4% relative RMS. Finite output alone is insufficient.

A fresh config-7 all-pass rerun independently confirms there is no hidden QuACK integration margin. Packed NVFP4 arithmetic was exact on the all-ones smoke and finite on every production shape, but sequential fprop+dgrad+wgrad measured 80.814 ms and the optimistic deferred/batched-wgrad lower bound measured 76.032 ms. They miss the absolute 63.463 ms break-even ceiling by 17.351 and 12.569 ms respectively before quantization, scale, cache, bias, or distributed costs.

The directly deployable prequantized result had only 3.255 ms of normalized arithmetic margin and missed the <=59.524 ms / 3.3x integration target by 0.684 ms. The complete measurement closes that candidate. Deferred batched wgrad remains an optimistic arithmetic lower bound because it changes activation lifetime and reduce-scatter overlap; it cannot rescue the measured quantization and quality failures by itself.

## Fresh-allocation control and phase decomposition

The fresh allocations differ materially, so only paired same-node deltas and ratios are used for attribution. The historical external fixed-arena result was 421.850613 ms / 34.239882%. A corrected measured-head same-allocation gate supersedes that headline: 400.616610 ms / 36.054708% versus a 413.472777 ms source-control midpoint, saving 12.856167 ms / 3.109314% and adding 1.119589 MFU points with 1.295174% control drift. The best matched pushed-source B1 gate remains packed commit `d01623709`: 414.823197 ms / 34.819931% versus a 427.432493 ms split-projection control midpoint. Measured head `20c36acef` enables packing and two-module FSDP2 groups in the overfit recipe. Batch scaling reaches 43.623053% MFU on 8x/B3 with packing; larger-batch rows change global batch and remain separate from B1 attribution.

- Uninstrumented fixed-arena control: 547.683096 ms wall, 547.350 ms GPU, 26.373125% MFU.
- Event-instrumented control: 545.251277 ms; the 2.432 ms difference validates that the phase instrumentation does not materially perturb the step.
- Median event decomposition: data 0.221 ms; exposed all-gather wait 0.202 ms; forward compute 160.339 ms; backward plus overlapped reduce-scatter 345.321 ms; exposed reduce-scatter wait 3.637 ms; grad-shard copy 3.578 ms; norm/clip 6.746 ms; AdamW 14.678 ms; BF16 copy/all-gather launch 3.171 ms; GPU other 7.272 ms; host other 0.087 ms.

Only 3.838 ms is exposed collective waiting. The large backward interval is occupied by compute plus overlapped communication; it is not a 133 ms idle bubble that a graph or launch fusion can erase.

## Input-pipeline productionization

The original four-row overfit fixture exhausted each rank's epoch every step, rebuilding the stateful iterator and leaving input work on the critical path. The dataset already supports virtual path repetition, so no source abstraction or physical data copy is needed.

| 4x exact-source path | Median slowest-rank wall | MFU | Delta |
|---|---:|---:|---:|
| original four-row fixture, worker 0 | 493.569924 ms | 29.264577% | control |
| virtual repeat `:32`, worker 1 | 466.474781 ms | 30.964407% | -27.095143 ms / -5.489% |

Production commit `7c58950a9` configures the 300-step LTX-2 overfit recipe with a virtual repeat count of 300 and one worker, giving each of four ranks exactly 300 samples without expanding the 12 MB fixture. It also removes the nightly launcher's redundant scalar `data_path` override so the mapping survives. The GB200 environment parsed the exact committed YAML as `{'data/ltx2_overfit_preprocessed': 300}` with one worker, and the existing structured-path config test passed. The performance result is from exact source `7f139e2b2`; later commits do not touch the dataset path.

## Deferred gradient-norm materialization

Gradient clipping already computes the norm and enqueues the clipping work before the optimizer, but converting that CUDA scalar to a Python float forced the host to wait before fused AdamW could launch. A scratch-only A/B/A kept clipping and per-step logging intact while retaining the norm tensor until after the optimizer. Both the original trainer timer and an independent synchronized wall timer were recorded:

| 4x FA4 run | True-wall slowest-rank median | MFU | Matched result |
|---|---:|---:|---:|
| control A | 460.133193 ms | 31.391161% | control |
| scratch deferred candidate | 435.467763 ms | 33.169195% | candidate |
| control B | 459.621117 ms | 31.426134% | control |
| control midpoint | 459.877155 ms | 31.408638% | **-24.409393 ms / -5.3078%** candidate saving |

The candidate won all 20 measured true-wall steps, and 99.65% of its internal-timer delta survived the independent timer, ruling out a bookkeeping artifact. Production commit `7f6c290c9` implements the same idea in only three files: the clipping helper returns the device tensor, `GradNormClipCallback` queues it, and the existing `on_training_step_end` hook clears then materializes/logs it after the optimizer and zero-grad path. No new trainer lifecycle or configuration surface is added.

The exact committed-source confirmation selected FA4 and reached **431.193091 ms / 33.498021% MFU / 9.276586 samples/s**, saving **28.684064 ms / 6.2373%** and adding **2.08938 percentage points MFU** against the same two-control midpoint. Every rank recorded exactly 30 clips, 30 optimizer calls, and 30 gradient-norm logs; every norm was materialized after the optimizer. The three-file change passed 49 focused tests in the GB200 environment and finished review-clean. Exact-source log SHA-256: `87347668ad28ab200bf3540da01ff1c7744854c2ce2a3a0a7d1c0cb1d1788d01`.

## Singleton timestep memory gate

Plain LTX-2 T2V uses one uniform sigma per sample. At SP=1, AdaLN broadcasts one timestep embedding across all 4,290 video tokens, so commit `49508050b` passes `[B, 1]` instead of eagerly materializing `[B, 4290]`; SP>1 still expands before sequence sharding. The proof recorded 4,290 semantic tokens and one model timestep token on every rank, with finite AdaLN gradient probes. Its proof script SHA-256 is `05836dda4426309f23e1c797281f0e69cc343c522006679fef6b34e6c7d3f865`, and proof log SHA-256 is `efe7e529860771bc3564adf82b2cfa2a712dc60cf83691b174769aff11760317`.

The 4x GB200 A/X/B timing is deliberately classified as neutral. Controls measured 435.652742 and 440.625450 ms, a 438.139096 ms midpoint with 1.14% control drift; the candidate measured 434.923236 ms. The corresponding true-mean and measured-window changes were only about 0.1%. Peak allocated memory fell **13.0057 GiB / 11.416%**, and peak reserved memory fell **13.6719 GiB / 10.043%**, so the change is retained for capacity rather than timing credit. Raw log SHA-256 values are `983b52f96ceace4ddd1fa01c36f6b851c9bb74fd6c4033e8932bc99abf9f1c3a` (A), `189f0ffc65ff4ae2a449990c026a385a79222e45e95ade94977bbb981c23f70a` (X), and `505d0062845b2c1662384ab7f5e60acdebd4e6971ac66bcb521b7ab0b2bb41f5` (B).

The singleton path passed seven lightweight tests plus real LTX-2.0 and LTX-2.3 model tests in the GB200 environment. Current head passed its 80-test focused suite in 3.92 s, parsed the committed packed recipe with `PACKED_RECIPE_OK`, and remained pre-commit/review clean.

## Current-stack batch capacity

The memory reduction makes larger local batches profitable on the cumulative stack. These split-projection accumulation-1 runs were frozen at `49508050b` and inherited through `002ec0771`; the packing gate below supersedes their current timing. The harness's printed throughput/MFU numerator was fixed at B1, so the B2/B3 values below are recomputed from the measured true-wall median and the actual samples per step.

| Topology | Local batch/GPU | Median step | Aggregate throughput | MFU | Peak allocated / reserved per rank |
|---|---:|---:|---:|---:|---:|
| 4x B1 midpoint | 1 | 0.4246335145 s | 9.419888 samples/s | 34.015485% | - |
| 4x B2 | 2 | 0.7105401774 s | 11.259040 samples/s | 40.656716% | 140.496 / 163.723 GiB |
| 8x B1 midpoint | 1 | 0.4165137762 s | 19.207048 samples/s | 34.678601% | - |
| 8x B2 | 2 | 0.6963156550 s | 22.978085 samples/s | 41.487262% | 122.319 / 135.307 GiB |
| 8x B2 second-gate midpoint | 2 | 0.6996195166 s | 22.869574 samples/s | 41.291344% | - |
| 8x B3 | 3 | 0.9994570174 s | 24.013039 samples/s | 43.355887% | 160.862 / 175.172 GiB |
| 8x B4, forward reshard enabled | 4 | OOM before step 1 | - | - | 177.86 GiB allocated / 184.17 GiB total in use |

B2 gains 19.524% throughput / 6.641 MFU points on 4x and 19.634% / 6.809 points on 8x. B3 adds 5.000% throughput / 2.065 points over its 8x B2 midpoint. B4 does not fit even after enabling forward resharding, closing the capacity ladder at B3. These configurations increase global batch, so convergence and LR scaling remain separate gates.

An exact-final-head equal-global-batch gate at `20c36acef` removes that confound by comparing B1/gradient-accumulation-2 with B2/gradient-accumulation-1; both process eight samples per optimizer step. The B1 controls measured 0.7426303995 and 0.7441010120 seconds, a 0.7433657057-second midpoint / 10.761874 samples/s / 38.861435% MFU. B2 measured **0.7078659144 seconds / 11.301575 samples/s / 40.810314% MFU**, saving **35.499791 ms / 4.775549%**, raising throughput **5.014942%**, and adding **1.948878 MFU points** with only 0.198028% control drift. B2 peak allocation/reservation was 140.865/166.404 GiB, 16.109/15.293 GiB above the B1 midpoint. All runs covered 927 FP32 parameters / 13,041,520,768 elements, retained FP32 Adam moments, recorded zero master-writeback mismatches, and had finite sampled AdaLN gradients on all ranks. This is an efficiency result at unchanged optimizer batch; stochastic-run gradient hashes are not a parity claim.

The exact B2/group-2 Kineto trace is diagnostic rather than MFU evidence, but it closes the remaining systems hypothesis. Its two profiled steps were 97.2415% GPU-busy; 120.8658 of 131.3196 ms/step communication was overlapped (92.0394%), leaving only 10.4538 ms exposed. The per-step self-CUDA bands were 389.300 ms for GEMMs, 133.329 ms for FA4, and 14.134 ms for fused AdamW. The compute-event union alone was 688.602 ms, already 110.838 ms above the 577.765 ms 50%-MFU target. Further large gains must accelerate kernels, principally GEMMs and then attention; host/collective tuning cannot close this gap.

## Persistent LTX projection packing

Commit `d01623709` adds opt-in persistent packing for the video path: self-attention Q/K/V use one `to_qkv`, and text cross-attention K/V use one `to_kv`. Across 48 blocks this removes 432 projection GEMM launches over forward, dgrad, and wgrad and reduces parameter objects from 1,215 to 927 without changing the 13,041,520,768 total parameter elements. The loader records every source key and split size so DCP/Diffusers export restores strict split HF keys. Audio/cross-modal paths are unchanged; linear quantization and enabled LoRA are rejected. Current head `20c36acef` enables the otherwise-default-false option in the overfit recipe.

| Gate | Split control A | Packed | Split control B | Control midpoint | Packed delta |
|---|---:|---:|---:|---:|---:|
| 4x/B1 step | 0.4284051351 s / 33.716017% | 0.4148231971 s / 34.819931% | 0.4264598510 s / 33.869812% | 0.4274324930 s | -12.609296 ms / -2.950009%; +1.027191 pp |
| 8x/two-tray B3 step | 0.998659810983 s / 43.390496% | 0.993335918058 s / 43.623053% | 0.999399903463 s / 43.358364% | 0.999029857223 s | -5.693939 ms / -0.569947%; +0.248622 pp |

The 4x candidate reaches 9.642662 samples/s (+3.039680%) with 0.455109% control drift; the 8x candidate reaches 24.161011 samples/s with 0.074081% drift. Both are memory-neutral and completed 30 forward/backward steps with finite loss/norms, strict split-HF loading, FP32 registered masters/moments, and exact optimizer coverage. The final-head focused suite passed 80 tests in 3.92 s; a separate two-tray run passed 3 split/packed SP-gradient and fused-export tests in 150.97 s.

A real FP32 production export/reload gate printed `REAL_PACK_EXPORT_RELOAD_OK`. It wrote 52,166,223,624 bytes and 1,215 HF keys; all 192 merged internal tensors became 480 split projection tensors bit-exact against their sources. Strict split reload restored 1,215 parameter objects from the packed model's 927 while preserving 13,041,520,768 total parameter elements. Final log/script SHA-256: `c3ec531c9cf00c569be30ce1460e224225c617e17e8224af7385b9a82955f08c` / `2a4ba1280710c026fefdcd188ce0207f99a5828c277dcaf265758a8e46c06367`.

### Rejected regional max-autotune mode

A packed 4x/B1 A/B/A tested `max-autotune-no-cudagraphs` without changing source. Default controls measured 0.415032553021 s / 34.802366% MFU and 0.412693266000 s / 34.999638%; their midpoint was 0.413862909511 s / 34.901002% with 0.565232343% drift. Max-autotune measured 0.416518588027 s / 34.678200%, a **2.655679 ms / 0.641680724% regression and -0.222802 MFU points**. Triton repeatedly discarded invalid choices requiring 262,160 registers above the SM100 hardware limit of 232,448, then fell back. That regional gate did **not** enable TorchInductor's CUTLASS backend, so it does not contradict the separate CUTLASS measurements below. Keep the regional compile default; both alternatives are now rejected.

### Attention-compile environment diagnostic

`FASTVIDEO_DISABLE_ATTENTION_COMPILE=0` does not change LTX execution: its `LocalAttention.forward` and `DistributedAttention.forward` overrides bypass the decorated base forward. The packed 4x A/candidate/B medians were 0.415846481 / 0.413138084 / 0.413302450 s. The apparent -1.436381 ms / -0.346471% candidate delta is inside 0.613649% control drift and cannot be attributed to the environment variable. This is a confirmed no-op; make no source or configuration change.

## Raw velocity benchmark-only rejection

A scratch path returning raw velocity avoids the BF16 x0-to-velocity round trip and numerically improves reconstruction error at sigma=1e-3, but it has no performance or capacity case. Its true-wall A/X/B medians were 540.702049 / 542.334403 / 542.752936 ms: the candidate is 0.607 ms / 0.112% slower than the 541.727492 ms control midpoint, with no memory reduction. It remains benchmark-only and is not in the source stack. The round-trip proof script SHA-256 is `f6924a9277723fe51c2d210f5ea3fb66aafbb845d142ee50ef92effd1cb5fa4d`; raw log SHA-256 values are `83ebfafb10c4c9f10011b5cf746b7c6fdde6414d8247774ade8013f4a57a2f78` (A), `b421b2e0bcb1c9b9ac8f4871f3ad41587249a1f520faed6795fc0b10319c0c45` (X), and `6a1875640a2528a8bd126a9037138f03d34a06bf188e1fc10e06005c26cab514` (B).

## NCCL occupancy gate

`NCCL_MAX_CTAS=16` was first tested with the identical 10-warmup/20-measure workload on the same allocation:

| Run | Median slowest-rank wall | MFU | Delta |
|---|---:|---:|---:|
| paired control | 547.683096 ms | 26.373125% | control |
| `NCCL_MAX_CTAS=16` | 546.032818 ms | 26.452833% | -1.650278 ms / 0.301% |

That historical 0.301% single pair predated the final symmetric-memory stack and lacked a closing control. A fresh A/B/A on clean, frozen then-head `7c58950a9`, with FA4, regional compile, no forward reshard, symmetric memory, and BF16 reductions, reversed the sign:

| Frozen-head 4x run | Median slowest-rank wall | MFU | Delta |
|---|---:|---:|---:|
| control A, `NCCL_MAX_CTAS` unset | 558.356668 ms | 25.868976% | control |
| `NCCL_MAX_CTAS=16` | 559.553032 ms | 25.813666% | candidate |
| control B, `NCCL_MAX_CTAS` unset | 559.054499 ms | 25.836685% | control |
| control midpoint | 558.705584 ms | 25.852820% | +0.847448 ms / +0.151681% candidate regression |

All 30 losses and gradient norms were finite in every row, and control drift was only 0.124980%. Do not add the setting: it is withheld because the repeatable frozen-head result is negative, not because the effect is small. Raw log hashes are `00cc6b6632185f1f378996fdda9e62bd2ce285e57042b1c746a178a2c105b1e9` (A), `cf7351572e7ff7d585cf8ac3e4a3ee0463c03dd7ae0f9051f1423669bc85d4fd` (candidate), and `5dd6d20cbeafb70b9f62c4f37ae4391ca5937cb570f96d58a369610ac98d2ecf` (B).

## Native FSDP2 symmetric-memory gate

PyTorch 2.12 symmetric-memory communication was enabled after sharding on all 49 FSDP2 groups (root plus 48 blocks), with forced SUM reductions and `NCCL_CTA_POLICY=2` set before process-group initialization. This enables the native symmetric-memory all-gather and reduce-scatter paths and makes Copy Engine all-gather eligible on the single-node NVLink domain.

The 10-warmup/20-measure A/B/A sandwich used the same source, input, and slowest-rank timing contract:

| Run | Median slowest-rank wall | MFU | Delta |
|---|---:|---:|---:|
| baseline A | 573.191594 ms | 25.199454% | control |
| symmetric-memory AG/RS | 554.638125 ms | 26.042413% | candidate |
| baseline B | 571.166225 ms | 25.288812% | control |
| baseline midpoint | 572.178910 ms | 25.244053% | -17.540784 ms / 3.066% |

The candidate completed normally and printed proof that all 49 groups were configured. Its 17.541 ms saving established the first repeatable signal. The 30 ms threshold is retained only as a research-priority boundary, not as a rule against sound incremental optimizations.

Production commit `e42cfa5e5` adds strict `training.distributed.fsdp_symmetric_memory`, installs `NCCL_CTA_POLICY=2` before process-group initialization, forces SUM reductions, and enables native symmetric-memory communication on every transformer FSDP wrapper. Candidates deliberately launched without the environment variable and configured all 49 wrappers.

| Topology / policy | Control midpoint | Production candidate | MFU | Delta |
|---|---:|---:|---:|---:|
| 4x FULL_SHARD | 594.420553 ms | 571.917264 ms | 25.255602% | -22.503288 ms / -3.786% |
| 4x SHARD_GRAD_OP | 571.403084 ms | 553.293423 ms | 26.105705% | -18.109661 ms / -3.169% |
| 8x/two-tray FULL_SHARD | 471.689844 ms | 460.362467 ms | 31.375527% | -11.327377 ms / -2.401% |
| 8x/two-tray SHARD_GRAD_OP | 448.385441 ms | 445.343413 ms | 32.433656% | -3.042028 ms / -0.678% |

The two-tray allocation was verified as one eight-GPU MNNVL clique. A direct two-rank NCCL all-reduce test found equal large-message performance inside and across trays: 514.923 versus 513.070 GB/s at 256 MiB, and 577.143 versus 577.151 GB/s at 1 GiB. Cross-tray overhead was visible only at small sizes (1 MiB: 82.97 versus 72.73 us; 16 MiB: 76.68 versus 70.83 us).

## Conventional FSDP follow-up gates

The research-priority threshold is not an acceptance threshold. Sound, composable optimizations are kept even when their individual savings are smaller. Four additional public FSDP2 paths were tested on the exact production stack.

The accumulation-1 4x control sandwich measured 555.859681 and 551.755290 ms, or a 553.807485 ms midpoint:

| Path | Median slowest-rank wall | Delta from midpoint | Decision |
|---|---:|---:|---|
| explicit one-module forward/backward prefetch | 554.107704 ms | +0.300219 ms / +0.054% | neutral; do not add |
| explicit two-module forward/backward prefetch | 552.395538 ms | -1.411947 ms / -0.255% | drift-sensitive; do not add |
| process-group communication allocator, replacing symmetric memory | 569.133412 ms | +15.325927 ms / +2.767% | slower and mutually exclusive; reject |

On the 8x/two-tray allocation, existing HSDP `replicate=2, shard=4` maps each four-way shard group within a tray and the replicate pair across trays. It measured 465.559134 ms /31.025307% MFU versus 442.891026 ms /32.613248% for the eight-way shard, a 22.668108 ms /5.118% regression. The extra replicate all-reduce outweighs the smaller intra-tray AG/RS groups, so the existing 1x8 layout remains preferred.

A Kineto trace also exposed 49 size-one `mesh_replicate` all-reduces because a `(replicate=1, shard=4)` device mesh selects PyTorch's HSDP code path. The FP32-era trace attributed 12.146 GiB and 13.660-16.465 ms summed per rank to those events. A standards-aligned 1-D FSDP mesh removed the degenerate topology and passed 8 focused unit tests plus an exact old-2-D-to-new-1-D four-rank DCP model/AdamW resume smoke, but the current BF16/symmetric-memory A/B/A timing was neutral:

| 4x mesh path | Median slowest-rank wall | MFU | Matched result |
|---|---:|---:|---:|
| historical 2-D control A | 454.177486 ms | 31.802798% | control |
| 1-D FSDP candidate | 457.564300 ms | 31.567399% | +0.158553 ms / +0.034664% vs control midpoint |
| historical 2-D control B | 460.634008 ms | 31.357031% | control |
| historical control midpoint | 457.405747 ms | 31.578342% | midpoint |

The source candidate is not retained. Removing a trace-visible no-op is not credited as a speed optimization when the current production sandwich is indistinguishable from drift.

Gradient accumulation exposed one accepted general optimization. Native FSDP2 was previously synchronizing every microstep. Production commit `0b324d0a4` now uses `set_requires_gradient_sync(False)` and `set_is_last_backward(False)` on non-final microsteps, restoring synchronization only for the final backward. The code path is skipped entirely when accumulation is 1, so headline MFU is unchanged.

The matched 4x accumulation-2 sandwich used 30 optimizer steps, 10 warmup + 20 measured, and the same standard FP32-master/BF16-working stack:

| Path | Median slowest-rank wall | Effective MFU | Global samples/s |
|---|---:|---:|---:|
| no-sync candidate A | 1.069807470 s | 27.003205% | 7.477981 |
| forced synchronization every microstep | 1.090327118 s | 26.495012% | 7.337248 |
| no-sync candidate B | 1.074732777 s | 26.879454% | 7.443711 |
| no-sync midpoint | 1.072270124 s | 26.941187% | 7.460807 |

Suppressing the redundant first reduction saves **18.056994 ms / 1.656%**, raises effective MFU by 0.446175 points, and recorded exactly 30 non-sync plus 30 final-sync backwards in each candidate. Focused trainer tests pass in the GB200 environment (3 passed); accumulation-1 performs no additional dispatch.

Commit `002ec0771` then uses the public `set_reshard_after_backward` control to retain already-unsharded parameters across non-final backwards. This path is deliberately gated by the existing `reshard_after_forward: false` opt-in; default-policy jobs never touch backward reshard state, and the final backward restores resharding before optimization.

| 4x accumulation-2, no-sync in both paths | Median optimizer step | Effective MFU | Global samples/s |
|---|---:|---:|---:|
| force backward reshard, control A/B midpoint | 1.0214430510 s | 28.281782% | 7.83206 |
| retain params between microsteps | 1.0055416964 s | 28.729023% | 7.95591 |

This saves **15.901355 ms / 1.55675%**, adds **0.447240 MFU points**, and raises throughput **1.58137%**. Control drift was 0.12197%. Every run recorded 30 non-final plus 30 final sync calls; controls forced 30 backward reshards and the candidate forced none. Peak allocation did not increase. The two-file change passed pre-commit, focused GB200 tests (3 passed), and independent review.

## BF16 registered shards with FP32 TE masters

A source-built Transformer Engine 2.16 scratch path tested whether persistent BF16 FSDP shards can remove enough FP32-to-BF16 pack work while retaining sharded FP32 master weights and FP32 Adam moments. The stock/TE/stock 8x B1 sandwich used the same frozen `49508050b` source plus one optimizer-only scratch patch, dense FA4, singleton timesteps, BF16 reductions, symmetric-memory collectives, and 10-warmup/20-measure true-wall timing.

| 8x B1 path | Median slowest-rank wall | MFU | Global samples/s | Peak allocated / reserved |
|---|---:|---:|---:|---:|
| stock FP32 registered/master control A | 0.415380163 s | 34.773242% | 19.259466 | 82.646 / 96.264 GiB |
| TE BF16 registered + FP32 master candidate | 0.409804175 s | 35.246383% | 19.521519 | 85.666 / 93.465 GiB |
| stock FP32 registered/master control B | 0.416044141 s | 34.717746% | 19.228729 | 82.651 / 96.184 GiB |
| stock control midpoint | 0.415712152 s | 34.745472% | 19.244085 | 82.648 / 96.224 GiB |

The candidate saves **5.907977 ms / 1.421%**, raises throughput **1.442%**, and adds **0.500911 MFU points** with only 0.160% control drift. The strengthened post-timing probe checked all 1,215 trainable parameters on every rank: registered parameters were BF16 DTensors; `master_param`, `exp_avg`, and `exp_avg_sq` were FP32 DTensors with matching mesh, placements, shapes, and strides; optimizer coverage was exact; and every registered shard exactly equaled its FP32 master rounded to BF16. Losses and sampled gradients were finite.

This is not a good production trade despite the real small speedup. Peak allocation rises **3.018 GiB/rank**, registered gradients become BF16 instead of FP32, and the checkpoint has 194 FP32 tensors / 3,256,320 elements that the scratch BF16 load rounds before TE creates its masters. TE 2.16 also has no compatible prebuilt binding for this Torch 2.12/CUDA 13 environment, so it required a source build; production support would additionally need TE-specific resume initialization, FP32-master Diffusers export, checkpoint-backend compatibility checks, and guards or support for EMA and validation optimizer-state offload. The stock FP32 registered/master path remains the production convention. No TE dependency or source path is added.

## Whole-step CUDA-graph localization

Moving the complete static batch to CUDA before capture and replacing the sigma sampler's device-created constants with Python scalars advanced capture to `VideoLatentPatchifier.get_patch_grid_bounds`, where `torch.tensor(self._patch_size, device=patch_starts.device)` attempted a CPU-to-CUDA copy during capture.

A scratch-only patchifier override pre-staged that constant and passed bit-exact eager equivalence on the real shape:

```json
{"equal": true, "max_error": 0, "shape": [1, 3, 4290, 2]}
```

Capture then advanced to `_get_pixel_coords`, where another `torch.tensor(scale_factors, device=latent_coords.device)` caused the same error on all four ranks. The bounded retry stopped at this third distinct graph blocker. No graph was created or replayed, so graph latency and MFU credit remain unavailable. The repeated pattern suggests source graphability can be fixed systematically by pre-registering immutable device constants, but it does not establish a material speedup.

## Rejected FlashAttention fused-dense GELU gate

The fused-dense extension from pinned FlashAttention `82d6441` built and loaded on SM100 with CUDA 13, and executable heuristic h0 passed output and gradient parity for cuBLASLt `GELU_AUX_BIAS` plus `DGELU_BGRAD`. It was substantially slower than the unfused reference:

| Local batch | Reference | h0 fused | Reference - fused/block | Projected over 48 blocks | Fused regression |
|---|---:|---:|---:|---:|---:|
| B1 | 0.895942 ms | 1.615837 ms | -0.719894 ms | -34.554927 ms | 80.35% |
| B3 | 2.637349 ms | 4.774322 ms | -2.136973 ms | -102.574704 ms | 81.03% |

h0 also added 32 MiB peak allocation. Heuristic h1 took 15.566678 ms/block at B1 and 46.050705 ms/block at B3; h2-h4 fail `bias_act_linear_dgrad_bgrad`. The exact build and parity gate therefore reject this path before trainer integration. No source integration or FlashAttention dependency change is made.

## Existing SM100 all-pass projection gate

The node's installed packages are structurally skewed: QuACK 0.6.1 pins released `nvidia-cutlass-dsl==4.6.0`, while the FA4 environment contains `4.6.0.dev0`. Runtime shims reached incompatible shared-memory and pipeline APIs and were abandoned. A PYTHONPATH-only QuACK 0.5 shadow is compatible with the installed DSL after two moved-type aliases (`cute.core.ThrMma` and `cute.core.ThrCopy`); the environment was not installed into or modified.

The correctness smoke used packed NVFP4 inputs, BF16 output, batch `L=2`, and exact all-ones arithmetic. It passed with `max_abs=0.0`. Every exact production shape produced finite output.

The harness covers the seven valid packed projection GEMMs per block, all 48 blocks, and forward + dgrad + wgrad: 336 occurrences per phase / 1,008 GEMMs total / 300.096 TFLOP. Unique-shape latency is multiplied by its exact 48 or 144 occurrence count.

| QuACK tile / cluster | Paired BF16 | NVFP4 sequential | Speedup | Deferred-wgrad lower bound | Speedup |
|---|---:|---:|---:|---:|---:|
| config 5, 256x128 / 2x1 | 256.934 ms | 93.723 ms | 2.741x | 90.439 ms | 2.841x |
| config 6, 256x192 / 2x1 | 257.331 ms | 83.220 ms | 3.092x | 79.049 ms | 3.255x |
| config 7, 256x256 / 2x1 | 256.380 ms | 81.288 ms | 3.154x | 76.422 ms | 3.355x |
| best exact-shape dispatch, configs 5/6/7 | 257.025 ms | 78.781 ms | 3.263x | 73.916 ms | 3.477x |

Config 7 phase totals are 23.894 ms forward, 23.721 ms dgrad, and 33.673 ms sequential wgrad. The slow shapes are wgrad: text KV is 2.349x, self-QKV is 2.560x, video DD is 2.595x, and the two FFN wgrads are about 2.68x. Cross-block batched wgrad improves the aggregate, but even its self-QKV/FFN cases remain about 2.93x.

Config 6 improves most forward/dgrad shapes while config 7 remains best for every wgrad shape; config 5 wins only text dgrad. Exact-shape dispatch therefore lowers the sequential total to 78.781 ms without changing the operator boundary or requiring a new kernel. No 1CTA sweep was run.

Normalizing the paired ratios to the historical exact packed BF16 pass:

- sequential best-of: `196.431 / 3.262522 = 60.208 ms`, passing the 63.463 ms bare break-even by 3.255 ms but missing the 59.524 ms integration target by 0.684 ms;
- deferred best-of: `196.431 / 3.477274 = 56.490 ms`, passing the integration target by 3.034 ms, but not deployable without proving memory and communication overlap.

These are prequantized arithmetic lower bounds. They exclude quantization, global scale, bias/postscale epilogues, weight-cache refresh, and integration overhead. They justify an operator prototype, not a 50% end-to-end claim.

The later all-pass gate reran config 7 without ratio normalization and included exact single-shape plus batched-wgrad correctness checks. BF16 measured 256.166 ms. Sequential NVFP4 measured 80.814 ms (23.711 forward, 23.518 dgrad, 33.585 wgrad; 3.170x), while deferred batched wgrad reduced the total to 76.032 ms (3.369x). Both fail the absolute 63.463 ms break-even bound; the deferred result is still only an optimistic lower bound because it excludes every packing/epilogue cost and changes gradient lifetime. This closes the all-pass library-kernel route independently of the much slower complete-operator result below.

### Complete projection operator result

The complete scratch gate preserved the standard training contract (resident FP32 masters/moments, BF16 working tensors), refreshed both weight orientations, quantized row/transposed activation and output-gradient layouts, ran forward/dgrad/wgrad, and counted postscale, BF16 output, bias, and dbias. Immutable text-context packs were charged once across 48 blocks.

| Metric | Result | Gate |
|---|---:|---:|
| Paired bare BF16 GEMMs | 250.556930 ms | reference |
| Complete BF16 GEMMs + bias/dbias | 292.985852 ms | reference |
| Complete QuACK NVFP4 | 2,072.035073 ms | <=80.950 ms same-run / <=63.463 ms normalized |
| Ratio-normalized complete time | 1,624.428915 ms | <=63.463 ms |
| Speedup vs bare BF16 | 0.120923x | >=3.095x |

This is a hard reject for the current separate quantize/transpose/pack/epilogue path. It is about 8.27x slower than paired BF16, before autograd dispatch or distributed overlap.

## Current-head follow-up gates

### Repeated no-autocast RMSNorm

The repeated 4x A/X/B gate measured 418.399114 / 415.219851 / 413.452594 ms. The 415.925854 ms control midpoint leaves a 0.706003 ms / 0.1697% candidate saving inside 1.189% control drift. Native unweighted and weighted semantics were exact, BF16 output was preserved, the refine path still delegated, and the learned weight remained effective. Reject the source change as neutral.

### Grouped BF16 weight gradients

Exact packed LTX-2 shapes across all 48 blocks were bit-identical (`max_abs=0`), but grouping adjacent wgrad GEMMs cannot clear the 10 ms integration gate. Groups of 2/4 saved only 2.512/2.073 ms under the optimistic prepacked arithmetic ceiling; staging separate inputs regressed 11.199/11.436 ms and staging plus gradient scatter regressed 19.859/20.566 ms. This already excludes autograd hooks, buffer lifetime, and likely lost FSDP reduce-scatter overlap. Reject.

### Short-text attention dispatch

FA4 remains faster for LTX-2's short text attention: at B1 it measured 0.416 ms versus 0.809 ms FA2 and 0.852 ms SDPA; at B3 it measured 0.825 ms versus 2.341 ms FA2 and 2.468 ms SDPA. All paths were finite and passed the declared parity checks. Reject a short-text FA2/SDPA hybrid dispatch.

### QuACK 0.6.1 fused norms

The QuACK fused AdaLN and joint QK-normalization experiment regressed the projected 48-block path by 30.274514 ms on tray 0 and 31.686422 ms on tray 1. It also provided only loose BF16 parity (AdaLN output 64.3% exact, max absolute 0.03125; scale-gradient 49.7% exact, max absolute 1.0). Reject the dependency and integration.

### Fused clipping through AdamW

Passing the reciprocal clip coefficient to fused AdamW preserved the existing distributed norm and deferred logging, supported role-mapped DMD2 optimizers, and was bit-exact for parameters, moments, and step tensors on both trays. Timing did not replicate: tray 0 saved 3.263339 ms, tray 1 regressed 0.299465 ms, and the equal-tray aggregate saved only 1.481937 ms / 0.360336% (+0.127012 MFU points). The signs conflict, so reject it as neutral; fused AdamW's scaled-gradient writeback plausibly absorbs the removed foreach pass.

### Corrected current-head fixed arena

The 4x A/X/B source controls measured 416.150371 and 410.795182 ms, for a 413.472777 ms midpoint with 1.295174% drift. The fixed-arena candidate measured **400.616610 ms**, saving **12.856167 ms / 3.109314%** and moving MFU **34.935119% -> 36.054708% (+1.119589 points)**. It keeps FP32 master weights and Adam moments with a BF16 working parameter/gradient arena, covers all 927 packed parameter objects, reports zero sampled replica error, and measures a 0.015617 maximum master-precision delta. Peak allocation rises **100.832486 -> 136.661506 GiB/rank (+35.829020 GiB)**.

This is a real architecture result, not production source. The scratch runtime is limited to world size 4, accumulation 1, and LTX-specific bucket/layer ordering, and it lacks DCP/export, EMA, and generic optimizer/checkpoint lifecycle support. Keep it as the next systems architecture reference; do not fold it into PR #1630 without those gates.

### Purpose-built Triton norms

The synthetic wall projection appeared to save 2.244263 ms, but GPU kernel sums show the candidate is slower than the current compiled path: AdaLN 0.192063 versus 0.081024 ms and joint QK norm 0.353639 versus 0.157087 ms. AdaLN parity was also loose (57.4% exact output, max absolute 0.0625; scale-gradient max absolute 1.0). The wall signal is a host/autograd artifact, so reject the kernels and claim no MFU credit.

### Public FSDP2 boundary grouping

A root-only public `fully_shard` boundary is a hard reject. Its 4x A/X/B controls measured 415.065887 and 410.919773 ms, or a 412.992830 ms midpoint, while the root-only candidate measured 466.765127 ms. That is a **53.772297 ms / 13.02% regression**, moving MFU from about **34.975% to 30.945%**. Peak allocation rose by about 23.8 GiB/rank and peak reservation by about 56 GiB/rank because the entire transformer was materialized as one unit and communication overlap was lost.

Grouping consecutive block modules through PyTorch's public FSDP2 list API retains per-block decoration while reducing collective launches. The exploratory A/G2/G4/B gate chose **two modules per group** because it was faster and used less memory than four. Both final four-GB200 A/X/B gates used exact clean head `20c36acef`:

| Exact-head gate | Control A | Group-2 candidate | Control B | Control midpoint | Matched result |
|---|---:|---:|---:|---:|---:|
| optimized, no checkpointing | 414.894377 ms | **404.962008 ms / 35.667827% MFU** | 410.968906 ms | 412.931642 ms | **-7.969633 ms / -1.930%; about +0.688 pp** |
| committed recipe, full checkpointing | 777.312431 ms | **753.134414 ms / 19.178668% MFU** | 776.550259 ms | 776.931345 ms | **-23.796931 ms / -3.063%; about +0.587 pp** |

The optimized candidate added 0.500 GiB peak allocation and 2.063 GiB reservation. The full-checkpoint recipe added no peak allocation and 3.719 GiB reservation; its controls differed by only 0.098%. Both gates completed with finite losses and gradients while proving FP32 registered weights and Adam moments. The proof covers all 13,041,520,768 parameter elements, retains 48 decorated blocks in 24 block groups plus the 152,096,896-element root group, and uses BF16 working parameters, gradients, and reductions.

An exact 8x/two-tray B3 A/X/B on replacement allocation `1629519` measured 1.418004981/1.421743279/1.421487014 seconds for control A/group 2/control B. Against the 1.419745998-second / 30.521241%-MFU control midpoint, group 2 is neutral at +1.997281 ms / +0.140679% and -0.042922 MFU points, smaller than the 0.245257% control drift. The native eight-rank health gate sustained 463.195 GB/s derived bus bandwidth with MNNVL/NVLS enabled and no NCCL errors. All paths completed 30 steps with finite gradients and FP32 registered weights/moments. Absolute performance on this replacement pair was about 30.5% MFU, so it is not compared across allocations with the earlier 43.623053% pair; only the same-pair bracket is attribution evidence.

Current source exposes the generic positive `training.distributed.fsdp_modules_per_group` setting and selects `2` in the LTX-2 overfit recipe. Before the full timing gate, that exact recipe also passed a two-step smoke with full activation checkpointing. Smoke log SHA-256: `66cb9fe7d6b95d22319ea772dafa54cde434095fa6d02c184a46b474a69056ae`; runner SHA-256: `4ccb2cc9c163357180c673daf46213be02172a4a01fb6048489d8dd937d1b593`.

### CUTLASS exact-shape BF16 GEMMs

The packed LTX-2 projection mix contains five unique shapes and 15 forward/dgrad/wgrad cases, weighted to 1,008 GEMM calls and 300.095807 TFLOP per optimizer step. A cold-cache A/X/B microgate measured 280.932863 ms for current kernels, 212.677120 ms for the ATen+CUTLASS candidate, and 288.790019 ms for the closing current-kernel control. Against the **284.861441 ms** midpoint, the candidate saves **72.184321 ms / 25.3402%** and raises effective projection throughput from 1.053480 to 1.411039 PFLOP/s.

All 15 parity checks passed. Runtime proof found actual CUTLASS kernels in 8 of the 15 shape/phase cases, with the remaining winners using current NVJet kernels. The one-time cold compilation cost was 948.13 seconds; it is excluded from steady-state timing and requires persistent compiler caching in production.

The unrestricted end-to-end attempt is a hard safety reject, not a timing result. Four independent rank-local caches emitted 8,757 illegal-memory-access warnings, 105 CUTLASS errors, and 343 failed autotune choices, then produced no `BF16_RESULT`. One directly inspected FFN-up candidate (`128x128x64_0x0x1_0_tnt_align8_stream_k_2sm_epi_tma`, M/N/K 4290/16384/4096) returned CUTLASS `Error Internal` during initialization; an inspected FFN-down candidate (`128x256x64_2x2x1_0_tnt_align8_2sm_epi_tma`, 4290/4096/16384) was rejected after illegal-memory-access warnings. The failed log SHA-256 is `d6a7ae86e66fb36a3b9df1d191ab82d12c8d78fe7853f8ae7fb4c20e2660f99d`; no generic CUTLASS config or source staging is retained.

The constrained trainer gate also rejects the apparent opportunity. At B2, the current control completed at 720.135417 ms, while the exact-name candidate completed at 735.566620 ms: **+15.431203 ms / +2.142820%**, throughput -2.097866%, and corrected MFU 40.114997% -> 39.273438%. It emitted 7,330 illegal-access warnings, 18 failed-choice warnings, and four compiled-cache move tracebacks. The allowlist did not constrain generated backward layouts: logs selected unrequested `ntt` and `ttt` variants. This makes a regex-only production solution invalid even before the regression. Candidate log SHA-256: `0a12bc6ef750fab9b2c2fb43aad003f3be5c9a384a59246d3d2e44771087a8ce`.

A final direct B2-shape A/X/B microgate set TorchInductor's profiling swizzles to the singleton `[4]`, removed the optional epilogue suffix, and covered all 15 forward/dgrad/wgrad cases. It was clean: all parity checks passed and the candidate log contained zero illegal-access, failed-choice, traceback, RuntimeError, or CUDA-error strings. It was also decisively slower. Controls measured 393.934851 and 389.372927 ms, a 391.653889 ms midpoint; the candidate measured **422.406662 ms**, regressing **30.752773 ms / 7.852028%**. Only the small text projections won; the dominant video/FFN cases lost. Comparison/candidate log SHA-256 values are `7424fe05b7978d341c39e3f5a18f06a6c93f6a572a221f6b7479426cb66e7daf` and `23dae03bd6ae6405b40c3b55e07497ab355f1b464682ecf730fb616dc9f993b0`. This closes TorchInductor CUTLASS for B2; no source/config staging remains.

### Post-measurement validation-fix re-gate

Tracker head `52f1114dd` (validation compile isolation `0e60a0e9c`, video-only validation memory safety `3f3f06541`, and the tracker snapshot itself) was re-gated against measured head `20c36acef` with a 4x/B2 A/X/B sandwich on fresh allocation `1627918` (gb-nvl-118-compute02, 4x GB200, 1200 W power limit, 2062 MHz max SM clock, torch 2.12.0+cu130, CUDA 13.0, flash-attn-4 `4.0.0b20.dev2+g82d6441`). All three runs used the committed `runners/run_current.sh` and `harness/benchmark_fastvideo_train_pack_d016.py` from one container session with shared FA4/Inductor caches; only the `/mnt/FastVideo` source SHA changed between runs.

| 4x/B2 run | True-wall slowest-rank median | MFU | Matched result |
|---|---:|---:|---:|
| control A `20c36acef` | 0.721040 s | 40.064684% | control |
| candidate `52f1114dd` | 0.722525 s | 39.982318% | +1.326267 ms / +0.183898% vs midpoint |
| control B `20c36acef` | 0.721358 s | 40.047009% | control |
| control midpoint | 0.721199 s | 40.055847% | 0.044125% drift |

The candidate is within +0.184% of the control midpoint with byte-identical peak memory to control B (140.874/166.588 GiB allocated/reserved; control A measured 140.865/166.564 GiB on its cold-compile run) and no training-path mechanism: both post-measurement commits touch only validation code, and this harness removes the validation callback before training. All three runs passed the full embedded proofs — 927 FP32 DTensor parameters / 13,041,520,768 elements with FP32 Adam moments and exact optimizer coverage, finite AdaLN gradient probes and losses on every rank, and 4,290 semantic tokens with singleton model timesteps on all 30 steps. Load clocks/power were healthy (about 1.4-1.8 GHz SM, 0.91-1.17 kW against the 1.2 kW cap, no throttle reasons). Decision: current head inherits the stopping-point table as timing-neutral. The allocation-local absolute result (about 40.0-40.1% MFU at B2) is consistent with a healthy 4x tray and is not compared across allocations.

### MFU formula audit

A head-of-branch audit reverse-engineered and empirically verified the published MFU chain. The harness formula `MFU% = 14.444115 * local_batch * grad_accum / median_slowest_rank_step_sec` reproduces every published row from its recorded medians, and the README's `353.8808175 TFLOP/sample` is exactly `14.444115 * 24.5`: the harness constant is the primary value, rounded at the sixth decimal from the exact count below (relative rounding error 4e-9).

The numerator is the strict no-recompute model-FLOP count of the 48 transformer blocks at 4,290 video tokens, 1,024 text tokens, and hidden width 4,096:

| Component | Exact FLOPs/sample |
|---|---:|
| video-token block linears, `6*4290*234,881,024*48` | 290,200,202,772,480 |
| text KV projections, `6*1024*33,554,432*48` | 9,895,604,649,984 |
| self-attention, `12*4290^2*4096*48` | 43,420,719,513,600 |
| cross-attention, `12*4290*1024*4096*48` | 10,364,292,956,160 |
| total (353.880820 TFLOP) | 353,880,819,892,224 |

The first two rows sum to 300,095,807,422,464, the projection-gate figure above. The convention charges attention backward at twice forward (no flash recompute) and excludes the caption projection, patchifier, output head, and AdaLN MLP.

`probes/audit_train_flops_per_sample.py` then measured executed FLOPs with `FlopCounterMode` on allocation `1627918` at head `52f1114dd`, running the production recipe eagerly (TORCH_SDPA, compile off — compiled regions and the FA4 custom op bypass dispatch-mode counting; collectives are uncounted either way). At B1 it recorded 118,036,079,050,752 forward + 244,999,614,103,552 backward = 363,035,693,154,304 FLOPs/sample, identical on all four ranks and on both counted steps, and reconciled with the numerator integer-exactly in forward and backward separately: the counter additionally charges the flash-attention backward QK recompute, `(2*4290^2 + 2*4290*1024)*4096*48 = 8,964,168,744,960`, plus the excluded non-block layers, `190,704,517,120` (caption MLP 3840->4096->4096 and AdaLN/patchifier/output head, with no-grad inputs skipping first-layer dgrads). The numerator is therefore confirmed and is 0.054% conservative. A B2 executed-FLOP repeat was capacity-blocked: eager mode allocated 177.80 GiB and OOMed (the first attempt also exposed a transient `NCCLSymmetricMemory.cu:455` init failure), but batch linearity is structural — every counted operator's FLOPs are linear in the batch dimension — and the B2 timing gates already verify per-sample token semantics.

The denominator is a house convention. The device reports 152 SMs (SM100) with a 2,062 MHz max SM clock at a 1,200 W limit; NVIDIA's GB200 NVL72 materials give 360 PFLOPS sparse FP16/BF16 across 72 GPUs, i.e. 2,500 TFLOP/s dense per GPU, and no vendor source quotes 2,450. Published MFU values are therefore 1.020408x their vendor-peak equivalents: 43.623053% reads 42.750592%, 40.810314% reads 39.994108%, and the head re-gate 39.982318% reads 39.182672% against 2,500 TFLOP/s. `nsys-ai`'s 2,250 TFLOP/s default matches the 1,000 W HGX B200 part, not these 1,200 W GB200s. Keep 2,450 for continuity with every published row — multiply reported MFU by 0.98 for the vendor-peak value — and relabel the baseline per the README rule if the convention is ever changed.

## BF16 systems campaign toward 50% (2026-07-22)

Scope: standard training conventions only (FP32 registered masters and Adam moments, BF16 working compute and reductions); quantization deferred. All 4x gates ran the committed packed harness at B2 on allocation `1631376` (gb-nvl-118-compute04, healthy class, 2,062 MHz max SM clock). Four interleaved controls measured 0.732045/0.732073/0.731982/0.729492 s (39.462/39.461/39.466/39.600% allocation-local MFU); each candidate below is judged against its bracketing control midpoint.

A fresh Kineto category rollup (`harness/profile_fastvideo_train_pack_head.py`, B2, rank 0) decomposed the step: about 391 ms of nvjet GEMMs (band efficiency about 62% of the 2,450 convention), 133 ms FA4, 100.5 ms compiled pointwise/reduction kernels, 33.4 ms optimizer band (14.2 fused AdamW, 8.3 `_foreach_copy_`, 4.1 clip `_foreach_mul_`, 2.2 `_foreach_norm`), 4.6 ms of 203 eager `aten::sum` calls at compile-region boundaries (broadcast AdaLN/timestep gradient reductions), and mostly-overlapped FSDP staging (17.1 chunk_cat + 7.9 split_with_sizes + 25.8 PtoP). Eliminating the entire non-GEMM/attention main-stream band would land near the 50% target, which framed the gates below.

| 4x/B2 gate | Candidate median | vs bracket midpoint | Verdict |
|---|---:|---:|---|
| `TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1` | 0.727078 s / 39.732% | -4.981 ms / -0.680%; +0.270 pp (0.0039% drift) | **accept** |
| `CUBLASLT_WORKSPACE_SIZE=76800` alone | 0.732398 s / 39.443% | +0.370 ms / +0.051% | neutral alone |
| workspace + coordinate descent | 0.724103 s / 39.895% | -7.925 ms / -1.083%; +0.432 pp | **best stack; carry both envs** |
| cuDNN video self-attention (scratch swap) | 0.730204 s / 39.562% | -0.533 ms / -0.073% (0.341% drift) | neutral; reject source change |
| cuDNN + workspace + coordinate descent | 0.724667 s / 39.864% | -6.070 ms / -0.831% | no additive value over env stack |

The coordinate-descent candidate re-tunes Inductor pointwise/reduction configs only; the previously rejected `max-autotune-no-cudagraphs` gate additionally autotuned GEMMs into SM100 register-pressure fallbacks, so the two results do not conflict. The 2.98 ms spread between the two accepted-stack rows is within compile-lottery variance across coordinate-descent searches. Memory was unchanged in all rows.

Supporting microgates, all committed: `probes/benchmark_ltx2_video_attention_backends.py` measured cuDNN SDPA at the exact (B, 4290, 32, 128) fwd+bwd shape 2.31-3.45% faster per call than FA4 (about -3.2 ms/step projected) with FA2 and SDPA-flash 2-3x slower, but the trainer gate above shows the per-call win does not survive end to end; reject without source change. `probes/bench_ltx2_tunableop_gemm.py` reproduced the GEMM band at 416.5 ms single-GPU (validating the profile) and found `PYTORCH_TUNABLEOP` non-functional on this CUDA 13 build (tune pass writes no CSV; replay delta inside noise): reject. The same probe measured the 75 MB cuBLASLt workspace at -3.2% band in isolation, which motivated the workspace rows above; the trainer shows it is only additive under coordinate descent.

Batch capacity at 4x is rejected decisively. `harness/benchmark_fastvideo_train_blockskip.py` forces `block_skip` checkpointing with a stride (the LTX-2 wrapper does not plumb `n_layer`, so plain `block_skip` degenerates to full). B3 with every-6th-block checkpointing fits memory (160.9/179.9 GiB allocated/reserved) and passes all embedded proofs, but on the same tray whose B2 re-gate measured 0.7212 s / 40.06%, it ran 1.511623 s / 28.666% with the native allocator and 5.261937 s / 8.235% with `expandable_segments:True` — an allocator-pathology collapse at the reservation ceiling, not recompute cost (which models at about +2%). Do not pursue 4x/B3 or near-ceiling batch capacity without first freeing tens of GiB of real activation memory.

The 8x confirmation is blocked by rack `gb-nvl-057` infrastructure, diagnosed to root cause across two allocation pairs and recorded for the next attempt. Pair `1631584` (compute01/08, heterogeneous 4-vs-8 NIC trays) and pair `1631863` (compute01/04) both draw the rack's 1,965 MHz max-clock bin (baseline-ineligible against the 2,062 MHz rows even when working). Findings, in dependency order:

1. `/etc/hosts` on these hosts maps each host's own name to `127.0.1.1`, so Gloo full-mesh advertises loopback and peers get connection-refused. Fix: `GLOO_SOCKET_IFNAME=enP5p9s0`. Real, but not the main blocker.
2. Cross-tray TCP is healthy: torchrun static rendezvous, arbitrary fixed ports, ten sequential connections, and the c10d TCPStore phase (server on node0, both clients connected) all pass.
3. The actual hang: a `faulthandler` dump shows every rank blocked in its first `all_reduce` inside lazy NCCL comm creation, with zero NCCL output even at `NCCL_DEBUG=INFO` — a silent driver-level wait in the MNNVL/IMEX registration path. `nvidia-smi` reports fabric state Completed/Success but with sentinel `CliqueId 32766 (0x7ffe)` on both trays; NCCL's own detection line reads `MNNVL ... cliqueId 0x7ffe state 3 healthMask 0x11a9`.
4. Discriminator: `NCCL_MNNVL_ENABLE=0` makes the identical cross-tray all-reduce complete immediately over `Using network Socket`. Socket transport is diagnosis-only — a B3 step would be comm-bound and meaningless as an MFU row — so rack-057 pairs are unusable for the confirmation until IMEX is repaired.

Working preamble for two-tray pairs: `GLOO_SOCKET_IFNAME=enP5p9s0 NCCL_SOCKET_IFNAME=enP5p9s0`, a fresh `MASTER_PORT`, and `MASTER_ADDR` resolved to a raw IP (rank0's own `enP5p9s0` address on node 0; DNS resolution of the master hostname elsewhere) to stay immune to the hosts-file mapping.

A third allocation roll drew rack-059 pair `1632122` (gb-nvl-059-compute05/06), which passed the native health gate with that preamble: all-rank scalar correctness and 458.914 GB/s slowest-rank sustained all-reduce bus bandwidth, matching the healthy 463 GB/s reference, so the MNNVL failure is rack-057-specific. The pair is nevertheless the 1,965 MHz clock bin and its absolute B3 result sits in the degraded-allocation class alongside the `1629519` row: the exact-head A/X/B measured 1.442031/1.443601/1.444706 s for control A / env stack / control B — a 1.443369 s / 30.021699% MFU control midpoint with 0.185% drift, and an env-stack delta of **+0.232 ms / +0.016%, i.e. neutral** (memory unchanged, 161.38 GiB peak allocated in all rows). On a degraded pair the pointwise-tuning win measured at 4x is inside noise, so the environment stack remains accepted for 4x work, is harmless on 8x, and stays **out of `run_current.sh`** until a healthy-class (about 43% baseline) 8x pair re-gates it. This row is baseline-ineligible and does not supersede the stopping-point table.

## Decision boundary

1. Do not write a whole-transformer mega-kernel or integrate the current QuACK wrapper.
2. Keep production FSDP2 symmetric memory, repeated/prefetched input, deferred gradient-norm materialization, both scoped accumulation optimizations, singleton timestep embedding, persistent projection packing, and two-module public-FSDP2 grouping. The singleton path is a capacity optimization; `d01623709` remains the matched B1 source attribution. Keep the corrected fixed arena only as architecture evidence until it has generic lifecycle, DCP/export, EMA, accumulation, and topology coverage. Reject root-only sharding, raw velocity, TE BF16 registered shards, fused dense GELU, no-autocast RMSNorm, grouped BF16 wgrad, short-text hybrid attention, QuACK/Triton norms, AdamW-integrated clipping, and current TorchInductor CUTLASS schedules. No currently confirmed production-ready stack reaches 50%; treat 50% as research, not a PR #1630 completion gate.
3. If BF16-equivalent all-pass NVFP4 remains wanted, the next admissible experiment is a purpose-built projection pipeline that fuses dual-orientation quantization with scale packing and QMM alpha/bias epilogues, and overlaps cache production with GEMM execution. This is projection-band kernel work, not a monolithic transformer kernel.
4. Require <=63.463 ms for the complete 48-block equivalent and acceptable sampled fprop/dgrad/wgrad cosine/norm before autograd or distributed integration. The current result fails both gates.
5. Keep CUDA graph work separate and secondary until it produces a measured >=32 ms bare saving.

## Artifact provenance

Hashes were captured immediately after each run. Scratch paths explicitly marked `recorded` were later reused or cleaned, so those entries are immutable provenance records rather than claims that the old bytes still exist at that path.

```text
b427c267440ae30b99e32588e155b81bbd4c67f9ade67065d0d3eea24fc26abc  /tmp/pr1630_fsdp_root_control_a.log
8bf84e651bad50f559da45d2ed7c1f4df22c86c4e559b9e3569495ffda2ec65d  /tmp/pr1630_fsdp_root_candidate.log
b1e45a3dce91565a151762a3897203dad7eac240be5f30924734d4d2cfaae8b4  /tmp/pr1630_fsdp_root_control_b.log
6f7a4e5138e51a7bb1a0e2f319cf30eb714d5affd7da8c1470acbed925e738a8  /tmp/run_pr1630_fsdp_root_only_4x_aba.sh
276c57e3120a8cfbaa54bcb4951ea14892d6d41da39d813eb51000e882b679de  /tmp/pr1630_fsdp_buckets_control_a_fa47ce1.log
4b1e0246ccfcd9c88ccf1270dbcc6b0092fe3875069158a1ecbd193f27044182  /tmp/pr1630_fsdp_buckets_group2_fa47ce1.log
6f4cfb5c8a5cef467707d5a400ab049494eae7c12ba9931517d3e9cf735b7875  /tmp/pr1630_fsdp_buckets_group4_fa47ce1.log
2086997a99ed42f6235150437cdcfedf787f15d58fae13e8d37a2b0ff2511e09  /tmp/pr1630_fsdp_buckets_control_b_fa47ce1.log
0392370dbc0aec3bedc4b600184cd36cdd5e7c60043c19433f262e085a2d08ad  /tmp/benchmark_fastvideo_train_fsdp_buckets_fa47ce1.py
9babec2096727dd05fcc261c4d1589c80bc6ba0491d35cfa9fdbf6982ce92d04  /tmp/run_pr1630_fsdp_buckets_4x.sh
66cb9fe7d6b95d22319ea772dafa54cde434095fa6d02c184a46b474a69056ae  /mnt/pr1630_group2_recipe_smoke_20c36.log
4ccb2cc9c163357180c673daf46213be02172a4a01fb6048489d8dd937d1b593  /tmp/run_pr1630_group2_recipe_smoke_20c36.sh
93f0147b706ba9ea21887d4f6d8ebf684b81ec368628247cb35aa374ad1fc739  [exact-source optimized A/X/B runner] /tmp/run_pr1630_fsdp_group_source_4x_20c36.sh
9c0db1bc60feea788a8cadb069d674c56c58c588c3ea268c035751a55efcf064  /tmp/ltx2_bf16_cutlass_fa47ce1_valid/current_a.log
836da870f8356e0d126028906913988e3e612f204d238163da90dec1477c94fe  /tmp/ltx2_bf16_cutlass_fa47ce1_valid/cutlass.log
44578c630d98f339269e3138d77bca6f5612e3ba38feb0194b4f2914b8d8402f  /tmp/ltx2_bf16_cutlass_fa47ce1_valid/current_b.log
02c563717bdbea7249fc92a0141b6d755fd718627e3fd470bc7dd0df9014a192  /tmp/ltx2_bf16_cutlass_fa47ce1_valid/comparison.log
1864f0776c5929c75b95262a7273c81d07765a2dc64a1193eaad74bc058227ff  /tmp/benchmark_ltx2_bf16_cutlass_gemm_fa47ce1.py
1bdb44c617162728bb33c33019d4514c6b9598e7ca75fd6d21ffe24173c129b7  /tmp/run_ltx2_bf16_cutlass_gemm_gate_fa47ce1.sh
6a419a1e0c7cb35b64bd622852d3847935fd9728a4d14d8fd45cf952ec55fa99  /tmp/pr1630_cutlass_allowlist_b2_current_a_20c36.log
0a12bc6ef750fab9b2c2fb43aad003f3be5c9a384a59246d3d2e44771087a8ce  /tmp/pr1630_cutlass_allowlist_b2_allowlist_20c36.log
0f2ca616e34f40030df669a426fd0c207e70abb772c08ebee524a9b5307254b0  /tmp/pr1630_ltx2_b2_cutlass_swizzle4_20c36/current_a.log
23dae03bd6ae6405b40c3b55e07497ab355f1b464682ecf730fb616dc9f993b0  /tmp/pr1630_ltx2_b2_cutlass_swizzle4_20c36/cutlass.log
19f0b55e97316dfa7661e6bee17aade8afc87da3cc97f749f06d8162f02f1a7b  /tmp/pr1630_ltx2_b2_cutlass_swizzle4_20c36/current_b.log
7424fe05b7978d341c39e3f5a18f06a6c93f6a572a221f6b7479426cb66e7daf  /tmp/pr1630_ltx2_b2_cutlass_swizzle4_20c36/comparison.log
4f937bc1ebc80dd2c07dadbecec84c7c7eb2d7da59aaedf52aad36f95e025add  /tmp/benchmark_ltx2_bf16_cutlass_gemm_fa47ce1.py
4fe47986aad94b5cf042a82c037ad854f69ec5d2951207ba80cd7bf83fbd59ef  /tmp/run_ltx2_bf16_cutlass_gemm_gate_fa47ce1.sh
ff56f6ddf8e322e4f15d242c0367982b8745118470310f6bf4c9e54bb7693156  /tmp/pr1630_rmsnorm_repeat2_control_a.log
9640fd71cf69a022a7f4e32b00b6a74203d9567747291340a776c0a9a31bc2af  /tmp/pr1630_rmsnorm_repeat2_candidate.log
8ec42a9a064a01377cde9e22dbd7a44712729ee13676fcc486ccfd21de16b2c0  /tmp/pr1630_rmsnorm_repeat2_control_b.log
0f76a190a2496aac31bd26efba14957fef41910a361c166ee7c8ddb37db6ed90  [recorded before scratch-path reuse] /tmp/benchmark_fastvideo_train.py
3a1db6f243e72af1b7bd276f8b571351a073d8cdd5ddaaf8476e2332a0061eba  /tmp/run_pr1630_rmsnorm_no_autocast_4x_aba.sh
1eea3f150c18b8eddc9019edd957cd1e522af0d41c8678a537d1959a625289a2  /private/tmp/bench_ltx2_bf16_grouped_wgrad.py
874c17cdfd805ca1f2e1c2e3f9df80c00aa433a215269135fb4d7814d0abeed7  /private/tmp/run_ltx2_bf16_grouped_wgrad.sh
731a0f72b031d8768b199a6cd9990192d7e7dd5ae07a5238b8be5e3da21ea631  /private/tmp/bench_ltx2_bf16_grouped_wgrad_job1623561.jsonl
8f2615727cc6e6d73921bb8e0891e9564ef0e5d259e1136123da4339c90ae249  /tmp/pr1630_ltx2_text_attention_gate.log
40489af84ec0b8232d9b97705078243a9f36aee791595ffe3c02023f205e1a9e  /tmp/benchmark_ltx2_text_attention_backends.py
74e2d30e1ec505545c5ae4192c6a8e572a1426d99d8c6aaf0fe39f2fb77a23ac  /tmp/run_ltx2_text_attention_gate.sh
d5660696c77170c0fbcc61e5a9539aab1bcbb95d4ae32ad5b30bd342d81bdf7d  /tmp/pr1630_ltx2_quack_norm_gate_node0.log
e5af3d5f9e89c566a740b0538d53afde99c95c826b63df0705fa497b7b4a0681  /tmp/pr1630_ltx2_quack_norm_gate_node1.log
fc43cbb03c67f87e20c216d1f3dfb5fcfcea0a8b0adea3ba70dc7abb095eb4f6  /tmp/benchmark_ltx2_quack_norm_gate_fa47ce1.py
f5da0421aa8c9c81adbb36c6639926f919046444f905ecd672703ba0ac404ec9  /tmp/run_ltx2_quack_norm_gate.sh
12ccb47c662016184e9959dc9f20d7163bc06f4e3382b9f4d3d32acacc7930b0  /tmp/benchmark_fastvideo_train_fused_clip_fa47ce1.py
2dda641fb5f97fd62c695c678a7b608665c09c036de3cd36ade4a363f90644c6  /tmp/run_pr1630_fused_clip_dual_4x_aba.sh
6bae2ddb903abb13816ef79ef1819c2788d31073063832c0f745cf5c2d8a7347  /tmp/pr1630_fused_clip_parity_node0.latest.log
8187717e6f30b0d1e931175543d62818bfeab3b571b2a5de927a5ae77d2abb54  /tmp/pr1630_fused_clip_control_a_node0.partial.log
1075beb1bbc33a127a915be1067c14fda8603c2ff81b4c22a123e9c29ddc5d3b  /tmp/pr1630_fused_clip_candidate_node0.latest.log
0b7c61091024cb62dfee3b4cc1acd6c7c98aab6cd39c4253d28b2fd23f4bd295  /tmp/pr1630_fused_clip_control_b_node0.latest.log
e85b49fa8121766ef1debc57f9455b967e9cd6f6efecd8accd5fc7f195882443  /tmp/pr1630_fused_clip_parity_node1.latest.log
e15a993d043fb90367e6ebc577da961fbf6fd90c3c9296a51a78859fbac9905e  /tmp/pr1630_fused_clip_control_a_node1.partial.log
d30eb902f93b01f4868dd45ae5449bcd4c7ebbd6ce0d4e7f6ad434664be4ff90  /tmp/pr1630_fused_clip_candidate_node1.latest.log
4becc78eff103866f5affdb467cbfe7bf9e4f786adba031dc4e62d8c18eb75c7  /tmp/pr1630_fused_clip_control_b_node1.latest.log
7b5852a45482e58df9088015d173f7d3fc049ec80fcaa65c89480acfa8859f10  /tmp/pr1630_fa47_fixed_arena_control_a.log
c27069a696d57a46cc3374574bf0d563f8f9387161498858935265104c1f3ce2  /tmp/pr1630_fa47_fixed_arena_candidate.log
3caa215e0ad0884965da38fc4c8a71e313d19aaf02f62c0e94a07126c4a62659  /tmp/pr1630_fa47_fixed_arena_control_b.log
46e6a123ad2980fe0ef70f178015b6c9a75abc2eab6d9fab0831606554c46127  /tmp/pr1630_ltx2_triton_norm_gate.log
08e113c7aa52ed13a5bfba07a9de3394d4a7abee1f4255ad3781ea3ec23ddfb5  /tmp/benchmark_ltx2_triton_norm_gate_fa47ce1.py
7c76497e07a0c2a39acf2bd223acf74885d20e0cd40b227b5289746181d8dcdc  /tmp/run_ltx2_triton_norm_gate_fa47ce1.sh
4c69503260a9e2351037161e32138fc6d6d7e133993ea8f1edbfb62866bfee3d  /tmp/zero2_ltx2_nccl_base_a.log
879d43933d958ffae3f3e1e5b94ba40a0f4bdf9448c217e0af0c4e00f869e8ae  /tmp/zero2_ltx2_phase_repeat32.log
76e4eb26f4bbdbd173b884b379b78d10ae89a93f54119cc2db9e21edc654a49c  /tmp/zero2_ltx2_nccl_cta16.log
375eac7aa12965034ca1629415b7e605bdda100b10aabb954ee985b3dfd80ab4  /tmp/bench_quack_sm100_nvfp4_v050_config5.jsonl
8b6b843d32476cbb6afc6150ebfbd2bc39dc7b8beaf9ffb71216035cdd55e963  /tmp/bench_quack_sm100_nvfp4_v050_config7.jsonl
3cdd55ebd363438f0bc5af8650eab7f12f05da6baba65aa5af1a7ee27cf0b961  /tmp/bench_quack_sm100_nvfp4_v050_config6.jsonl
2f901b87a1e227bf9b1a71b5b9945aaebb15420a1ec4f0412f1b22f28c1e7efb  /tmp/bench_quack_sm100_nvfp4_v050_smoke.log
61577ec757260cb54d6df8260c8e633837d7c141e450966ab559573ca762c492  /tmp/bench_quack_sm100_nvfp4_quack050.py
a89a44b0fa17c2741817d43bb77bac2fcb43c221eea5b1e5591afe1b03b54f6a  /tmp/zero2_ltx2_graph_staged_minimal.log
9037cfd2050f78a53967fa6f530ede2f0e1ebb4ed5d09cfa1e7d781740186d8c  /tmp/zero2_ltx2_graph_patchifier_minimal.log
0aa302e5cdef20f543897ba767077e7ffbb97fccbaaa77eb77e0749cd8bd61fe  /tmp/bf16_fsdp_symm_base_a.log
907d9526fb4b6c8e48aac3ac9e42920d5dffc06e00656aa32ffc855a52399fdc  /tmp/bf16_fsdp_symm_candidate.log
7883956935bec2494372b4fd044042c32da901e21e3e35349bf7fb495c887eed  /tmp/bf16_fsdp_symm_base_b.log
86209d328970e05a9da7849a6e8e9080015d7c20488d1cc0614e0b16b96ec3f1  /tmp/benchmark_fastvideo_train.py
3499dbf7080426d33bb13a7121fd45e83326eadddbc545769418200890b65f6e  /tmp/benchmark_fastvideo_train_symm_mem.py
5fcd10ba65ebfdf7c2d37a2b4411a99c464ad9e39fe09444f44fd840696d1784  /tmp/bench_quack_sm100_nvfp4_complete_projection.jsonl
61e5fa6ed8aa26ed15c825cf9c7ad5aa4f34151d0803766ada7fa87866f2c258  /tmp/bench_quack_sm100_nvfp4_complete_projection_smoke.jsonl
56bae046d674a7c1f4396f117f6faca83241377c850b4e409b8fe01cb15425d5  /tmp/bench_quack_sm100_nvfp4_complete_projection.py
cfd2f8621a2cc04120caf78434087b1ac345e8357b979d3df34443fcc88554d9  /tmp/bench_quack_sm100_nvfp4_complete_projection_DESIGN.md
aa260691ce2847aa1e5cf5c852fec8c2af67de306b4bf393557bdbc3a4477d54  /tmp/pr1630_mfu_1x_fa4_no_vae.log
b8a6444936ae98641dd9bf947dc2fd513faa5650d4eedc4137b78bdbda8eb43a  /tmp/pr1630_mfu_4x_full_shard_symm_production.log
0c4850071806aaaec637aee0a3201cfc21cb8bf44d195da775ff8c17c2ab9a52  /tmp/pr1630_mfu_4x_no_reshard_symm_production.log
5be182649bdbd78c6e49e862422fa11968fec7ddc2112de91e24834f80cc35fa  /tmp/pr1630_mfu_8x_full_shard_node0.log
fc7c9c6e6783281aa372d17f94fb48bff005d1bcffbceed79ae5288657cf752d  /tmp/pr1630_mfu_8x_full_shard_symm_production_node0.log
715fcb9385bbcc81ba605508828e4114bc159d82d31c3001afcb9f633353b31e  /tmp/pr1630_mfu_8x_full_shard_cta2_b_node0.log
218902dc7568fd690bfd1cc43e5f5c02667d5fde37cfba72d815736d26420d5a  /tmp/pr1630_mfu_8x_no_reshard_cta2_a_node0.log
a81f1899a00d02ee5861322649e4702981f0592ed555bc17866a1e7f9b7aef61  /tmp/pr1630_mfu_8x_no_reshard_symm_production_node0.log
f3c368eef644dfd0d806dfb8e4293c8ab3d3bb139939e03539343be1b9300592  /tmp/pr1630_mfu_8x_no_reshard_cta2_b_node0.log
f217ea1756a281b11c4be8b1b8d17a37aa6e3d43160ef8c94b54fb4f0d7261c0  /tmp/pr1630_nvlink_intra_inter_node0.log
a11d544802989fa716f2c44ffa288af3255e5b0ae0467f044dd745ab85b21b60  /tmp/bench_nvlink_intra_inter.py
2b249d4ac775494a08aa704b51bbad7b41f3fd586a6bdf774143188b7bc7ef23  /tmp/pr1630_fsdp_knobs_base_a.log
07b5e0c13c8800de85c978d1395e3ce2e52a818747cf9187d9f35cf98978da36  /tmp/pr1630_fsdp_knobs_base_b.log
c93dda310617d44464dd3aa5efa9792a81d7bb679b3fc4edc25d0f7ae84bc7ab  /tmp/pr1630_fsdp_prefetch1.log
5844e6838734032785c7df4dadfacf92fcc78d884f0aff146850dd23420713d2  /tmp/pr1630_fsdp_prefetch2.log
14535b726719c1a4a8d393237c260fbc33d98eb3aaa5e8996d265b3bc2192224  /tmp/pr1630_fsdp_pg_alloc_alt.log
8eead593a61f9f74e09592aea7cef90564b7cd9596d41839e74e529709f3408b  /tmp/pr1630_hsdp_base_a_node0.log
21d43573ad3224f9d3426368e677af50fbda8d6573ac355fdb6165ea18077aeb  /tmp/pr1630_hsdp_2x4_node0.log
dc1cd934099e7ee72bed564213a1bcf415343d8183aa0ce3c5a4851ee9a964cb  /tmp/pr1630_accum2_no_sync.log
caad7d10cbb211ca534d4120621ad2e39b85cb95935ab5bdb903a5473f8eca98  /tmp/pr1630_accum2_force_sync.log
4d86bb7ec6600c9925d9223129f2c15955e579a817e1c97624a9ae5e5aad4b4b  /tmp/pr1630_accum2_no_sync_b.log
abde9be2305a70f9787f7bceec18fd2b197ad0f778bed1634ceb3e310aca8a76  [recorded before scratch-path reuse] /tmp/benchmark_fastvideo_train_fsdp_knobs.py
5e1ffa7fbbeeff10f20814f2315e4175df79d76f2a97665ca62553af235ec8e0  /tmp/bf16_input_control_maxrank2_7f139e2b.log
05f753036f10f6475c21ac7ffb2deb6c8374e0028ddc2f9c3413f85c63368405  /tmp/bf16_input_repeat32_maxrank_7f139e2b.log
b254de8912d9556c2b28040593216ae3cf5b6e496b8d2602482090b6a35df303  /tmp/pr1630_mesh_base_a.log
3d747e320d05995b0a30fd49b7ac89b1a611e3b074cc6531c26af6b75a20c41c  /tmp/pr1630_mesh_1d.log
da372dae90eb840709853e7601bc8a4fe04291872a470b370cb3f5d6f1b2f5af  /tmp/pr1630_mesh_base_b.log
31baeaec642fe690c564a337a87e145339a3c3cfd7a8053a26d6c7fefa59cde4  /tmp/run_fsdp_mesh_benchmark.py
230e0a749be30c7c23cac5fd8c76782c32ed3bcff282b2647d3dedaa6c193bb5  /tmp/run_fsdp_2d_control_benchmark.py
37b17e37d7ae2491ffa36ee382957664fbd51806b44a34cb35f553425c4a6360  /tmp/test_fsdp_2d_to_1d_dcp.py
cfaa897a6dcce741a72ffe826992c773323e791d83ca7d40c0f5f7b3db13649d  /tmp/pr1630_mesh_dcp_smoke.log
5917084f5745e8c3d94ff7c3449ab4000c59657f347eaa3ad7bd6245427113e6  /tmp/bench_quack_sm100_nvfp4_allpass_job1622676_config7.jsonl
3286866342ef63334dc26cabfa57ea393e3888da272ba4fed7268674a56441b1  /tmp/bench_quack_sm100_nvfp4_allpass.py
8a523326e703310f36219f5fedb108d009c2887b149e135a332645e6e06daad0  /tmp/pr1630_gradnorm4_fa4_base_a.log
63ad8b88c8e2e434087e95e126217e91333fe047ae65bbd31230034fe371f0a4  /tmp/pr1630_gradnorm4_fa4_deferred.log
49fc52400bcb65485888b5dc159644ba18fc6e8eb5b785a92c070ab226aaff28  /tmp/pr1630_gradnorm4_fa4_base_b.log
87347668ad28ab200bf3540da01ff1c7744854c2ce2a3a0a7d1c0cb1d1788d01  /tmp/pr1630_gradnorm4_source_deferred.log
683f6716b8912e26c974286228404f1d4a9d7e1e458654ee10068697c1e0b681  [recorded before scratch-path reuse] /tmp/benchmark_fastvideo_train_grad_norm_sync.py
05836dda4426309f23e1c797281f0e69cc343c522006679fef6b34e6c7d3f865  /tmp/check_ltx2_singleton_timestep_parity.py
efe7e529860771bc3564adf82b2cfa2a712dc60cf83691b174769aff11760317  /tmp/singleton_gate_7f6.0.log
b3a317fc18813c34aca7b5d0998f8e5a5b5e85d94fe707720da8c47b0a614752  /tmp/benchmark_fastvideo_train_ltx2_singleton_timestep.py
983b52f96ceace4ddd1fa01c36f6b851c9bb74fd6c4033e8932bc99abf9f1c3a  /tmp/pr1630_singleton_control_a.log
189f0ffc65ff4ae2a449990c026a385a79222e45e95ade94977bbb981c23f70a  /tmp/pr1630_singleton_candidate.log
505d0062845b2c1662384ab7f5e60acdebd4e6971ac66bcb521b7ab0b2bb41f5  /tmp/pr1630_singleton_control_b.log
f6924a9277723fe51c2d210f5ea3fb66aafbb845d142ee50ef92effd1cb5fa4d  /tmp/check_ltx2_raw_velocity_roundtrip.py
ee3b1ae78578e0226082230f566e244c6aa7c9063439218a47af0542d1211d22  /tmp/benchmark_fastvideo_train_ltx2_raw_velocity.py
83ebfafb10c4c9f10011b5cf746b7c6fdde6414d8247774ade8013f4a57a2f78  /tmp/pr1630_raw_velocity_control_a.log
b421b2e0bcb1c9b9ac8f4871f3ad41587249a1f520faed6795fc0b10319c0c45  /tmp/pr1630_raw_velocity_candidate.log
6a1875640a2528a8bd126a9037138f03d34a06bf188e1fc10e06005c26cab514  /tmp/pr1630_raw_velocity_control_b.log
c14ff07a8eb9785ae59f789fef1da9cfeb830ec29afc4e41936dc932f5637dbd  /tmp/run_pr1630_batch2_aba.sh
521853c83e20eb99fe2ad459d4ee8a180e34075e49c387918203a952844c55c8  /tmp/pr1630_batch2_control_a.log
bed7081a443163326ec6401d8c8a085af4069f6a47bad34ac66882856aec9e54  /tmp/pr1630_batch2_candidate.log
d867ca9be8ed9f1a72fd2c55dee8e48d60a2a0a71dff346f27bbb456930219d3  /tmp/pr1630_batch2_control_b.log
e8c710ecc8bc247051395cd10b69f895c0ab589755d82ea616122e50a1e12962  [recorded; scratch runner cleaned] /tmp/run_pr1630_batch2_8x_aba.sh
e5b52d00895f097e88bf5f71fc890f6709a935b8b50abfeeccf7c41f3c0cf285  /tmp/pr1630_batch2_8x_control_a_node0.log
5c825d2a9dad33566669ec3324b4499af54be6e275c3abd30692ebbcec61632e  /tmp/pr1630_batch2_8x_control_a_node1.log
3cc50b482a9a1ec1fd5a9ed405e6abab8c34608b90958f4166ef4886a67e824c  /tmp/pr1630_batch2_8x_candidate_node0.log
073550a2f6109acdaa94c72a1f9a53bcbaee4468957d65cd36de539d45baed9b  /tmp/pr1630_batch2_8x_candidate_node1.log
602656ed804d6c453d2e3a16f30d613f5569534d4dd984e2354f039b18962050  /tmp/pr1630_batch2_8x_control_b_node0.log
6f619a490cd0046c14a37a14d83f29161947df35dfea9bbb9471f7cfa7e7668a  /tmp/pr1630_batch2_8x_control_b_node1.log
15dc2f617c139c05fc8e8c45dfd6d06f3f3fecc6d6999b61ced78dddb9c1f44c  [recorded; scratch runner cleaned] /tmp/run_pr1630_batch3_8x_aba.sh
604804ad3139a6fbc345ba31b58c575ed82ad1d87ef55bfdeac0b90027103af3  /tmp/pr1630_batch3_8x_control_a_node0.log
48b8c8b7a65c906e7f043b9e0dbaebd829e7abdaaf85b67956145f687d0c9db6  /tmp/pr1630_batch3_8x_control_a_node1.log
117264799fba09c8e267e5f30bb8dc2a0f53e3618109e1d13c91e0347bcbb770  /tmp/pr1630_batch3_8x_candidate_node0.log
1d86886b79c59e309ea452b6e9c323a2b2a134468a8adb3962c71a323ae4a45b  /tmp/pr1630_batch3_8x_candidate_node1.log
049c822078ee57013bd1be3020970ea3bfdaf084be4e91afed8331fcd904a92d  /tmp/pr1630_batch3_8x_control_b_node0.log
c295dca59a641d5ea628f9f697d2db1cb3d55cc494356e2c2bda4d51ac0eeda2  /tmp/pr1630_batch3_8x_control_b_node1.log
cbb3fd5a95ab0ba2081c9fffd49767a9004a74770d1ccc0b5d83c57a12cbd1d1  /tmp/run_pr1630_batch4_full_shard_8x_aba.sh
4e21524538adc0fa7fde4cd117d6e09380cd2714b3a2e9e3fcda5ac81bc75a7e  /tmp/pr1630_batch4_full_shard_8x_control_a_node0.log
9af9a84e58b446723a585cf95c7433894b876d54450ff4b05b970c87cf97362d  /tmp/pr1630_batch4_full_shard_8x_control_a_node1.log
cfcf400c687750ef45bd84c24b8ef68b67b6e75d48aaa3024a58b794549d513d  /tmp/pr1630_batch4_full_shard_8x_candidate_node0.log
57ef2efb41b85b09dedf8801d705d42429f98309b3850b04392810aba814216d  /tmp/pr1630_batch4_full_shard_8x_candidate_node1.log
38f064416dfd68699fd58ee694dfe8d8b01a93c76d5d75b018b64ca9f6fd29cf  /tmp/benchmark_fastvideo_train_fsdp_knobs.py
ab97235175edfd90a8576920b3456414bedab07eb31dde9279cf37cb4b19fcf8  /tmp/run_pr1630_accum_reshard_aba.sh
a36c89911353f21398f0f927f0490d4dfbd4bda1adac48b38bb72f2a4cfd1e6b  /tmp/pr1630_accum_reshard_control_a.log
e5156b7d066912b9ad1489dcf0b0cb3146861121dcf39103b6a3ff6516a946c2  /tmp/pr1630_accum_reshard_candidate.log
e254c0ec799944f870e329c18313fb1eca8d2e1af85df3df86115eef7882a87d  /tmp/pr1630_accum_reshard_control_b.log
adacec37acc19f436f289bb2ea7ff184263a39255b0412cc7ed58a0769204c97  /tmp/pr1630_te_master_8x_control_a_node0.log
d819ab40b55a546d7649d430914950f1d738a3d5a50ddda2f8a266d27fdc6722  /tmp/pr1630_te_master_8x_control_a_node1.log
6c9dc52db52ef7ae7571ea1a0f02072aa44b9f283e0b7ee7a53fbda8e8b42923  /tmp/pr1630_te_master_8x_candidate_node0.log
cc2a138f8dfdfa84233e3e5e32f4855ad3a57ea72c7413a61bf50c7744c3ff82  /tmp/pr1630_te_master_8x_candidate_node1.log
4c8e91889e9eef21f938382b21e230b928e71c44d71da8685927bfd14bb1c35b  /tmp/pr1630_te_master_8x_control_b_node0.log
115a5d0e18d21683da40d3a1dbc87f11842cda23822999039067285c3507c980  /tmp/pr1630_te_master_8x_control_b_node1.log
bf0861ff481c499fd65350d2f0f16c86487db88cb03c332e4be5c76540fcc7fd  /private/tmp/benchmark_fastvideo_train_ltx2_singleton_timestep.py
63e702e2b258184ffa7ae668e4e53a8b4e7f9779a0a27b93a4ad045e27212b65  /private/tmp/run_pr1630_te_master_8x_aba.sh
eac2355c930ce7aae541b97d9bedb636e0b76250b30d2348f83bca6eabc3a2e1  /private/tmp/optimizer_te_master_scratch.py
949a31c8c78b05e7529d0cd81d312740edecc5686783bc8d986fbe19c7e45d35  /mnt/pr1630_pack_4x_control_a_d016.log
ad29304eaffbd56ef2053790d310f56c3acdfd75b89113136192fb8f2b62a93e  /mnt/pr1630_pack_4x_candidate_d016.log
f03b4578e1ac8931e002adcb5c00f9c1aa656cd4256138697935a1e795ead818  /mnt/pr1630_pack_4x_control_b_d016.log
bf0861ff481c499fd65350d2f0f16c86487db88cb03c332e4be5c76540fcc7fd  /mnt/benchmark_fastvideo_train_pack_d016.py
78d41f61e46710ff6d4c10e2758c37ff9aff08768ed211cdb2b4525ee303a295  /mnt/run_pr1630_pack_4x_aba.sh
d4857775e66717b9ec43434d4147f01bf16c0bbfb8f9e721aff3a30755376568  /mnt/pr1630_pack_8x_b3_control_a_d016_node0.log
0265f372cb18d07812fc0ff5dc08fefff5ba720fc89b159385981ea0843c0834  /mnt/pr1630_pack_8x_b3_control_a_d016_node1.log
5e1f03976d834465cd60072459f960f40b4d0ac4dfba7d24e7b4856f4e615bcc  /mnt/pr1630_pack_8x_b3_candidate_d016_node0.log
65ea51b8b98c21da82eb0a93d64dbc341b748dc5ecefdf6b4d2f033de56a4324  /mnt/pr1630_pack_8x_b3_candidate_d016_node1.log
8a8727d2777625dbabe9cbd8fd862f12aa0996e761138b1518668269166419c8  /mnt/pr1630_pack_8x_b3_control_b_d016_node0.log
22e7626f076d5e07ab70b57abbfe9df2d39e8d44e713897802ce6abc58659fc2  /mnt/pr1630_pack_8x_b3_control_b_d016_node1.log
534f201aadfc3cb02df2ad130a1193c01ebfa97ddcfd38fd4ee05a5a1e4965a5  /mnt/benchmark_fastvideo_train_pack_b3.py
ea5a9fa8898fba7904c9a649ee7fde28b1821a07747b8ba89480d877a1876592  /mnt/run_pr1630_pack_8x_b3_aba.sh
808d90a77c1d52d234a3eee30fa145b4c8ee85f4e064f0b051bd3c6255d231f9  /mnt/pr1630_compile_mode_4x_control_a_fa47.log
4da9de0d042c20b06ec465059ed6ecbcae0b019094d473f32c0c3d82c6fc2fdb  /mnt/pr1630_compile_mode_4x_candidate_fa47.log
8d6573c2b13341064a96efb081cf292e87d43fd2e155d2cfb24c96eff207c327  /mnt/pr1630_compile_mode_4x_control_b_fa47.log
d7e571ac7043e4252f44260617e0de8f3467fd97ca3393393a04c64d1af482f3  /mnt/benchmark_fastvideo_train_compile_mode.py
ebc6a2493ee8a109b02561a8f6c8d9a5f5b9c09bd84241c9285b449d4f3b70b3  /mnt/run_pr1630_compile_mode_4x_aba.sh
c3ec531c9cf00c569be30ce1460e224225c617e17e8224af7385b9a82955f08c  /mnt/pr1630_real_pack_export_keep_fa47.log
2a4ba1280710c026fefdcd188ce0207f99a5828c277dcaf265758a8e46c06367  /mnt/validate_ltx_pack_export.py
46b31dd97b14a28fd5da06efcd8893e8cb404908dafbc1d46e6108bc6628c62c  /mnt/pr1630_attention_compile_4x_control_a_fa47.log
62aedbef5b90f9ebd162e8b45a06dbc8c74650ddd12f4a3539ef8e23fbba4580  /mnt/pr1630_attention_compile_4x_candidate_fa47.log
e81bc6f1640ad0c23b74df1fb22eb38f5c37f402d3d294baff691bb9291ad1e6  /mnt/pr1630_attention_compile_4x_control_b_fa47.log
b2fabf6ffa2d174f449c44568373a758c93c39ce2e2da01151412a6e8531dabb  /mnt/run_pr1630_attention_compile_4x_aba.sh
ebb28a2930d4f4d3372b10717d54b7b60d18e006ed720e19458af6ae299da9f9  /mnt/flash-attention-82d6441e.tar.gz
578486b918fe205f17f449f254975207c3b9443b4e12d8e77e461d443d63d672  /mnt/flash-attention-82d6441eec5d4dfec120153db2c0145ae855a083/csrc/fused_dense_lib/fused_dense_lib.cpython-312-aarch64-linux-gnu.so
589a2f030bb42ea48aec9c981f638fef05c43dc4146275f83b0c0476240f6299  /mnt/bench_ltx2_fused_dense_gelu.py
c2ba5a88e215bd81199122aa5298dfc36f3d05770410ad1c491bed71eff115f6  /mnt/build_and_run_ltx2_fused_dense_gelu.sh
4fb7528407e75718c86a2596544643d4a9b2e942a6f078e511d59d0b104d1080  /mnt/pr1630_fused_dense_gelu_82d6441.log
ae32a4fb16cb4c735ba055c6ef6556ed71560d77cfb0770ea3559c78fb2b0d3b  /mnt/pr1630_fused_dense_gelu_h1.log
17ac709ec74445a4ad3bb440e3bab6c77f0fca80783545644dc0f71fa94ab5a5  /mnt/pr1630_fused_dense_gelu_h2.log
17ac709ec74445a4ad3bb440e3bab6c77f0fca80783545644dc0f71fa94ab5a5  /mnt/pr1630_fused_dense_gelu_h3.log
17ac709ec74445a4ad3bb440e3bab6c77f0fca80783545644dc0f71fa94ab5a5  /mnt/pr1630_fused_dense_gelu_h4.log
1edd404b1fb886ca92439761a255861a9119878becef74f2e7e3455ccbf1b34e  /mnt/pr1630_regate_52f1114/control_a.log
0e2bb3834933ab1b0aabbe65b3905e8615fc3f8d7a8ae76ac1b07f448cbff296  /mnt/pr1630_regate_52f1114/candidate.log
98e17c84a4dda1c43e5726b827bfc34e91a3721460ca999fe01d1ebb5abe1deb  /mnt/pr1630_regate_52f1114/control_b.log
098b29f24cc1858ae4237f3a9dce5830ac1484f6c77c23615b9d428bfb966a41  /mnt/pr1630_regate_52f1114/health_before.log
7154bfe19e0b8b1971fae454f85b76fe4228b04b8a8a3eb34e955b4edc0488e5  /mnt/pr1630_regate_52f1114/health_during.csv
9b3f76ba4b0b6e79e04e461638d660f08c9a63880d4e433848d9af9932583384  /mnt/pr1630_regate_52f1114/health_after.log
c21dc0fff404a7b7950610a854bb14316b23899a120eca3068f9960a804f1a1c  [committed runners/run_current.sh as staged for the re-gate] /mnt/pr1630_tracker_52f1114/runners/run_current.sh
ec4cd5092a691de0f0c630b5ed349f98e7d94d5795dfd11de9e242765bdcb790  [committed harness/benchmark_fastvideo_train_pack_d016.py as staged for the re-gate] /mnt/pr1630_tracker_52f1114/harness/benchmark_fastvideo_train_pack_d016.py
f39045adab530f07cbd080abc08e1b3b45bb8b80aa1ce16f605366625f492d32  /mnt/pr1630_flop_audit/b1.log
9a353a98d3697ed84b3b485cb20d52fad8acfabcb6d11cc808db88e20a48afec  [failed base-patch first attempt, then NCCL symmetric-memory init failure at B2] /mnt/pr1630_flop_audit/b2.log
3b8537b9dbe4d2a707a16bef65165b2fbba1645d67ca2628a36b45b0c96a7241  [eager B2 OOM, capacity-blocked] /mnt/pr1630_flop_audit/b2_nosymm.log
64feb60e3deb87bb76564a30326922470c2113bd729b2ce85fdacf40743ddd25  [committed probes/audit_train_flops_per_sample.py] /mnt/audit_train_flops_per_sample.py
1a2d98385c77b491aed2009f9bf23e004459c97c0b6f8dcf3ab17dbc245eba52  /mnt/pr1630_head_profile_b2.log
89826925326da5da851a5766eff4b7e98fc66139851d470bd6a5e58f264cf75d  /mnt/pr1630_head_profile_b2.summary.json
50cc112f6bc88990154db11880b0788933467655c4aa4ac6829e2385c87e1147  /mnt/pr1630_cdt_gate/control_a.log
b031e3152d022816eabf450f0f619a15496291aadad92207e88579e19df6c3b5  /mnt/pr1630_cdt_gate/candidate.log
2b268a684fc265810703898480799e12c88d21807543104cff6e37d899dc6c8e  /mnt/pr1630_cdt_gate/control_b.log
1accd7c721551ab800ddbfe82e1a99723d99aaa14899fe7e065a2238ec3f5d16  /mnt/pr1630_ws_gate/ws75.log
00e998a4498652c4416c19b35b9b68db046b1e24256bb0491f4a3e2a0667ba79  /mnt/pr1630_ws_gate/ws75_cdt.log
132b7939e026cec3d7194ac2d10c19f56ed37b30d7bbca22b3fe28622dcf1db5  /mnt/pr1630_ws_gate/control_c.log
d5ee5c7e85c4f31583676c8af795b5882dc637f7d11bd32256642b95f75ad888  /mnt/pr1630_cudnn_gate/cudnn_alone.log
c6cbdd0e447afa14f0eb5ed37183151ce2f4705fbcffe812ca1182e2897a60e7  /mnt/pr1630_cudnn_gate/cudnn_cdt_ws.log
cf116038960d0edbe2a94982d42d5a18feb34629fb85cfdd260d889985304ed5  /mnt/pr1630_cudnn_gate/control_d.log
35a882dbb4972476e3dfb0207970a6140c1dad31e8f8a864ebfb5da0a273ee0f  /mnt/pr1630_video_attn_gate_b2.log
0cc840410699cb6b28840134007eff8f28d2a2a45f9385863124aa3277e90793  /mnt/pr1630_video_attn_gate_b3.log
6fc72a3d3c1af53712662575c012df386cbbb77d4cec0a573ebe6df1901587eb  [committed probes/benchmark_ltx2_video_attention_backends.py] /mnt/benchmark_ltx2_video_attention_backends.py
286afc02d989a72567c573532a45db61782c90c6b9f8b9bb8ba85d12ed26c718  /mnt/pr1630_gemm_band_default.log
653824bad44bb39142ed74c228cc338a4ef9f990e5c5338224ed6f5cb34dfee1  /mnt/pr1630_gemm_band_ws75.log
37f5c465f534b1c32a0080c0585ab7c5c705214ad8f33dea912ce63179e256cc  /mnt/pr1630_gemm_band_tune.log
b973790131bf7a6f04799c482ad919b03094489ef8ac56478f271febf93fbbc0  /mnt/pr1630_gemm_band_replay.log
47cd9bf4c7a7833ad3dfcf4e86e0eb10627499748992da89e15831bf6634d56e  [committed probes/bench_ltx2_tunableop_gemm.py] /mnt/bench_ltx2_tunableop_gemm.py
4ed684c2858008e4e02ca70943974b21856172dd730d18cd8ea8d84334b9eaf0  /mnt/pr1630_b3skip6_smoke.log
8a8ea4c8eabd20188f75d2f5aee98f0b4f6065d7e09f70df4f6fe052c2d8f87b  /mnt/pr1630_b3skip6_expandable.log
fc2c97a86e1d0ae082b6d8bb3340e554063919943eda52f6fce1864f19d88f2e  /mnt/pr1630_8x_health_1631863_node0.log
1f75874659db76179a0e3e79e7e95259999ed2989035b6f88cbe8efc00bcb205  /mnt/pr1630_8x_health_1631863_node1.log
13b309424061d50611a5e9da5ffc6f0cf7c7d10ab73f40ba655f9fd76235c6ac  /mnt/pr1630_8x_stack_node0.log
f0e16c6bb91d9889d450ff072673c65076d2d0d826b6e8ea8ebeb4c07b4984fd  /mnt/pr1630_8x_stack_node1.log
9e824e8988f3a63d0d53a27ce76da6f99d1eefa6aafa3dd0fefc3954e6198ca2  /mnt/pr1630_8x_nomnnvl_node0.log
0a0093e71b52f804963710606f1f77ece99d37d90b6924b7ce1c82c59c948541  /mnt/pr1630_8x_nomnnvl_node1.log
69787402d008711ea68f3b3289aa2dd05e7ee12034d4e7337d8a792e051f29f8  /mnt/pr1630_8x_crossdebug_node0.log
f3203e22560a7cc1e041b36d406eee1faef580ea5d402470dffae8a1726c259c  /mnt/pr1630_8x_crossdebug_node1.log
f6bdae8f1e6aef72285ddcb15a0639a28549e67d12e0c2b9db18536255efa8a9  /mnt/pr1630_8x_health_1632122_node0.log
7ef78badee4d7077955e129d9df3ff29bc95353ca2f28dab74d08ebf6616b6d0  /mnt/pr1630_8x_health_1632122_node1.log
77f6911fa369b850477a6026e86b026a10236d990bc38b8942aa948f85f99a18  /mnt/pr1630_8x_envstack_control_a_node0.log
6df94ec852ef76334d8f45e991b8cde5bd2e38dc85fb9f3a342d8b5d1f16c16f  /mnt/pr1630_8x_envstack_control_a_node1.log
15a99f49f50925df8e53c8ce628328c1cfee819bd14fa4276b630c775ef5bd67  /mnt/pr1630_8x_envstack_envstack_node0.log
4704d335a688c2eab66ad6eb4e033c12b709938689abac82398e52326105cca6  /mnt/pr1630_8x_envstack_envstack_node1.log
10eebf0d0eb9c71e5aca4eee51c66e68454767b0838af61516cce0d4e3a231f4  /mnt/pr1630_8x_envstack_control_b_node0.log
f681ac1b60beb31aa3dd868e9005684810e05ddb26966bd78885141acd7864bb  /mnt/pr1630_8x_envstack_control_b_node1.log
```

PR #1630's optimization stack was review-clean through measured head `20c36acef`; validation fixes followed at `0e60a0e9c` and `3f3f06541`. No dependency was added or installed. The operational GB200 launcher now forwards allocated IMEX character devices into multi-node containers so the existing MNNVL fabric is reachable.
