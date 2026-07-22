# LTX-2 training MFU tracker

This directory makes PR #1630 an intentionally cumulative, resumable scratch branch for LTX-2 training-performance work. It preserves enough source, configuration, and compact evidence to reproduce accepted results, understand rejected ideas, and continue toward 50% model FLOP utilization (MFU). Focused production changes can be extracted into smaller PRs later.

Start with this file, then read [`REPORT.md`](REPORT.md). The production recipe is [`examples/train/configs/overfit_ltx2_t2v.yaml`](../../../examples/train/configs/overfit_ltx2_t2v.yaml). Runner flags are the experiment-specific configuration record; there were no additional temporary YAML configs worth preserving.

## Current stopping point

The accepted measurement contract is full-parameter LTX-2 training with FP32 registered/master parameters and FP32 Adam moments, BF16 working compute and reductions, 81x480x832 video, dense FA4 attention, regional compile, no activation checkpointing, slowest-rank inclusive step time, 10 warmup steps, and the median of 20 measured steps. Do not compare a run that changes this contract as if it were the same baseline.

The harness numerator is `14.444115 percentage-seconds/sample`, derived from `353.8808175 TFLOP/sample / 2,450 TFLOP/s * 100`. Both constants are audited (see the REPORT's MFU formula audit): the FLOP figure is the exact strict no-recompute, blocks-only model-FLOP count `353,880,819,892,224` at 4,290 video/1,024 text tokens, reconciled integer-exactly against a `FlopCounterMode` measurement by `probes/audit_train_flops_per_sample.py`; the denominator is a house convention 2% below the 2,500 TFLOP/s GB200 vendor dense-BF16 peak, so reported MFU times 0.98 gives vendor-peak MFU. A different model, resolution, frame count, or FLOP convention requires a new numerator before its MFU is reportable.

| Topology | Local/global batch | Median step | Throughput | MFU | Status |
|---|---:|---:|---:|---:|---|
| 1x GB200 | 1/1 | no valid step | - | - | OOM when FP32 Adam state is materialized, even with full activation checkpointing |
| 4x GB200 | 2/8 | 0.707865914 s | 11.301575 samples/s | 40.810314% | final equal-global-batch result |
| 8x GB200, healthy two-tray allocation | 3/24 | 0.993335918 s | 24.161011 samples/s | 43.623053% | fastest accepted result |
| 8x GB200, degraded allocation | 3/24 | 1.421743279 s | 16.880685 samples/s | 30.478319% | same recipe; clocks/power were degraded, so baseline-ineligible |

The measured optimization source is `20c36acef`. Later commits `0e60a0e9c` and `3f3f06541` isolate compiled validation and unload video-only validation state; a 4x run completed steps 0-50, validations at 0 and 50, and resumed at step 51 without a compile reset or OOM ([W&B run `iopu4dwm`](https://wandb.ai/wlsaidhi/fastvideo_ltx2/runs/iopu4dwm)). A 4x/B2 A/X/B on allocation `1627918` re-gated tracker head `52f1114dd` as timing-neutral (+1.326 ms / +0.184% against a 0.044%-drift control midpoint, memory unchanged), so the table remains valid at current head; re-gate again after any future source change.

Historical 1x W&B runs `5h2cy7m9`, `zj6wqtpg`, `nzh2k20t`, and `gbjiw7og` reported 11.71-13.46% MFU but used `training.dit_precision=bf16`. Their persistent parameters and Adam state were BF16, so they are non-comparable capacity diagnostics, not standard-training MFU results.

## Resume here

1. Use a healthy GB200 allocation and record source SHA, GPU count/topology, clocks, power limit, throttle state, CUDA/PyTorch/FA4 versions, and W&B/job IDs. Run [`probes/pr1630_realloc1629519_native_health.py`](probes/pr1630_realloc1629519_native_health.py) and the topology/NCCL probes before accepting an 8x result.
2. Run [`runners/run_current.sh`](runners/run_current.sh). For the final configurations set `LOCAL_BATCH_SIZE=2` on 4x or `LOCAL_BATCH_SIZE=3` on 8x. This timing harness intentionally uses a dummy tracker; its structured `BF16_*` records, not W&B, are the MFU source of truth. Use [`runners/run_observed.sh`](runners/run_observed.sh) for an observable 4x training/validation run with the same production contract.
3. Attribute changes only with an A/candidate/B sandwich on one allocation. Use the slowest rank per step, discard 10 warmup steps, and report the 20-step median plus peak allocated/reserved memory. Never attribute a cross-allocation delta.
4. Check loss/gradient finiteness, FP32 parameter and Adam-state coverage, batch/accumulation semantics, and relevant export/resume or validation behavior before accepting a speedup.
5. Append the result—including rejections—to `REPORT.md`, add the exact driver or runner if it is new, update the stopping-point table only when superseded, and commit the handoff to this branch.

For profiling, use the existing FastVideo profiler/NVTX regions, PyTorch Profiler/Kineto, TorchInductor logs, CUDA-event microbenchmarks, and NCCL diagnostics. `nsys-ai` can read an exported Nsight Systems SQLite database offline but is not a collector and uses a different default GB200 FLOP convention; keep this harness authoritative.

## Artifact map

- `harness/`: end-to-end training instrumentation and Kineto drivers. `benchmark_fastvideo_train_pack_d016.py` is the detailed batch-aware driver behind the accepted rows; `benchmark_fastvideo_train_dynamic_world.py` is a lighter current-world diagnostic.
- `runners/`: `run_current.sh` is the maintained MFU entrypoint and `run_observed.sh` is the maintained W&B/validation entrypoint. Other files are frozen launch snapshots for historical A/B/A, topology, kernel, validation, and profiler gates. They deliberately retain source hashes and cluster assumptions; do not assume they run unchanged at current head.
- `probes/`: semantic/parity checks, distributed diagnostics, fixed-arena and graph prototypes, and exact-shape kernel gates.
- `reports/`: focused subreports and designs. `REPORT.md` is the authoritative chronological decision record.

Historical runners may reference `/mnt/FastVideo`, `/mnt/fv-pr1630-*`, `/mnt/fa4-cache`, `enP5p9s0`, `/mnt/te216*`, `/mnt/cutlass`, or old temporary harness aliases. Adapt checkout/data/cache paths for a new allocation. Most referenced drivers are preserved here; runners that expect the missing `bf0861ff...` harness are archival configuration records, not exact rerun entrypoints. `benchmark_fastvideo_train_pack_b3.py` was an alias of `harness/benchmark_fastvideo_train_ltx2_singleton_timestep.py`; `benchmark_fastvideo_train_pack_ga_ec4c.py` was an alias of `harness/benchmark_fastvideo_train_pack_d016.py`. The final validation runners are historical and intentionally assert the pre-fix source/diff; use the committed validation callbacks at current head for a new gate.

## Decision index

| Family | Decision | Where to continue |
|---|---|---|
| no activation checkpointing, FA4, regional compile | accepted; largest practical wins | current recipe and `run_current.sh` |
| fused AdamW, BF16 reductions, batched/deferred grad norm | accepted | production commits and full-step harnesses |
| warm repeated input, singleton timestep, packed LTX projections | accepted | recipe, packing harness, export/parity probes |
| symmetric-memory FSDP2, accumulation no-sync/retention, two-module FSDP groups | accepted, topology/policy sensitive | grouped and distributed runners |
| validation compile isolation and video-only unload | accepted correctness/capacity fixes | current source; repeat 51-step validation gate after related changes |
| raw velocity, max-autotune, attention compile flag, RMSNorm autocast removal, prefetch, 1-D mesh | rejected or timing-neutral | do not repeat without a new mechanism |
| CUTLASS/cuBLASLt fused GELU, QuACK/NVFP4 complete projection, current TP layouts | rejected by speed, safety, or quality gates | focused reports and exact-shape probes |
| fixed-arena ZeRO-2-style runtime and whole-step CUDA graphs | research only; not lifecycle-complete | fixed-arena and graph reports/probes |
| 1x standard full-parameter training | capacity blocked | add an explicit optimizer/parameter offload mode or use a clearly labeled nonstandard state precision |

## Artifact and credential policy

Commit source-like artifacts and compact reports only. Do not commit W&B keys, checkpoints, generated videos, raw logs, profiler traces, telemetry dumps, compiled binaries, generated CUDA, or copied third-party source. `run_observed.sh` inherits W&B authentication from the job environment or the user's W&B login. Record W&B/job IDs, package or upstream commit versions, command lines, and SHA-256 hashes in the report. Output files from historical runs are intentionally excluded.
