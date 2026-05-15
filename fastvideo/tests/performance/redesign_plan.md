# FastVideo Scheduled Main Performance Regression Tracking Redesign

## Purpose

FastVideo needs a scheduled main-branch performance benchmark that detects real inference regressions without confusing intentional optimizations, runtime upgrades, hardware changes, or quality-changing recipe changes with ordinary code regressions.

The current design compares recent results by `(benchmark_id, gpu_type)` and mainly tracks end-to-end generation latency. That is too coarse. A PyTorch upgrade, attention backend change, fewer inference steps, different precision, or different parallelism setup can enter the same baseline even though the result is not comparable.

This redesign makes comparability explicit.

## Core Principle

A benchmark result is comparable only when all of these match:

- workload identity
- variant identity
- benchmark schema/version
- inference recipe fingerprint
- hardware profile
- software/runtime profile

If any required identity does not match, the system reports `CALIBRATION_NEEDED` or `RECIPE_MISMATCH`, not a regression.

## Result Statuses

Each scheduled run should produce one of these statuses:

- `PASS`: comparable baseline exists and no metric regressed beyond threshold.
- `REGRESSION`: comparable baseline exists and one or more metrics regressed.
- `CALIBRATION_NEEDED`: no comparable baseline exists for this hardware/software cohort.
- `RECIPE_MISMATCH`: same variant was used with a different recipe fingerprint.
- `INFRA_ERROR`: benchmark failed for infrastructure reasons.
- `QUALITY_BLOCKED`: faster/new variant exists but lacks required quality validation.

## Identity Model

Use this comparison key:

```text
workload_id + variant_id + benchmark_version + hardware_profile_id + software_profile_id + recipe_fingerprint
```

Definitions:

- `workload_id`: stable benchmark family, for example `wan-t2v-1.3b`.
- `variant_id`: intentional recipe family, for example `canonical`, `fast-4step`, `flash-attn`, `torch-sdpa`.
- `benchmark_version`: version of the benchmark schema and comparison policy.
- `recipe_fingerprint`: hash of inference settings that affect comparability.
- `hardware_profile_id`: normalized GPU/hardware cohort.
- `software_profile_id`: intentionally versioned runtime cohort used for comparison.
- `environment_fingerprint`: full audit hash, stored but not used as the primary comparison key.

## Recipe Fingerprint

The `recipe_fingerprint` should include:

- model path and model revision
- pipeline or preset name/version
- prompt set digest
- negative prompt digest
- height, width, frame count, fps
- seed
- number of inference steps
- scheduler settings
- guidance scale and embedded CFG scale
- attention backend
- SP/TP size and number of GPUs
- text encoder, DiT, and VAE precision
- VAE tiling/SP/offload settings
- output type
- any benchmark-specific overrides

If the same `variant_id` produces a different recipe fingerprint, the system should not compare it to the existing baseline. It should report `RECIPE_MISMATCH`.

## Runtime And Software Cohorts

A PyTorch, CUDA, Triton, FlashAttention, container, or driver upgrade can change performance without a FastVideo code regression. These must be handled as software cohort changes.

Use two runtime fields:

- `software_profile_id`: compare-affecting runtime cohort.
- `environment_fingerprint`: full audit trail.

`software_profile_id` should include only major performance-affecting runtime fields:

- Python version
- PyTorch version
- CUDA runtime version
- Triton version
- FlashAttention/SageAttention/xFormers versions when used
- container performance profile version
- optional CUDA driver major/minor if runner stability requires it

`environment_fingerprint` can include:

- full container image digest
- all package versions
- driver details
- OS/kernel details
- dependency lock hash
- full environment metadata

A new `software_profile_id` should produce `CALIBRATION_NEEDED` until maintainers manually seed or promote reviewed records through the reseed-performance workflow.

## Runtime Upgrade Workflow

For PyTorch/CUDA/container upgrades:

- Create a new `software_profile_id`.
- Do not compare new runtime results against old runtime baselines.
- Run a bridge report when possible:
  same FastVideo commit, same benchmark recipe, same hardware, old runtime vs new runtime.
- Use the bridge report to attribute performance shifts to the runtime upgrade.
- Seed the new runtime cohort only after the shift is reviewed, using the manual reseed-performance workflow.

## Hardware Profile

`hardware_profile_id` should normalize:

- GPU name
- GPU count
- GPU memory size
- relevant interconnect/topology when available
- Modal/runner machine class if stable and meaningful

For multi-GPU benchmarks, collect rank-level metrics and compare rank-max values where appropriate.

## Measurement Model

Keep end-to-end latency, but do not make it the only primary regression signal.

Primary metrics:

- `pipeline_total_s`
- `text_encode_s`
- `denoise_total_s`
- `denoise_step_mean_s`
- `denoise_step_p50_s`
- `denoise_step_p90_s`
- `transformer_forward_mean_s`
- `scheduler_step_mean_s`
- `vae_decode_s`
- `postprocess_cpu_s`
- `peak_memory_mb`
- `throughput_fps`

Informational metrics:

- `request_total_s`
- video write/export time
- generated artifact size
- individual run timings

For canonical inference compute benchmarks, prefer `save_video=false` to avoid disk and video encoding noise. Video writing/export can be tracked in a separate benchmark if needed.

## Instrumentation

Use existing stage logging as the coarse stage timing source.

Extend instrumentation to capture:

- text encoding time
- denoising total time
- denoising per-step timings
- transformer forward timings
- scheduler step timings
- VAE decode time
- CPU postprocess time
- peak memory per measurement run
- rank-level memory/timing for distributed runs

Avoid adding heavy synchronization inside every denoising sub-step by default. Prefer CUDA events or a dedicated low-overhead perf instrumentation path. Coarse stage timing may use synchronization, but per-step timing should not distort the scheduled benchmark.

Reset CUDA peak memory stats before each measured run on every rank.

## Baseline Policy

For each exact comparable identity:

- Use scheduled `main` records only.
- Persist only from scheduled main runs.
- Keep failed records for audit.
- Exclude failed records from baseline medians.
- Use the last 5 successful records as the rolling baseline.
- Compare current median against rolling median.
- Treat `CALIBRATION_NEEDED` records as audit data only. They must not automatically seed or advance a comparable baseline.
- Initialize new comparable baselines manually from reviewed scheduled-main artifacts through the reseed-performance workflow.

To prevent slow drift, also keep a promoted baseline:

- `rolling_baseline`: last 5 successful scheduled records.
- `promoted_baseline`: explicitly accepted reference cohort/version.

A run should alert or fail if it exceeds thresholds against either baseline.

## Threshold Policy

Use metric-specific thresholds instead of one global threshold.

Suggested defaults:

- 5% for core compute metrics: denoise total, denoise per-step, transformer forward, VAE decode.
- 5% for peak memory.
- 8-10% for noisy end-to-end/request metrics.
- Minimum absolute delta floor:
  - 250ms for stage totals.
  - 50ms for per-step metrics.
  - reasonable MB floor for memory.

A regression should require both:

```text
percent_delta > threshold_percent
absolute_delta > threshold_absolute
```

This avoids failing on tiny noisy changes.

## Optimization And Variant Workflow

Same-recipe code optimizations stay in the same variant.

Examples:

- Faster implementation of the same attention backend: same variant.
- Reduced inference steps: new variant.
- Different default attention backend: new variant.
- Different precision recipe: new variant.
- DMD/distilled recipe: new variant.

A faster quality-affecting recipe should be created as a candidate variant:

```text
workload_id = wan-t2v-1.3b
variant_id = fast-4step
status = candidate
```

Candidate lifecycle:

```text
candidate -> calibrating -> canonical
candidate -> deprecated
```

Promotion requires:

- quality validation
- stable scheduled performance records
- explicit maintainer approval

## Quality Policy

Performance tracking must not decide "same quality" by latency alone.

Candidate variants require quality evidence, such as:

- SSIM/latent similarity reference
- existing evaluation metric
- reviewed generated artifacts
- documented human approval when automated quality metrics are unavailable

If quality validation is missing, the benchmark should report `QUALITY_BLOCKED` for promotion, even if performance improved.

## Data Schema

Each raw result should include:

```text
result_schema_version
workload_id
variant_id
benchmark_version
recipe_fingerprint
hardware_profile_id
software_profile_id
environment_fingerprint
trigger_type
schedule_id
build_id
build_url
branch
commit_sha
timestamp
status
quality_status
recipe
hardware
software
metrics
runs
comparison
```

Do not make PR fields required. This system is for scheduled main benchmarks.

Optional ad-hoc provenance can be stored as:

```text
change_request.type
change_request.number
```

but it should not be part of the scheduled-main schema requirements.

## Storage Layout

Change HF storage from:

```text
<model_id>/<record>.json
```

to:

```text
<workload_id>/<variant_id>/<hardware_profile_id>/<software_profile_id>/<record>.json
```

Store full result JSONs under this path.

Keep legacy v1 records readable for dashboard history, but do not compare v2 records against v1 records unless an explicit migration script creates compatible v2 identities.

## Dashboard Requirements

Dashboard should group by:

- workload
- variant
- hardware profile
- software profile
- benchmark version

Dashboard should show:

- current status
- rolling baseline comparison
- promoted baseline comparison
- recipe fingerprint
- software profile
- environment fingerprint
- quality status
- metric trend charts
- calibration-needed cohorts
- recipe mismatch incidents
- runtime upgrade bridge reports

Legacy v1 data can be shown separately as historical context.

## Scheduled Main Behavior

Scheduled main runs should:

- run every configured interval
- execute benchmarks on `main`
- write raw result artifacts
- compare against exact comparable baselines
- persist successful, failed, calibration, and mismatch records for audit
- update rolling baseline only with successful comparable records
- never silently initialize a baseline as passing

When no baseline exists, status is `CALIBRATION_NEEDED`. That record is persisted for review, but it does not become a baseline seed unless a maintainer explicitly reseeds or promotes it.

## Implementation Plan

1. Update benchmark JSON configs to include:
   `workload_id`, `variant_id`, `benchmark_version`, comparison policy, quality policy, and recipe fields.

2. Update `test_inference_performance.py` to:
   emit v2 schema, collect runtime metadata, compute recipe/hardware/software IDs, collect median metrics, reset memory per run, and avoid PR-required fields.

3. Extend pipeline instrumentation to:
   capture stage metrics and denoising internals with low overhead.

4. Update distributed executor paths to:
   return rank-level timing and memory metrics for multi-GPU runs.

5. Update `compare_baseline.py` to:
   load by the new identity, reject recipe mismatches, report calibration for missing cohorts, compare rolling and promoted baselines, apply metric-specific thresholds, and avoid treating calibration records as baseline seeds.

6. Update `hf_store.py` to:
   support the new storage layout and preserve legacy reads.

7. Update `dashboard.py` to:
   group by workload/variant/hardware/software and show status, quality, and runtime cohort changes.

8. Update the reseed workflow to:
   reseed only one exact identity tuple, require provenance especially for runtime upgrades, and mark reviewed records as accepted baseline seeds.

9. Add unit and integration tests for:
   identity, recipe mismatch, software cohort changes, regression detection, dashboard grouping, and v2 result serialization.

## Migration Plan

- Treat current records as legacy schema v1.
- Start v2 with the current Wan benchmark:
  - `workload_id = wan-t2v-1.3b`
  - `variant_id = canonical`
  - `benchmark_version = 1`
- Seed v2 baselines manually from reviewed scheduled-main result artifacts through the reseed-performance workflow.
- Keep legacy charts for reference only.
- Do not compare v2 records against v1 records by default.

## Test Plan

Required tests:

- Same config produces stable recipe fingerprint.
- Changed `num_inference_steps` under same variant produces `RECIPE_MISMATCH`.
- Changed PyTorch/software profile produces `CALIBRATION_NEEDED`.
- Same recipe with slower denoise metric produces `REGRESSION`.
- Same recipe with faster metrics produces `PASS`.
- Candidate variant does not compare against canonical.
- Missing quality evidence blocks candidate promotion.
- Dashboard separates software cohorts and variants.
- Multi-GPU metrics use rank-level aggregation.
- Current Wan config writes valid v2 JSON with populated metrics.

## Defaults

- HF dataset remains `FastVideo/performance-tracking`.
- Only scheduled main runs persist canonical tracking records.
- Missing exact baseline means `CALIBRATION_NEEDED`.
- Calibration records are not baseline seeds by default.
- New cohorts become comparable only after manual reseed or promotion.
- Runtime upgrades create new software cohorts.
- Fewer-step optimizations create new variants.
- End-to-end latency remains visible but is not the sole regression signal.
- Video export timing is separate from canonical inference compute timing.
