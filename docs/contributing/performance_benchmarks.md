# Performance Benchmarks

FastVideo's performance benchmark suite measures end-to-end inference latency,
throughput, peak GPU memory, and component-level pipeline timings for
representative pipeline configurations. It tracks those metrics over time
against a rolling baseline stored on the Hugging Face Hub.

It serves three audiences:

* **CI** — gates pull requests against a per-GPU static threshold and a
  rolling-median regression check.
* **Maintainers** — surfaces regressions in a Markdown summary on every
  performance build and a long-form Plotly dashboard.
* **Local developers** — lets you run the same benchmark on your own machine,
  then compare against the historical baseline for the same model and GPU.

## Quick start (local)

```bash
# Run all benchmarks; writes raw perf_*.json under
# fastvideo/tests/performance/results/
pytest fastvideo/tests/performance/ -vs

# Optional: compare against the rolling HF baseline.
# PERF_REPORTS_DIR defaults to /root/data/perf_reports for Modal/CI, so
# override it when running outside the container.
PERF_REPORTS_DIR=/tmp/fastvideo_perf_reports \
python fastvideo/tests/performance/compare_baseline.py

# Optional: explicitly upload a passing local/manual run.
HF_TOKEN=hf_... \
PERF_RUN_SOURCE=local \
PERF_UPLOAD_POLICY=pass \
PERF_REPORTS_DIR=/tmp/fastvideo_perf_reports \
python fastvideo/tests/performance/compare_baseline.py

# Optional: build the Plotly dashboard locally.
PERF_REPORTS_DIR=/tmp/fastvideo_perf_reports \
python fastvideo/tests/performance/dashboard.py
```

The pytest run never uploads anything. `compare_baseline.py` uploads only when
`PERF_UPLOAD_POLICY` is set. Local uploads are explicit opt-in and require HF
credentials. PR/direct performance runs upload passing records for dashboard
visibility, while scheduled-main runs upload both pass and fail records. The
report directory default is container-oriented; set `PERF_REPORTS_DIR` to a
writable local path when generating dashboards or when you want local
Markdown/normalized-result artifacts from the comparator.
`compare_baseline.py` reads every `perf_*.json` currently present in
`fastvideo/tests/performance/results/`; remove stale result files if you only
want to compare the latest local run.

## Local live dashboard

For an app-style local dashboard backed by the same HF performance-tracking
records, see `performance_dashboard/README.md`. The dashboard provides a
FastAPI API plus a React UI and can be exposed with `ngrok` after building the
frontend.

## Architecture

```
.buildkite/performance-benchmarks/tests/*.json
    └── per-benchmark configs: model, gen kwargs, per-GPU thresholds

fastvideo/tests/performance/
    ├── test_inference_performance.py
    │       └── pytest test that runs each config, writes perf_*.json
    │           with latency, memory, throughput, and component timings
    ├── compare_baseline.py
    │       └── normalizes raw results, compares against HF rolling baseline,
    │           writes Markdown summary + (optionally) uploads new records
    ├── dashboard.py
    │       └── builds time-series Plotly HTML from HF history

fastvideo/performance/
    ├── hf_store.py               # shared HF I/O + DataFrame helpers
    └── metric_policy.py          # shared rolling-baseline threshold policy
```

The HF dataset (`FastVideo/performance-tracking` by default) holds one
normalized JSON per `(model_id, gpu_type, run)` tuple. The rolling baseline is
the median of the last 5 successful, baseline-eligible records for that
model+GPU. PR and local records are visible in the dashboard but are not
baseline eligible.

## Planned Coverage

The current rollout tracks a small set of representative inference workloads.
Broader coverage is planned for additional models, GPU types, attention
backends, workload shapes, and inference recipes. As that coverage lands, the
performance tracking system will also add environment-specific considerations
so comparisons remain meaningful across hardware, runtime, attention backend,
and recipe changes instead of treating all records for a model as equivalent.

## Metrics

Each benchmark records six metrics. The rolling-baseline comparator also has a
per-metric policy with direction, percent threshold, absolute threshold, and a
`gated` flag.

| Metric | Raw key | Normalized key | Direction | Default rolling policy |
|---|---|---|---|---|
| End-to-end generation latency | `avg_generation_time_s` | `latency` | Lower is better | 8% and 0.5 s |
| Video throughput | `throughput_fps` | `throughput` | Higher is better | 8% and 0.05 FPS |
| Peak GPU memory | `max_peak_memory_mb` | `memory` | Lower is better | 5% and 256 MB |
| Text encoder time | `text_encoder_time_s` | `text_encoder_time_s` | Lower is better | 5% and 0.25 s |
| DiT denoising time | `dit_time_s` | `dit_time_s` | Lower is better | 5% and 0.25 s |
| VAE decode time | `vae_decode_time_s` | `vae_decode_time_s` | Lower is better | 5% and 0.25 s |

`test_inference_performance.py` temporarily sets `FASTVIDEO_STAGE_LOGGING=1`
while it runs so pipeline stage execution times are available in
`generate_video(...).logging_info`. Stage logs use pipeline-unique keys such as
`prompt_encoding_stage` so duplicate stage classes do not collide. For
`PipelineStage` entries, the extractor maps the `stage_class` field:
`TextEncodingStage` maps to `text_encoder_time_s`, `DenoisingStage` and
`DmdDenoisingStage` map to `dit_time_s`, and `DecodingStage` maps to
`vae_decode_time_s`, with a fallback for older logs that used the class name as
the stage key. Generator-side timings such as `PostDecodeFrameProcessStage`,
`VideoSaveStage`, and `AudioMuxStage` are intentionally ignored. If a pipeline
does not report one of the mapped stages, that component metric is stored as
`null` and is skipped by the static threshold and rolling baseline checks.

## The two gates

There are **two independent regression gates** — they protect against
different failure modes and are not redundant.

### Static thresholds (per-GPU)

Defined in `.buildkite/performance-benchmarks/tests/<benchmark>.json` under
`thresholds`. Example:

```json
"thresholds": {
  "L40S": {
    "max_generation_time_s": 34.0,
    "max_peak_memory_mb": 11000.0,
    "max_text_encoder_time_s": 5.0,
    "max_dit_time_s": 10.0,
    "max_vae_decode_time_s": 10.0
  },
  "default": { "max_generation_time_s": 120.0, "max_peak_memory_mb": 30000.0 }
}
```

Selection: `_get_thresholds(cfg)` matches the current GPU name (substring
match) against keys; falls back to `default` if no GPU matches.

`max_generation_time_s` and `max_peak_memory_mb` are required for every
selected threshold block. Component limits are optional: if
`max_text_encoder_time_s`, `max_dit_time_s`, or `max_vae_decode_time_s` is
absent, the pytest static-threshold gate skips that component.

These are **fail-safes** — they catch order-of-magnitude regressions,
unrealistic memory growth, and optionally large component-specific slowdowns
even when the rolling baseline is empty. They are hand-set with generous
headroom and almost never need touching.

### Rolling baseline (per `(model_id, gpu_type)`)

`compare_baseline.py` loads the last 5 successful, baseline-eligible records
for the same `(model_id, gpu_type)` from the HF dataset, computes the median
for each available metric, and evaluates the current run with the metric's
rolling regression policy. For latency, memory, and component times, higher
values are regressions. For throughput, lower values are regressions.

A metric exceeds its rolling threshold when both of these are true:

```text
percent_delta > threshold_percent
absolute_delta > threshold_absolute
```

Gated metrics fail CI when that threshold crossing happens. Set `gated: false`
for metrics that should remain visible in reports and the dashboard without
failing CI. Dashboard/API payloads expose `threshold_exceeded` separately from
`regressed`, where `regressed` means a gated CI failure. Missing or `null`
metrics are skipped.

This is the **drift detector** — it catches sub-threshold regressions that
slowly add up. Only scheduled-main successful records are baseline eligible.
Local and pull-request runs can upload dashboard-visible records, but they do
not update future gating baselines.

When the baseline shifts for a legitimate reason (torch upgrade, kernel
change, etc.) and CI starts failing, use the
[`reseed-performance-baseline`](https://github.com/hao-ai-lab/FastVideo/blob/main/.agents/skills/reseed-performance-baseline/SKILL.md)
agent skill to advance the rolling median.

## Schemas

### Benchmark config (`.buildkite/performance-benchmarks/tests/*.json`)

Benchmark configs without `config_schema_version` are treated as legacy v1
configs and remain loadable. New or migrated configs should use
`config_schema_version: 2` and include explicit comparable identity fields:

```jsonc
{
  "benchmark_id": "wan-t2v-1.3b-2gpu",
  "config_schema_version": 2,
  "workload_id": "wan-t2v-1.3b",
  "variant_id": "canonical",
  "benchmark_version": 1
}
```

`benchmark_id` is still required in this phase because raw artifact names,
generated-video directories, normalized record paths, and the current rolling
baseline comparator still depend on it. The v2 identity fields are config
metadata that make the measured workload explicit:

| Field | Purpose |
|---|---|
| `workload_id` | Stable benchmark family, such as `wan-t2v-1.3b`. |
| `variant_id` | Intentional recipe family, such as `canonical`. |
| `benchmark_version` | Version of the measurement protocol and comparison policy. |

If a config declares `config_schema_version: 2`, loading fails clearly when any
required v2 identity field is missing. If v2 identity or metadata fields are
added without `config_schema_version: 2`, loading also fails so partial
migrations do not silently run as v1 configs. Optional v2 metadata fields
reserved for follow-up work, such as `recipe`, `metric_threshold_policy`, and
`quality_metadata`, must be JSON objects when present.

Recipe fingerprinting, hardware/software profile IDs, exact-identity
comparison, metric-specific threshold policy behavior, promoted baselines, and
dashboard regrouping are separate follow-up changes. Until those land, rolling
baseline comparison remains keyed by `(model_id, gpu_type)`.

### Raw record (`results/perf_*.json`)

Written by `test_inference_performance.py`. One file per benchmark run.

```jsonc
{
  "benchmark_id": "wan-t2v-1.3b-2gpu",
  "config_schema_version": 2,
  "workload_id": "wan-t2v-1.3b",
  "variant_id": "canonical",
  "benchmark_version": 1,
  "model_short_name": "Wan2.1-T2V-1.3B-Diffusers",
  "device": "NVIDIA L40S",
  "num_gpus": 2,
  "num_warmup_runs": 1,
  "num_measurement_runs": 3,
  "avg_generation_time_s": 28.4,
  "individual_times_s": [28.5, 28.3, 28.4],
  "throughput_fps": 1.58,
  "max_peak_memory_mb": 10840.0,
  "individual_peak_memories_mb": [10840.0, 10822.0, 10833.0],
  "thresholds": {
    "max_generation_time_s": 34.0,
    "max_peak_memory_mb": 11000.0,
    "max_text_encoder_time_s": 5.0,
    "max_dit_time_s": 10.0,
    "max_vae_decode_time_s": 10.0
  },
  "regression_thresholds": {
    "latency": {
      "threshold_percent": 0.10,
      "threshold_absolute": 1.0,
      "gated": true
    }
  },
  "commit": "<full sha>",
  "pr_number": "1234",
  "timestamp": "2026-05-08T22:00:00+00:00",
  "text_encoder_time_s": 2.141,
  "dit_time_s": 8.437,
  "vae_decode_time_s": 3.208,
  "recipe": {
    "recipe_schema_version": 1,
    "benchmark": {
      "benchmark_id": "wan-t2v-1.3b-2gpu",
      "workload_id": "wan-t2v",
      "variant_id": "1.3b-sp2",
      "benchmark_version": 2
    },
    "model": { "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" },
    "init_kwargs": { "num_gpus": 2, "sp_size": 2, "tp_size": 1 },
    "generation_kwargs": { "height": 480, "width": 832, "num_frames": 45 },
    "inputs": { "prompt_count": 1, "prompt_sha256": ["<measured-prompt-sha256>"] },
    "attention": { "requested_backend": "FLASH_ATTN", "resolved_backend": "FLASH_ATTN" }
  },
  "recipe_fingerprint": "<sha256>",
  "hardware_profile": {
    "device_type": "cuda",
    "gpu_count": 2,
    "gpus": [{ "name": "NVIDIA L40S", "memory_gb": 48, "compute_capability": "8.9" }],
    "interconnect": "none_or_partial"
  },
  "hardware_profile_id": "hw-<sha256-prefix>",
  "software_profile": {
    "python": "3.12",
    "pytorch": "2.12",
    "cuda": "13.0",
    "packages": {
      "fastvideo_kernel": "0.3.2",
      "flashinfer": "0.2.11",
      "nvidia_cutlass_dsl": "4.5.0",
      "triton": "3.4.1"
    }
  },
  "software_profile_id": "sw-<sha256-prefix>",
  "environment_metadata": { "env": { "IMAGE_VERSION": "py3.12-cuda13.0.0" } },
  "environment_fingerprint": "env-<sha256-prefix>"
}
```

### Normalized record (HF dataset, also dumped as `normalized_perf_*.json`)

Written by `compare_baseline.py:_normalize_record`. One file per benchmark
result, used as the rolling-baseline source of truth.

```jsonc
{
  "model_id": "wan-t2v-1.3b-2gpu",
  "workload_id": "wan-t2v",
  "variant_id": "1.3b-sp2",
  "benchmark_version": 2,
  "timestamp": "2026-05-08T22:00:00+00:00",
  "commit_sha": "<full sha>",
  "gpu_type": "NVIDIA L40S",
  "latency": 28.4,
  "throughput": 1.58,
  "memory": 10840.0,
  "text_encoder_time_s": 2.141,
  "dit_time_s": 8.437,
  "vae_decode_time_s": 3.208,
  "regression_thresholds": {
    "latency": {
      "threshold_percent": 0.08,
      "threshold_absolute": 0.5,
      "gated": true
    }
  },
  "recipe_fingerprint": "<sha256>",
  "hardware_profile_id": "hw-<sha256-prefix>",
  "software_profile_id": "sw-<sha256-prefix>",
  "environment_fingerprint": "env-<sha256-prefix>",
  "success": true
}
```

### Compatibility with legacy records

Older records in the HF dataset may not have component timing fields. The
comparator ignores missing or `null` metrics when computing a median, and the
dashboard lists skipped plots for metric series that have no non-null values.
Records missing both `run_source` and `baseline_eligible` are treated as legacy
successful main/full-suite uploads and remain eligible for rolling baselines.

New records compare only against the same `model_id`, `gpu_type`,
`workload_id`, `variant_id`, `benchmark_version`, `recipe_fingerprint`,
`hardware_profile_id`, and `software_profile_id` cohort.
`environment_metadata` and `environment_fingerprint` are audit data and are not
part of the comparison key.
The recipe prompt digests describe the prompts actually measured by the
benchmark run; extra configured prompts are ignored unless the benchmark runner
executes them.
Software profile package cohorts keep exact versions for relevant
attention/kernel packages, including FastVideo kernels, FlashAttention,
FlashInfer, Cutlass DSL, SageAttention, Triton, and xFormers when installed.

## Environment variable reference

| Variable | Default | Used by | Purpose |
|---|---|---|---|
| `PERFORMANCE_TRACKING_ROOT` | `/tmp/perf-tracking` | `compare_baseline.py`, `dashboard.py` | Local directory the HF dataset is synced to. |
| `PERF_REPORTS_DIR` | `/root/data/perf_reports` | `compare_baseline.py`, `dashboard.py` | Where the Markdown summary and Plotly HTML get written for Buildkite to pick up. |
| `HF_REPO_ID` | `FastVideo/performance-tracking` | `fastvideo/performance/hf_store.py` | HF dataset repo holding rolling-baseline records. |
| `HF_API_KEY`, `HUGGINGFACE_HUB_TOKEN`, `HF_TOKEN` | unset | `fastvideo/performance/hf_store.py` | Required for upload or private dataset reads. |
| `PERF_RUN_SOURCE` | inferred | `compare_baseline.py` | Source metadata for uploaded records: `pr`, `local`, `scheduled_main`, or `unknown`. |
| `PERF_UPLOAD_POLICY` | `never` | `compare_baseline.py` | Upload policy: `never`, `pass`, or `always`. |
| `PERF_PYTEST_RC` | unset | `compare_baseline.py` | Static-threshold pytest exit code, used so scheduled-main failures can be uploaded with `success=false`. |
| `TEST_SCOPE` | unset | `compare_baseline.py` | CI context used to infer scheduled-main runs together with `BUILDKITE_BRANCH=main`. |
| `BUILDKITE_BRANCH`, `BUILDKITE_COMMIT`, `BUILDKITE_PULL_REQUEST` | unset | `compare_baseline.py`, `test_inference_performance.py` | CI metadata stamped into records. |
| `DASHBOARD_DAYS` | `30` | `dashboard.py` | Lookback window for the Plotly trend pages. |
| `PERFORMANCE_TRACKING_SYNC_REUSE_TTL_SECONDS` | `3600` | `fastvideo/performance/hf_store.py` | Freshness window for reusing an existing HF sync when requested by dashboard consumers. |
| `FASTVIDEO_STAGE_LOGGING` | set by the pytest test | `test_inference_performance.py` | Enables pipeline stage timing capture for component metrics during benchmark runs. |

## CI integration

The performance step can run on demand with `/test performance` and as part of
the Full Suite (see [CI/CD Architecture](ci_architecture.md)). The Modal entry
point is `fastvideo/tests/modal/pr_test.py:run_performance_tests` and the
Buildkite artifact upload is in
`.buildkite/scripts/pr_test.sh:upload_performance_artifacts`.

Each performance build runs pytest first. If that fixed-threshold phase fails,
PR/direct runs skip `compare_baseline.py` because they only upload passing
records. Scheduled-main runs still execute `compare_baseline.py` with
`PERF_PYTEST_RC` set so the failed canonical attempt is visible in normalized
JSON and dashboard history. The dashboard runs best-effort for observability.
When the rolling-baseline phase runs, it emits:

* **Markdown summary** — appended to `$GITHUB_STEP_SUMMARY` when that variable
  is set, and written as `perf_<sha>_<ts>.md` for Buildkite upload. Contains a
  per-benchmark row with current vs. baseline values for latency, throughput,
  memory, text encoder time, DiT time, and VAE decode time.
* **Plotly dashboard** — `dashboard_<sha>_<ts>.html` showing time-series for
  each metric grouped by `(model_id, gpu_type)`.
* **Normalized records** — `normalized_perf_*.json`, one per benchmark.
  Useful as input to the
  [`reseed-performance-baseline`](https://github.com/hao-ai-lab/FastVideo/blob/main/.agents/skills/reseed-performance-baseline/SKILL.md)
  skill.

## Adding a new benchmark

1. Drop a new JSON config into
   `.buildkite/performance-benchmarks/tests/<name>.json`. New configs should
   use v2 identity fields:

   ```json
   {
     "benchmark_id": "<unique-id>",
     "config_schema_version": 2,
     "workload_id": "<stable-workload-id>",
     "variant_id": "canonical",
     "benchmark_version": 1,
     "model": { "model_path": "...", "model_short_name": "..." },
     "init_kwargs": { "num_gpus": 1, ... },
     "generation_kwargs": { "num_frames": 45, ... },
     "test_prompts": ["..."],
     "run_config": { "required_gpus": 1,
                     "num_warmup_runs": 1, "num_measurement_runs": 3 },
     "thresholds": {
       "L40S": {
         "max_generation_time_s": 34.0,
         "max_peak_memory_mb": 11000.0,
         "max_text_encoder_time_s": 5.0,
         "max_dit_time_s": 10.0,
         "max_vae_decode_time_s": 10.0
       },
       "default": { "max_generation_time_s": 120.0, "max_peak_memory_mb": 30000.0 }
     },
     "regression_thresholds": {
       "latency": { "threshold_percent": 0.10, "threshold_absolute": 1.0, "gated": true }
     }
  }

  ```

   Legacy v1 configs without `config_schema_version` still load, but should not
   gain v2 identity or metadata fields until they are migrated to
   `config_schema_version: 2`. For v2 configs, `workload_id`, `variant_id`,
   and `benchmark_version` are part of the comparison key; benchmark runs
   fail if any of these identity fields are missing.

2. The pytest test auto-discovers all configs — no test code needed. CI
   picks it up on the next `/test performance` run.

3. The first persisted main-branch run with no HF history initializes the
   baseline (passes automatically). Subsequent runs compare against it. Local
   and pull-request runs with no HF history also pass, but they do not seed the
   shared baseline.

4. If the benchmark targets a GPU not currently in `thresholds`, either add
   that GPU as a key or rely on the `default` block. Note that `default` is
   intended for slower fallback GPUs, so its values should be relaxed
   relative to the fastest entry.

5. Add component thresholds only when the stage timing is stable enough to be
   a useful fixed gate. The rolling baseline will still track component times
   when static component thresholds are omitted.

6. Omit `regression_thresholds` to use the default rolling-baseline policy, or
   include only benchmark-specific deviations. Tune these independently from
   the fixed thresholds when a metric is noisy or should be informational. The
   fixed `thresholds` block is an absolute pytest ceiling. The
   `regression_thresholds` block controls rolling-baseline comparisons against
   recent scheduled-main records.

## Troubleshooting

**"No baseline for ... Initializing"** — first run for this `(model_id,
gpu_type)`. Run will pass and (if persisting) seed the first record.

**Persistent failure right after a torch / kernel / image upgrade** —
genuine regression *or* baseline drift. Compare the failing normalized record
with recent successful records in the HF dataset. If the shift is expected and
reviewed, use the `reseed-performance-baseline` skill.

**Dashboard reports skipped metric plots** — the loaded records do not have
non-null values for that metric. This is expected for older records or for
pipelines that did not report a mapped component stage.

**Component timing is `null`** — the generated result did not include a mapped
stage in `logging_info.stages`. Check that the pipeline emits stage logging
and that the stage name is listed in `STAGE_METRIC_MAP` in
`test_inference_performance.py`.
