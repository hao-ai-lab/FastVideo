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
    └── hf_store.py               # shared HF I/O + DataFrame helpers
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

Each benchmark records six metrics:

| Metric | Raw key | Normalized key | Direction |
|---|---|---|---|
| End-to-end generation latency | `avg_generation_time_s` | `latency` | Lower is better |
| Video throughput | `throughput_fps` | `throughput` | Higher is better |
| Peak GPU memory | `max_peak_memory_mb` | `memory` | Lower is better |
| Text encoder time | `text_encoder_time_s` | `text_encoder_time_s` | Lower is better |
| DiT denoising time | `dit_time_s` | `dit_time_s` | Lower is better |
| VAE decode time | `vae_decode_time_s` | `vae_decode_time_s` | Lower is better |

`test_inference_performance.py` temporarily sets `FASTVIDEO_STAGE_LOGGING=1`
while it runs so pipeline stage execution times are available in
`generate_video(...).logging_info`. It maps `TextEncodingStage` to
`text_encoder_time_s`, `DenoisingStage` and `DmdDenoisingStage` to
`dit_time_s`, and `DecodingStage` to `vae_decode_time_s`. If a pipeline does
not report one of those stages, that component metric is stored as `null` and
is skipped by the static threshold and rolling baseline checks.

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
for each available metric, and fails if the current run regresses by more than
`PERF_MAX_REGRESSION` (default 5%). For latency, memory, and component times,
higher values are regressions. For throughput, lower values are regressions.

This is the **drift detector** — it catches sub-threshold regressions that
slowly add up. Only scheduled-main successful records are baseline eligible.
Local and pull-request runs can upload dashboard-visible records, but they do
not update future gating baselines.

When the baseline shifts for a legitimate reason (torch upgrade, kernel
change, etc.) and CI starts failing, use the
[`reseed-performance-baseline`](https://github.com/hao-ai-lab/FastVideo/blob/main/.agents/skills/reseed-performance-baseline/SKILL.md)
agent skill to advance the rolling median.

## Schemas

### Raw record (`results/perf_*.json`)

Written by `test_inference_performance.py`. One file per benchmark run.

```jsonc
{
  "benchmark_id": "wan-t2v-1.3b-2gpu",
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
  "commit": "<full sha>",
  "pr_number": "1234",
  "timestamp": "2026-05-08T22:00:00+00:00",
  "text_encoder_time_s": 2.141,
  "dit_time_s": 8.437,
  "vae_decode_time_s": 3.208
}
```

### Normalized record (HF dataset, also dumped as `normalized_perf_*.json`)

Written by `compare_baseline.py:_normalize_record`. One file per benchmark
result, used as the rolling-baseline source of truth.

```jsonc
{
  "model_id": "wan-t2v-1.3b-2gpu",
  "timestamp": "2026-05-08T22:00:00+00:00",
  "commit_sha": "<full sha>",
  "gpu_type": "NVIDIA L40S",
  "latency": 28.4,
  "throughput": 1.58,
  "memory": 10840.0,
  "text_encoder_time_s": 2.141,
  "dit_time_s": 8.437,
  "vae_decode_time_s": 3.208,
  "success": true
}
```

### Compatibility with legacy records

Older records in the HF dataset may not have component timing fields. The
comparator ignores missing or `null` metrics when computing a median, and the
dashboard lists skipped plots for metric series that have no non-null values.

## Environment variable reference

| Variable | Default | Used by | Purpose |
|---|---|---|---|
| `PERF_MAX_REGRESSION` | `0.05` | `compare_baseline.py` | Per-metric regression fraction that fails the build. |
| `PERFORMANCE_TRACKING_ROOT` | `/tmp/perf-tracking` | `compare_baseline.py`, `dashboard.py` | Local directory the HF dataset is synced to. |
| `PERF_REPORTS_DIR` | `/root/data/perf_reports` | `compare_baseline.py`, `dashboard.py` | Where the Markdown summary and Plotly HTML get written for Buildkite to pick up. |
| `HF_REPO_ID` | `FastVideo/performance-tracking` | `hf_store.py` | HF dataset repo holding rolling-baseline records. |
| `HF_API_KEY`, `HUGGINGFACE_HUB_TOKEN`, `HF_TOKEN` | unset | `hf_store.py` | Required for upload or private dataset reads. |
| `PERF_RUN_SOURCE` | inferred | `compare_baseline.py` | Source metadata for uploaded records: `pr`, `local`, `scheduled_main`, or `unknown`. |
| `PERF_UPLOAD_POLICY` | `never` | `compare_baseline.py` | Upload policy: `never`, `pass`, or `always`. |
| `PERF_PYTEST_RC` | unset | `compare_baseline.py` | Static-threshold pytest exit code, used so scheduled-main failures can be uploaded with `success=false`. |
| `TEST_SCOPE` | unset | `compare_baseline.py` | CI context used to infer scheduled-main runs together with `BUILDKITE_BRANCH=main`. |
| `BUILDKITE_BRANCH`, `BUILDKITE_COMMIT`, `BUILDKITE_PULL_REQUEST` | unset | `compare_baseline.py`, `test_inference_performance.py` | CI metadata stamped into records. |
| `DASHBOARD_DAYS` | `30` | `dashboard.py` | Lookback window for the Plotly trend pages. |
| `PERFORMANCE_TRACKING_SYNC_REUSE_TTL_SECONDS` | `3600` | `hf_store.py` | Freshness window for reusing an existing HF sync when requested by dashboard consumers. |
| `FASTVIDEO_STAGE_LOGGING` | set by the pytest test | `test_inference_performance.py` | Enables pipeline stage timing capture for component metrics during benchmark runs. |

## CI integration

The performance step can run on demand with `/test performance` and as part of
the Full Suite (see [CI Architecture](ci_architecture.md)). The Modal entry
point is `fastvideo/tests/modal/pr_test.py:run_performance_tests` and the
Buildkite artifact upload is in
`.buildkite/scripts/pr_test.sh:upload_performance_artifacts`.

Each performance build runs pytest first. If that fixed-threshold phase fails,
`compare_baseline.py` is skipped, so Markdown summaries and normalized JSON
artifacts are not emitted. The dashboard still runs best-effort for
observability. When pytest passes, the rolling-baseline phase emits:

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
   `.buildkite/performance-benchmarks/tests/<name>.json`. Required keys:

   ```json
   {
     "benchmark_id": "<unique-id>",
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
     }
   }
   ```

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
