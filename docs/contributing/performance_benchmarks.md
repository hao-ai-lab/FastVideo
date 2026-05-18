# Performance Benchmarks

FastVideo's performance benchmark suite measures end-to-end inference latency,
throughput, and peak GPU memory for representative pipeline configurations,
and tracks them over time against a rolling baseline stored on the Hugging
Face Hub.

It serves three audiences:

* **CI** — gates pull requests against a per-GPU static threshold and a
  rolling-median regression check.
* **Maintainers** — surfaces regressions in a Markdown summary on every
  performance build and a long-form Plotly dashboard.
* **Local developers** — lets you run the same benchmark on your own machine,
  with environment-aware comparison so a different GPU or torch build doesn't
  produce misleading "regressions".

## Quick start (local)

```bash
# Run all benchmarks; writes raw perf_*.json under
# fastvideo/tests/performance/results/
pytest fastvideo/tests/performance/ -vs

# Optional: compare against the rolling HF baseline (read-only outside CI).
# By default this compares your record against records from any environment.
python fastvideo/tests/performance/compare_baseline.py

# Only compare against records produced on a matching environment
# (same GPU model, torch major.minor, CUDA major, attention backend):
PERF_STRICT_ENV=1 python fastvideo/tests/performance/compare_baseline.py
```

The pytest run never uploads anything. `compare_baseline.py` only writes to
the HF dataset when `TEST_SCOPE=full` *and* `BUILDKITE_BRANCH=main`, so local
runs are always read-only.

## Architecture

```
.buildkite/performance-benchmarks/tests/*.json
    └── per-benchmark configs: model, gen kwargs, per-GPU thresholds

fastvideo/tests/performance/
    ├── env_capture.py            # runtime env metadata + compare-tuple helpers
    ├── test_inference_performance.py
    │       └── pytest test that runs each config, writes perf_*.json
    ├── compare_baseline.py
    │       └── normalizes raw results, compares against HF rolling baseline,
    │           writes Markdown summary + (optionally) uploads new records
    ├── dashboard.py
    │       └── builds time-series Plotly HTML from HF history
    └── hf_store.py               # shared HF I/O + DataFrame helpers
```

The HF dataset (`FastVideo/performance-tracking` by default) holds one
normalized JSON per `(model_id, gpu_type, run)` tuple. The rolling baseline is
the median of the last 5 successful records for that model+GPU.

## The two gates

There are **two independent regression gates** — they protect against
different failure modes and are not redundant.

### Static thresholds (per-GPU)

Defined in `.buildkite/performance-benchmarks/tests/<benchmark>.json` under
`thresholds`. Example:

```json
"thresholds": {
  "L40S":    { "max_generation_time_s": 34.0, "max_peak_memory_mb": 11000.0 },
  "default": { "max_generation_time_s": 120.0, "max_peak_memory_mb": 30000.0 }
}
```

Selection: `_get_thresholds(cfg)` matches the current GPU name (substring
match) against keys; falls back to `default` if no GPU matches.

These are **fail-safes** — they catch order-of-magnitude regressions and
unrealistic memory growth even when the rolling baseline is empty. They are
hand-set with generous headroom and almost never need touching.

### Rolling baseline (per `(model_id, gpu_type)`)

`compare_baseline.py` loads the last 5 successful records for the same
`(model_id, gpu_type)` from the HF dataset, computes the median for each
metric, and fails if the current run regresses by more than
`PERF_MAX_REGRESSION` (default 5%).

This is the **drift detector** — it catches sub-threshold regressions that
slowly add up. It only persists new records when running the full suite on
`main`.

When the baseline shifts for a legitimate reason (torch upgrade, kernel
change, etc.) and CI starts failing, use the
[`reseed-performance-baseline`](https://github.com/hao-ai-lab/FastVideo/blob/main/.agents/skills/reseed-performance-baseline/SKILL.md)
agent skill to advance the rolling median.

## Environment-aware comparison

Every record carries an `env` block describing the runtime — GPU model and
count, CUDA toolkit, PyTorch version, attention backend, key package
versions, container image. The comparator can use this to decide whether two
records are "comparable".

### Default mode (env-tuple as metadata)

By default the comparator filters baselines only by `(model_id, gpu_type)`.
The env tuple is captured and surfaced in the Markdown summary and the
dashboard, but does not affect baseline selection.

This is the right default for CI on a fixed machine image — the env tuple
rarely changes, and any change is a deliberate operator action.

### Strict mode (`PERF_STRICT_ENV=1`)

When the environment variable is set, baseline records are additionally
filtered to those whose env tuple matches the current run on the keys
returned by `env_capture.get_compare_keys()`. The default tuple is:

```
(gpu.name, gpu.count, torch.version_major_minor, cuda.runtime_major,
 attention_backend)
```

Override via `PERF_COMPARE_KEYS` (comma-separated dotted paths into the `env`
block):

```bash
PERF_STRICT_ENV=1 \
PERF_COMPARE_KEYS="gpu.name,torch.version_major_minor" \
python fastvideo/tests/performance/compare_baseline.py
```

When strict mode finds zero matching records, it logs `no env-comparable
records on <model_id>` and skips the regression check (the run still passes).
This is the correct behavior for local dev on a non-CI GPU — you get a clear
"no comparable baseline" message instead of a false-positive failure against
a different machine's median.

## Schemas

### Raw record (`results/perf_*.json`)

Written by `test_inference_performance.py`. One file per benchmark run.

```jsonc
{
  "schema_version": 1,
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
  "thresholds": { "max_generation_time_s": 34.0, "max_peak_memory_mb": 11000.0 },
  "commit": "<full sha>",
  "pr_number": "1234",
  "timestamp": "2026-05-08T22:00:00+00:00",
  "env": {
    "schema_version": 1,
    "gpu":   { "name": "NVIDIA L40S", "count": 2, "compute_capability": "8.9",
               "memory_total_mb": 46068.0, "driver_version": "550.54.15",
               "uuids": ["GPU-...", "GPU-..."] },
    "cuda":  { "runtime": "12.4", "runtime_major": "12" },
    "torch": { "version": "2.5.1+cu124", "version_major_minor": "2.5",
               "built_with_cuda": "12.4" },
    "python": "3.12.3",
    "os":    { "system": "Linux", "release": "6.5.0-x", "machine": "x86_64" },
    "attention_backend": "FLASH_ATTN",
    "key_packages": { "torch": "2.5.1+cu124", "transformers": "...", ... },
    "container_image": "<modal image digest, when on Modal>"
  }
}
```

### Normalized record (HF dataset, also dumped as `normalized_perf_*.json`)

Written by `compare_baseline.py:_normalize_record`. One file per benchmark
result, used as the rolling-baseline source of truth.

```jsonc
{
  "schema_version": 1,
  "model_id": "wan-t2v-1.3b-2gpu",
  "timestamp": "2026-05-08T22:00:00+00:00",
  "commit_sha": "<full sha>",
  "gpu_type": "NVIDIA L40S",
  "latency": 28.4,
  "throughput": 1.58,
  "memory": 10840.0,
  "env": { ...same as raw.env... },
  "success": true
}
```

### Compatibility with legacy records

Records produced before `schema_version: 1` exist in the HF dataset with no
`env` block. The comparator and dashboard treat them as having
`extract_compare_tuple(...) == (None, None, None, ...)`. In default mode they
participate in the rolling median normally; in strict mode they only match
*other* legacy records, which is the conservative behavior.

## Environment variable reference

| Variable | Default | Used by | Purpose |
|---|---|---|---|
| `PERF_MAX_REGRESSION` | `0.05` | `compare_baseline.py` | Per-metric regression fraction that fails the build. |
| `PERF_STRICT_ENV` | unset | `compare_baseline.py` | When `1`/`true`/`yes`, filter baseline records by env-tuple equality. |
| `PERF_COMPARE_KEYS` | `gpu.name,gpu.count,torch.version_major_minor,cuda.runtime_major,attention_backend` | `compare_baseline.py`, `dashboard.py` | Comma-separated dotted paths into `env` for strict-mode filtering and dashboard grouping. |
| `PERFORMANCE_TRACKING_ROOT` | `/tmp/perf-tracking` | `compare_baseline.py` | Local directory the HF dataset is synced to. |
| `PERF_REPORTS_DIR` | `/root/data/perf_reports` | `compare_baseline.py`, `dashboard.py` | Where the Markdown summary and Plotly HTML get written for Buildkite to pick up. |
| `HF_REPO_ID` | `FastVideo/performance-tracking` | `hf_store.py` | HF dataset repo holding rolling-baseline records. |
| `HF_API_KEY` | unset | `hf_store.py` | Required for upload (main-branch full-suite only); reads work without it. |
| `TEST_SCOPE` | unset | `compare_baseline.py` | Set to `full` together with `BUILDKITE_BRANCH=main` to enable HF persistence. |
| `BUILDKITE_BRANCH`, `BUILDKITE_COMMIT`, `BUILDKITE_PULL_REQUEST` | unset | `compare_baseline.py`, `test_inference_performance.py` | CI metadata stamped into records. |
| `DASHBOARD_DAYS` | `30` | `dashboard.py` | Lookback window for the Plotly trend pages. |
| `FASTVIDEO_ATTENTION_BACKEND` | unset | `env_capture.py` | Captured into `env.attention_backend` for env-tuple comparison. |
| `MODAL_IMAGE_DIGEST`, `DOCKER_IMAGE_DIGEST`, `DOCKER_IMAGE` | unset | `env_capture.py` | First non-empty value is captured into `env.container_image`. |

## CI integration

The performance step runs in the Full Suite (see
[CI Architecture](ci_architecture.md)). The Modal entry point is
`fastvideo/tests/modal/pr_test.py:run_performance_tests` and the Buildkite
artifact upload is in `.buildkite/scripts/pr_test.sh:upload_performance_artifacts`.

Each performance build produces:

* **Markdown summary** — appended to `$GITHUB_STEP_SUMMARY` and uploaded as
  `perf_<sha>_<ts>.md`. Contains the env tuple banner and a per-benchmark row
  with current vs. baseline.
* **Plotly dashboard** — `dashboard_<sha>_<ts>.html` showing time-series for
  each `(model_id, gpu_type, torch_version, cuda_runtime, attention_backend)`
  bucket.
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
       "L40S":    { "max_generation_time_s": 34.0, "max_peak_memory_mb": 11000.0 },
       "default": { "max_generation_time_s": 120.0, "max_peak_memory_mb": 30000.0 }
     }
   }
   ```

2. The pytest test auto-discovers all configs — no test code needed. CI
   picks it up on the next `/test performance` run.

3. The first run with no HF history initializes the baseline (passes
   automatically). Subsequent runs compare against it.

4. If the benchmark targets a GPU not currently in `thresholds`, either add
   that GPU as a key or rely on the `default` block. Note that `default` is
   intended for slower fallback GPUs, so its values should be relaxed
   relative to the fastest entry.

## Troubleshooting

**"No baseline for ... Initializing"** — first run for this `(model_id,
gpu_type)`, or strict env mode and no env-comparable history. Run will
pass and (if persisting) seed the first record.

**Persistent failure right after a torch / kernel / image upgrade** —
genuine regression *or* baseline drift. Read the `env` block of the failing
record and the most recent baseline record in the HF dataset to compare. If
the env shifted and the metric shift is within the upgrade's expected cost,
use the `reseed-performance-baseline` skill.

**Local strict-mode comparison returns 0 records** — your local env
(GPU/torch/CUDA/backend) doesn't match any historical record. This is
expected — the comparator skips with a clear log line. Either disable strict
mode (`PERF_STRICT_ENV=` ) to compare against the closest available history
on the same model+GPU, or build a local baseline by running a few times with
`PERFORMANCE_TRACKING_ROOT=/some/local/dir` against itself.

**Markdown summary `Env` column shows mostly `None=`** — records lack the
`env` block (legacy `schema_version=0`). Re-run the benchmark to produce a
new record on the current schema, or accept the legacy bucket until enough
new records accumulate.
