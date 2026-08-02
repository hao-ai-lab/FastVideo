# FastVideo Performance Dashboard

Local FastAPI + React dashboard for records stored in the Hugging Face
performance tracking dataset.

## Data Source

The dashboard reads the same normalized JSON records used by
`fastvideo/tests/performance/compare_baseline.py`.

Defaults:

- `HF_REPO_ID=FastVideo/performance-tracking`
- `PERFORMANCE_TRACKING_ROOT=/tmp/fastvideo-perf-dashboard`

Records can include source metadata and rolling-baseline policy context:

- `run_source`: `pr`, `local`, `scheduled_main`, or `unknown`
- `baseline_eligible`: only successful scheduled-main records should be true
- Buildkite metadata such as branch, PR number, build URL, build ID, and job ID
- `regression_thresholds`: per-metric rolling-baseline percent and absolute
  floors used for recomputed status context

Dashboard/API metric payloads expose `threshold_exceeded` for raw threshold
crossings; `regressed` remains the gated CI-failure signal.

Set one of `HF_API_KEY`, `HUGGINGFACE_HUB_TOKEN`, or `HF_TOKEN` if the
configured dataset repo requires authenticated access:

```bash
export HF_TOKEN=hf_...
```

If Hugging Face returns `401 Unauthorized`, confirm that `HF_REPO_ID` points to
the dataset repo you expect and that your token has access to it.

## Development

Run the API:

```bash
python -m fastvideo.performance_dashboard --host 127.0.0.1 --port 8000 --reload
```

Run the React dev server:

```bash
cd performance_dashboard/frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api/*` to the FastAPI server on
port 8000.

## Single-Port Mode For ngrok

Build the frontend:

```bash
cd performance_dashboard/frontend
npm install
npm run build
```

Serve API and built frontend from one FastAPI process:

```bash
python -m fastvideo.performance_dashboard --host 0.0.0.0 --port 8000
```

Expose it:

```bash
ngrok http 8000
```

The ngrok URL will serve the dashboard UI and all `/api/performance/*`
endpoints from the same local port.

## Dashboard Behavior

The dashboard supports cascading Model, GPU configuration, Benchmark cohort,
Source, and day-window filters. The backend owns the canonical cohort key used
for grouping, API filtering, React keys, and URL state. V2 keys use the exact CI
comparison identity; legacy records use a separate legacy identity and are
never merged with v2 records.

The Benchmark cohort selector defaults to the cohort with the most recent
successful baseline-eligible observation, falling back to the cohort with the
most recent record. `All cohorts` remains an explicit option. Model constrains
GPU configurations, and Model plus GPU constrain cohorts. Source and Date
change displayed observations without redefining cohort availability.

Selected filters are encoded in the URL. Unknown or stale Model, GPU, and
cohort values fall back safely to available options.

Selecting `All cohorts` shows a compact, sortable overview with one row per
exact comparison cohort and deliberately hides the detailed trend grid. Failed
cohorts appear first by default, followed by the newest observation. Status,
latest run, latency, throughput, and memory columns are sortable; unavailable
metrics remain explicit instead of being rendered as zero. Select a row with a
pointer, Enter, or Space to open that exact cohort and its detailed trends.

Trend charts show metric-specific axes and exact point details on hover/focus:

- metric value and unit
- timestamp
- commit SHA
- run source
- stored status
- baseline eligibility
- PR number, branch, and Buildkite URL when present

The latest status table uses the stored JSON `success` value. Recomputed
baseline context applies each metric's percent and absolute regression floors
and does not override stored status.

## API

- `GET /api/performance/health`
- `POST /api/performance/refresh`
- `GET /api/performance/cohorts?model_id=wan-t2v-1.3b-2gpu&gpu_key=gpu%3A...`
- `GET /api/performance/summary?days=90&run_source=pr`
- `GET /api/performance/trends?days=90&run_source=scheduled_main`
- `GET /api/performance/records?days=90&run_source=local`

Summary, trend, and raw-record endpoints also accept opaque `gpu_key` and
`cohort_key` values returned by `/api/performance/cohorts`. Cohort descriptors
include readable recipe, hardware, software, and GPU configuration labels plus
the full raw hardware, software, and recipe identifiers for debugging.

V2 records use the same comparison cohort as CI: `workload_id`, `variant_id`,
`benchmark_version`, `recipe_fingerprint`, `hardware_profile_id`, and
`software_profile_id`. `model_id` and `gpu_type` remain display/filter
metadata, so renaming either does not split history. Legacy records still group
by `(model_id, gpu_type)`. Dashboard baselines use the latest five previous
successful, baseline-eligible records in each group. Summary and trend filters
match the latest display metadata after grouping, while the raw records endpoint
continues to filter individual records.
