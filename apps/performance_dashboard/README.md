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

The dashboard supports model, GPU, source, and day-window filters.

Trend charts show metric-specific axes and exact point details on hover/focus:

- metric value and unit
- timestamp
- commit SHA
- run source
- stored status
- canonical comparison status and reason
- baseline status
- baseline eligibility
- PR number, branch, and Buildkite URL when present

The latest status table shows the stored JSON `success` value alongside the
canonical comparison status and reason. Recomputed baseline context applies
each metric's percent and absolute regression floors and does not override
either stored result.

## API

- `GET /api/performance/health`
- `POST /api/performance/refresh`
- `GET /api/performance/summary?days=90&run_source=pr`
- `GET /api/performance/trends?days=90&run_source=scheduled_main`
- `GET /api/performance/records?days=90&run_source=local`

V2 records use the same comparison cohort as CI: `workload_id`, `variant_id`,
`benchmark_version`, `recipe_fingerprint`, `hardware_profile_id`, and
`software_profile_id`. `model_id` and `gpu_type` remain display/filter
metadata, so renaming either does not split history. Legacy records still group
by `(model_id, gpu_type)` under the explicit `Legacy v1` label. Records that
claim or partially use v2 identity without all required fields are labeled
`Invalid v2` and remain scoped by model, GPU, and their available identity
values. Dashboard baselines use the latest five previous successful,
baseline-eligible records in each group.

Summary rows, trend groups, and trend points expose `cohort_kind`,
`result_schema_version`, `baseline_status`, `comparison_status`, and
`comparison_status_reason`. Summary and trend filters match the latest display
metadata after grouping, while the raw records endpoint continues to filter
individual records.
