# Performance Dashboard Memory

Date: 2026-06-16
Branch: `ci/dashboard`

## Purpose

This branch adds a local live dashboard for FastVideo performance benchmark
history. It is intended for maintainer/operator use: inspect latest benchmark
status, compare current values with recent baseline context, and view trends
from the Hugging Face performance-tracking dataset.

The dashboard is a FastAPI + React app. It is separate from the existing
Svelte `ui/` app.

## Main Files Added Or Changed

Backend:

- `fastvideo/performance_dashboard/__init__.py`
- `fastvideo/performance_dashboard/__main__.py`
- `fastvideo/performance_dashboard/api.py`
- `fastvideo/performance_dashboard/metrics.py`
- `fastvideo/performance_dashboard/service.py`

Frontend:

- `performance_dashboard/frontend/package.json`
- `performance_dashboard/frontend/package-lock.json`
- `performance_dashboard/frontend/tsconfig.json`
- `performance_dashboard/frontend/vite.config.ts`
- `performance_dashboard/frontend/index.html`
- `performance_dashboard/frontend/scripts/build.mjs`
- `performance_dashboard/frontend/src/main.tsx`
- `performance_dashboard/frontend/src/api.ts`
- `performance_dashboard/frontend/src/App.tsx`
- `performance_dashboard/frontend/src/styles.css`

Docs/tests:

- `performance_dashboard/README.md`
- `docs/contributing/performance_benchmarks.md`
- `fastvideo/tests/performance/test_dashboard_service.py`
- `fastvideo/tests/performance/test_dashboard_api.py`

Shared HF utility change:

- `fastvideo/tests/performance/hf_store.py`

## Data Source

The source of truth remains the Hugging Face dataset repo used by existing
performance CI:

```text
HF_REPO_ID=FastVideo/performance-tracking
```

The dataset stores normalized JSON records emitted by
`fastvideo/tests/performance/compare_baseline.py`. The current v1 normalized
schema includes:

- `model_id`
- `timestamp`
- `commit_sha`
- `gpu_type`
- `latency`
- `throughput`
- `memory`
- `text_encoder_time_s`
- `dit_time_s`
- `vae_decode_time_s`
- `success`

Records are grouped by `(model_id, gpu_type)` for v1 dashboard behavior.

## Local Cache

The backend syncs the HF dataset to a local cache directory:

```text
PERFORMANCE_TRACKING_ROOT=/tmp/fastvideo-perf-dashboard
```

If `PERFORMANCE_TRACKING_ROOT` is not set, the dashboard defaults to:

```text
/tmp/fastvideo-perf-dashboard
```

The sync is performed through the existing helper:

```python
fastvideo.tests.performance.hf_store.sync_from_hf(...)
```

The dashboard then loads JSON files from the local cache through:

```python
fastvideo.tests.performance.hf_store.load_records(...)
```

## Authentication

Originally `hf_store.py` only read `HF_API_KEY`. This caused local dashboard
runs to fail when users had standard Hugging Face token variables set.

`hf_store.py` now resolves tokens from the first available variable in:

```text
HF_API_KEY
HUGGINGFACE_HUB_TOKEN
HF_TOKEN
```

For local use:

```bash
export HF_TOKEN=hf_...
```

If the HF repo is private or gated, the token must have dataset read access.

## Backend API

The FastAPI app is created by:

```python
fastvideo.performance_dashboard.api:create_app
```

The module-level app is:

```python
fastvideo.performance_dashboard.api:app
```

Endpoints:

- `GET /api/performance/health`
- `POST /api/performance/refresh`
- `GET /api/performance/records?days=90`
- `GET /api/performance/summary?days=90`
- `GET /api/performance/trends?days=90`

`POST /api/performance/refresh` forces a fresh HF sync.

## Status Semantics

Important: the dashboard intentionally separates stored CI status from
recomputed context.

Stored status:

- Comes directly from the latest JSON record's `success` field.
- This is what the dashboard displays as `Stored Status`.
- This is the primary latest status.

Recomputed status:

- Calculated locally from cached records for explanatory context.
- Uses the latest record's metric values compared to the median of the latest
  five previous successful records in the same `(model_id, gpu_type)` group.
- Displayed separately as `Recomputed`.
- Does not override the stored JSON `success` status.

This distinction was added after observing that recomputing pass/fail from the
local cache can disagree with the status originally uploaded by CI.

## Time Window Behavior

The default dashboard time window is 90 days.

The selected `days` value affects:

- trend charts
- record browsing/filtering

The selected `days` value does not affect:

- latest stored status
- latest summary baseline context

Reason: latest status should not change when users widen or narrow the trend
window. The API keeps `days` on `/summary` only for shared frontend filter
state, but summary loading uses all cached records.

This fixed a bug where changing from roughly 35 days to 42 days could change
the latest status from pass to fail because older records entered the local
baseline window.

## Metric Logic

Dashboard metric definitions live in:

```text
fastvideo/performance_dashboard/metrics.py
```

Tracked metrics:

- `latency` lower is better
- `throughput` higher is better
- `memory` lower is better
- `text_encoder_time_s` lower is better
- `dit_time_s` lower is better
- `vae_decode_time_s` lower is better

Baseline context uses the median of up to five previous successful records for
the same `(model_id, gpu_type)`.

## Frontend Behavior

The React app:

- fetches `/api/performance/summary`
- fetches `/api/performance/trends`
- displays summary cards
- displays latest rows by model/GPU
- displays native SVG trend charts
- has model/GPU/day filters
- includes a refresh button
- auto-refreshes every five minutes

The UI is implemented without a charting library. Trend charts are native SVG
in `performance_dashboard/frontend/src/App.tsx`.

The production frontend build uses `esbuild` through
`performance_dashboard/frontend/scripts/build.mjs`. Vite is still used for the
dev server and `/api` proxy.

Why esbuild for production build:

- Vite/Rollup hit a local macOS native optional dependency code-signing issue
  in this environment.
- Direct esbuild worked reliably and is sufficient for this small dashboard.

## Static Serving

After frontend build, the FastAPI server serves:

- static JS/CSS from `performance_dashboard/frontend/dist/assets`
- `performance_dashboard/frontend/dist/index.html` for the dashboard page

This allows a single local port to serve both the API and UI.

## Local Run Workflow

Build frontend:

```bash
cd performance_dashboard/frontend
conda run -n fastvideo env PATH=/Applications/Codex.app/Contents/Resources/cua_node/bin:/usr/local/bin:/usr/bin:/bin \
  /Applications/Codex.app/Contents/Resources/cua_node/bin/npm install
conda run -n fastvideo env PATH=/Applications/Codex.app/Contents/Resources/cua_node/bin:/usr/local/bin:/usr/bin:/bin \
  /Applications/Codex.app/Contents/Resources/cua_node/bin/npm run build
```

Run dashboard:

```bash
export HF_TOKEN=hf_...
python -m fastvideo.performance_dashboard --host 0.0.0.0 --port 8000
```

Open locally:

```text
http://127.0.0.1:8000
```

## ngrok Workflow

`python -m fastvideo.performance_dashboard --host 0.0.0.0 --port 8000`
starts the actual local dashboard server.

`ngrok http 8000` does not start the dashboard. It exposes the already-running
local server through a temporary public URL.

Typical flow:

```bash
python -m fastvideo.performance_dashboard --host 0.0.0.0 --port 8000
ngrok http 8000
```

Use the HTTPS URL printed by ngrok to view the dashboard remotely.

## Verification Commands

Backend tests:

```bash
conda run -n fastvideo python -m pytest \
  fastvideo/tests/performance/test_dashboard_service.py \
  fastvideo/tests/performance/test_dashboard_api.py \
  -q
```

Expected after latest changes:

```text
8 passed
```

Frontend build:

```bash
cd performance_dashboard/frontend
conda run -n fastvideo env PATH=/Applications/Codex.app/Contents/Resources/cua_node/bin:/usr/local/bin:/usr/bin:/bin \
  /Applications/Codex.app/Contents/Resources/cua_node/bin/npm run build
```

Expected:

```text
tsc && node scripts/build.mjs
```

with exit code 0.

## Known Notes

- `performance_dashboard/frontend/node_modules/` and
  `performance_dashboard/frontend/dist/` are ignored by git.
- `npm install` reported two high-severity audit findings in dependency tree.
  `npm audit fix --force` was not run because it can introduce breaking
  dependency upgrades.
- Existing `fastvideo` package imports may emit platform warnings such as NPU
  or macOS torch distributed messages. These are not dashboard-specific errors.

