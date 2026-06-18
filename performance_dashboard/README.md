# FastVideo Performance Dashboard

Local FastAPI + React dashboard for records stored in the Hugging Face
performance tracking dataset.

## Data Source

The dashboard reads the same normalized JSON records used by
`fastvideo/tests/performance/compare_baseline.py`.

Defaults:

- `HF_REPO_ID=FastVideo/performance-tracking`
- `PERFORMANCE_TRACKING_ROOT=/tmp/fastvideo-perf-dashboard`
- `PERF_MAX_REGRESSION=0.05`

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

## API

- `GET /api/performance/health`
- `POST /api/performance/refresh`
- `GET /api/performance/summary?days=90`
- `GET /api/performance/trends?days=90`
- `GET /api/performance/records?days=90`

The current v1 grouping key is `(model_id, gpu_type)`. Baselines are computed
from the latest five previous successful records in each group.
