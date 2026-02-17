# FastVideo Job Runner UI

A lightweight web-based UI for creating and managing FastVideo video generation
jobs.

## Features

- Select from supported FastVideo text-to-video models
- Enter a prompt and configure generation parameters (steps, frames, resolution,
  guidance scale, seed, GPU count)
- Create, start, stop, and delete jobs via the UI
- Live-polling job status updates
- In-browser video preview for completed jobs
- Generated videos are saved to a configurable output directory

## Quick Start

### Option 1: Combined Server (Default)

Run both the API and web server together:

```bash
# From the repository root
pip install fastvideo fastapi uvicorn

# Launch the combined server (defaults to http://0.0.0.0:8188)
python -m ui.server

# Or with a custom output directory
python -m ui.server --output-dir /path/to/videos --port 8080
```

Then open [http://localhost:8188](http://localhost:8188) in your browser.

### Option 2: Separate API and Web Servers

Run the API server and web server separately for better scalability:

```bash
# Terminal 1: Start the API server (defaults to http://0.0.0.0:8189)
python -m ui.api_server --output-dir /path/to/videos

# Terminal 2: Start the web server (defaults to http://0.0.0.0:8188)
# With API proxy (recommended):
python -m ui.web_server --api-url http://localhost:8189

# Or without proxy (requires CORS on API server):
python -m ui.web_server
```

Then open [http://localhost:8188](http://localhost:8188) in your browser.

## API Endpoints

| Method   | Path                         | Description                        |
| -------- | ---------------------------- | ---------------------------------- |
| `GET`    | `/api/models`                | List available models              |
| `GET`    | `/api/jobs`                  | List all jobs (newest first)       |
| `GET`    | `/api/jobs/{id}`             | Get a single job's details         |
| `POST`   | `/api/jobs`                  | Create a new job                   |
| `POST`   | `/api/jobs/{id}/start`       | Start a pending/stopped/failed job |
| `POST`   | `/api/jobs/{id}/stop`        | Request a running job to stop      |
| `DELETE` | `/api/jobs/{id}`             | Delete a job                       |
| `GET`    | `/api/jobs/{id}/video`       | Stream the generated video/image   |

### Create Job Request Body

```json
{
  "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "prompt": "A curious raccoon in a sunflower field",
  "num_inference_steps": 50,
  "num_frames": 81,
  "height": 480,
  "width": 832,
  "guidance_scale": 5.0,
  "seed": 1024,
  "num_gpus": 1
}
```

## Architecture

```
ui/
├── server.py            # Combined FastAPI server (API + static files)
├── api_server.py        # API-only server (REST endpoints)
├── web_server.py        # Web-only server (static files + optional API proxy)
├── requirements.txt     # Python dependencies (fastapi, uvicorn, httpx)
└── static/
    ├── index.html       # Single-page application
    ├── style.css        # Dark-themed responsive styles
    └── app.js           # Frontend logic (fetch API, polling, rendering)
```

- **API Server** (`api_server.py`): A FastAPI server that manages an in-memory job store. Each job runs
  in a daemon thread that uses `fastvideo.VideoGenerator` to generate videos.
  Model instances are cached so switching between prompts on the same model
  doesn't reload weights. Provides REST endpoints under `/api/*`.
- **Web Server** (`web_server.py`): Serves static HTML/CSS/JS files. Optionally proxies API requests
  to a separate API server or relies on CORS for cross-origin requests.
- **Combined Server** (`server.py`): Legacy combined server that serves both API and static files
  from a single process. Use this for simple deployments.
- **Frontend**: A vanilla HTML/CSS/JS single-page app. Jobs are polled every
  2 seconds and rendered as cards with status badges and action buttons. The API
  base URL can be configured via a meta tag injected by the web server.
