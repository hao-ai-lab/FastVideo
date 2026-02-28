# FastVideo Job Runner UI

A lightweight web-based UI for creating and managing FastVideo video generation
jobs.

## Quick Start

The easiest way to run the application is with a single command that starts both
the API server and the Next.js frontend:

```bash
cd ui
npm install
npm run build
npm run start
```

This runs the API server at [http://localhost:8189](http://localhost:8189) and the Next.js app
at [http://localhost:3000](http://localhost:3000) via `concurrently`.

### Running API and Web Separately

To run the API server only:

```bash
python -m ui.server --output-dir /path/to/videos --log-dir /path/to/logs
```

The API server defaults to port 8189. Configure the frontend to point to it:

```bash
cd ui
cp .env.example .env.local
# Set NEXT_PUBLIC_API_BASE_URL=http://localhost:8189/api in .env.local
```

Then run the Next.js dev server (with hot reload):

```bash
npm install && npm run dev
```

## Features

- Select from supported FastVideo text-to-video models
- Enter a prompt and configure generation parameters (steps, frames, resolution,
  guidance scale, seed, GPU count)
- Create, start, stop, and delete jobs via the UI
- Live-polling job status updates
- Download video for completed jobs
- Generated videos are saved to a configurable output directory

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
| `GET`    | `/api/jobs/{id}/logs`        | Get job logs (polling, supports `?after=`) |
| `GET`    | `/api/jobs/{id}/download_log`| Download the job's log file        |

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
├── server.py            # FastAPI server (REST endpoints for job management)
├── job_runner.py        # Job lifecycle, execution, and generator caching
├── package.json         # npm scripts (start, dev, build)
└── src/
    ├── app/             # Next.js app router (page.tsx, layout.tsx)
    ├── components/      # React components (JobCard, CreateJobModal)
    └── lib/             # API client and types
```

- **API Server** (`server.py`): A FastAPI server that manages an in-memory job store. Each job runs
  in a daemon thread that uses `fastvideo.VideoGenerator` to generate videos.
  Model instances are cached so switching between prompts on the same model
  doesn't reload weights. Provides REST endpoints under `/api/*`.
  - **Error Handling**: Jobs that crash are automatically marked as `FAILED` without
    crashing the server. Error details are stored in the job's `error` field.
  - **Log Files**: Each job maintains a persistent log file (`{job_id}.log`) in a
    dedicated log directory (configurable via `--log-dir`), containing all logs from
    model loading through completion or failure. Log files are named after the job ID
    for easy identification.
- **Frontend**: A Next.js (React) application. Jobs are polled when running and
  rendered as cards with status badges and action buttons. The API base URL is
  configured via `NEXT_PUBLIC_API_BASE_URL` in `.env.local`.
