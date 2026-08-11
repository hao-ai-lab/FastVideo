# FastVideo Studio

A lightweight web-based UI for interacting with FastVideo.

## Features

The UI currently supports:

- Inference
- Finetuning & Distillation
- Datasets
- Gallery View

## Project Structure

```
apps/fastvideo_studio/
├── server.py / job_runner.py / database.py   # FastAPI backend + job lifecycle
├── mock_server.py                            # In-memory API mock for e2e tests
├── models/                                   # Pydantic request models (shared with the mock)
├── training_config.py                        # Studio workloads → fastvideo/train YAML configs
├── tests/                                    # Backend unit tests (pytest)
├── e2e/                                      # Playwright specs (run against the mock)
└── src/
    ├── app/                                  # Next.js App Router pages (thin routes)
    ├── components/
    │   ├── shell/                            # App chrome: header, sidebars, layout
    │   ├── jobs/                             # Job queue, cards, create-job modal, log sidebar
    │   ├── datasets/                         # Dataset cards, upload, captions
    │   └── ui/                               # shadcn-style primitives (shared theme)
    ├── stores/                               # Framework-agnostic state + React bridge (hooks/)
    ├── lib/                                  # API client, types, option persistence
    └── test/                                 # Vitest setup + factories
```

The visual theme (slate light/dark palettes, IBM Plex type, `#356cff` accent)
is shared with `apps/dreamverse`; the toggle in the header persists the choice
per browser.

## Testing

```bash
npm run typecheck        # tsc
npm test                 # vitest unit tests
npm run e2e              # Playwright against the in-memory mock backend
python -m pytest tests/  # backend unit tests (from apps/fastvideo_studio)
```

## Quick Start

For local development, install dependencies and start the Next.js dev server:

```bash
cd apps/fastvideo_studio
npm install
npm run dev
```

You can then access the app at [http://localhost:3000](http://localhost:3000).
Start the Python API server (default port 8189) in a separate terminal — see below.

For a production build of the web app together with the API server:

```bash
cd apps/fastvideo_studio
npm install
npm run build
npm run start:all
```

### Running API and Web Separately

The UI is composed of two separate components:

- API Server
- Web Server

To run each component separately, you can use the commands `npm run start:web` (after `npm run build`) and `npm run start:api`.

You can also run the API server with this command from the `apps/` directory:

```bash
python -m fastvideo_studio.server --output-dir /path/to/videos --log-dir /path/to/logs
```

Running it this way allows you to pass command line parameters.

The API server defaults to port 8189. To configure this, you can edit `.env.local` file.
Refer to `.env.example` for reference.

### API Endpoints

| Method   | Path                          | Description                                |
| -------- | ----------------------------- | ------------------------------------------ |
| `GET`    | `/api/models`                 | List available models                      |
| `GET`    | `/api/jobs`                   | List all jobs (newest first)               |
| `GET`    | `/api/jobs/{id}`              | Get a single job's details                 |
| `POST`   | `/api/jobs`                   | Create a new job                           |
| `POST`   | `/api/jobs/{id}/start`        | Start a pending/stopped/failed job         |
| `POST`   | `/api/jobs/{id}/stop`         | Request a running job to stop              |
| `DELETE` | `/api/jobs/{id}`              | Delete a job                               |
| `GET`    | `/api/jobs/{id}/video`        | Stream the generated video/image           |
| `GET`    | `/api/jobs/{id}/logs`         | Get job logs (polling, supports `?after=`) |
| `GET`    | `/api/jobs/{id}/download_log` | Download the job's log file                |
