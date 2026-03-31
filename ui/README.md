# FastVideo UI

A lightweight web-based UI for interacting with FastVideo.

## Features

The UI currently supports:

- Inference
- Finetuning & Distillation
- Datasets
- Gallery View

## Quick Start

The easiest way to run the application is:

```bash
cd ui
npm install
npm run build
npm run start
```

You can then access the app at [http://localhost:3000](http://localhost:3000).

### Running API and Web Separately

The UI is composed of two separate components:

- API Server
- Web Server

To run each component separately, you can use the commands `npm run start:web` and `npm run start:api`.

You can also run the API server with this command from the root directory:

```bash
python -m ui.server --output-dir /path/to/videos --log-dir /path/to/logs
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
