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
| `GET`    | `/api/models`                 | List models with capabilities and recipes  |
| `GET`    | `/api/jobs`                   | List all jobs (newest first)               |
| `GET`    | `/api/jobs/{id}`              | Get a single job's details                 |
| `POST`   | `/api/jobs`                   | Create a new job                           |
| `POST`   | `/api/jobs/{id}/start`        | Start a pending/stopped/failed job         |
| `POST`   | `/api/jobs/{id}/stop`         | Request a running job to stop              |
| `PATCH`  | `/api/jobs/{id}`              | Edit a pending/stopped/failed job          |
| `POST`   | `/api/jobs/{id}/duplicate`    | Copy a job's configuration as a new job    |
| `DELETE` | `/api/jobs/{id}`              | Delete a job                               |
| `GET`    | `/api/jobs/{id}/video`        | Stream the generated video/image           |
| `GET`    | `/api/jobs/{id}/logs`         | Get job logs (polling, supports `?after=`) |
| `GET`    | `/api/jobs/{id}/download_log` | Download the job's log file                |
| `POST`   | `/api/upload-media`           | Upload an image/video/audio reference      |
| `GET`    | `/api/uploads/preview`        | Serve an upload for previews (`?path=`)    |

`GET /api/models` reports, per model, `capabilities` (references and their
limits, start/end frame, frame and GPU limits); the create-job form shows
inputs from these, not from model names. For a FastH3-Omni export it also
reports the locked `recipe`, the `hardware` verdict, and any load `error`.

### FastH3-Omni

FastH3-Omni (a FastH3 OmniRef PDD export) is offered only when the API server
is started with it; a request can never name an arbitrary directory:

```bash
python -m fastvideo_studio.server \
  --fasth3-omni-export /path/to/omniref_export \
  --fasth3-omni-export SomeOrg/FastH3-Omni@<revision> \
  --fasth3-omni-base /path/to/MiniMax-H3   # optional; default: the base the export pins
```

`FASTVIDEO_STUDIO_FASTH3_OMNI_EXPORTS` (comma-separated) and
`FASTVIDEO_STUDIO_FASTH3_OMNI_BASE` do the same for `npm run start:api`.

- **Recipe.** The server reads each export's `fastvideo_inference.json` with
  FastVideo's own validator. Steps (fused PDD blocks, one transformer forward
  each), CFG, VSA sparsity/tile, fps, compile and parallel layout are locked to
  it; the form shows them read-only and the server overwrites whatever a
  request sends. Frames, resolution, seed, GPU count and encoder offload stay
  editable. An export that is not a Ref2VA PDD export (for example FastH3 V2,
  a T2AV DMD export) is listed as unavailable.
- **Weights.** On the first run the export is composed with its base
  (`transformer_ref` and schedulers from the export; text encoder, processor
  and both VAEs from the base) using the official
  `examples/inference/basic/basic_fasth3_omniref_pdd.py` logic, into
  `<data-dir>/models/fasth3_omni/<key>/` with a `<key>.json` provenance file.
  The key covers both snapshots, so a new export or base revision never
  reuses an old directory. A missing component fails the job as a weights
  error.
- **Hardware.** The PDD-8 contract's 128-token VSA tiles run only on the
  sm_100a/sm_103a kernel (B200/B300/GB200/GB300) with the Blackwell VSA
  extension built. On other GPUs the model is marked unsupported and a job
  fails before any weights are composed or loaded.
  `FASTVIDEO_STUDIO_SKIP_HARDWARE_CHECK=1` hands that check back to FastVideo.
- **References** are ordered (labels `<Picture 1>`, `<Video 1>`, … follow the
  list order) and need at least one image or video. Order, recipe and failure
  category (`weights`, `parameters`, `kernel`, `memory`, `runtime`) are stored
  with the job and survive edits, duplication and restarts.
