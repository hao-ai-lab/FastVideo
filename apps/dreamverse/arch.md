# Dreamverse Architecture

## Overview

Dreamverse currently has two main runtime pieces:

- `apps/dreamverse/web/`: Next.js frontend
- `apps/dreamverse/dreamverse/`: Python FastAPI runtime

Today, the browser talks directly to the Dreamverse runtime over HTTP and a
single websocket on `/ws`. The frontend owns UI state and interaction flow. The
server owns generation state, prompt rewrite, prompt safety, websocket session
semantics, and GPU-backed execution.

Near-term OSS note:

- `apps/dreamverse/dreamverse/` is the current runtime implementation.
- A future `controller/` layer is planned for local-only compute management and
  provider orchestration, but it does not exist yet.

## Repo Map

### Frontend

- `apps/dreamverse/web/src/app/page.tsx`: main client orchestration,
  websocket connect, init payloads, send paths, and top-level app behavior
- `apps/dreamverse/web/src/lib/ws/reducer.ts`: reduces normalized websocket
  events into client stores
- `apps/dreamverse/web/src/stores/session.ts`: connection, mode, and top-level
  session UI
- `apps/dreamverse/web/src/stores/promptWindow.ts`: editable prompt window and
  seed prompt UI state
- `apps/dreamverse/web/src/stores/rewrite.ts`: rewrite activity timeline and
  inspection state
- `apps/dreamverse/web/src/stores/stream.ts`: playback and stream-related
  client state
- `apps/dreamverse/web/src/lib/prompts/promptWindowSnapshot.ts`: prompt-window snapshot
  building for rewrite requests

### Server

- `apps/dreamverse/dreamverse/main.py`: websocket endpoint, request handling,
  session state machine, rewrite orchestration, REST routes, and stream relay
- `apps/dreamverse/dreamverse/gpu_pool.py`: GPU worker processes, warmup, model
  loading, and `generate_video()` calls through FastVideo
- `apps/dreamverse/dreamverse/prompt_enhancer.py`: prompt enhancement, rollout
  rewrite execution, provider selection, and timeout/fallback behavior
- `apps/dreamverse/dreamverse/rewrite_prompt_payload.py`: canonical rewrite request payload
  building
- `apps/dreamverse/dreamverse/config.py`: runtime flags, prompt file paths, provider settings, and
  warmup config
- `apps/dreamverse/dreamverse/session_init_image.py`: validates and persists uploaded initial
  images for segment 1

## Current Split Of Responsibility

### Frontend owns

- local UI state and client-side stores
- prompt drafts and prompt window editing
- websocket connection management
- deciding which user action to send:
  - `session_init_v2`
  - `project_init_v1`
  - `append_prompt`
  - `rewrite_seed_prompts`
  - `simple_generate`
- showing rewrite progress, stream status, prompt history, and devtools views

### Server owns

- websocket session lifecycle and protocol
- GPU assignment and worker lifecycle
- the authoritative seed prompt memory used for generation
- prompt rewrite execution and prompt safety
- actual generation queue semantics
- stream chunk emission and segment lifecycle events
- prompt config and preset persistence routes
- health and readiness endpoints

Important rule:

- The frontend may propose prompt-window state for rewrite, but the server is
  the source of truth for the rewritten rollout and the active prompt memory
  used for generation.

## End-To-End Flow

1. The frontend opens `/ws`.
2. The frontend sends `session_init_v2` with the initial prompt-window state,
   preset metadata, and current toggles.
3. The server validates init data, persists an optional initial image, acquires
   a GPU slot, and emits session status such as `gpu_assigned`.
4. The server starts or resumes project generation and emits events like
   `ltx2_stream_start`, `ltx2_segment_start`, media init/chunks, and completion
   events.
5. The frontend reduces those websocket events into its stores and updates the
   UI.
6. User actions such as appending prompts, rewriting seed prompts, or starting
   a single custom clip go back to the server over the same websocket.

## Frontend Architecture

The frontend is store-driven.

- `page.tsx` wires together websocket setup, send helpers, reducer
  application, and top-level interaction flows.
- Store modules separate concerns like session state, rewrite state, prompt
  window state, and stream state.
- The websocket reducer is responsible for turning normalized runtime events
  into store updates. If the server event schema changes, the reducer must
  change with it.

The frontend is intentionally not responsible for:

- generating rewritten prompts locally
- deciding final prompt safety outcomes
- reconstructing server session state from scratch
- inventing its own generation semantics independent of the runtime

## Server Architecture

The current runtime is a FastAPI app with a single long-lived websocket per
session.

`apps/dreamverse/dreamverse/main.py` manages:

- websocket connect/init
- prompt queues
- project and segment state
- prompt enhancement/rewrite triggers
- stream relay from GPU workers to the browser
- session logging and REST endpoints

`apps/dreamverse/dreamverse/gpu_pool.py` manages:

- model loading through FastVideo
- one or more worker processes
- startup warmup
- user join/leave commands
- `USER_STEP` execution for each segment
- continuation state between segments

`apps/dreamverse/dreamverse/prompt_enhancer.py` manages:

- prompt enhancement for user-submitted prompts
- rollout rewrite requests for the prompt window
- provider selection and fallback across configured prompt providers
- response normalization and safety-aware failure handling

## FastAPI Surface

The server is a single FastAPI application created in
`apps/dreamverse/dreamverse/main.py`.

Current built-in FastAPI docs are enabled:

- `/docs`: Swagger UI
- `/redoc`: ReDoc
- `/openapi.json`: OpenAPI schema

The runtime also mounts the frontend static build at `/` when one of the
configured frontend static directories exists. It does not expose the backend
Python package as static content.

## HTTP API

The current HTTP API is small. Most realtime behavior still goes through the
websocket.

### Core health and status routes

- `GET /healthz`
  - process liveness probe
  - returns a small payload with `status`, `service`, and timestamp
- `GET /readyz`
  - readiness probe
  - returns `503` until prompt services are initialized and at least one GPU
    worker is ready
  - returns readiness and GPU pool summary fields such as ready workers, total
    GPUs, warmup counts, and queue size
- `GET /status`
  - returns the current GPU pool status payload from `gpu_pool`
- `GET /internal/monitor/sessions`
  - internal monitoring payload for session dashboards
  - includes pending session count, max available sessions, prompt provider
    success counts, and timestamp

### Prompt config routes

- `GET /prompt-system-config`
  - returns the editable prompt-system configuration currently loaded by
    `PromptEnhancer`
- `POST /prompt-system-config`
  - saves prompt-system configuration to disk and reloads prompt config in the
    runtime
  - current editable fields include:
    - next-segment system prompt
    - auto-extension system prompt
    - rewrite-window system prompt
    - rewrite-user system prompt
    - rewrite model
    - rewrite temperature

### Devtools-only preset routes

These exist only when `DEVTOOLS_ENABLED` is true in
`apps/dreamverse/dreamverse/config.py`.

- `GET /curated-presets`
  - returns merged curated presets, applying the local overlay file on top of
    the fallback file when both exist
- `POST /curated-presets/append`
  - appends a new curated preset to the overlay presets file
  - validates non-empty label, normalized id, and at least two non-empty
    segment prompts

## Websocket API

`WS /ws` is the main runtime API.

The websocket owns:

- session init
- project init and reset
- prompt append
- prompt rewrite
- generation toggles
- segment lifecycle events
- media stream delivery
- runtime error delivery

The websocket is the authoritative API for realtime Dreamverse behavior. The
HTTP routes mainly support health checks, devtools persistence, and monitoring.

## Prompt Rewrite Architecture

Prompt rewrite is a shared flow with strict ownership boundaries.

### Frontend responsibilities

- collect the rewrite instruction
- build the prompt-window snapshot from current client state
- send `rewrite_seed_prompts`
- show rewrite progress, raw output, fallback state, and resulting prompt list

### Server responsibilities

- validate and normalize the prompt-window payload
- choose rewrite model, system prompt, timeout, and temperature
- build the canonical prompt payload in
  `apps/dreamverse/dreamverse/rewrite_prompt_payload.py`
- execute rewrite through `PromptEnhancer`
- apply safety filtering to rewritten prompts
- replace the authoritative seed prompt memory when rewrite succeeds
- emit `seed_prompts_updated` and `rewrite_seed_prompts_complete`

Important rule:

- The frontend owns editable drafts.
- The server owns the accepted rollout.

After a successful rewrite, the frontend should replace its prompt-window view
from the server payload instead of preserving a locally-derived version.

## Prompt Modes

There are three related prompt paths in the current system:

### Initial rollout

- The frontend sends seed prompts during `session_init_v2`.
- The server uses those prompts as the initial seed prompt memory.
- If the rollout starts from an empty prompt window plus an initial rewrite
  instruction, the server can pause generation until rewrite completes.

### Live append

- The frontend sends `append_prompt`.
- The server may enhance that prompt, safety-check it, enqueue it, and use it
  as the next generated segment.

### Rewrite

- The frontend sends `rewrite_seed_prompts`.
- The server rewrites the entire seed prompt window or generates a new rollout,
  depending on the payload and current state.

## Initial Image And Segment Handling

The frontend currently sends `initial_image` as part of session init or
`simple_generate`.

The server:

- validates and persists the image
- uses it only for segment 1 when present
- keeps continuation state for later segments in the GPU worker

This means the runtime, not the frontend, decides how segment 1 image
conditioning and later continuation conditioning are applied.

## Websocket Contract

The websocket is the main integration surface between UI and runtime.

Typical incoming messages from the frontend:

- `session_init_v2`
- `project_init_v1`
- `append_prompt`
- `rewrite_seed_prompts`
- `simple_generate`
- `set_enhancement`
- `set_auto_extension`
- `set_loop_generation`

Typical outgoing messages from the server:

- `gpu_assigned`
- `ltx2_stream_start`
- `ltx2_segment_start`
- `segment_prompt_source`
- `prompt_received`
- `prompt_ready`
- `prompt_enhancing`
- `seed_prompts_updated`
- `rewrite_seed_prompts_complete`
- `media_init`
- `media_segment_complete`
- `project_idle`
- `error`

Binary websocket frames carry media chunks for playback.

## Current And Planned Architecture

Current architecture:

- browser -> `apps/dreamverse/web`
- `apps/dreamverse/web` -> `apps/dreamverse/dreamverse/main.py`
- `apps/dreamverse/dreamverse/main.py` ->
  `apps/dreamverse/dreamverse/gpu_pool.py`
- `gpu_pool.py` -> FastVideo runtime

Planned architecture:

- browser -> `apps/dreamverse/web`
- `apps/dreamverse/web` -> local `controller/`
- `controller/` -> local or remote Dreamverse runtime
- runtime -> FastVideo runtime

That future controller split should not move prompt rewrite, session state, or
generation semantics out of the runtime.
