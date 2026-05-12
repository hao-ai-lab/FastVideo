# Dreamverse OSS Design

## Overview

Dreamverse should ship as a local-first open source application.

- The browser talks only to a local Dreamverse control plane on the user's
  machine.
- Provider credentials stay local to that machine.
- Dreamverse may provision compute on the user's behalf, but Dreamverse does
  not host that control path as a service.

This keeps the UX simple without turning Dreamverse into a credential-holding
hosted platform.

## Goals

- Support three compute modes behind one product surface:
  - local GPU
  - managed remote GPU via Runpod
  - managed remote GPU via Modal
- Keep prompt rewrite, websocket session state, and generation behavior
  consistent across providers.
- Keep provider API keys out of browser state and out of any hosted service.
- Make the existing runtime reusable as the common serving contract.
- Minimize provider-specific code and isolate it behind a narrow interface.

## Non-goals

- Do not make the frontend call provider APIs directly.
- Do not unify providers at the level of SSH, VM, serverless, or pod
  semantics.
- Do not move prompt rewrite logic into the frontend or controller.
- Do not require remote compute for the basic product path.

## Current State

Today the repo contains two major pieces:

- `apps/dreamverse/web/`: Next.js frontend
- `apps/dreamverse/dreamverse/`: FastAPI runtime that owns websocket state,
  prompt rewrite, prompt safety, and GPU-backed generation

The current runtime already exposes useful health and streaming surfaces such
as `/healthz`, `/readyz`, `/status`, and `/ws`.

## Target Architecture

The target open source structure should be:

```text
Dreamverse/
├── apps/dreamverse/
│   ├── web/          # browser UI
│   ├── dreamverse/   # current FastAPI websocket/generation runtime
│   ├── controller/   # local control plane and provider lifecycle
│   ├── providers/    # provider adapters
│   ├── tests/
│   │   ├── contract/
│   │   ├── controller/
│   │   └── smoke/
│   └── design.md
└── ...
```

Near-term note:

- `apps/dreamverse/dreamverse/` is the current runtime implementation.
- We can keep the code there initially and rename it to `runtime/` only after
  the controller lands.

## Trust Model

Dreamverse is local-only for control and secrets.

- The user launches Dreamverse on their own machine.
- Provider API keys are entered into the local app or local CLI.
- The controller uses those credentials to provision or connect to compute.
- The browser never talks to Modal or Runpod directly.
- Dreamverse-hosted infrastructure is not involved.

This is the key reason the provider-based path is acceptable for OSS.

## Responsibility Split

### `apps/dreamverse/web`

The frontend should own:

- UI state, drafts, and local interaction state
- websocket event reduction into client stores
- selection of compute mode and display of cost/health/status
- local forms for provider configuration
- sending prompt requests and rewrite requests to the local controller

The frontend should not own:

- provider credentials after submission
- provider API calls
- runtime lifecycle
- authoritative prompt window after rewrite
- prompt safety or generation policy

### `controller`

The local controller should own:

- provider credential loading and local-only storage
- compute mode selection
- provisioning, reuse, shutdown, and health monitoring of runtimes
- reverse proxying HTTP and websocket traffic from the frontend to the active
  runtime
- user-visible status such as provisioning, ready, failed, and idle shutdown
- local persistence for user settings that must survive ephemeral runtimes

The controller should not own:

- prompt rewrite logic
- seed prompt memory semantics
- generation queue behavior
- provider-specific UI state

### `runtime`

The runtime should remain the authoritative owner of:

- `/ws` session state
- prompt rewrite execution
- prompt safety
- seed prompt memory and prompt-window state used for generation
- generation orchestration and GPU worker lifecycle
- websocket event schemas

This preserves the current model and avoids splitting state across layers.

## Runtime Contract

Provider abstraction should happen around a stable Dreamverse runtime contract,
not around infrastructure details.

Minimum runtime surface:

- `GET /healthz`
- `GET /readyz`
- `GET /status`
- `GET/POST /prompt-system-config` if devtools persists config through the
  runtime
- curated preset routes if those remain runtime-backed
- `WS /ws`

Important rule:

- The controller only needs to know how to reach a healthy runtime.
- The runtime remains provider-agnostic.

## Provider Abstraction

Use a narrow provider interface:

```python
class ComputeProvider(Protocol):
    async def ensure_runtime(self, spec: RuntimeSpec) -> RuntimeHandle: ...
    async def wait_until_ready(self, handle: RuntimeHandle) -> None: ...
    async def stop_runtime(self, handle: RuntimeHandle) -> None: ...
```

`RuntimeHandle` should include:

- `provider`
- `runtime_id`
- `base_url`
- `ws_url`
- runtime auth headers or tokens if needed
- lifecycle metadata
- cost or hardware metadata for UI display

The controller should work only with `RuntimeHandle`, never with raw SSH hosts
or provider-specific payloads after resolution.

## Provider Notes

### Local

Local mode should be the reference implementation.

- Start the runtime as a local subprocess or connect to an already-running
  local runtime URL.
- Reuse the same runtime contract as remote providers.
- Make this the first supported path and the main smoke-test target.

### Runpod

Runpod should be treated as pod lifecycle plus runtime reachability.

- Prefer prepared images or templates that auto-start the Dreamverse runtime.
- Prefer exposed HTTP/TCP ports for steady-state traffic.
- Use SSH only for bootstrap fallback, diagnostics, or repair.
- Avoid a design where the controller shells into the pod for every action.

### Modal

Modal should be treated as deployment-based runtime hosting.

- Wrap the Dreamverse runtime in a thin Modal entrypoint if needed.
- Reuse the same runtime behavior behind that wrapper.
- Do not model Modal as a machine that Dreamverse logs into.
- Do not force the websocket runtime into a per-request serverless handler
  shape.

## Config and Persistence

Remote compute may be ephemeral, so mutable user configuration should not live
only inside remote runtimes.

Keep durable state local to the user's machine unless there is a strong reason
otherwise:

- provider selection
- provider credentials or credential references
- default hardware preferences
- editable prompt presets
- prompt system prompt overrides
- idle shutdown policy

Runtime-local state should be treated as disposable unless explicitly synced.

## Prompt Rewrite Ownership

Prompt rewrite remains runtime-owned even after the controller is added.

Frontend responsibilities:

- collect the rewrite instruction
- build the prompt-window snapshot
- display rewrite activity and results

Runtime responsibilities:

- validate and normalize the prompt window
- choose the rewrite system prompt and model settings
- execute rewrite
- apply safety filtering
- replace authoritative seed prompt memory
- emit the canonical completion events

Controller responsibilities:

- proxy the request and response
- surface runtime availability and failure state

This boundary should not move.

## Recommended Rollout

1. Finish the path reorg so docs and code agree on `apps/dreamverse/web`.
2. Introduce `controller/` as a local-only API/proxy process.
3. Keep `apps/dreamverse/dreamverse/` as the runtime and adapt it behind the
   controller.
4. Add `local` provider first.
5. Add "bring your own runtime URL" as an escape hatch.
6. Add automated Runpod provisioning.
7. Add Modal deployment support.
8. Rename `apps/dreamverse/dreamverse/` to `runtime/` once the split is stable.

## Implementation Plan

The implementation should start with the smallest milestone that gives users a
working local GPU setup without forcing the controller/provider architecture
into the first patch series.

### Milestone 0: Make local GPU the official baseline

Goal:

- A user with a working `fastvideo` install can run the Dreamverse backend on a
  local GPU and connect to it from `apps/dreamverse/web`.

Non-goals for this milestone:

- no controller process yet
- no provider abstraction yet
- no Runpod or Modal support yet
- no secret-management UI yet

Reasoning:

- `apps/dreamverse/dreamverse/` already is the real local GPU runtime.
- `apps/dreamverse/web` already knows how to talk to a backend over `/ws` and
  REST rewrites.
- The shortest path is to make the existing local path explicit, reliable, and
  tested before adding another layer.

### Milestone 0 work items

#### 0.1 Fix repo path assumptions after the frontend move

Current issue:

- Some paths still assume `prod-ui/`, but the frontend now lives at
  `apps/dreamverse/web/`.

Required changes:

- update prompt/preset path resolution in
  `apps/dreamverse/dreamverse/config.py`
- update docs that still mention `prod-ui`
- audit any frontend build settings that assume the old repo root

This is prerequisite cleanup. Local GPU mode should not depend on stale
monorepo paths.

#### 0.2 Make local runtime startup the primary supported entrypoint

Required outcome:

- one documented backend command
- one documented frontend command
- one clear env contract for local development

Expected shape:

```bash
uv pip install -e ".[dreamverse]"
dreamverse-server --host 0.0.0.0 --port 8009

cd apps/dreamverse/web
npm ci
BACKEND_HOST=localhost BACKEND_PORT=8009 npm run dev
```

Optional but useful:

- add a small root helper script or Make target for local startup
- add a `dreamverse-doctor` or lightweight startup check later

#### 0.3 Define the minimum local runtime contract

For Milestone 0, the frontend should rely only on the current runtime surface:

- `/ws`
- `/status`
- `/healthz`
- `/readyz`
- existing prompt/devtools routes

Do not add a second local API layer yet unless the current runtime surface is
proven insufficient.

#### 0.4 Make failure states explicit in the UI

Local GPU mode fails in a few predictable ways:

- backend not reachable
- backend reachable but not ready
- `fastvideo` or model runtime missing
- no compatible GPU available

Minimum implementation:

- show a clear connection error when `/ws` or `/status` fails
- surface readiness failures in a human-readable way
- avoid silent retry loops that hide backend startup failures

This is a small UI pass, not a controller project.

#### 0.5 Add a minimal local smoke test path

At this milestone, local GPU support is "done" only if there is a repeatable
test path for the local runtime contract.

Minimum test additions:

- backend tests for `/healthz`, `/readyz`, and `/status`
- a frontend integration test that assumes a reachable backend URL and verifies
  connection lifecycle behavior
- one local smoke script that starts the backend and verifies readiness before
  the frontend is launched

### Milestone 1: Introduce a thin local controller

Goal:

- Preserve the same local GPU behavior, but place a stable local control-plane
  API in front of the runtime.

This should happen only after Milestone 0 is stable.

Scope:

- add `controller/`
- proxy `/ws` and the needed REST routes to `apps/dreamverse/dreamverse/`
- expose controller-owned status for "backend starting", "runtime ready", and
  "runtime failed"
- optionally spawn the local runtime as a subprocess

Non-goal:

- do not add remote provider logic yet

Reasoning:

- the controller earns its complexity only once it stabilizes the local
  contract that future providers will share

### Milestone 2: Provider abstraction on top of the controller

Goal:

- Keep the same frontend contract while allowing the controller to resolve a
  runtime via `local`, then later `runpod` and `modal`.

At this point:

- define `ComputeProvider`
- implement `providers/local.py`
- move local-runtime subprocess management behind the provider interface

The first provider should be `local`, because it is cheapest to debug and
matches the runtime most closely.

## Minimal Code Change Order

If we want the shortest path to a working local GPU milestone, the change order
should be:

1. Fix `apps/dreamverse/dreamverse/config.py` and any remaining path
   assumptions from `prod-ui` to `apps/dreamverse/web`.
2. Update `README.md` to document the real local GPU startup flow.
3. Confirm `apps/dreamverse/web` connects cleanly to the local wrapper-backed
   backend.
4. Improve frontend error handling for backend-not-ready and backend-missing
   cases.
5. Add a local smoke test and keep existing backend/frontend tests green.
6. Only then introduce `controller/`.

## Test Plan for the Local GPU Milestone

### Backend

Keep the current Python test suite as the base:

- `apps/dreamverse/dreamverse/tests/test_health_endpoints.py`
- `apps/dreamverse/dreamverse/tests/test_mock_server.py`
- `apps/dreamverse/dreamverse/tests/test_prompt_enhancer.py`
- `apps/dreamverse/dreamverse/tests/test_rewrite_prompt_payload.py`
- related config and logging tests

Add or tighten tests for:

- path resolution in `apps/dreamverse/dreamverse/config.py`
- readiness behavior when GPU pool initialization fails
- startup error messaging when `fastvideo` is unavailable

### Frontend

Keep the current Vitest suite as the base:

- websocket reducer tests
- prompt-window snapshot tests
- integration tests under `apps/dreamverse/web/src/app/`

Add or tighten tests for:

- connection failure UX when backend is down
- readiness failure UX when backend returns non-ready status
- backend routing configuration through `BACKEND_HOST` and `BACKEND_PORT`

### Manual smoke path

The first manual smoke checklist should be:

1. start `dreamverse-server`
2. confirm `GET /healthz` returns 200
3. confirm `GET /readyz` returns 200 after warmup
4. start `apps/dreamverse/web`
5. confirm the UI opens and the websocket connects
6. submit a prompt and verify the first generation starts

This checklist should be written down in the README once the milestone is
implemented.

## Test Strategy

The test suite should preserve one rule: provider changes must not be able to
break prompt rewrite, websocket semantics, or runtime behavior silently.

### 1. Runtime unit and integration tests

Keep and expand the current `pytest` coverage in
`apps/dreamverse/dreamverse/tests/test_*.py`.

Focus areas:

- config loading
- prompt rewrite payload normalization
- prompt enhancement and prompt safety
- health/readiness endpoints
- websocket session behavior
- session logging
- mock runtime behavior

These tests should remain provider-agnostic.

### 2. Controller unit tests

Add a Python test suite for the controller state machine.

Key cases:

- provider selection and validation
- credential loading from local config or env
- runtime lifecycle transitions:
  - idle
  - provisioning
  - ready
  - failed
  - stopping
- idle timeout and cleanup behavior
- retry and backoff behavior
- HTTP and websocket proxy routing

These tests should use fake providers and fake runtimes by default.

### 3. Provider contract tests

Each provider should pass the same contract tests.

Examples:

- `ensure_runtime()` returns a usable `RuntimeHandle`
- `wait_until_ready()` surfaces timeout vs readiness correctly
- `stop_runtime()` is safe to call twice
- provider errors are mapped into stable controller error types

Use recorded fixtures or fakes wherever possible to avoid spend in CI.

### 4. Web contract tests

The frontend already has useful Vitest coverage under
`apps/dreamverse/web/src`.
Preserve that and expand around the controller split.

Priority areas:

- websocket event reduction
- prompt-window snapshot construction
- rewrite request shaping
- compute-status UI
- failure and reconnect UX

The frontend should mock the local controller API, not provider APIs.

### 5. Cross-layer protocol tests

Add contract fixtures that validate shared payloads across layers.

Important fixtures:

- websocket event payloads
- rewrite request payloads
- runtime status payloads
- controller status payloads

These can be simple JSON fixtures validated by both Python and TypeScript
tests. They will catch drift earlier than end-to-end tests.

### 6. Smoke tests

Add a small number of high-signal smoke tests:

- local provider + mock runtime
- local provider + real runtime when GPU is available
- controller startup + frontend health path

These should be cheap enough for routine local use.

### 7. Provider-backed manual or nightly tests

Real Modal and Runpod tests should be opt-in.

- Do not run them in default CI.
- Gate them behind explicit credentials and flags.
- Keep them focused on provisioning and reachability, not full product
  regression.

This avoids flaky and expensive CI while still validating real provider flows.

## Testing Recommendations for the Next Step

The next practical additions should be:

1. A controller test suite in Python using fake providers.
2. Shared contract fixtures for websocket and rewrite payloads.
3. A smoke test that starts the local controller against the existing mock
   runtime.

I would not add Playwright yet. The current web stack already has Vitest and
integration-style component tests, which are cheaper and better aligned with
the immediate reorg. Add browser automation only after the controller path is
stable.
