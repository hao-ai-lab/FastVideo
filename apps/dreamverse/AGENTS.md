# Dreamverse Agent Notes

## Repo-local skills

- `.agents/skills/bootstrap-fastvideo-private-fork/`: temporary setup skill for
  cloning `git@github.com:hao-ai-lab/FastVideo-internal.git` at
  `will/rebase-nbv` into `../FastVideo-internal`, then running
  `uv sync --extra server`.
- Prefer the bundled script in that skill instead of inventing a new private
  FastVideo bootstrap flow.

## Repo layout

Current paths:

- `apps/dreamverse/web/`: Next.js frontend, client-side stores, websocket
  event reduction, prompt-window editing, devtools UI.
- `apps/dreamverse/dreamverse/`: current Python FastAPI runtime, websocket
  protocol, prompt enhancement, prompt rewrite orchestration, GPU worker
  lifecycle.
- `apps/dreamverse/dreamverse/tests/`: backend unit and integration-oriented
  tests.
- `apps/dreamverse/dreamverse/benchmarks/`: prompt-provider latency/token
  benchmarking scripts.

Planned paths during the OSS reorg:

- `controller/`: local control plane for provider credentials, compute
  lifecycle, and proxying.
- `runtime/`: eventual rename of `apps/dreamverse/dreamverse/` once the
  controller/runtime split is stable.
- `providers/`: provider adapters for local, Runpod, and Modal.

Important rule:

- Until the split lands, treat `apps/dreamverse/dreamverse/` as the
  authoritative runtime and keep provider orchestration out of
  `apps/dreamverse/web`.

## System split

Dreamverse is moving toward a three-part local-first architecture.

- Frontend owns local UI state, drafts, inspection tools, and user-triggered
  actions.
- Controller will own local-only credentials, compute provisioning, runtime
  lifecycle, and HTTP/websocket proxying.
- Runtime owns generation sessions, prompt rewrite, prompt safety, and
  websocket semantics.

The browser should only talk to the local Dreamverse process, never directly to
Modal or Runpod.

## Current runtime responsibilities

The runtime in `apps/dreamverse/dreamverse/` is responsible for:

- websocket session lifecycle on `/ws`
- queueing, GPU assignment, worker startup, and stream chunk emission
- seed prompt memory and the active prompt window used for generation
- prompt enhancement and prompt rewrite execution
- prompt safety checks
- persistence and reload of prompt system prompt files
- curated preset append/read routes in devtools mode
- health and readiness endpoints

Relevant files:

- `apps/dreamverse/dreamverse/main.py`: websocket protocol, session state
  machine, REST routes
- `apps/dreamverse/dreamverse/gpu_pool.py`: FastVideo-backed generation
  workers
- `apps/dreamverse/dreamverse/prompt_enhancer.py`: provider clients, prompt
  enhancement, rewrite execution
- `apps/dreamverse/dreamverse/rewrite_prompt_payload.py`: canonical rewrite
  request body format
- `apps/dreamverse/dreamverse/config.py`: prompt file paths, provider
  configuration, runtime flags

## Frontend responsibilities

The frontend in `apps/dreamverse/web/` is responsible for:

- collecting user input and deciding whether to send raw prompts or rewrite
  requests
- maintaining client-side stores for session, prompt-window, stream, rewrite,
  and UI state
- rendering prompt history, playback state, devtools controls, and rewrite
  inspection
- building the prompt-window snapshot sent with rewrite requests
- reducing websocket events into UI state
- showing compute status and controller-driven errors once the controller lands

Relevant files:

- `apps/dreamverse/web/src/app/page.tsx`: main orchestration, websocket
  connect/send paths
- `apps/dreamverse/web/src/lib/ws/reducer.ts`: applies normalized websocket
  events to stores
- `apps/dreamverse/web/src/stores/promptWindow.ts`: prompt window and
  preset/editor state
- `apps/dreamverse/web/src/stores/rewrite.ts`: rewrite activity timeline and
  flags
- `apps/dreamverse/web/src/lib/prompts/promptWindowSnapshot.ts`: rewrite snapshot
  normalization and padding

## Planned controller responsibilities

The future local controller should own:

- local-only provider credential loading and storage
- provider selection
- runtime provisioning, reuse, shutdown, and health checks
- proxying frontend HTTP and websocket traffic to the active runtime
- surfacing provisioning, ready, failed, and idle states to the frontend
- durable local settings that should survive ephemeral remote runtimes

The controller should not own:

- prompt rewrite logic
- seed prompt memory
- generation queue semantics
- websocket event schemas

## Prompt rewrite contract

Prompt rewrite is a shared flow with a strict ownership split.

Frontend responsibilities:

- decide when a user action should trigger `rewrite_seed_prompts` instead of
  `append_prompt`
- send `rewrite_instruction` and a snapshot of the current prompt window
- pad the rewrite snapshot to the runtime-expected segment count using
  `buildRewritePromptWindowSnapshotFromPrompts(...)`
- show rewrite activity and raw LLM output in local inspection UI

Runtime responsibilities:

- validate and normalize `prompt_window_prompts`
- choose the rewrite system prompt and provider/model/temperature
- build the canonical LLM request body in
  `apps/dreamverse/dreamverse/rewrite_prompt_payload.py`
- run the rewrite through `PromptEnhancer.rewrite_prompt_sequence(...)`
- apply safety filtering to rewritten prompts
- replace the authoritative seed prompt memory when rewrite succeeds
- emit `seed_prompts_updated` and `rewrite_seed_prompts_complete`

Controller responsibilities:

- proxy the request and response
- surface runtime availability and provider lifecycle failures

Important rule:

- The frontend may suggest the prompt window to rewrite, but the runtime owns
  the actual rewritten rollout and the authoritative prompt window after
  acceptance.

## Rewrite modes

There are two runtime rewrite modes:

- edit existing rollout: when `prompt_window_prompts` is non-empty, rewrite the
  current rollout while preserving segment count and ordering
- new rollout: when the prompt window is empty but there is a
  `rewrite_instruction`, generate a fresh rollout

The frontend should not emulate runtime rewrite behavior locally. It should
prepare the snapshot, send it, and display the result.

## Prompt window ownership

- Frontend owns editable drafts, selected preset UI, and prompt-window
  inspection state.
- Runtime owns the active seed prompt memory used for actual generation.
- After any runtime event with reason `rewrite`, the frontend must replace its
  prompt window from the server payload instead of keeping a locally-derived
  version.

## Devtools and persistence ownership

Prompt config editing is runtime-owned persistence with frontend-owned forms
today.

- Frontend loads and edits drafts through `/prompt-system-config`.
- Runtime reads and writes prompt files and reloads runtime prompt config.

Curated presets follow the same pattern:

- frontend submits append requests and may update local UI optimistically from
  the response
- runtime persists the JSON file and resolves overlay vs fallback file paths

During the controller reorg, avoid moving durable user settings into ephemeral
remote runtimes. Controller-owned local persistence is preferred for anything
that must survive provider restarts.

## Editing guidance

- Do not move rewrite logic into the frontend or controller.
- Do not make the frontend the source of truth for the generated prompt window
  after rewrite.
- If you change websocket message types or payload fields in
  `apps/dreamverse/dreamverse/main.py`, update the reducer in
  `apps/dreamverse/web/src/lib/ws/reducer.ts` in the same change.
- If you change rewrite request shape, update both
  `apps/dreamverse/web/src/lib/prompts/promptWindowSnapshot.ts` and
  `apps/dreamverse/dreamverse/rewrite_prompt_payload.py`.
- If you add controller-managed status or error payloads, keep them separate
  from runtime websocket events unless there is a strong reason to merge them.
- If you change prompt file paths or devtools persistence, update
  `apps/dreamverse/dreamverse/`, the frontend devtools UI, and any
  controller-owned local persistence logic together.
- Keep provider adapters focused on runtime lifecycle and reachability, not on
  prompt or session semantics.
