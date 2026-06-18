# Exploration Log: InterleaveThinker FastVideo Integration

## Status: draft

## Context
User requested implementation of the proposed InterleaveThinker integration plan
without touching the active dirty checkout in `/home/toolbox/FastVideo`.

Hard constraints from the user:

- Do not overwrite or edit the current branch/directory. Another agent is
  actively working there.
- Create a new branch and use `git worktree`.
- Make commits as frequently as useful.
- Keep this progress/plan/handoff file updated with enough detail that work can
  be resumed after interruption or context compaction.
- Before compacting context, update this file.
- If tests need to run, run them on Modal using
  `fastvideo/tests/modal/launch_l40s_job.py`; local machine is not capable.
- User pre-approved uploading codebase contents, including uncommitted code, to
  Modal and doing necessary work on the remote Modal machine for this job.

Isolation setup:

- Original checkout: `/home/toolbox/FastVideo`
- Original branch observed: `feat-mixgrpo-promptrl`
- Original checkout dirty: yes, unrelated RL/MixGRPO edits. Must ignore.
- New worktree: `/tmp/fastvideo-interleavethinker`
- New branch: `interleavethinker-fastvideo`
- Base: committed `main` at `633d39356804e63478d242611e992dc8e1af3caa`

## Goal
Add a minimal, native FastVideo integration surface for InterleaveThinker-style
workflows:

1. A FastVideo-backed generator service compatible with the kind of `/edit`
   API InterleaveThinker expects for FLUX.2-klein-style generation.
2. A small native interleaved orchestration package that can run planner/critic
   loops without putting planner/critic training into `fastvideo/train`.
3. Example config/script and focused tests for request translation and
   orchestration logic.

Out of scope for this first pass unless explicitly needed:

- Vendoring InterleaveThinker, EasyR1, LLaMA-Factory, or their training stack.
- Training planner/critic LLMs inside FastVideo.
- Adding InterleaveThinker rewards to FastVideo RL. That belongs in a later
  `fastvideo/train/methods/rl/rewards/` change if training FastVideo diffusion
  policies with those rewards.
- Heavy GPU/image-quality validation beyond smoke tests unless required.

## Architecture Plan

Layering:

- `fastvideo.entrypoints.interleave`: new app/orchestration layer.
- FastVideo `VideoGenerator`: owns model loading and generation.
- Planner and critic: provider-style interfaces, with deterministic/fake
  implementations for tests and later wrappers for HF/HTTP LLMs.
- Interleave-compatible service: small FastAPI app exposing health and image
  generation/edit endpoints that translate to `GenerationRequest`.

Proposed modules:

- `fastvideo/entrypoints/interleave/__init__.py`
- `fastvideo/entrypoints/interleave/schema.py`
  - dataclasses for `InterleaveConfig`, `InterleaveStep`, `InterleaveTrace`,
    planner/critic decisions, and service request/response shapes.
- `fastvideo/entrypoints/interleave/generator.py`
  - `FastVideoImageGeneratorBackend` that maps prompt/image/seed/size/steps to
    `VideoGenerator.generate(...)` and returns PIL/base64/path data.
- `fastvideo/entrypoints/interleave/orchestrator.py`
  - provider protocols plus `InterleaveOrchestrator`.
  - retry/refine loop around planner, generator, critic.
- `fastvideo/entrypoints/interleave/server.py`
  - FastAPI-compatible app factory for InterleaveThinker-like generator API.
- `fastvideo/entrypoints/cli/main.py`
  - possibly add `interleave-serve` or `interleave` subcommand if the local CLI
    pattern is small and stable.
- `examples/interleave/`
  - example config/readme/script for FLUX.2-klein.
- `tests/local_tests/test_interleave_*.py`
  - pure unit tests for schema translation and orchestration.

Testing plan:

- Focused unit tests should not require GPU or model downloads; they should use
  fake generator/planner/critic providers.
- If import-level or CLI tests need repo dependencies, run through Modal via
  `fastvideo/tests/modal/launch_l40s_job.py`.
- For pre-commit, run `pre-commit run --files <changed paths>` from the new
  worktree. If local dependency/runtime is insufficient, record and use Modal
  for validation.

## Progress

- [x] Confirmed original checkout is dirty and on a different branch.
- [x] Created isolated worktree `/tmp/fastvideo-interleavethinker`.
- [x] Created branch `interleavethinker-fastvideo` from committed `main`.
- [x] Read root `AGENTS.md`, `fastvideo/AGENTS.md`, onboarding, codebase map,
  skills index, workflows list, lessons list, and exploration template.
- [x] Inspect `fastvideo/entrypoints/cli`, `VideoGenerator`, schema, serve, and
  streaming prompt provider patterns.
- [x] Decide exact CLI/service shape based on existing FastVideo conventions.
- [x] Implement first minimal app/service layer.
- [x] Add unit tests with fakes.
- [x] Run Modal-backed validation.
- [x] Commit first coherent checkpoint.
- [x] Add trace serialization and a native single-prompt example runner.

## Findings

- `main` already has FLUX.2-klein registered in `fastvideo/registry.py`, so a
  compatibility service can initially target existing T2I generation rather than
  adding any model code.
- PR #1450's RL surfaces are useful later for generator RL, but the first
  InterleaveThinker integration should stay in `fastvideo/entrypoints/` and not
  under `fastvideo/train`.
- Existing prompt orchestration under `fastvideo/entrypoints/streaming/prompt`
  provides a good provider/fallback style to mirror for planner/critic clients.
- Existing OpenAI image routes expose `/v1/images` and `/v1/images/edits`,
  but InterleaveThinker reward code posts JSON to `/edit` and expects
  `{"success": true, "edited_image": "<base64 png>"}`. Add a compatibility
  app instead of changing the OpenAI-compatible API surface.
- Use `ServeConfig` for the new compatibility server rather than inventing a
  new top-level config. `fastvideo interleave-serve --config <serve.yaml>`
  can reuse `generator`, `server`, and `default_request`.
- The first code slice will include:
  - `fastvideo/entrypoints/interleave/schema.py` for Pydantic request/response
    models and dataclasses shared by the orchestrator.
  - `fastvideo/entrypoints/interleave/generator.py` for `VideoGenerator` request
    translation plus base64 image IO.
  - `fastvideo/entrypoints/interleave/orchestrator.py` for planner/critic loop
    protocols and a minimal retry/refine implementation.
  - `fastvideo/entrypoints/interleave/server.py` for `/health`, `/edit`, and
    `/generate` alias endpoints.
  - `fastvideo/entrypoints/cli/interleave_serve.py` and CLI registration.
  - example config/docs under `examples/interleave/`.
  - pure tests under `tests/local_tests/` using fakes.
- Implemented the package and CLI registration:
  - `fastvideo/entrypoints/interleave/__init__.py`
  - `fastvideo/entrypoints/interleave/schema.py`
  - `fastvideo/entrypoints/interleave/generator.py`
  - `fastvideo/entrypoints/interleave/orchestrator.py`
  - `fastvideo/entrypoints/interleave/server.py`
  - `fastvideo/entrypoints/cli/interleave_serve.py`
  - `fastvideo/entrypoints/cli/main.py`
  - `examples/interleave/README.md`
  - `examples/interleave/flux2_klein_interleave_serve.yaml`

## Mistakes / Dead Ends

- Initial attempt to create branch `feat/interleavethinker-fastvideo` failed
  because this repo's refs layout cannot create that nested branch path.
- Retried with flat branch name `interleavethinker-fastvideo`. Sandbox required
  escalation because `git worktree add` writes `.git` metadata.
- A first relative-path `apply_patch` attempt was rejected because it would have
  targeted the original checkout. All manual edits must use absolute paths under
  `/tmp/fastvideo-interleavethinker`.
- First Modal validation attempt failed before tests. Cause: `--apply-local-patch`
  included `.agents/exploration/interleavethinker-fastvideo-integration.md` as a
  tracked modification relative to the local-only checkpoint commit
  `7e70f859`, but the Modal job checked out upstream `main`
  `633d39356804e63478d242611e992dc8e1af3caa`, where that file does not exist.
  Fix: rerun Modal with `--patch-paths` limited to code, examples, and tests,
  excluding `.agents`.

## Current Handoff Notes

Resume from `/tmp/fastvideo-interleavethinker`, not `/home/toolbox/FastVideo`.

Useful commands:

```bash
cd /tmp/fastvideo-interleavethinker
git status --short --branch
git log --oneline -5
```

Before editing code, continue reading:

- Already inspected:
  - `fastvideo/entrypoints/cli/main.py`
  - `fastvideo/entrypoints/cli/serve.py`
  - `fastvideo/entrypoints/cli/generate.py`
  - `fastvideo/entrypoints/cli/inference_config.py`
  - `fastvideo/entrypoints/openai/image_api.py`
  - `fastvideo/entrypoints/openai/video_api.py`
  - `fastvideo/entrypoints/openai/api_server.py`
  - `fastvideo/entrypoints/openai/protocol.py`
  - `fastvideo/entrypoints/streaming/prompt/enhancer.py`
  - `fastvideo/entrypoints/streaming/server.py`
  - `fastvideo/entrypoints/video_generator.py`
  - `fastvideo/api/schema.py`
  - `fastvideo/api/compat.py`

Do not use or modify the dirty `/home/toolbox/FastVideo` checkout.

Next immediate steps:

1. Add pure tests:
   - request schema accepts `num_inference_step` and `num_inference_steps`.
   - backend builds a `GenerationRequest` with one image frame and optional
     decoded input image.
   - server returns InterleaveThinker-compatible JSON with a fake backend.
   - orchestrator retries with critic-provided `refine_prompt`.
2. Run formatting/pre-commit on changed files if possible.
3. Run focused tests on Modal using
   `fastvideo/tests/modal/launch_l40s_job.py` from this worktree, not from the
   original checkout.
   - Important: pass `--git-repo https://github.com/hao-ai-lab/FastVideo.git`
     and `--git-commit 633d39356804e63478d242611e992dc8e1af3caa`.
   - Important: pass `--patch-paths` excluding `.agents/...` until the next
     local commit is pushed anywhere reachable by Modal.
4. Commit the first code checkpoint after tests or at least after static import
   checks.

Validation completed:

- First Modal attempt:
  - Command: `pytest tests/local_tests/test_interleave_entrypoint.py -q &&
    pre-commit run --files ...`
  - Failed before tests because `.agents` handoff diff could not apply to
    upstream `main`.
- Second Modal attempt:
  - Command: `pytest tests/local_tests/test_interleave_entrypoint.py -q &&
    pre-commit run --files fastvideo/entrypoints/cli/main.py
    fastvideo/entrypoints/cli/interleave_serve.py
    fastvideo/entrypoints/interleave/__init__.py
    fastvideo/entrypoints/interleave/schema.py
    fastvideo/entrypoints/interleave/generator.py
    fastvideo/entrypoints/interleave/orchestrator.py
    fastvideo/entrypoints/interleave/server.py`
  - Modal app URL shown by CLI:
    `https://modal.com/apps/hao-ai-lab/main/ap-kB8zs3gMtKj66loYptASBw`
  - Result: `4 passed, 14 warnings in 22.54s`.
  - Pre-commit result: `yapf`, `ruff`, `codespell`, `mypy`, filename check all
    passed. PyMarkdown/actionlint skipped with no files to check.

Commits so far:

- `7e70f859` — `[misc] track InterleaveThinker integration plan`
- `eb33639c` — `[feat] add InterleaveThinker compatibility service`

Next slice:

- Added `fastvideo/entrypoints/interleave/trace.py` with JSON-safe
  serialization for `InterleaveTrace`, excluding base64 image payloads by
  default.
- Added `examples/interleave/interleave_single_prompt.py` that uses
  `VideoGenerator`, `FastVideoImageGeneratorBackend`, `SinglePromptPlanner`,
  and `AcceptAllCritic` to generate a one-step interleave trace.
- Added focused tests for trace serialization in
  `tests/local_tests/test_interleave_entrypoint.py`.

Validation note for the next Modal run:

- Commit the trace/example slice locally.
- Create a temporary validation worktree from upstream `main`.
- Apply `git diff --binary main..interleavethinker-fastvideo` into that
  temporary worktree so all branch changes are uncommitted relative to `main`.
- Run Modal from the temporary validation worktree with `--apply-local-patch`.
  This avoids the launcher's limitation that it only captures local diffs from
  the current `HEAD`, not local-only commits.

Full branch validation completed:

- Temporary validation worktree: `/tmp/fastvideo-interleavethinker-validation`
- Full diff applied from `main..interleavethinker-fastvideo`.
- Modal app URL shown by CLI:
  `https://modal.com/apps/hao-ai-lab/main/ap-cREtgFJyncsIfsXn1B5hLL`
- Command:
  `pytest tests/local_tests/test_interleave_entrypoint.py -q &&
  pre-commit run --files fastvideo/entrypoints/cli/main.py
  fastvideo/entrypoints/cli/interleave_serve.py
  fastvideo/entrypoints/interleave/__init__.py
  fastvideo/entrypoints/interleave/schema.py
  fastvideo/entrypoints/interleave/generator.py
  fastvideo/entrypoints/interleave/orchestrator.py
  fastvideo/entrypoints/interleave/server.py
  fastvideo/entrypoints/interleave/trace.py`
- Result: `5 passed, 14 warnings in 13.24s`.
- Pre-commit result: `yapf`, `ruff`, `codespell`, `mypy`, filename check all
  passed. PyMarkdown/actionlint skipped with no files to check.
- Temporary validation worktree was removed after validation.

## Proposed Standardization

If this integration lands cleanly, promote this exploration into a workflow or
skill for adding agentic app layers over FastVideo generators:

- provider protocols for external LLM/planner/critic systems,
- `VideoGenerator` backend adapters,
- compatibility API service wrappers,
- fake-provider unit tests,
- Modal smoke-test pattern.
