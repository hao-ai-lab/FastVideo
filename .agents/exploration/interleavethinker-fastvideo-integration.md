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
- [ ] Inspect `fastvideo/entrypoints/cli`, `VideoGenerator`, schema, serve, and
  streaming prompt provider patterns.
- [ ] Decide exact CLI/service shape based on existing FastVideo conventions.
- [ ] Implement first minimal app/service layer.
- [ ] Add unit tests with fakes.
- [ ] Run Modal-backed validation.
- [ ] Commit first coherent checkpoint.

## Findings

- `main` already has FLUX.2-klein registered in `fastvideo/registry.py`, so a
  compatibility service can initially target existing T2I generation rather than
  adding any model code.
- PR #1450's RL surfaces are useful later for generator RL, but the first
  InterleaveThinker integration should stay in `fastvideo/entrypoints/` and not
  under `fastvideo/train`.
- Existing prompt orchestration under `fastvideo/entrypoints/streaming/prompt`
  provides a good provider/fallback style to mirror for planner/critic clients.

## Mistakes / Dead Ends

- Initial attempt to create branch `feat/interleavethinker-fastvideo` failed
  because this repo's refs layout cannot create that nested branch path.
- Retried with flat branch name `interleavethinker-fastvideo`. Sandbox required
  escalation because `git worktree add` writes `.git` metadata.
- A first relative-path `apply_patch` attempt was rejected because it would have
  targeted the original checkout. All manual edits must use absolute paths under
  `/tmp/fastvideo-interleavethinker`.

## Current Handoff Notes

Resume from `/tmp/fastvideo-interleavethinker`, not `/home/toolbox/FastVideo`.

Useful commands:

```bash
cd /tmp/fastvideo-interleavethinker
git status --short --branch
git log --oneline -5
```

Before editing code, continue reading:

- `fastvideo/entrypoints/cli/main.py`
- `fastvideo/entrypoints/cli/serve.py`
- `fastvideo/entrypoints/openai/image_api.py`
- `fastvideo/entrypoints/openai/video_api.py`
- `fastvideo/entrypoints/streaming/prompt/enhancer.py`
- `fastvideo/entrypoints/video_generator.py`
- `fastvideo/api/schema.py`

Do not use or modify the dirty `/home/toolbox/FastVideo` checkout.

## Proposed Standardization

If this integration lands cleanly, promote this exploration into a workflow or
skill for adding agentic app layers over FastVideo generators:

- provider protocols for external LLM/planner/critic systems,
- `VideoGenerator` backend adapters,
- compatibility API service wrappers,
- fake-provider unit tests,
- Modal smoke-test pattern.
