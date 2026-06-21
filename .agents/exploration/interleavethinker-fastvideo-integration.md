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

## Model / API Backend Integration Extension

Status: in progress as of the user request to integrate the remaining
InterleaveThinker models and wrap closed-source Nano Banana / Gemini APIs.

User constraints still active:

- Work only in `/tmp/fastvideo-interleavethinker`.
- Do not edit or overwrite `/home/toolbox/FastVideo`; that checkout belongs to
  another agent.
- Commit useful checkpoints and push each commit immediately.
- Run validation on Modal through `fastvideo/tests/modal/launch_l40s_job.py`.
- Keep this handoff file current before interruptions or context compaction.

Scope for this slice:

1. Replace the placeholder InterleaveThinker critic actor shell with a concrete
   Hugging Face Qwen3-VL-compatible `ModelBase` wrapper.
   - Load `InterleaveThinker/Critic-SFT-8B`,
     `InterleaveThinker/InterleaveThinker-Critic-8B`, or a local
     `ckpt/critic_sft` / `ckpt/critic_rl` checkpoint via Transformers.
   - Preserve FastVideo's `ModelBase` role-module/checkpoint visibility through
     `self.transformer`.
   - Implement `generate_interleave_responses(...)` for grouped RL rollouts.
   - Implement `train_interleave_rollouts(...)` as a lightweight
     advantage-weighted policy-gradient update over generated critic responses.
     This is not a full EasyR1/vLLM/FSDP parity port, but it makes the FastVideo
     RL loop executable with a real Qwen3-VL actor backend.
   - Add a small JSON/JSONL dataloader for InterleaveThinker critic RL records
     so the standard FastVideo trainer can receive batches.
2. Add closed-source API wrappers:
   - Nano Banana image generation/editing via the official `google-genai`
     `models.generate_content` API, with model aliases for Nano Banana,
     Nano Banana Pro, and Nano Banana 2.
   - Gemini VLM scoring via the same SDK, returning InterleaveThinker
     semantic/quality scores.
   - A composite `GeminiNanoBananaEditScorer` callable that the RL reward
     scorer can instantiate from YAML.
3. Wire the API scorer into `InterleaveThinkerRLMethod` through an optional
   `_target_` config block while keeping existing offline/fake-score tests
   credential-free.
4. Update examples and focused tests.

Official API facts verified against Google AI docs on 2026-06-19:

- Nano Banana image generation/editing is exposed through Gemini API native
  image models. Current documented model IDs include:
  `gemini-3.1-flash-image` (Nano Banana 2), `gemini-3-pro-image`
  (Nano Banana Pro), and `gemini-2.5-flash-image` (Nano Banana).
- Python SDK pattern:
  `from google import genai`; `client = genai.Client(...)`;
  `client.models.generate_content(model=..., contents=[prompt, image])`.
- Image responses are returned as response parts with inline data / `as_image`.
- Structured Gemini outputs use `response_mime_type="application/json"` and a
  response schema in `GenerateContentConfig`.

Implementation boundary:

- The Qwen3-VL critic is a local/open model wrapper because it must train in the
  RL loop.
- Nano Banana and Gemini remain API models; FastVideo should not pretend they
  are local trainable modules.
- The API wrappers must import `google-genai` lazily and raise clear dependency
  / API-key errors only when used.
- Tests must stub the SDK; no real API calls in CI/Modal validation.

Implementation completed for this slice:

- Replaced the placeholder
  `fastvideo/train/models/interleave_thinker/critic.py` adapter with a
  Qwen3-VL-compatible `InterleaveThinkerCriticModel`.
  - Loads a processor and model through Transformers when `load_backend=true`.
  - Keeps `load_backend=false` for import/config/unit-test dry runs.
  - Exposes `self.transformer` for FastVideo role-module/checkpoint visibility.
  - Freezes visual tower and multimodal projector by default, matching upstream
    critic SFT.
  - Implements `build_messages(...)` with the InterleaveThinker two-image
    critic prompt.
  - Implements `generate_interleave_responses(...)` for grouped critic
    rollouts.
  - Implements `train_interleave_rollouts(...)` as an advantage-weighted
    policy-gradient update over response token NLL.
  - Adds a small JSON/JSONL dataloader for critic RL records.
- Added `NanoBananaImageGeneratorBackend` to
  `fastvideo/entrypoints/interleave/generator.py`.
  - Supports model aliases: `nano-banana`, `nano-banana-pro`,
    `nano-banana-2`.
  - Uses the official `google-genai` `models.generate_content` API lazily.
  - Implements the existing `ImageGeneratorBackend` protocol.
- Added `fastvideo/train/methods/rl/rewards/interleave_api.py`.
  - `GeminiInterleaveImageScorer` wraps Gemini VLM JSON scoring.
  - `GeminiNanoBananaEditScorer` generates edits with Nano Banana and scores
    them with Gemini, matching `EditScoreProvider`.
  - `ConstantInterleaveEditScorer` gives tests/offline configs a deterministic
    scorer.
- Wired `InterleaveThinkerRLMethod` to accept an optional `method.edit_scorer`
  `_target_` block.
- Updated `examples/train/configs/rl/interleave_thinker/critic_grpo.yaml` to
  point at the Qwen3-VL critic wrapper and the Gemini/Nano Banana scorer.
- Added optional dependency extra: `fastvideo[interleave-api] = google-genai`.
- Documented the API-backed path in `examples/interleave/README.md`.
- Added tests:
  - `tests/local_tests/test_interleave_thinker_api_models.py`
  - `tests/local_tests/test_interleave_thinker_critic_model.py`
  - extended `tests/local_tests/test_interleave_thinker_method.py`

Validation for this slice:

- Initial Modal pytest run:
  - App: `https://modal.com/apps/hao-ai-lab/main/ap-rGgMjcQcI2bNAl3Viy0eES`
  - Result: one test assertion failure in
    `test_interleave_thinker_critic_builds_qwen_vl_messages_without_loading_backend`.
  - Cause: test assumed fixed content indices, but adjacent `<image><image>`
    placeholders produce adjacent image parts.
  - Fix: assert by part type instead of index.
- Second Modal pytest run:
  - App: `https://modal.com/apps/hao-ai-lab/main/ap-SIztyW6PC3yRAKdybFxEEE`
  - Result: `27 passed, 14 warnings in 19.71s`.
- First Modal pre-commit run:
  - App: `https://modal.com/apps/hao-ai-lab/main/ap-E0mnD27dHXDDnOquRHU5QD`
  - Result: yapf and ruff modified files; mypy passed on Modal.
- Local formatter convergence:
  - Installed `pre-commit==4.0.1` locally with hook cache under `/tmp`.
  - Local hook run converged for yapf/ruff/codespell.
  - Local mypy cannot run from `/tmp/fastvideo-interleavethinker` because the
    hyphenated directory name is interpreted as an invalid package name; Modal
    mypy remains the authoritative result.
- Post-format Modal pytest:
  - App: `https://modal.com/apps/hao-ai-lab/main/ap-ykgfYTZyJgO4EgRlHv6l81`
  - Result: `27 passed, 14 warnings in 17.04s`.
- Final Modal pre-commit:
  - App: `https://modal.com/apps/hao-ai-lab/main/ap-QOKlzapm5bSAo3c21lprwv`
  - Result: yapf, ruff, codespell, mypy, filename check, and suggestion hooks
    passed; PyMarkdown/actionlint skipped with no files to check.

Pending:

- Commit this model/API integration slice.
- Push the commit immediately after committing, per user instruction.

Commit completed:

- `fc04d019` — `[feat] integrate InterleaveThinker model backends`
- Pushed to `origin/interleavethinker-fastvideo`.

## Critic Backend Hardening Extension

Status: completed and pushed.

Goal for this slice:

- Tighten the Qwen3-VL critic wrapper from config/import-ready toward
  executable-backend-ready without pulling real Qwen weights into unit tests.
- Preserve user constraints:
  - work only in `/tmp/fastvideo-interleavethinker`,
  - do not touch `/home/toolbox/FastVideo`,
  - validate via Modal,
  - commit and push immediately after commit.

Planned changes:

1. Add `image_dir` handling to `InterleaveThinkerCriticModel`, matching
   upstream EasyR1's convention that relative `origin_image_path` and
   `edited_image_path` are resolved under a configured image root.
2. Normalize image paths at dataset load/collate time, and expose an explicit
   before/after image pair:
   - before: `previous_image_path`, then `origin_image_path`, then
     `input_image_path`;
   - after: `edited_image_path`, then `generated_image_path`, then
     `output_image_path`.
3. Improve actor-update scaling so per-rollout backward computes the mean batch
   policy gradient instead of summing all rollout losses.
4. Add fake Qwen backend tests that exercise:
   - `generate_interleave_responses(...)`,
   - `train_interleave_rollouts(...)`,
   - image path resolution,
   - public YAML parse for the new config knobs.

Validation plan:

- Modal pytest:
  `pytest tests/local_tests/test_interleave_thinker_reward.py
  tests/local_tests/test_interleave_thinker_method.py
  tests/local_tests/test_interleave_thinker_api_models.py
  tests/local_tests/test_interleave_thinker_critic_model.py
  tests/local_tests/test_train_rl_sampling.py -q`
- Modal pre-commit on changed files.

Implemented changes:

- `InterleaveThinkerCriticModel` accepts `image_dir` and passes it into the
  JSON/JSONL dataset loader.
- Dataset loading normalizes relative image paths under `image_dir` while
  leaving absolute paths, URI-style paths, and `data:` URLs unchanged.
- Qwen3-VL message assembly now selects an explicit before/after image pair.
  This avoids sending `origin`, `previous`, and `edited` together when the
  critic task expects two images.
- Actor update scaling now averages over `len(rollouts) *
  gradient_accumulation_steps`, preventing the loss from growing with rollout
  count.
- Added fake-backend tests for generation and response-token-only actor
  training without downloading real Qwen weights.
- Added `image_dir: data` to
  `examples/train/configs/rl/interleave_thinker/critic_grpo.yaml`.

Validation completed:

- Modal pytest:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-fRlUUdFx19OWpSvDSrWoAe`
  - Command:
    `pytest tests/local_tests/test_interleave_thinker_reward.py
    tests/local_tests/test_interleave_thinker_method.py
    tests/local_tests/test_interleave_thinker_api_models.py
    tests/local_tests/test_interleave_thinker_critic_model.py
    tests/local_tests/test_train_rl_sampling.py -q`
  - Result: `30 passed, 14 warnings in 19.47s`.
- Modal pre-commit:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-DplMFq23YYfBx34e6TcsRc`
  - Command:
    `pre-commit run --files
    examples/train/configs/rl/interleave_thinker/critic_grpo.yaml
    fastvideo/train/models/interleave_thinker/critic.py
    tests/local_tests/test_interleave_thinker_critic_model.py
    tests/local_tests/test_interleave_thinker_method.py`
  - Result: yapf, ruff, codespell, mypy, filename check, and suggestion hooks
    passed; PyMarkdown/actionlint skipped with no files to check.

Commit completed:

- `ace421bc` — `[feat] harden InterleaveThinker critic backend`
- Pushed to `origin/interleavethinker-fastvideo`.

Next recommended integration step:

- Run a real small-checkpoint smoke for `InterleaveThinkerCriticModel` on Modal
  with a Qwen3-VL-compatible checkpoint and a two-image JSONL fixture. The
  current slice validates the FastVideo contract with a fake backend; it does
  not download or execute the real Qwen critic weights.

## Real Critic Checkpoint Smoke

Status: completed and Modal-validated.

Goal:

- Run `InterleaveThinkerCriticModel` with a real Qwen3-VL-compatible checkpoint
  on Modal L40S, using the pushed `interleavethinker-fastvideo` branch.
- First target the upstream public critic SFT checkpoint:
  `InterleaveThinker/Critic-SFT-8B`, with processor
  `Qwen/Qwen3-VL-8B-Instruct`.
- Use a tiny two-image fixture generated on the Modal worker to validate:
  - backend load through the FastVideo wrapper,
  - dataset `image_dir` resolution,
  - Qwen3-VL chat-template/image preprocessing,
  - one short `generate_interleave_responses(...)` call.

Smoke command shape:

- Modal launcher:
  `python -m modal run fastvideo/tests/modal/launch_l40s_job.py`
- GPU: `L40S:1`
- Install extra: `dev`
- Git commit: current pushed branch head.
- Remote command: create two 128x128 PNGs and one JSONL row, instantiate
  `InterleaveThinkerCriticModel(load_backend=True, trainable=False,
  torch_dtype="bf16", device_map="cuda:0", attn_implementation="sdpa")`, build
  the dataloader, run one generation with `max_new_tokens=16`.

Success criteria:

- Model and processor load successfully.
- The dataloader yields one row with image paths resolved under the fixture
  image directory.
- `generate_interleave_responses(...)` returns one rollout with a non-empty
  response.

Known risk:

- This is intentionally a real-weight smoke. It may fail because the upstream
  HF checkpoint is unavailable/gated, because the Modal image lacks a Qwen3-VL
  runtime dependency, or because the wrapper needs a Qwen3-specific model class.
  Any of those failures should be captured as integration work, not ignored.

Validation completed:

- Modal app URL:
  `https://modal.com/apps/hao-ai-lab/main/ap-hDxj5MhLgdnGq22mRLjgIK`
- Commit tested:
  `9973307b11b837a280cc81f1a0e61676fb55acf8`
- Local patch applied: `false`; the job used the pushed branch state.
- Modal volume commit: `true`; downloaded HF weights were saved back to the
  `hf-model-weights` volume.
- Remote smoke details:
  - Created two 128x128 PNGs under
    `/tmp/interleave_real_critic_smoke/images`.
  - Created one JSONL row with relative `origin_image_path` and
    `edited_image_path`.
  - Loaded model `InterleaveThinker/Critic-SFT-8B`.
  - Loaded processor `Qwen/Qwen3-VL-8B-Instruct`.
  - Instantiated `InterleaveThinkerCriticModel(load_backend=True,
    trainable=False, image_dir=<fixture images>, torch_dtype="bf16",
    device_map="cuda:0", attn_implementation="sdpa")`.
  - Ran `init_preprocessors(...)`.
  - Ran `generate_interleave_responses(..., num_generations=1,
    max_new_tokens=16)`.
- Observed output:
  - CUDA device: `NVIDIA L40S`.
  - Backend class: `Qwen3VLForConditionalGeneration`.
  - Resolved paths:
    `/tmp/interleave_real_critic_smoke/images/before.png` and
    `/tmp/interleave_real_critic_smoke/images/after.png`.
  - Rollout count: `1`.
  - Response prefix began with `<think>`.
  - Smoke marker printed: `SMOKE_OK`.

Conclusion:

- The critic wrapper is properly set up for real checkpoint load, image-root
  resolution, Qwen3-VL chat/image preprocessing, and short rollout generation
  on Modal L40S.
- This smoke did not run a full real-weight optimizer update. The fake-backend
  tests cover `train_interleave_rollouts(...)`; a real 8B optimizer-step smoke
  should be done separately with LoRA or sharding because full-parameter Adam on
  one L40S is likely too memory-heavy.

## Training Loop Extension

Status: in progress as of the user request to "Add a training loop" for
InterleaveThinker's RL integration.

User constraints still active:

- Continue using `/tmp/fastvideo-interleavethinker`.
- Do not edit `/home/toolbox/FastVideo`; that checkout is dirty and belongs to
  another agent.
- Commit useful checkpoints and push each commit immediately after committing.
- Run tests on Modal through `fastvideo/tests/modal/launch_l40s_job.py`; the
  local machine is not suitable for the relevant test environment.
- Keep this handoff file current before interruptions or context compaction.

Skills / guidance applied for this slice:

- `.agents/skills/rlhf-training-abstractions/SKILL.md`
- `.agents/skills/add-rl-method/SKILL.md`
- `.agents/skills/add-reward-model/SKILL.md`
- `fastvideo/train/AGENTS.md`
- `fastvideo/tests/AGENTS.md`

Upstream InterleaveThinker RL facts gathered from
`/tmp/InterleaveThinker-src`:

- RL launch script:
  `train/EasyR1/local_scripts/run_interleave_thinker_rl.sh`.
- It runs EasyR1/verl GRPO with
  `python3 -m verl.trainer.main`.
- The actor checkpoint is `${ROOT}/ckpt/critic_sft`, not a diffusion model.
- `critic_sft` is produced by LLaMA-Factory SFT from
  `Qwen/Qwen3-VL-8B-Instruct`.
- The SFT config freezes the visual tower and multimodal projector and trains
  the language side of a Qwen3-VL critic.
- The RL script overrides EasyR1's default model path with
  `worker.actor.model.model_path="${MODEL_PATH}"`.
- The EasyR1 config uses `algorithm.adv_estimator: grpo`,
  `worker.rollout.n: 8`, `worker.rollout.temperature: 1.0`,
  `worker.rollout.top_p: 1.0`, `algorithm.use_kl_loss: true`, and
  `algorithm.kl_coef: 1.0e-2`.
- The reward function is
  `train/EasyR1/verl/reward_function/interleave_thinker_reward.py:compute_score`.
- The reward posts JSON to a FastVideo-compatible `/edit` API using
  `EDIT_MODEL_NAME="klein"`, `num_inference_step=4`,
  `guidance_scale=1.0`, `width=1024`, `height=1024`, and
  `enhance_prompt=false`.
- Reward components:
  - XML/JSON format reward: response must include `<think>...</think>` and an
    `<answer>...</answer>` JSON object with `previous_step_success` and
    `refine_prompt`.
  - Judge accuracy reward: predicted `previous_step_success` must match the
    ground-truth previous-step success flag.
  - Image edit improvement: generate an edited image from `refine_prompt` and
    score semantic / quality improvement with Gemini; upstream combines these
    with `overall = 0.5 * format + 0.5 * (0.2 * accuracy + 0.6 * semantic +
    0.2 * quality)`.

Implementation plan for the FastVideo training loop:

1. Add reusable InterleaveThinker reward utilities under
   `fastvideo/train/methods/rl/rewards/interleave_thinker.py`.
   - Keep parser/format/accuracy/edit aggregation independent from the method.
   - Allow fake scorer injection in tests.
   - Do not include Gemini credentials or network calls in the reusable core.
2. Add a managed RL method under
   `fastvideo/train/methods/rl/interleave_thinker.py`.
   - Subclass `TrainingMethod`.
   - Return `manages_optimization() == True`.
   - Implement `managed_train_step(data_stream, iteration)` as a genuine
     outer loop: consume prompts, ask the actor role-model adapter for grouped
     responses, score rewards, compute GRPO-style group-normalized advantages,
     and call the actor adapter's update hook.
   - Keep Qwen3-VL / tokenizer / logprob details behind model hooks instead of
     embedding them in the method. FastVideo currently has no native Qwen3-VL
     actor wrapper, so this slice creates the method contract and tests it with
     fakes rather than vendoring EasyR1.
3. Add a public skeleton config under `examples/train/configs/rl/` showing the
   method target and knobs. It should be parseable but clearly rely on a future
   actor `ModelBase` wrapper.
4. Add focused tests under `tests/local_tests/`:
   - reward parser and score aggregation,
   - managed loop calls actor generation/update and computes grouped
     advantages,
   - config parsing for the new method target,
   - sanity that existing non-RL methods still default to trainer-managed
     optimization if touched by the tests.
5. Validate via Modal:
   - `pytest tests/local_tests/test_interleave_thinker_reward.py
     tests/local_tests/test_interleave_thinker_method.py
     tests/local_tests/test_train_rl_sampling.py -q`
   - `pre-commit run --files <changed FastVideo files>` with the repository's
     configured hooks.

Design boundary for reviewers:

- This is intentionally not a full native Qwen3-VL/EasyR1 port. Adding a
  first-class Qwen3-VL actor model belongs in `fastvideo/train/models/` and can
  reuse this method once available.
- This does provide a FastVideo-native RL method and reward surface, so the
  InterleaveThinker RL loop can be integrated like other training methods:
  YAML target, trainer lifecycle, method-managed step, metrics, callbacks, and
  checkpoint visibility through role models.

Current training-loop edits:

- [x] Add `fastvideo/train/methods/rl/rewards/interleave_thinker.py`.
- [x] Export the new reward utilities.
- [x] Add `fastvideo/train/methods/rl/interleave_thinker.py`.
- [x] Export `InterleaveThinkerRLMethod`.
- [x] Add `fastvideo/train/models/interleave_thinker/` adapter shell so the
  example YAML has an importable FastVideo model target.
- [x] Add config example:
  `examples/train/configs/rl/interleave_thinker/critic_grpo.yaml`.
- [x] Add local tests:
  - `tests/local_tests/test_interleave_thinker_reward.py`
  - `tests/local_tests/test_interleave_thinker_method.py`
- [x] Run Modal validation.
- [x] Commit and push.

Implementation details added:

- `InterleaveThinkerRewardScorer` parses `<think>` / `<answer>` responses,
  accepts JSON or Python-literal answer dicts, computes format reward, judge
  accuracy reward, and normalized semantic/quality edit rewards.
- Reward scoring accepts optional injected `edit_scorer` or per-rollout
  `edit_score(s)` metadata. It does not load Gemini or call edit APIs.
- `InterleaveThinkerRLMethod` subclasses `TrainingMethod`, returns
  `manages_optimization() == True`, and implements a real
  `managed_train_step`:
  1. collect rollouts from `student.generate_interleave_responses(...)` or
     offline `response/responses` batch fields,
  2. score InterleaveThinker rewards,
  3. compute GRPO-style group-normalized advantages by `group_key`,
  4. call `student.train_interleave_rollouts(...)` with rollouts, rewards,
     advantages, optional FastVideo optimizer/scheduler, grad accumulation, and
     max grad norm.
- `InterleaveThinkerCriticModel` is deliberately an adapter shell. It documents
  the required hooks and raises until a native Qwen/VLM backend is implemented.
- `git diff --check` passed locally before Modal validation.
- First Modal validation attempt for this slice reached the remote L40S
  container:
  - Modal app URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-tIqT7DuddSWzb3O9Un1w66`
  - Tests passed: `22 passed, 14 warnings in 20.75s`.
  - Pre-commit failed only on Ruff `SIM108` in
    `fastvideo/train/methods/rl/interleave_thinker.py`.
  - Fixed by replacing a small `if`/`else` response selection with the ternary
    Ruff requested.
- Second Modal validation attempt passed:
  - Modal app URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-2Z2sH2UfhMoPmKolG0KY6t`
  - Command:
    `pytest tests/local_tests/test_interleave_thinker_reward.py
    tests/local_tests/test_interleave_thinker_method.py
    tests/local_tests/test_train_rl_sampling.py -q && pre-commit run --files
    fastvideo/train/methods/rl/__init__.py
    fastvideo/train/methods/rl/interleave_thinker.py
    fastvideo/train/methods/rl/rewards/__init__.py
    fastvideo/train/methods/rl/rewards/interleave_thinker.py
    fastvideo/train/models/interleave_thinker/__init__.py
    fastvideo/train/models/interleave_thinker/critic.py`
  - Result: `22 passed, 14 warnings in 19.80s`.
  - Pre-commit result: `yapf`, `ruff`, `codespell`, `mypy`, filename check,
    and suggestion hooks passed. PyMarkdown/actionlint skipped with no files to
    check.
- Commit pushed:
  - `9521077f` — `[feat] add InterleaveThinker RL training loop`
  - Pushed to `origin/interleavethinker-fastvideo`

Pending validation command:

```bash
python fastvideo/tests/modal/launch_l40s_job.py run-l40s-1 \
  --git-repo https://github.com/hao-ai-lab/FastVideo.git \
  --git-commit 633d39356804e63478d242611e992dc8e1af3caa \
  --apply-local-patch \
  --patch-paths fastvideo/train/methods/rl/__init__.py \
    fastvideo/train/methods/rl/interleave_thinker.py \
    fastvideo/train/methods/rl/rewards/__init__.py \
    fastvideo/train/methods/rl/rewards/interleave_thinker.py \
    fastvideo/train/models/interleave_thinker/__init__.py \
    fastvideo/train/models/interleave_thinker/critic.py \
    examples/train/configs/rl/interleave_thinker/critic_grpo.yaml \
    tests/local_tests/test_interleave_thinker_reward.py \
    tests/local_tests/test_interleave_thinker_method.py \
  --cmd 'pytest tests/local_tests/test_interleave_thinker_reward.py tests/local_tests/test_interleave_thinker_method.py tests/local_tests/test_train_rl_sampling.py -q && pre-commit run --files fastvideo/train/methods/rl/__init__.py fastvideo/train/methods/rl/interleave_thinker.py fastvideo/train/methods/rl/rewards/__init__.py fastvideo/train/methods/rl/rewards/interleave_thinker.py fastvideo/train/models/interleave_thinker/__init__.py fastvideo/train/models/interleave_thinker/critic.py'
```

## Full Integration Plan

Status: planning record requested by the user after the real critic checkpoint
smoke passed. This section is the durable roadmap for a complete
InterleaveThinker integration into FastVideo: planner, critic, generator
orchestration, rewards, SFT, RL, evaluation, and docs.

Working constraints for all stages:

- Continue work only in `/tmp/fastvideo-interleavethinker`.
- Do not edit `/home/toolbox/FastVideo`; that checkout belongs to another
  agent.
- Keep this file updated before context compaction or interruption.
- Make focused commits, and push immediately after each commit.
- Run GPU/model validation on Modal through
  `fastvideo/tests/modal/launch_l40s_job.py`.
- Prefer FastVideo's modular `fastvideo/train` stack for new training work.
- Keep planner/critic VLM details inside `ModelBase` wrappers. RL methods own
  algorithm logic, rewards own scoring, and orchestration lives under
  `fastvideo/entrypoints/interleave`.

Definition of "full integration":

- A user can run InterleaveThinker-style inference from FastVideo using a
  planner, an image generator/edit backend, and a critic.
- A user can train or fine-tune the planner and critic through FastVideo YAML
  configs and the modular trainer.
- A user can run critic RL with InterleaveThinker rewards through FastVideo
  rather than EasyR1/verl.
- Closed-source services used by the paper are represented by API wrappers with
  testable fake backends.
- The integration has real-checkpoint smoke coverage, fake-backend unit tests,
  and at least one Modal training smoke that exercises optimizer/checkpoint
  plumbing.

Current baseline on branch `interleavethinker-fastvideo`:

- FastVideo-compatible `/edit` service and interleave orchestration shell.
- API wrappers for Gemini / Nano Banana style reward and image backends.
- InterleaveThinker reward parser/scorer.
- Initial `InterleaveThinkerRLMethod` with grouped rollouts and advantages.
- `InterleaveThinkerCriticModel` Qwen3-VL wrapper.
- Real Modal L40S critic smoke:
  - model: `InterleaveThinker/Critic-SFT-8B`;
  - processor: `Qwen/Qwen3-VL-8B-Instruct`;
  - backend: `Qwen3VLForConditionalGeneration`;
  - result: non-empty response and `SMOKE_OK`.

### Stage 0: Scope, Contracts, And Branch Hygiene

Goal:

- Convert the existing exploratory branch into a reviewable, staged integration
  plan with clear acceptance gates.

Work items:

- Keep this handoff as the canonical progress document.
- Add a short public design doc under `docs/` or `examples/interleave/` that
  defines the integration surfaces:
  - inference/orchestration;
  - planner model;
  - critic model;
  - SFT methods/configs;
  - RL method/configs;
  - reward backends;
  - evaluation scripts.
- Identify which upstream artifacts are required:
  - `InterleaveThinker/InterleaveThinker-Planner-8B`;
  - `InterleaveThinker/Critic-SFT-8B`;
  - `InterleaveThinker/InterleaveThinker-Critic-8B`;
  - `InterleaveThinker/Train-Data`;
  - upstream `demo_klein.py`;
  - upstream `train/EasyR1/local_scripts/run_interleave_thinker_rl.sh`;
  - upstream reward function.
- Decide and document the model-port shape:
  - planner and critic are FastVideo `ModelBase` training adapters around
    Transformers Qwen3-VL, not native FastVideo DiT components;
  - no checkpoint conversion is needed for upstream HF checkpoints unless later
    FastVideo-native Qwen3-VL support is explicitly required;
  - LoRA adapter save/load should use existing FastVideo/PEFT mechanisms where
    possible.

Files likely touched:

- `.agents/exploration/interleavethinker-fastvideo-integration.md`
- `docs/` or `examples/interleave/README.md`
- `examples/train/configs/rl/interleave_thinker/README.md`

Acceptance gates:

- Design doc reviewed in the branch.
- Handoff lists exact upstream checkpoints and current validation status.
- `git status --short --branch` clean after commit and push.

### Stage 1: Public Package Contract And Examples

Goal:

- Make the intended public surface obvious and stable before adding more
  implementation.

Work items:

- Define package-level exports:
  - `fastvideo.entrypoints.interleave`;
  - `fastvideo.train.models.interleave_thinker`;
  - `fastvideo.train.methods.rl.interleave_thinker`;
  - `fastvideo.train.methods.rl.rewards.interleave_thinker`.
- Add or update examples:
  - single-prompt interleaved inference;
  - FastVideo-compatible `/edit` server;
  - planner smoke config;
  - critic smoke config;
  - critic RL config;
  - planner SFT config;
  - critic SFT config.
- Ensure example names separate inference, SFT, and RL:
  - `examples/interleave/run_interleave_prompt.py`;
  - `examples/interleave/flux2_klein_interleave_serve.yaml`;
  - `examples/train/configs/interleave_thinker/planner_sft_lora.yaml`;
  - `examples/train/configs/interleave_thinker/critic_sft_lora.yaml`;
  - `examples/train/configs/rl/interleave_thinker/critic_grpo_lora.yaml`.
- Add README content for required credentials:
  - Hugging Face token, if needed for model/dataset access;
  - Google/Gemini API key for closed-source reward backends;
  - FastVideo local generator endpoint for open generator backends.

Acceptance gates:

- Config parse tests prove every public YAML target is importable.
- Example docs name which commands require Modal/GPU/API credentials.

### Stage 2: Shared Qwen3-VL Actor Base

Goal:

- Remove critic-only duplication and create the shared runtime both planner and
  critic need.

Work items:

- Add a shared base module, for example:
  `fastvideo/train/models/interleave_thinker/qwen_actor.py`.
- Move shared functionality out of `critic.py`:
  - `AutoProcessor` loading;
  - model-class selection, preferring `Qwen3VLForConditionalGeneration` when
    available;
  - `torch_dtype`, `device_map`, `attn_implementation`, `trust_remote_code`;
  - chat-template application;
  - image path normalization;
  - device movement;
  - response-token masking for SFT/RL;
  - trainable/freezing policy;
  - LoRA enablement.
- Keep role-specific prompt/message construction in planner and critic
  subclasses.
- Add explicit model/runtime diagnostics:
  - loaded processor class;
  - loaded model class;
  - trainable parameter count;
  - frozen vision/projector counts;
  - dtype/device summary.
- Preserve `load_backend=false` for unit tests.

Files likely touched:

- `fastvideo/train/models/interleave_thinker/qwen_actor.py`
- `fastvideo/train/models/interleave_thinker/critic.py`
- `fastvideo/train/models/interleave_thinker/__init__.py`
- `tests/local_tests/test_interleave_thinker_critic_model.py`
- new `tests/local_tests/test_interleave_thinker_qwen_actor.py`

Acceptance gates:

- Existing critic tests still pass.
- Fake-backend tests cover shared generation and response-token masking.
- Modal real critic smoke still passes after the refactor.

### Stage 3: Planner Model Wrapper

Goal:

- Add a first-class FastVideo planner model adapter for
  `InterleaveThinker/InterleaveThinker-Planner-8B`.

Work items:

- Add `fastvideo/train/models/interleave_thinker/planner.py`.
- Implement `InterleaveThinkerPlannerModel` with hooks:
  - `generate_interleave_plan(batch_or_prompt, **kwargs)`;
  - `train_interleave_supervised(...)` or shared SFT loss hook;
  - optional `build_messages(...)` for text-only and image-conditioned planning.
- Port upstream planner prompt formats from:
  - `UEval/system.py`;
  - `demo_klein.py`;
  - data generation scripts if they define the canonical JSON format.
- Add plan dataclasses or Pydantic models:
  - execution plan;
  - step number;
  - step name;
  - instruction;
  - generator prompt;
  - auxiliary text.
- Add strict and permissive parsing modes:
  - strict for training/evaluation;
  - permissive for inference recovery when the model emits near-valid JSON.
- Support optional input image paths for guidance-style tasks.

Files likely touched:

- `fastvideo/train/models/interleave_thinker/planner.py`
- `fastvideo/entrypoints/interleave/schema.py`
- `fastvideo/entrypoints/interleave/orchestrator.py`
- `tests/local_tests/test_interleave_thinker_planner_model.py`
- `tests/local_tests/test_interleave_orchestrator.py`

Acceptance gates:

- Fake backend planner tests pass.
- Modal real planner smoke:
  - load `InterleaveThinker/InterleaveThinker-Planner-8B`;
  - use processor `Qwen/Qwen3-VL-8B-Instruct`;
  - generate a plan for one text prompt;
  - parse at least one valid execution step.

### Stage 4: Full Inference Orchestration

Goal:

- Make FastVideo run the whole InterleaveThinker loop, not just individual
  model calls.

Work items:

- Expand `InterleaveOrchestrator` to match upstream flow:
  1. accept user prompt and optional input image;
  2. call planner to create execution plan;
  3. for each plan step, select generation or edit mode;
  4. call generator/edit backend;
  5. call critic with before/after image pair and prompt state;
  6. retry failed step with `refine_prompt`;
  7. stop when the step passes or retry budget is exhausted;
  8. emit final interleaved text/image sequence and full trace.
- Support multiple generator backend types:
  - FastVideo local model;
  - FastVideo `/edit` HTTP service;
  - upstream-style generator API;
  - Gemini/Nano Banana wrappers for closed-source comparison;
  - fake deterministic backend.
- Trace schema should record:
  - original user prompt;
  - planner raw response and parsed plan;
  - every step attempt;
  - prompt used for generation/edit;
  - before and after image paths;
  - critic raw response and parsed answer;
  - success/failure;
  - timing and backend metadata;
  - random seed and generation parameters.
- Add CLI:
  - `fastvideo interleave-run --config ... --prompt ... --output-dir ...`;
  - optional `--input-image`;
  - optional `--max-step-iterations`;
  - optional `--save-trace-json`.
- Add server route if useful:
  - `/interleave/run` for a complete prompt-to-trace request.

Files likely touched:

- `fastvideo/entrypoints/interleave/orchestrator.py`
- `fastvideo/entrypoints/interleave/schema.py`
- `fastvideo/entrypoints/interleave/generator.py`
- `fastvideo/entrypoints/interleave/server.py`
- `fastvideo/entrypoints/cli/main.py`
- `fastvideo/entrypoints/cli/interleave_run.py`
- `examples/interleave/`
- `tests/local_tests/test_interleave_orchestrator.py`
- `tests/local_tests/test_interleave_cli.py`

Acceptance gates:

- Pure fake-backend orchestrator tests pass:
  - one-step success;
  - retry then success;
  - retry exhaustion;
  - plan parse failure;
  - trace serialization.
- Modal smoke with real planner and critic plus fake generator passes.
- Modal smoke with real planner, real critic, and FastVideo `/edit` service is
  added when resources permit.

### Stage 5: Official Data Loaders And Converters

Goal:

- Make upstream InterleaveThinker data directly consumable by FastVideo SFT and
  RL configs.

Work items:

- Add dataset utilities under one of:
  - `fastvideo/train/models/interleave_thinker/data.py`;
  - `fastvideo/train/datasets/interleave_thinker.py` if a dataset bucket exists
    or is preferred locally.
- Support upstream files:
  - `planner_sft.json`;
  - `critic_sft.json`;
  - `critic_rl.jsonl`.
- Support `image_dir` and relative path resolution for all image keys:
  - `origin_image_path`;
  - `previous_image_path`;
  - `edited_image_path`;
  - `generated_image_path`;
  - `input_image_path`;
  - `output_image_path`;
  - upstream aliases discovered during reference study.
- Add schema normalization:
  - planner row -> prompt/messages/completion;
  - critic SFT row -> two-image evaluation prompt/completion;
  - critic RL row -> rollout seed record with evaluation labels.
- Add validation and clear errors for missing files, malformed JSON, empty
  datasets, missing image paths, and non-image file extensions.
- Add optional conversion script for local inspection:
  - `scripts/interleave_thinker/inspect_dataset.py`;
  - `scripts/interleave_thinker/convert_dataset_preview.py`.

Files likely touched:

- `fastvideo/train/models/interleave_thinker/data.py`
- `tests/local_tests/test_interleave_thinker_data.py`
- `examples/train/configs/interleave_thinker/*.yaml`
- `examples/train/configs/rl/interleave_thinker/*.yaml`

Acceptance gates:

- Unit tests cover minimal upstream-like planner, critic SFT, and critic RL
  rows.
- Modal dataset smoke loads a tiny generated fixture with relative image paths.
- Optional Modal smoke can load the HF dataset if credentials and storage are
  available.

### Stage 6: Planner And Critic SFT

Goal:

- Train planner and critic with supervised fine-tuning inside FastVideo.

Work items:

- Add an SFT method or reuse an existing generic supervised method if it fits
  multimodal causal LM response-token masking.
- Preferred shape if no existing method fits:
  - `fastvideo/train/methods/fine_tuning/interleave_thinker_sft.py`;
  - method delegates tokenization and loss to planner/critic model wrappers;
  - trainer still owns normal optimizer flow unless the method needs custom
    batching.
- Add model hooks:
  - `prepare_interleave_sft_batch(...)`;
  - `compute_interleave_sft_loss(...)`;
  - shared label masking for assistant response tokens only.
- Add LoRA-first training configs:
  - planner LoRA SFT;
  - critic LoRA SFT.
- Add full-finetune configs only after LoRA path is stable.
- Ensure checkpoint save/resume works:
  - LoRA adapter checkpoint;
  - optimizer/scheduler state;
  - training state.

Files likely touched:

- `fastvideo/train/methods/fine_tuning/interleave_thinker_sft.py`
- `fastvideo/train/methods/fine_tuning/__init__.py`
- `fastvideo/train/models/interleave_thinker/qwen_actor.py`
- `fastvideo/train/models/interleave_thinker/planner.py`
- `fastvideo/train/models/interleave_thinker/critic.py`
- `examples/train/configs/interleave_thinker/planner_sft_lora.yaml`
- `examples/train/configs/interleave_thinker/critic_sft_lora.yaml`
- `tests/local_tests/test_interleave_thinker_sft_method.py`

Acceptance gates:

- Fake-backend SFT loss tests verify:
  - prompt tokens are masked with `-100`;
  - response tokens are trainable;
  - gradients flow to trainable language parameters;
  - frozen vision/projector parameters remain frozen.
- Modal one-step LoRA planner SFT smoke.
- Modal one-step LoRA critic SFT smoke.
- Pre-commit passes on changed files.

### Stage 7: Critic RL Upgrade To EasyR1-Parity GRPO

Goal:

- Move from the current useful GRPO-like skeleton to an EasyR1-parity critic RL
  method that is credible for real training.

Work items:

- Add per-token logprob support to the critic actor wrapper:
  - generated response token ids;
  - attention masks;
  - response masks;
  - current policy logprobs;
  - old policy logprobs captured at rollout time;
  - optional reference policy logprobs.
- Upgrade `InterleaveThinkerRLMethod` objective:
  - group-normalized advantages;
  - PPO/GRPO ratio;
  - clipping;
  - optional KL penalty;
  - configurable KL coefficient;
  - reward/advantage normalization metrics;
  - rollout microbatching and update microbatching.
- Add optional reference model role:
  - `models.reference`;
  - frozen Qwen3-VL critic SFT checkpoint;
  - shared processor.
- Add rollout controls:
  - `num_generations`;
  - `rollout_batch_size`;
  - `global_batch_size`;
  - `micro_batch_size_per_device_for_update`;
  - `micro_batch_size_per_device_for_experience`;
  - `temperature`;
  - `top_p`;
  - `max_new_tokens`.
- Add memory controls:
  - LoRA mode;
  - gradient checkpointing;
  - flash attention when available;
  - FSDP/HSDP compatibility investigation;
  - tensor parallel only if it fits FastVideo's training stack cleanly.
- Make optimizer behavior explicit:
  - method-managed optimization remains appropriate;
  - trainer callbacks/checkpointing should still see optimizer/scheduler state.
- Add parity notes against upstream EasyR1:
  - fields intentionally matching;
  - fields intentionally not supported yet;
  - semantic differences in distributed execution.

Files likely touched:

- `fastvideo/train/methods/rl/interleave_thinker.py`
- `fastvideo/train/models/interleave_thinker/qwen_actor.py`
- `fastvideo/train/models/interleave_thinker/critic.py`
- `examples/train/configs/rl/interleave_thinker/critic_grpo_lora.yaml`
- `tests/local_tests/test_interleave_thinker_method.py`
- new `tests/local_tests/test_interleave_thinker_grpo_math.py`

Acceptance gates:

- Unit tests prove GRPO math on deterministic fake logprobs:
  - advantage grouping;
  - ratio/clipping;
  - KL term;
  - mask handling;
  - metric names.
- Fake actor update test proves only response tokens contribute to loss.
- Modal one-step LoRA critic RL smoke with real checkpoint and fake edit scorer.
- Modal one-step critic RL smoke with API reward backend only if credentials are
  available and cost/rate limits are acceptable.

### Stage 8: Reward Backend Completion

Goal:

- Fully represent the InterleaveThinker paper reward setup in FastVideo with
  testable open and closed backends.

Work items:

- Keep reusable reward aggregation in
  `fastvideo/train/methods/rl/rewards/interleave_thinker.py`.
- Keep network/API clients in
  `fastvideo/train/methods/rl/rewards/interleave_api.py`.
- Complete backends:
  - format reward;
  - judge accuracy reward;
  - semantic reward;
  - quality reward;
  - Gemini judge wrapper;
  - Nano Banana image generation/edit wrapper;
  - Gemini image generation/edit wrapper;
  - FastVideo `/edit` HTTP wrapper;
  - local fake scorer for tests.
- Add operational features:
  - retries with exponential backoff;
  - request timeout;
  - max concurrency;
  - disk cache keyed by prompt/image/model params;
  - artifact output directory;
  - redacted logging for API keys;
  - fallback reward behavior when backend fails.
- Add reward request/response schema versioning so saved traces remain
  inspectable.

Files likely touched:

- `fastvideo/train/methods/rl/rewards/interleave_thinker.py`
- `fastvideo/train/methods/rl/rewards/interleave_api.py`
- `fastvideo/train/methods/rl/rewards/__init__.py`
- `tests/local_tests/test_interleave_thinker_reward.py`
- `tests/local_tests/test_interleave_thinker_api_models.py`

Acceptance gates:

- Unit tests cover all parsing and aggregation branches.
- API-wrapper tests use fake clients, not live credentials.
- Optional Modal live API smoke is recorded only when credentials are present.

### Stage 9: Generator Backend Integration

Goal:

- Make InterleaveThinker generation/edit calls work against both FastVideo and
  closed-source backends.

Work items:

- Finish generator backend interface:
  - `generate(prompt, ...)`;
  - `edit(prompt, image, ...)`;
  - `supports_edit`;
  - `supports_text_to_image`;
  - output image path/base64/PIL handling.
- Implement or complete backends:
  - FastVideo local `VideoGenerator` backend;
  - FastVideo HTTP `/edit` backend;
  - upstream-compatible HTTP backend;
  - Gemini image backend;
  - Nano Banana backend;
  - fake backend.
- Ensure request parameter mapping handles:
  - `num_inference_step` and `num_inference_steps`;
  - `guidance_scale`;
  - `width`;
  - `height`;
  - `seed`;
  - `enhance_prompt`;
  - input image.
- Define behavior for pure text generation step:
  - use blank canvas for image-only generators when needed;
  - preserve auxiliary text-only steps in the trace without forcing image
    generation.

Files likely touched:

- `fastvideo/entrypoints/interleave/generator.py`
- `fastvideo/entrypoints/interleave/server.py`
- `fastvideo/train/methods/rl/rewards/interleave_api.py`
- `examples/interleave/`
- `tests/local_tests/test_interleave_generator_backend.py`

Acceptance gates:

- Fake backend tests cover request translation and image serialization.
- Modal import/server smoke starts the `/edit` service.
- Real FastVideo generator smoke is added for the smallest practical model
  target available on Modal.

### Stage 10: Evaluation And Benchmark Harness

Goal:

- Provide enough evaluation tooling to compare FastVideo InterleaveThinker
  behavior with upstream and to catch regressions.

Work items:

- Add trace-level evaluator:
  - load saved trace;
  - validate schema;
  - summarize success rate, retries, failure reasons, token counts, timing.
- Add small prompt-set runner:
  - JSONL prompt list;
  - output trace directory;
  - resume from partial results.
- Add optional UEval/WISE/RISE adapters if datasets are available.
- Add visual artifact organization:
  - per-sample directory;
  - per-step attempt images;
  - final contact sheet or HTML report.
- Add comparison mode:
  - compare two trace directories;
  - summarize pass/fail deltas and retry counts.

Files likely touched:

- `scripts/interleave_thinker/evaluate_traces.py`
- `scripts/interleave_thinker/run_prompt_set.py`
- `examples/interleave/eval_prompts.jsonl`
- `tests/local_tests/test_interleave_evaluation.py`

Acceptance gates:

- Unit tests validate trace metrics on fake traces.
- Modal smoke runs a tiny fake-backend prompt set and writes metric JSON.
- Optional real planner/critic/generator evaluation is documented separately
  because it may require substantial GPU/API cost.

### Stage 11: Distributed And Checkpointing Hardening

Goal:

- Make the training path usable beyond toy smoke tests.

Work items:

- Verify FastVideo trainer checkpoint lifecycle with:
  - planner LoRA SFT;
  - critic LoRA SFT;
  - critic LoRA RL.
- Add resume tests or smoke commands:
  - run one step;
  - save checkpoint;
  - resume;
  - run one additional step.
- Investigate memory strategy for full 8B training:
  - LoRA default;
  - FSDP/HSDP support;
  - gradient checkpointing;
  - reference model memory impact;
  - whether tensor parallelism is compatible with Transformers Qwen3-VL in this
    wrapper.
- Add clear config comments for practical hardware:
  - one L40S LoRA smoke;
  - multi-GPU full or larger LoRA run;
  - expected storage for HF weights and outputs.

Files likely touched:

- training configs under `examples/train/configs/interleave_thinker/`
- RL configs under `examples/train/configs/rl/interleave_thinker/`
- docs/readmes
- possibly trainer/checkpoint integration if current callbacks cannot see
  method-managed optimizer state.

Acceptance gates:

- Modal checkpoint/resume smoke for one LoRA SFT config.
- Modal checkpoint/resume smoke for one LoRA RL config if memory permits.
- Documentation clearly states unvalidated full-finetune paths.

### Stage 12: Final Documentation And Review Package

Goal:

- Leave the branch ready for review or decomposition into PRs.

Work items:

- Write final docs:
  - quickstart inference;
  - planner SFT;
  - critic SFT;
  - critic RL;
  - reward backend configuration;
  - evaluation;
  - troubleshooting.
- Add a concise design note explaining why planner/critic are Transformers
  `ModelBase` wrappers rather than native diffusion pipeline components.
- Add a validation matrix with exact Modal URLs and commit hashes.
- Remove stale exploratory command snippets that reference old base commits or
  obsolete Modal invocation shapes.
- Ensure every public config has:
  - importable targets;
  - comments for credentials/hardware;
  - safe defaults for smoke tests where possible.

Acceptance gates:

- Full focused test suite passes on Modal.
- Pre-commit passes on all changed non-excluded files.
- Branch is clean and pushed.
- Handoff file includes:
  - latest commit hash;
  - validation evidence;
  - unresolved risks;
  - next recommended PR split.

### Recommended PR Stack

The full integration should be decomposed. Recommended stack:

1. Shared Qwen3-VL actor base plus planner model wrapper.
2. Official data loaders and SFT method/configs.
3. Full inference orchestrator, CLI, trace schema, and generator backends.
4. Reward backend completion with closed-source API wrappers and caching.
5. EasyR1-parity GRPO upgrade for critic RL.
6. Evaluation harness and final docs.

Each PR should include:

- focused code changes;
- config parse tests;
- fake-backend unit tests;
- at least one Modal smoke when the PR touches real model loading or training;
- an update to this handoff file until the work is merged or superseded.

### Open Risks And Decisions

- Full native Qwen3-VL port:
  - Current plan wraps Transformers Qwen3-VL. A native FastVideo Qwen3-VL port
    should be considered only if checkpoint conversion, distributed execution,
    or performance requirements justify the extra work.
- Full-parameter 8B optimizer memory:
  - One L40S is enough for inference smoke but likely not for full-parameter
    Adam RL. LoRA should be the first supported training path.
- Closed-source backend reproducibility:
  - Gemini/Nano Banana behavior can change. Unit tests must use fake clients,
    and live API results should be recorded as smoke evidence rather than
    deterministic regression tests.
- Upstream data availability:
  - HF dataset/model access may require tokens or change over time. Dataset
    loaders should have tiny checked-in fixtures for tests.
- EasyR1 parity:
  - FastVideo does not need to clone EasyR1 internals exactly, but any semantic
    difference in GRPO objective, KL handling, rollout batching, or distributed
    behavior must be documented.

### Immediate Next Step After This Plan

Start with Stage 2 and Stage 3 together:

- refactor `InterleaveThinkerCriticModel` onto a shared Qwen3-VL actor base;
- add `InterleaveThinkerPlannerModel`;
- add fake-backend planner tests;
- run a real Modal planner smoke with
  `InterleaveThinker/InterleaveThinker-Planner-8B`.

This is the smallest next slice that moves from critic-only integration toward
the complete planner + critic + orchestrator system while preserving the already
validated critic path.

## Stage 2/3 Execution: Shared Qwen Actor And Planner

Status: in progress as of the user request to "Execute the plan".

Scope for this implementation slice:

- Stay inside `/tmp/fastvideo-interleavethinker`; do not edit the dirty
  `/home/toolbox/FastVideo` checkout.
- Refactor shared Qwen3-VL runtime out of
  `InterleaveThinkerCriticModel` into a shared actor base.
- Preserve the existing critic public API and previously validated real
  checkpoint path.
- Add `InterleaveThinkerPlannerModel` for
  `InterleaveThinker/InterleaveThinker-Planner-8B`.
- Reuse upstream planner prompts from `/tmp/InterleaveThinker-src/UEval/system.py`:
  - `NARRATIVE_PROMPT_JSON` for text-only planning;
  - `GUIDANCE_GLOBAL_PROMPT_JSON` for image-conditioned sequence planning.
- Add parser utilities for planner `<answer>{execution_plan: [...]}</answer>`
  responses.
- Add fake-backend tests for:
  - shared placeholder/backend guard behavior;
  - critic still generates and trains through the refactored base;
  - planner message construction;
  - planner raw-response generation;
  - planner plan parsing;
  - public config import/parse for a planner smoke config.

Expected files:

- `fastvideo/train/models/interleave_thinker/qwen_actor.py`
- `fastvideo/train/models/interleave_thinker/critic.py`
- `fastvideo/train/models/interleave_thinker/planner.py`
- `fastvideo/train/models/interleave_thinker/__init__.py`
- `examples/train/configs/interleave_thinker/planner_smoke.yaml`
- `tests/local_tests/test_interleave_thinker_critic_model.py`
- `tests/local_tests/test_interleave_thinker_planner_model.py`
- this handoff file

Validation plan:

- Local lightweight checks:
  - `python -m py_compile` on changed Python files;
  - `git diff --check`;
  - local pre-commit where the environment can run it.
- Modal focused pytest:
  - `pytest tests/local_tests/test_interleave_thinker_reward.py
    tests/local_tests/test_interleave_thinker_method.py
    tests/local_tests/test_interleave_thinker_api_models.py
    tests/local_tests/test_interleave_thinker_critic_model.py
    tests/local_tests/test_interleave_thinker_planner_model.py
    tests/local_tests/test_train_rl_sampling.py -q`
- Modal pre-commit on changed non-excluded files.
- Modal real planner smoke:
  - load `InterleaveThinker/InterleaveThinker-Planner-8B`;
  - use `Qwen/Qwen3-VL-8B-Instruct` processor;
  - generate a short plan for one text prompt;
  - parse at least one execution step.

Known risks:

- Refactor could accidentally change the critic smoke path; rerun the real
  critic smoke if fake tests or planner smoke expose shared runtime issues.
- Planner real checkpoint may be larger or slower to fetch than critic, but the
  same Modal HF volume should cache it once downloaded.
- The planner may emit Python-literal JSON-like dicts rather than strict JSON;
  parser should support both, matching upstream parsing behavior.

Implemented changes:

- Added `fastvideo/train/models/interleave_thinker/qwen_actor.py` with shared
  Qwen3-VL actor loading, chat-template generation, response-token loss
  masking, image-path normalization, JSON/JSONL dataset loading, and
  non-diffusion `ModelBase` guard methods.
- Refactored `InterleaveThinkerCriticModel` to inherit from the shared actor
  base while preserving:
  - `generate_interleave_responses(...)`;
  - `train_interleave_rollouts(...)`;
  - critic-specific before/after image pairing;
  - `INTERLEAVE_CRITIC_PROMPT`;
  - compatibility re-export of `_PlaceholderActorModule`.
- Added `fastvideo/train/models/interleave_thinker/planner.py` with:
  - `InterleaveThinkerPlannerModel`;
  - `INTERLEAVE_PLANNER_PROMPT`;
  - `INTERLEAVE_GUIDANCE_PLANNER_PROMPT`;
  - `extract_interleave_plan(...)`;
  - dataclasses for parsed planner output and steps;
  - strict JSON and upstream Python-literal answer parsing.
- Added `examples/train/configs/interleave_thinker/planner_smoke.yaml`.
- Added `tests/local_tests/test_interleave_thinker_planner_model.py`.
- Updated package exports in
  `fastvideo/train/models/interleave_thinker/__init__.py`.

Validation completed:

- Local lightweight checks:
  - `python -m py_compile` passed for changed Python files.
  - `git diff --check` passed.
  - Local pre-commit passed `yapf`, `ruff`, `codespell`, filename check, and
    suggestion hooks. Local `mypy` failed only because
    `/tmp/fastvideo-interleavethinker` is not a valid package name; Modal mypy
    below is authoritative.
- Modal pytest:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-Q8Vc3IhXjRwr4fGm62v2vo`
  - Command:
    `pytest tests/local_tests/test_interleave_thinker_reward.py
    tests/local_tests/test_interleave_thinker_method.py
    tests/local_tests/test_interleave_thinker_api_models.py
    tests/local_tests/test_interleave_thinker_critic_model.py
    tests/local_tests/test_interleave_thinker_planner_model.py
    tests/local_tests/test_train_rl_sampling.py -q`
  - Result: `36 passed, 14 warnings in 18.51s`.
- Modal pre-commit:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-ZapOKZPOmhyMZFxZ0X1fQm`
  - Command:
    `pre-commit run --files
    examples/train/configs/interleave_thinker/planner_smoke.yaml
    fastvideo/train/models/interleave_thinker/__init__.py
    fastvideo/train/models/interleave_thinker/critic.py
    fastvideo/train/models/interleave_thinker/planner.py
    fastvideo/train/models/interleave_thinker/qwen_actor.py
    tests/local_tests/test_interleave_thinker_critic_model.py
    tests/local_tests/test_interleave_thinker_planner_model.py`
  - Result: yapf, ruff, codespell, mypy, filename check, and suggestion hooks
    passed; PyMarkdown/actionlint skipped with no files to check.
- First Modal real planner smoke:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-axZzqI8yL4pMTJlLQyCq15`
  - Model loaded successfully as `Qwen3VLForConditionalGeneration`.
  - Failed parser assertion because `max_new_tokens=512` cut off before the
    `<answer>` block. This confirmed the backend load path but not plan parsing.
  - Modal volume was committed, so planner weights were cached.
- Second Modal real planner smoke:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-BzH7QxVXoc5XFXBah5cJ2H`
  - Model: `InterleaveThinker/InterleaveThinker-Planner-8B`.
  - Processor: `Qwen/Qwen3-VL-8B-Instruct`.
  - Backend class: `Qwen3VLForConditionalGeneration`.
  - Generation budget: `max_new_tokens=2048`.
  - Parsed step count: `3`.
  - Smoke marker printed: `PLANNER_SMOKE_OK`.
  - Modal volume committed.
- Modal real critic refactor smoke:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-NGxUDBNJFiU30Wef0yAQN1`
  - Model: `InterleaveThinker/Critic-SFT-8B`.
  - Processor: `Qwen/Qwen3-VL-8B-Instruct`.
  - Backend class: `Qwen3VLForConditionalGeneration`.
  - Resolved relative fixture paths under `image_dir`.
  - Rollout count: `1`.
  - Smoke marker printed: `CRITIC_REFACTOR_SMOKE_OK`.
  - Modal volume committed.

Conclusion:

- Stage 2/3 is ready to commit: the shared actor base works for both real
  planner and real critic checkpoints, and fake/unit coverage verifies parser,
  message construction, generation, training loss masking, and config parsing.
- The planner smoke should use at least `max_new_tokens=2048` for this prompt;
  lower budgets can stop before the `<answer>` block.

Commit completed:

- `3b9ecb34` — `[feat] add InterleaveThinker planner actor`
- Pushed to `origin/interleavethinker-fastvideo`.

Next recommended integration step:

- Start Stage 4: wire the real planner and critic actor wrappers into the
  `fastvideo.entrypoints.interleave` orchestrator behind provider adapters.
  The first Stage 4 smoke should use the real planner + fake generator + real
  critic so the full loop can be validated without paying image-generation
  cost yet.

## Stage 4 Execution: Planner/Critic Provider Adapters

Status: completed and pushed.

Scope for this implementation slice:

- Add provider adapters that let `InterleaveOrchestrator` call
  `InterleaveThinkerPlannerModel` and `InterleaveThinkerCriticModel`.
- Keep the adapter layer under `fastvideo/entrypoints/interleave` so
  train-model wrappers remain model-specific and do not import orchestration
  concerns.
- Support fake-model unit tests first:
  - planner model response -> `PlannedInterleaveStep`;
  - critic model response -> `CriticDecision`;
  - orchestrator loop with planner adapter, fake generator, and critic adapter.
- Defer real planner + fake generator + real critic Modal smoke until the
  adapters pass unit and pre-commit validation. That smoke should create
  deterministic fixture images and avoid image-generation API cost.

Expected files:

- `fastvideo/entrypoints/interleave/providers.py`
- `fastvideo/entrypoints/interleave/__init__.py`
- `tests/local_tests/test_interleave_model_providers.py`
- this handoff file

Validation plan:

- Local syntax and `git diff --check`.
- Modal pytest including:
  - existing interleave tests;
  - new provider tests;
  - planner/critic model tests.
- Modal pre-commit on changed non-excluded files.

Implemented changes:

- Added `fastvideo/entrypoints/interleave/providers.py` with:
  - `InterleaveThinkerPlannerProvider`, adapting
    `InterleaveThinkerPlannerModel.generate_interleave_plan(...)` into
    `PlannedInterleaveStep` values for the native orchestrator;
  - `InterleaveThinkerCriticProvider`, adapting
    `InterleaveThinkerCriticModel.generate_interleave_responses(...)` into
    `CriticDecision` values via the shared `extract_interleave_answer(...)`
    parser;
  - request translation that preserves planner metadata on each step and sends
    previous/generated image paths to the critic under the fields used by the
    InterleaveThinker dataset path.
- Exported both providers from `fastvideo.entrypoints.interleave`.
- Added `tests/local_tests/test_interleave_model_providers.py` covering:
  - planner-step conversion from fake `InterleaveThinkerPlannerModel` output;
  - critic-answer conversion from fake `InterleaveThinkerCriticModel` output;
  - unparseable critic response handling;
  - a full `InterleaveOrchestrator` retry/refine loop using planner and critic
    adapters with a fake image generator.

Validation completed:

- Local lightweight checks:
  - `python -m py_compile` passed for
    `fastvideo/entrypoints/interleave/providers.py`,
    `fastvideo/entrypoints/interleave/__init__.py`, and
    `tests/local_tests/test_interleave_model_providers.py`.
  - `git diff --check` passed.
  - Local pre-commit passed `yapf`, `ruff`, `codespell`, filename check, and
    suggestion hooks. Local `mypy` still has the known false failure because
    `/tmp/fastvideo-interleavethinker` is not a valid package name; Modal mypy
    below is authoritative.
- First Modal provider pytest:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-HMmSpubEVYKYiJIH08PTOm`
  - Failed one orchestrator provider test because the fake generator returned a
    `file_path` but did not create the file. This was a test double bug, not a
    provider bug.
  - Fix: update the fake generator in
    `tests/local_tests/test_interleave_model_providers.py` to write a small
    deterministic file before returning `GeneratedImage`.
- Second Modal provider pytest:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-05SJShABrygVCKDhgiNLuL`
  - Command:
    `pytest tests/local_tests/test_interleave_entrypoint.py
    tests/local_tests/test_interleave_model_providers.py
    tests/local_tests/test_interleave_thinker_planner_model.py
    tests/local_tests/test_interleave_thinker_critic_model.py -q`
  - Result: `20 passed, 14 warnings in 15.99s`.
- Modal provider pre-commit:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-wfRX2DCt30DN903gETbDpj`
  - Command:
    `pre-commit run --files
    fastvideo/entrypoints/interleave/__init__.py
    fastvideo/entrypoints/interleave/providers.py
    tests/local_tests/test_interleave_model_providers.py`
  - Result: yapf, ruff, codespell, mypy, filename check, and suggestion hooks
    passed; PyMarkdown/actionlint skipped with no files to check.
- Modal real provider loop smoke:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-ZABadeyKBuGVcfy67LqmXt`
  - Checked out pushed commit `a2dd0b63988b575e534480ec27e2de90dfb97a4c` and
    applied the local Stage 4 patch.
  - Loaded planner model `InterleaveThinker/InterleaveThinker-Planner-8B` with
    processor `Qwen/Qwen3-VL-8B-Instruct`.
  - Loaded critic model `InterleaveThinker/Critic-SFT-8B` with the same
    processor.
  - Both backends loaded as `Qwen3VLForConditionalGeneration` on one NVIDIA
    L40S using bfloat16 and SDPA attention.
  - Ran `InterleaveOrchestrator` with real planner, fake deterministic PIL image
    generator, and real critic.
  - The trace produced one attempt. The critic returned a parsed decision with
    `success=False` and a non-empty `refine_prompt`, which is expected because
    the fake image contained a text label and off-white background.
  - Smoke marker printed: `INTERLEAVE_PROVIDER_REAL_LOOP_SMOKE_OK`.
  - Modal volume committed.

Conclusion:

- Stage 4 adapters are validated against fake provider tests, the existing
  entrypoint tests, real planner/critic model wrapper tests, Modal pre-commit,
  and a real planner + fake generator + real critic end-to-end orchestrator
  smoke.
- Commit completed:
  - `2d3bb7c7` — `[feat] wire InterleaveThinker model providers`
  - Pushed to `origin/interleavethinker-fastvideo`.
- Next step after commit: start Stage 5 by replacing the fake generator in the
  orchestrator smoke with `FastVideoImageGeneratorBackend` and a small
  FastVideo image-generation model/config, then decide whether the planner and
  critic should stay resident together or be unloaded/reloaded around the
  generator for tighter GPU memory budgets.

## Stage 5 Execution: Native Interleave Run CLI

Status: completed and pushed.

Scope for this implementation slice:

- Add a first-class `fastvideo interleave-run` command that runs the complete
  native orchestration path from a config file.
- Reuse existing pieces rather than adding another execution framework:
  - `InterleaveOrchestrator`;
  - `FastVideoImageGeneratorBackend`;
  - `NanoBananaImageGeneratorBackend`;
  - `InterleaveThinkerPlannerProvider`;
  - `InterleaveThinkerCriticProvider`;
  - `SinglePromptPlanner` and `AcceptAllCritic` for lightweight smoke paths.
- Add a typed interleave run config with blocks for:
  - `generator`: regular FastVideo `GeneratorConfig` for local generation;
  - `request`: regular `GenerationRequest` defaults for per-step generation;
  - `image_backend`: FastVideo or Nano Banana backend selection;
  - `planner`: single-prompt or InterleaveThinker planner backend;
  - `critic`: none, accept-all, or InterleaveThinker critic backend;
  - `interleave`: instruction, optional initial image, output directory, trace
    path, and trace image payload policy.
- Keep the command testable without loading GPU models by allowing runner tests
  to inject a fake image backend.

Expected files:

- `fastvideo/entrypoints/interleave/config.py`
- `fastvideo/entrypoints/interleave/runner.py`
- `fastvideo/entrypoints/cli/interleave_run.py`
- `fastvideo/entrypoints/cli/main.py`
- `fastvideo/entrypoints/interleave/orchestrator.py`
- `fastvideo/entrypoints/interleave/__init__.py`
- `examples/interleave/interleave_run.yaml`
- `examples/interleave/README.md`
- `tests/local_tests/test_interleave_run_cli.py`
- this handoff file

Validation plan:

- Local syntax and `git diff --check`.
- Modal pytest for interleave entrypoint/provider/run CLI tests.
- Modal pre-commit on changed non-excluded files.
- Modal import/config smoke for `fastvideo interleave-run` using injected/fake
  components in tests.
- A later GPU smoke should run real planner + real critic + real FastVideo
  generator once a small practical model/config is selected and GPU memory
  residency is measured.

Implemented changes:

- Added `fastvideo/entrypoints/interleave/config.py` with typed config blocks:
  - `InterleaveRunConfig`;
  - `InterleaveRunStateConfig`;
  - `InterleaveImageBackendConfig`;
  - `InterleavePlannerConfig`;
  - `InterleaveCriticConfig`.
- Added `load_interleave_run_config(...)` with:
  - FastVideo-style YAML/JSON loading;
  - dotted CLI overrides for `interleave.`, `image_backend.`, `planner.`,
    `critic.`, `request.`, and `generator.`;
  - CLI convenience fields for prompt, input image, output directory, and trace
    path;
  - explicit-path binding for the nested `GenerationRequest` so per-step
    generation defaults behave like existing `generate`/`serve` configs.
- Added `fastvideo/entrypoints/interleave/runner.py` with:
  - `run_interleave_config(...)`;
  - `build_planner(...)`;
  - `build_critic(...)`;
  - `build_image_backend(...)`;
  - trace path resolution and cleanup of local `VideoGenerator` instances.
- Added `fastvideo interleave-run` in
  `fastvideo/entrypoints/cli/interleave_run.py` and registered it in
  `fastvideo/entrypoints/cli/main.py`.
- Updated `SinglePromptPlanner` to accept a configurable retry count.
- Exported the new config/runner API from `fastvideo.entrypoints.interleave`.
- Added `examples/interleave/interleave_run.yaml` and updated the README with
  native CLI usage plus the real InterleaveThinker planner/critic config block.
- Added `tests/local_tests/test_interleave_run_cli.py` covering:
  - config loading from YAML;
  - CLI field overrides and dotted override normalization;
  - rejection of unsupported override prefixes;
  - full runner execution with an injected fake image backend;
  - configurable fallback planner attempt count.

Validation completed:

- Local lightweight checks:
  - `python -m py_compile` passed for the new/changed Python files.
  - `git diff --check` passed.
- Modal focused pytest:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-XyjjWxk63VMMAgyefDV9iz`
  - Command:
    `pytest tests/local_tests/test_interleave_entrypoint.py
    tests/local_tests/test_interleave_model_providers.py
    tests/local_tests/test_interleave_run_cli.py -q`
  - Result: `14 passed, 14 warnings in 15.93s`.
- Modal pre-commit:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-4lOvHURSxKqFmit8TqnhBe`
  - Command:
    `pre-commit run --files examples/interleave/README.md
    examples/interleave/interleave_run.yaml
    fastvideo/entrypoints/cli/interleave_run.py
    fastvideo/entrypoints/cli/main.py
    fastvideo/entrypoints/interleave/__init__.py
    fastvideo/entrypoints/interleave/config.py
    fastvideo/entrypoints/interleave/orchestrator.py
    fastvideo/entrypoints/interleave/runner.py
    tests/local_tests/test_interleave_run_cli.py`
  - Result: yapf, ruff, codespell, mypy, filename check, and suggestion hooks
    passed; PyMarkdown/actionlint skipped with no files to check.

Conclusion:

- FastVideo now has a native config-driven `interleave-run` entrypoint that can
  wire planner, generator, and critic backends without hand-written scripts.
- The default example provides a lightweight FastVideo image-backend smoke path;
  the same config can switch to the real InterleaveThinker planner and critic
  checkpoints.
- Commit completed:
  - `ee4021e5` — `[feat] add InterleaveThinker run CLI`
  - Pushed to `origin/interleavethinker-fastvideo`.
- Next step: run a GPU smoke with `fastvideo interleave-run` and a real
  FastVideo image backend. Use fallback planner/accept-all critic first to
  validate the CLI + generator path, then test real planner/critic residency or
  a split-process `/edit` generator to avoid loading all models in one process.

### Stage 5 Follow-up: CLI Config Deferral Fix

Status: completed and pushed.

Issue found during real generator smoke:

- Modal app URL: `https://modal.com/apps/hao-ai-lab/main/ap-unER4iqFrsNMavpmCOjiMx`
- Command attempted:
  `fastvideo interleave-run --config examples/interleave/interleave_run.yaml
  --prompt 'a simple centered red circle on a white background'
  --output-dir /tmp/interleave_run_smoke
  --trace-path /tmp/interleave_run_smoke/trace.json
  --request.sampling.width 512
  --request.sampling.height 512
  --request.sampling.num-inference-steps 2
  --request.sampling.seed 123`
- Result: failed before model load because `args.config` was empty in
  `InterleaveRunSubcommand.validate(...)`.
- Root cause: `FlexibleArgumentParser` auto-expands `--config` unless the
  subcommand is listed in `_DEFER_CONFIG_SUBCOMMANDS`. The list included
  `generate` and `serve` only, so it consumed the nested interleave config path
  before the subcommand validator saw it. This also affected the existing
  `interleave-serve` command.

Implemented fix:

- Updated `fastvideo/utils.py` so `_DEFER_CONFIG_SUBCOMMANDS` includes:
  - `generate`;
  - `interleave-run`;
  - `interleave-serve`;
  - `serve`.
- Added a parser regression test to
  `tests/local_tests/test_interleave_run_cli.py` verifying that
  `interleave-run --config <path>` preserves `args.config` and leaves dotted
  request overrides in the unknown list for the interleave config loader.

Validation completed:

- Local lightweight checks:
  - `python -m py_compile` passed for `fastvideo/utils.py`,
    `tests/local_tests/test_interleave_run_cli.py`,
    `fastvideo/entrypoints/cli/interleave_run.py`, and
    `fastvideo/entrypoints/cli/main.py`.
  - `git diff --check` passed.
- Modal focused pytest:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-bKcUOFZU95EC4jOaLzrqVV`
  - Command: `pytest tests/local_tests/test_interleave_run_cli.py -q`
  - Result: `6 passed, 14 warnings in 16.58s`.
- Modal pre-commit:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-CgQLuAvM35IBPru299v2re`
  - Command:
    `pre-commit run --files fastvideo/utils.py
    tests/local_tests/test_interleave_run_cli.py`
  - Result: yapf, ruff, codespell, mypy, filename check, and suggestion hooks
    passed; PyMarkdown/actionlint skipped with no files to check.
- Commit completed:
  - `375b944b` — `[bugfix] preserve interleave CLI config paths`
  - Pushed to `origin/interleavethinker-fastvideo`.

Next step after this fix is committed:

- Rerun the real FastVideo generator smoke for `fastvideo interleave-run` from
  the new pushed commit, using the same L40S command above.

### Stage 5 Follow-up: Real FastVideo Generator Smoke

Status: completed, pending handoff commit/push.

Validation completed:

- First rerun after parser fix:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-IyyHODKZKwUlDTbuKNVLf0`
  - Commit: `38513d896c42616bb24c1004ca789a3dba70811e`
  - Result: did not reach FastVideo code. `uv pip install -e '.[dev]'` failed
    downloading/extracting `plotly==6.8.0` due to the default 30 second
    `UV_HTTP_TIMEOUT`.
- Successful rerun:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-Tz4VQKximS0bSWx6Fb9aFe`
  - Commit: `38513d896c42616bb24c1004ca789a3dba70811e`
  - Env vars:
    `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA,UV_HTTP_TIMEOUT=120`
  - Command:
    `fastvideo interleave-run --config examples/interleave/interleave_run.yaml
    --prompt 'a simple centered red circle on a white background'
    --output-dir /tmp/interleave_run_smoke
    --trace-path /tmp/interleave_run_smoke/trace.json
    --request.sampling.width 512
    --request.sampling.height 512
    --request.sampling.num-inference-steps 2
    --request.sampling.seed 123`
    followed by a Python assertion that the trace has attempts, a final image
    path, and an existing output image file.
  - Loaded FastVideo FLUX.2-klein from
    `black-forest-labs/FLUX.2-klein-4B`.
  - Ran fallback `SinglePromptPlanner`, `FastVideoImageGeneratorBackend`, and
    `AcceptAllCritic`.
  - Generated image:
    `/tmp/interleave_run_smoke/interleave/2c860365e22440f4999e3c496b448341.png`
  - Trace:
    `/tmp/interleave_run_smoke/trace.json`
  - CLI output: `Success: True`.
  - Smoke marker printed:
    `INTERLEAVE_RUN_FASTVIDEO_SMOKE_OK
    /tmp/interleave_run_smoke/interleave/2c860365e22440f4999e3c496b448341.png`
  - Modal volume committed.

Conclusion:

- The native `fastvideo interleave-run` command is now validated with a real
  FastVideo image backend on Modal L40S.
- The remaining full-orchestration GPU gap is real planner + real critic +
  generator residency. Since the real planner+critic smoke already loaded both
  8B Qwen3-VL models together on L40S and this smoke loaded FLUX.2-klein
  separately, the next integration decision is whether to support all three
  models in one process or prefer a split generator service for memory
  isolation.

## Stage 6 Execution: Official Dataset Normalization

Status: completed and pushed.

Scope for this implementation slice:

- Add upstream InterleaveThinker dataset utilities for:
  - `planner_sft.json`;
  - `critic_sft.json`;
  - `critic_rl.jsonl`.
- Preserve upstream ShareGPT-style SFT rows while adding FastVideo-friendly
  aliases:
  - planner user instruction and assistant response;
  - critic origin/edited image path aliases;
  - critic `previous_prompt` / `rewritten_prompt` compatibility.
- Normalize critic RL rows to the reward-loop fields already used by
  `InterleaveThinkerRLMethod` and `InterleaveThinkerRewardScorer`:
  - `origin_prompt`;
  - `previous_prompt`;
  - `rewritten_prompt`;
  - `origin_image_path`;
  - `edited_image_path`;
  - `previous_image_path`;
  - `ground_truth`.
- Support `image_dir` path resolution and image-extension validation for scalar
  and list image fields.
- Add clear errors for missing files, empty datasets, malformed JSON, missing
  required prompt/image fields, and invalid image extensions.

Expected files:

- `fastvideo/train/models/interleave_thinker/data.py`
- `fastvideo/train/models/interleave_thinker/__init__.py`
- `tests/local_tests/test_interleave_thinker_data.py`
- this handoff file

Validation plan:

- Local syntax and `git diff --check`.
- Modal pytest for new data tests plus existing planner/critic model tests.
- Modal pre-commit on changed non-excluded files.

Implemented changes:

- Added `fastvideo/train/models/interleave_thinker/data.py` with:
  - `load_interleave_dataset(...)`;
  - `load_planner_sft_records(...)`;
  - `load_critic_sft_records(...)`;
  - `load_critic_rl_records(...)`;
  - role-specific normalizers for planner SFT, critic SFT, and critic RL rows;
  - `image_dir` path resolution for scalar and list image fields;
  - image extension validation and optional file-existence validation;
  - normalized critic RL `ground_truth` fields compatible with
    `InterleaveThinkerRewardScorer`.
- Exported dataset utilities from
  `fastvideo.train.models.interleave_thinker`.
- Added `tests/local_tests/test_interleave_thinker_data.py` covering:
  - upstream ShareGPT planner SFT rows;
  - critic SFT rows with `images` -> before/after aliases;
  - critic RL JSONL rows with `evaluation` -> `ground_truth`;
  - empty dataset rejection;
  - missing critic RL ground-truth success rejection;
  - invalid image extension rejection.

Validation completed:

- Local lightweight checks:
  - `python -m py_compile` passed for
    `fastvideo/train/models/interleave_thinker/data.py`,
    `fastvideo/train/models/interleave_thinker/__init__.py`, and
    `tests/local_tests/test_interleave_thinker_data.py`.
  - `git diff --check` passed.
- First Modal dataset pytest attempt:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-8PYqdvn1QFDZTIRBtrDIV9`
  - Failed before tests because the submitted `--git-commit` value had an
    incorrect full SHA for the pushed base commit.
- Modal dataset pytest rerun:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-bOKl6Q4p0uOjGmER7lvmc5`
  - Command:
    `pytest tests/local_tests/test_interleave_thinker_data.py
    tests/local_tests/test_interleave_thinker_critic_model.py
    tests/local_tests/test_interleave_thinker_planner_model.py -q`
  - Result: `17 passed, 14 warnings in 16.27s`.
- Modal pre-commit:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-ISuDU2lwc6Pl5NYDZnnBEb`
  - Command:
    `pre-commit run --files
    fastvideo/train/models/interleave_thinker/__init__.py
    fastvideo/train/models/interleave_thinker/data.py
    tests/local_tests/test_interleave_thinker_data.py`
  - Result: yapf, ruff, codespell, mypy, filename check, and suggestion hooks
    passed; PyMarkdown/actionlint skipped with no files to check.

Conclusion:

- FastVideo now has upstream-file-aware dataset normalization for
  InterleaveThinker planner SFT, critic SFT, and critic RL data.
- Commit completed:
  - `dcd82f93` — `[feat] add InterleaveThinker data normalizers`
  - Pushed to `origin/interleavethinker-fastvideo`.
- These utilities do not yet replace the generic `Qwen3VLActorDataset` loader;
  the next SFT/RL training integration slice should either call these utilities
  from model `init_preprocessors(...)` by dataset kind or add config fields that
  explicitly select the normalizer.

## Stage 7 Execution: Planner/Critic SFT Method

Status: completed and pushed.

Scope for this implementation slice:

- Add a native supervised fine-tuning method for InterleaveThinker Qwen3-VL
  actors.
- Reuse the shared Qwen actor response-token NLL path so prompt tokens stay
  masked and only assistant response tokens train.
- Let Qwen actor dataloaders opt into the new dataset normalizers via a model
  `dataset_kind` field:
  - `planner_sft`;
  - `critic_sft`;
  - `critic_rl`.
- Add LoRA-first planner and critic SFT config examples.
- Add fake-backend unit tests for SFT loss/backward behavior and dataset-kind
  loader selection.

Expected files:

- `fastvideo/train/models/interleave_thinker/qwen_actor.py`
- `fastvideo/train/methods/fine_tuning/interleave_thinker_sft.py`
- `fastvideo/train/methods/fine_tuning/__init__.py`
- `examples/train/configs/interleave_thinker/planner_sft_lora.yaml`
- `examples/train/configs/interleave_thinker/critic_sft_lora.yaml`
- `tests/local_tests/test_interleave_thinker_sft_method.py`
- this handoff file

Validation plan:

- Local syntax and `git diff --check`.
- Modal pytest for SFT method tests plus data/planner/critic tests.
- Modal pre-commit on changed non-excluded files.

Implemented changes:

- Added `InterleaveThinkerSFTMethod` under
  `fastvideo/train/methods/fine_tuning/`.
  - Subclasses the modular `TrainingMethod`.
  - Requires a trainable `student` role model.
  - Calls `student.init_preprocessors(...)`.
  - Builds optimizer and scheduler with the existing
    `build_optimizer_and_scheduler(...)` helper.
  - Delegates loss construction to
    `student.compute_interleave_sft_loss(...)` so model-specific message
    construction stays in the InterleaveThinker actor wrappers.
- Exported the method lazily from
  `fastvideo.train.methods.fine_tuning`.
- Extended the shared Qwen actor wrapper:
  - `InterleaveJSONLDataset` can now load records through
    `load_interleave_dataset(..., kind=...)`.
  - `Qwen3VLActorBase` accepts `dataset_kind`.
  - `compute_interleave_sft_loss(...)` reuses
    `response_nll_from_messages(...)`, masking prompt labels and training only
    assistant response labels.
  - `_sft_response_from_item(...)` accepts `response`, `completion`, `target`,
    or the first assistant message in ShareGPT-style data.
- Extended planner and critic model constructors with `dataset_kind`.
- Added LoRA-first example configs:
  - `examples/train/configs/interleave_thinker/planner_sft_lora.yaml`
  - `examples/train/configs/interleave_thinker/critic_sft_lora.yaml`
- Added `tests/local_tests/test_interleave_thinker_sft_method.py`, covering:
  - response-token-only label masking;
  - optimizer/backward path on a fake critic backend;
  - planner SFT dataset normalizer selection through `dataset_kind`;
  - public YAML parse checks for planner and critic SFT configs.

Validation completed:

- Local lightweight checks:
  - `python -m py_compile` passed for
    `fastvideo/train/models/interleave_thinker/qwen_actor.py`,
    `fastvideo/train/models/interleave_thinker/planner.py`,
    `fastvideo/train/models/interleave_thinker/critic.py`,
    `fastvideo/train/methods/fine_tuning/interleave_thinker_sft.py`,
    `fastvideo/train/methods/fine_tuning/__init__.py`, and
    `tests/local_tests/test_interleave_thinker_sft_method.py`.
  - `git diff --check` passed.
- Modal pytest on the formatted patch:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-X6a2Yhar42nkdfXIY4zvAD`
  - Repo: `https://github.com/macthecadillac/FastVideo.git`
  - Base commit: `033b75a6620316ecefdf956b7914ba0b933639f6`
  - Command:
    `pytest tests/local_tests/test_interleave_thinker_sft_method.py
    tests/local_tests/test_interleave_thinker_data.py
    tests/local_tests/test_interleave_thinker_critic_model.py
    tests/local_tests/test_interleave_thinker_planner_model.py -q`
  - Result: `20 passed, 14 warnings in 17.32s`.
- Modal pre-commit:
  - First app URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-Z9ReKctEaYcxSpM0AVI5rA`
  - Result: failed because `yapf` reformatted files; ruff, codespell, mypy,
    filename, and suggestion hooks passed.
  - Applied the same formatting locally via
    `env PRE_COMMIT_HOME=/tmp/pre-commit-cache pre-commit run --files ...`.
  - Local pre-commit caveat: after formatting, local mypy failed before normal
    checking with `fastvideo-interleavethinker is not a valid Python package
    name`, caused by the hyphenated temporary worktree path. Modal mypy is the
    authoritative result.
  - Final app URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-1jmIczO3KwZoP3WtLYOIxc`
  - Final command:
    `pre-commit run --files
    examples/train/configs/interleave_thinker/critic_sft_lora.yaml
    examples/train/configs/interleave_thinker/planner_sft_lora.yaml
    fastvideo/train/methods/fine_tuning/__init__.py
    fastvideo/train/methods/fine_tuning/interleave_thinker_sft.py
    fastvideo/train/models/interleave_thinker/critic.py
    fastvideo/train/models/interleave_thinker/planner.py
    fastvideo/train/models/interleave_thinker/qwen_actor.py
    tests/local_tests/test_interleave_thinker_sft_method.py`
  - Final result: yapf, ruff, codespell, mypy, filename, and suggestion hooks
    passed; PyMarkdown/actionlint skipped with no files to check.

Conclusion:

- FastVideo now has native planner/critic SFT entrypoints for the
  InterleaveThinker Qwen actors.
- The SFT loop can consume upstream planner/critic SFT files through the
  `dataset_kind` normalizers introduced in Stage 6.
- Commit completed:
  - `df88af31` — `[feat] add InterleaveThinker SFT method`
  - Pushed to `origin/interleavethinker-fastvideo`.
- The next integration stage is to upgrade the existing InterleaveThinker RL
  method from reward-scoring smoke coverage toward a real critic-policy
  optimization loop with rollouts, reward aggregation, and GRPO-style
  advantage/loss computation.

## Stage 8 Execution: Critic GRPO Policy Loss

Status: completed and pushed.

Scope for this implementation slice:

- Keep the existing managed `InterleaveThinkerRLMethod` outer loop.
- Add reusable, deterministic GRPO/PPO-ratio loss math under the RL common
  helpers.
- Add policy-loss controls to the InterleaveThinker RL method:
  - `clip_range`;
  - `kl_coef`;
  - update microbatch size aliases compatible with EasyR1-style configs.
- Upgrade Qwen critic actor training from advantage-weighted NLL to:
  - response-token logprobs;
  - old policy logprobs captured at rollout time when possible;
  - PPO/GRPO ratio;
  - clipping;
  - optional reference-logprob KL penalty;
  - response mask handling;
  - policy/ratio/KL/token metrics.
- Add focused fake-backend tests for GRPO math and critic actor update.

Expected files:

- `fastvideo/train/methods/rl/common/grpo.py`
- `fastvideo/train/methods/rl/common/__init__.py`
- `fastvideo/train/methods/rl/interleave_thinker.py`
- `fastvideo/train/models/interleave_thinker/qwen_actor.py`
- `fastvideo/train/models/interleave_thinker/critic.py`
- `examples/train/configs/rl/interleave_thinker/critic_grpo.yaml`
- `tests/local_tests/test_interleave_thinker_grpo_math.py`
- existing RL/SFT tests as needed

Validation plan:

- Local syntax and `git diff --check`.
- Modal pytest for new GRPO math tests plus existing InterleaveThinker method,
  reward, data, planner/critic model, and SFT tests.
- Modal pre-commit on changed non-excluded files.

Implemented changes:

- Added `fastvideo/train/methods/rl/common/grpo.py` with
  `compute_grpo_loss(...)` and `GRPOLossResult`.
  - Supports current/old logprobs, response masks, grouped advantages,
    PPO/GRPO ratio, clipping, optional reference-logprob KL, and scalar
    diagnostics.
- Exported GRPO helpers from `fastvideo.train.methods.rl.common`.
- Made `fastvideo.train.methods.rl` lazy-loaded so importing
  `fastvideo.train.methods.rl.common.grpo` does not initialize all RL methods
  and reward backends.
- Made `fastvideo.train.methods.rl.rewards` lazy-load optional frame scorers
  and Gemini/Nano Banana API wrappers while keeping the pure InterleaveThinker
  reward parser available immediately. This fixed a real circular import:
  `rewards -> interleave_api -> entrypoints.interleave.providers -> rewards`.
- Extended `InterleaveThinkerRLMethod` with:
  - `clip_range` (default `0.2`);
  - `kl_coef` (default `0.0`);
  - `micro_batch_size_per_device_for_update` /
    `update_micro_batch_size` aliasing.
- Extended `Qwen3VLActorBase` with:
  - `response_logprobs_from_messages(...)`;
  - logprob/mask coercion helpers;
  - 1D tensor padding for variable-length response token batches.
- Upgraded `InterleaveThinkerCriticModel`:
  - rollout generation now stores `old_logprobs` and `response_mask`;
  - `train_interleave_rollouts(...)` now computes the GRPO policy loss over
    response-token logprobs, with clipping, optional KL, update microbatching,
    and actor metrics.
- Updated `examples/train/configs/rl/interleave_thinker/critic_grpo.yaml`:
  - uses `dataset_kind: critic_rl`;
  - points `image_dir` at `data/InterleaveThinker/Train-Data`;
  - enables LoRA by default;
  - exposes `clip_range`, `kl_coef`, and
    `micro_batch_size_per_device_for_update`.
- Added `tests/local_tests/test_interleave_thinker_grpo_math.py`.
- Updated existing RL and critic fake-backend tests for the new policy-logprob
  training contract.

Validation completed:

- Local lightweight checks:
  - `python -m py_compile` passed for the changed RL method/common/reward
    modules, Qwen actor, critic actor, and updated tests.
  - `git diff --check` passed.
  - Local `pre-commit` with `PRE_COMMIT_HOME=/tmp/pre-commit-cache` applied
    yapf formatting. Local mypy still cannot run in this hyphenated worktree
    path and reports `fastvideo-interleavethinker is not a valid Python package
    name`; Modal mypy is authoritative.
- Modal pytest attempts:
  - `https://modal.com/apps/hao-ai-lab/main/ap-TrbqhbpPK1oQOVluAgnxQF`
    failed during collection because the new `rl.common.grpo` import exposed
    eager imports in `fastvideo.train.methods.rl.__init__`.
  - `https://modal.com/apps/hao-ai-lab/main/ap-EHjgwAiV9NRQU4STIsPA3c`
    failed during collection because `rewards.__init__` eagerly imported the
    optional API wrapper and hit a rewards/provider circular import.
  - `https://modal.com/apps/hao-ai-lab/main/ap-6QcN6Oa44Spgbwrky8ebOv`
    reached tests and failed only because old fake critic tests expected a
    label-loss-only Qwen fake.
  - `https://modal.com/apps/hao-ai-lab/main/ap-Wd3oTM0NvoHz0DrQihwOIz`
    passed the broad InterleaveThinker test set:
    `28 passed, 14 warnings in 14.90s`.
  - After yapf formatting:
    `https://modal.com/apps/hao-ai-lab/main/ap-1fLTfUteBKtC0HWIrPmdqM`
    passed the same broad test set:
    `28 passed, 14 warnings in 14.38s`.
  - After the final mypy type-guard fix:
    `https://modal.com/apps/hao-ai-lab/main/ap-I7hWfZT8l9dXqz39gmwr6M`
    passed focused GRPO/critic/method tests:
    `13 passed, 14 warnings in 17.75s`.
- Modal pre-commit:
  - `https://modal.com/apps/hao-ai-lab/main/ap-Iuh9l2XpAibion2oURXvd2`
    failed mypy on `qwen_actor.py` because `torch.is_tensor(...)` did not
    narrow `attention_mask`.
  - Fixed with `isinstance(attention_mask, torch.Tensor)`.
  - Final app URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-aYBz0F0ZiQ2nTGndudnGaH`
  - Final result: yapf, ruff, codespell, mypy, filename, and suggestion hooks
    passed; PyMarkdown/actionlint skipped with no files to check.

Conclusion:

- The critic RL path now has a real GRPO/PPO-ratio policy objective over
  response-token logprobs instead of the earlier advantage-weighted NLL
  placeholder.
- Commit completed:
  - `b7a923c0` — `[feat] add InterleaveThinker GRPO policy loss`
  - Pushed to `origin/interleavethinker-fastvideo`.
- The method still does not yet load or query a separate frozen reference model;
  it can consume per-rollout `reference_logprobs` when provided, and the next
  RL stage should add an optional `models.reference` path to compute them.
- The next full-integration step should be either:
  1. a one-step Modal LoRA critic RL smoke using the real
     `InterleaveThinker/Critic-SFT-8B` checkpoint and constant/fake edit
     scorer; or
  2. Stage 8 reward backend hardening, especially cache/concurrency/retry
     controls for live Gemini/Nano Banana calls.

### Stage 8 Follow-up: Real Critic RL Smoke And PEFT LoRA Gap

Status: completed and pushed.

Smoke attempted:

- App URL: `https://modal.com/apps/hao-ai-lab/main/ap-Uown6cfw8eGTKnXR48KVzK`
- Commit: `a312d7c6e138b843e15a4e8f1ee18aef83aea05c`
- Command shape:
  - created `/tmp/interleave_rl_smoke/critic_rl.jsonl` with one critic RL row;
  - created white/red PNG fixture images;
  - ran `torchrun --standalone --nproc_per_node=1 -m
    fastvideo.train.entrypoint.train`;
  - used the public
    `examples/train/configs/rl/interleave_thinker/critic_grpo.yaml`;
  - overrode the scorer to
    `fastvideo.train.methods.rl.rewards.ConstantInterleaveEditScorer`;
  - used two critic generations and one train step.
- Result:
  - distributed init succeeded;
  - `InterleaveThinker/Critic-SFT-8B` loaded successfully from 4 shards;
  - failure happened during LoRA setup before training:
    `No LoRA-compatible layers were found for target modules
    ['q_proj', 'k_proj', 'v_proj', 'o_proj']`.

Root cause:

- FastVideo's existing `fastvideo.train.utils.lora.enable_lora_training(...)`
  wraps FastVideo native linear layer classes such as `ColumnParallelLinear`,
  `QKVParallelLinear`, and `ReplicatedLinear`.
- Transformers Qwen3-VL checkpoints use standard HF modules, so the Qwen actor
  wrappers need PEFT LoRA rather than FastVideo's DiT LoRA wrapper.
- `peft>=0.15.0` is already a project dependency, so this can be fixed in the
  Qwen actor wrapper without adding a new dependency.

Planned follow-up implementation:

- In `Qwen3VLActorBase`, replace the call to
  `_enable_lora_if_configured(...)` with a Qwen-specific PEFT path.
- Use the same `models.<role>.lora` config block already present in examples.
- Default target modules remain `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Preserve `load_backend=false` behavior for tests.
- Validate with focused unit tests/pre-commit and rerun the real one-step critic
  RL smoke.

Implemented follow-up:

- Added a Qwen actor PEFT LoRA path in
  `fastvideo/train/models/interleave_thinker/qwen_actor.py`.
  - Uses the existing FastVideo `models.<role>.lora` config object.
  - Defaults target modules to `q_proj`, `k_proj`, `v_proj`, and `o_proj`.
  - Wraps real Transformers Qwen backends with `peft.get_peft_model(...)`.
  - Keeps `load_backend=false` placeholder actors untouched for unit tests.
  - Logs LoRA layer count and trainable/total parameter counts.
- Fixed real Qwen-VL assistant response formatting for response NLL/logprob
  recomputation.
  - The real processor expects assistant text in the same structured content
    format used by Qwen chat templates when image messages are present.
  - Added `make_assistant_text_message(...)` and used it in both
    `response_nll_from_messages(...)` and
    `response_logprobs_from_messages(...)`.

Validation completed:

- Local lightweight checks:
  - `python -m py_compile
    fastvideo/train/models/interleave_thinker/qwen_actor.py` passed.
  - `git diff --check` passed.
  - Local pre-commit still hits the known hyphenated-worktree mypy startup
    issue, but yapf/ruff/codespell passed.
- Modal focused pytest:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-yUmFCjyTWmxH77TqlhqKqq`
  - Result: `17 passed, 14 warnings in 15.77s`.
- Modal pre-commit after initial PEFT patch:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-Wd4zFHLoktBQHnAgm6tTCP`
  - Result: yapf, ruff, codespell, mypy, filename, and suggestion hooks
    passed.
- Real one-step critic RL smoke after initial PEFT patch:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-An28kFRstJ5ffKiSjREZjw`
  - Result:
    - PEFT LoRA enabled successfully:
      `trainable=3833856/8770957552`.
    - Failed during old-logprob recomputation because the assistant response
      was appended as a bare string:
      `TypeError: string indices must be integers`.
  - Fix: structured assistant text message in Qwen response helpers.
- Modal focused pytest after structured assistant-message fix:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-uWF2WjtiT1M1M2RwP2Ggog`
  - Result: `17 passed, 14 warnings in 14.23s`.
- Modal pre-commit after structured assistant-message fix:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-huN0Y2YY4ZRXvp5AEB89PM`
  - Result: yapf, ruff, codespell, mypy, filename, and suggestion hooks
    passed.
- Final real one-step critic RL smoke:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-eXMO3I81OcCyxj53XbPWj9`
  - Result:
    - `InterleaveThinker/Critic-SFT-8B` loaded from 4 shards.
    - PEFT LoRA enabled for `InterleaveThinkerCriticModel` with rank 4,
      alpha 8, targets `['q_proj', 'k_proj', 'v_proj', 'o_proj']`, and
      `3833856/8770957552` trainable parameters.
    - The trainer generated two critic rollouts, scored them with
      `ConstantInterleaveEditScorer`, recomputed response-token logprobs,
      completed one GRPO update step, and logged actor/interleave metrics.
    - The job printed `INTERLEAVE_CRITIC_RL_SMOKE_OK`.

Conclusion:

- The real Qwen critic RL loop now runs end-to-end for one step inside
  FastVideo with LoRA enabled.
- The smoke used the actual
  `InterleaveThinker/Critic-SFT-8B` checkpoint, not a fake actor.
- Commit completed:
  - `0cc04784` — `[feat] add PEFT LoRA for InterleaveThinker actors`
  - Pushed to `origin/interleavethinker-fastvideo`.
- Next full-integration stage: add optional frozen reference-policy support for
  InterleaveThinker GRPO so the method can compute KL against
  `models.reference` rather than only consuming precomputed reference logprobs.

## Stage 9 Execution: Optional Reference Policy KL For Critic GRPO

Status: completed and pushed.

Goal:

- Add first-class FastVideo `models.reference` support to the
  InterleaveThinker RL method.
- Keep the reference model frozen/eval-only.
- Compute per-response reference logprobs for generated critic rollouts before
  the student actor update.
- Feed those reference logprobs into the existing GRPO KL term.
- Keep offline/precomputed-rollout flows working when `models.reference` is not
  configured.

Planned implementation:

- `fastvideo/train/methods/rl/interleave_thinker.py`
  - Read optional `role_models["reference"]`.
  - Require `models.reference.trainable: false` when present.
  - Freeze and eval the reference transformer.
  - If present, attach `reference_logprobs` to rollouts that do not already
    include `reference_logprobs` or `ref_logprobs`.
  - Use a model-owned hook rather than putting Qwen tokenizer logic in the
    method.
  - Log how many rollouts received computed reference logprobs.
- `fastvideo/train/models/interleave_thinker/qwen_actor.py`
  - Add a reusable no-grad
    `reference_logprobs_for_interleave_rollouts(...)` hook that calls each
    actor wrapper's `build_messages(...)` and
    `response_logprobs_from_messages(...)`.
  - Preserve and restore the transformer's training/eval state.
- `examples/train/configs/rl/interleave_thinker/critic_grpo.yaml`
  - Add a frozen `models.reference` critic actor using the same base
    checkpoint and processor as the student.
  - Set `kl_coef` to a small nonzero value so the public config demonstrates
    actual reference-policy KL.
- Tests:
  - Method unit test proving a fake reference actor receives rollouts and the
    actor train call receives `reference_logprobs`.
  - Config parse test proving `models.reference` is present and frozen.
  - Critic/Qwen fake-backend test for the new reference-logprob hook.

Validation plan:

- Local syntax checks and `git diff --check`.
- Modal focused pytest for InterleaveThinker RL method, critic model, and GRPO
  math tests.
- Modal pre-commit on changed files.
- If the focused tests pass, run a one-step Modal smoke with a frozen reference
  actor on the same tiny critic RL fixture.

Implemented changes:

- `fastvideo/train/methods/rl/interleave_thinker.py`
  - Added optional `self.reference = role_models.get("reference")`.
  - Requires `models.reference.trainable=false` when a reference role is
    configured.
  - Freezes/evals the reference transformer during init and train start.
  - Computes reference logprobs for mutable rollout dictionaries that do not
    already contain `reference_logprobs` or `ref_logprobs`.
  - Keeps precomputed-reference offline rollouts working by leaving existing
    reference logprobs untouched.
  - Logs `interleave/reference_logprob_rollouts`.
- `fastvideo/train/models/interleave_thinker/qwen_actor.py`
  - Added `reference_logprobs_for_interleave_rollouts(...)`.
  - The hook is no-grad, uses the model wrapper's own `build_messages(...)`,
    returns JSON-serializable float rows, and restores the previous
    training/eval state.
- `examples/train/configs/rl/interleave_thinker/critic_grpo.yaml`
  - Added a frozen `models.reference` critic actor on
    `InterleaveThinker/Critic-SFT-8B`.
  - Set public `method.kl_coef` to `0.01`.
- Tests:
  - Added method coverage for attaching reference logprobs and rejecting a
    trainable reference model.
  - Updated public YAML parse coverage for `models.reference`.
  - Added fake Qwen critic coverage for the reference-logprob hook.

Validation completed:

- Local lightweight checks:
  - `python -m py_compile` passed for
    `fastvideo/train/methods/rl/interleave_thinker.py`,
    `fastvideo/train/models/interleave_thinker/qwen_actor.py`,
    `tests/local_tests/test_interleave_thinker_method.py`, and
    `tests/local_tests/test_interleave_thinker_critic_model.py`.
  - `git diff --check` passed.
- Modal focused pytest:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-zE4Vc67urAQpS5n5z28BWH`
  - Command:
    `pytest tests/local_tests/test_interleave_thinker_method.py
    tests/local_tests/test_interleave_thinker_critic_model.py
    tests/local_tests/test_interleave_thinker_grpo_math.py -q`
  - Result: `16 passed, 14 warnings in 15.84s`.
- Modal pre-commit:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-qJ3d8tDt6EklobhX1x0CTM`
  - Result: yapf, ruff, codespell, mypy, filename, and suggestion hooks
    passed; PyMarkdown/actionlint skipped with no files to check.
- Real one-step critic RL smoke with frozen reference actor:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-UQ38OTnymREO9bz0L1QzC5`
  - Result:
    - Loaded the trainable LoRA student
      `InterleaveThinker/Critic-SFT-8B`.
    - Enabled PEFT LoRA with rank 4 and alpha 8 on
      `['q_proj', 'k_proj', 'v_proj', 'o_proj']`.
    - Loaded the frozen reference
      `InterleaveThinker/Critic-SFT-8B`.
    - Generated two rollouts, computed old and reference response-token
      logprobs, scored with `ConstantInterleaveEditScorer`, and completed one
      GRPO update step with `kl_coef=0.01`.
    - The job printed `INTERLEAVE_CRITIC_RL_REFERENCE_SMOKE_OK`.

Conclusion:

- InterleaveThinker critic GRPO now has a first-class frozen reference-policy
  path in FastVideo.
- The path has both fake-unit coverage and a real one-step Modal smoke against
  actual InterleaveThinker critic checkpoints.
- Commit completed:
  - `42c2fe6a` — `[feat] add InterleaveThinker reference policy KL`
  - Pushed to `origin/interleavethinker-fastvideo`.
- Next full-integration stage: add a planner-policy RL path so the
  InterleaveThinker planner can be optimized with the same FastVideo GRPO loop,
  using planner-specific rollouts and rewards instead of only critic RL/SFT.

## Stage 10 Execution: Planner GRPO Path

Status: completed and pushed.

Goal:

- Let `InterleaveThinkerPlannerModel` participate in the same FastVideo
  managed GRPO loop as the critic actor.
- Keep Qwen response-token policy optimization shared between planner and
  critic wrappers.
- Add a planner-specific reward scorer for valid
  `<think>...</think><answer>{"execution_plan": [...]}</answer>` outputs, with
  optional external/fixture plan scores.
- Add a public planner GRPO YAML using a trainable planner student and frozen
  planner reference model.

Planned implementation:

- Move or duplicate the Qwen GRPO actor update onto `Qwen3VLActorBase` so both
  planner and critic actors can train over response-token logprobs.
- Add `InterleaveThinkerPlannerModel.generate_interleave_responses(...)`
  returning rollouts compatible with `InterleaveThinkerRLMethod`:
  `response`, parsed plan metadata, `old_logprobs`, `response_mask`,
  `sample_index`, `generation_index`, and `group_key`.
- Add `planner_rl` dataset normalization that can consume prompt-only planner
  RL files without requiring an assistant response.
- Add `InterleavePlannerRewardScorer` under
  `fastvideo/train/methods/rl/rewards/`.
- Add `examples/train/configs/rl/interleave_thinker/planner_grpo.yaml`.
- Add focused unit tests for:
  - planner RL data normalization;
  - planner reward parsing/scoring;
  - planner rollout generation with old logprobs;
  - planner public YAML parsing;
  - shared Qwen actor policy update behavior remains intact.

Validation plan:

- Local syntax checks and `git diff --check`.
- Modal focused pytest for planner model/data/reward/method/GRPO tests.
- Modal pre-commit on changed files.
- If feasible after unit validation, run a one-step Modal smoke with real
  `InterleaveThinker/InterleaveThinker-Planner-8B` student and frozen
  reference planner using format-only planner rewards.

Implemented changes:

- Shared Qwen actor update:
  - Moved the response-token GRPO actor update into
    `Qwen3VLActorBase.train_interleave_rollouts(...)`.
  - Moved `_grpo_logprob_batch(...)` and weighted actor-metric aggregation into
    `qwen_actor.py`.
  - Removed the duplicate critic-local GRPO update; the critic now inherits the
    shared Qwen actor path.
- Planner rollouts:
  - Added `InterleaveThinkerPlannerModel.generate_interleave_responses(...)`.
  - Planner rollouts now include `response`, parsed plan metadata, `steps`,
    `sample_index`, `generation_index`, `group_key`, `old_logprobs`, and
    `response_mask`.
  - Planner image prompt construction now accepts the upstream `images` list as
    an image-source alias.
- Planner RL data:
  - Added `planner_rl` to `InterleaveDatasetKind`.
  - Added `planner_rl.jsonl` default filename.
  - Added `load_planner_rl_records(...)` and
    `normalize_planner_rl_record(...)`.
  - Planner RL records can be prompt-only and do not require an assistant SFT
    response.
- Planner rewards:
  - Added `InterleavePlannerRewardScorer`.
  - Added `extract_interleave_plan_payload(...)`,
    `interleave_planner_format_reward(...)`, and
    `score_interleave_planner_rewards(...)`.
  - Exported planner reward utilities from
    `fastvideo.train.methods.rl.rewards`.
  - `InterleaveThinkerRLMethod` now accepts optional
    `method.reward_scorer` for scorer injection; when absent, it keeps the
    existing critic reward scorer behavior.
- Public config:
  - Added
    `examples/train/configs/rl/interleave_thinker/planner_grpo.yaml` with a
    trainable LoRA planner student, frozen planner reference, planner reward
    scorer, and GRPO/reference-KL knobs.
- Tests:
  - Added planner RL data normalization coverage.
  - Added planner reward parsing/scoring coverage.
  - Added planner RL rollout generation coverage with old logprobs.
  - Added planner GRPO public YAML parse coverage.
  - Reused existing critic GRPO tests to verify the shared Qwen refactor.

Validation completed:

- Local lightweight checks:
  - `python -m py_compile` passed for the changed InterleaveThinker model,
    reward, method, and test modules.
  - `git diff --check` passed.
  - Local pre-commit applied yapf/ruff formatting; local mypy still cannot run
    in this hyphenated worktree path and reports
    `fastvideo-interleavethinker is not a valid Python package name`.
    Modal mypy is authoritative.
- Modal focused pytest:
  - First app URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-aHGkQGhbijME5oLBFIdZOH`
  - Result: `36 passed`, `1 failed`.
  - Failure was a test fake issue: the fake planner processor returned one
    decoded response even when generation returned two sequences.
  - Fixed by making the fake `batch_decode(...)` return one response per
    sequence.
- Modal focused pytest after fake fix:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-CUch3nJsv7MWZfzik5y1wP`
  - Result: `37 passed, 14 warnings in 19.09s`.
- Modal pre-commit before local formatter application:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-TM7tfdNJ9ZfGReagJGpw8P`
  - Result: yapf and ruff modified files; codespell and mypy passed.
  - Applied the same formatter changes locally.
- Modal focused pytest after formatter changes:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-wccYJm9hNWdmEqyLZZfcnv`
  - Result: `37 passed, 14 warnings in 15.25s`.
- Final Modal pre-commit:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-xdYf7BFSVVyZqKx3Qdd0Yt`
  - Result: yapf, ruff, codespell, mypy, filename, and suggestion hooks
    passed; PyMarkdown/actionlint skipped with no files to check.
- Real one-step planner GRPO smoke:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-PDBijC8opxsMiMU0Uc064A`
  - Result:
    - Loaded the trainable LoRA student
      `InterleaveThinker/InterleaveThinker-Planner-8B`.
    - Enabled PEFT LoRA with rank 4 and alpha 8 on
      `['q_proj', 'k_proj', 'v_proj', 'o_proj']`.
    - Loaded the frozen planner reference
      `InterleaveThinker/InterleaveThinker-Planner-8B`.
    - Generated two planner rollouts, computed old and reference response-token
      logprobs, scored with `InterleavePlannerRewardScorer`, and completed one
      GRPO update step with `kl_coef=0.01`.
    - The job printed `INTERLEAVE_PLANNER_RL_SMOKE_OK`.

Conclusion:

- FastVideo now has planner SFT, critic SFT, critic GRPO, and planner GRPO
  paths for the InterleaveThinker Qwen actors.
- The planner and critic share the same Qwen response-token GRPO actor update.
- The planner and critic GRPO paths both have real one-step Modal smoke
  validation against actual InterleaveThinker 8B checkpoints.
- Commit completed:
  - `2cf9aa0d` — `[feat] add InterleaveThinker planner GRPO path`
  - Pushed to `origin/interleavethinker-fastvideo`.
- Next full-integration stage: add a first-class end-to-end
  planner-generator-critic training/evaluation workflow that wires the planner
  provider, FastVideo generation service, critic provider, and trace/reward
  logging into a single reproducible command or config.

## Stage 11 Execution: Prompt-Set Evaluation Workflow

Status: completed and pushed.

Goal:

- Add a first-class end-to-end evaluation workflow that runs the existing
  planner -> generator -> critic orchestration over a prompt set, writes one
  trace per prompt, and records aggregate success/attempt metrics.
- Keep the workflow provider/back-end driven so it can use fake providers in
  tests, a local FastVideo generator, Nano Banana, or real InterleaveThinker
  planner/critic models through config.

Implemented changes:

- Added `fastvideo/entrypoints/interleave/evaluation.py`.
  - Loads prompt sets from JSONL, JSON, or plain text.
  - Supports rows with `id`, `prompt` / `instruction`, optional
    `initial_image_path`, and arbitrary metadata.
  - Reuses planner, image backend, and critic instances across prompt rows.
  - Writes per-sample `trace.json` files under sanitized sample directories.
  - Writes aggregate `summary.json` with sample count, success count, success
    rate, total attempts, average attempts, resumed count, and per-sample
    trace paths.
  - `resume=true` skips existing traces; if all rows are already complete it
    summarizes without loading heavy backends.
- Added `fastvideo/entrypoints/cli/interleave_eval.py` and registered
  `fastvideo interleave-eval`.
  - CLI shape:
    `fastvideo interleave-eval --config <interleave_run.yaml> --prompts
    <prompts.jsonl> --output-dir <dir> [--limit N] [--resume]`.
  - Uses `load_interleave_run_config(..., require_instruction=false)` so the
    base config does not need a single prompt when a prompt set supplies rows.
- Extended `fastvideo/entrypoints/interleave/config.py` with an optional
  `require_instruction` validation flag. Existing `interleave-run` behavior
  remains strict by default.
- Exported prompt-set dataclasses and helpers from
  `fastvideo.entrypoints.interleave`.
- Added `examples/interleave/eval_prompts.jsonl` and README docs for
  prompt-set evaluation.
- Extended `tests/local_tests/test_interleave_run_cli.py` for:
  - promptless base config loading for eval;
  - `interleave-eval` CLI dotted override deferral;
  - JSONL prompt-set loading;
  - summary/trace writing with an injected fake image backend;
  - resume behavior.

Validation so far:

- Local `python -m py_compile` passed for changed Python files.
- Local `git diff --check` passed.
- Local `pytest` is unavailable (`pytest: command not found` /
  `No module named pytest`) and this task should validate on Modal anyway.
- Local import smoke is unavailable because local deps are not installed
  (`No module named torch` on FastVideo import).
- Local pre-commit with `PRE_COMMIT_HOME=/tmp/fastvideo-pre-commit-cache`
  passed yapf, ruff, codespell, filename, and suggestion hooks.
- Local mypy still fails for the known hyphenated worktree path issue:
  `fastvideo-interleavethinker is not a valid Python package name`.
  Modal mypy remains the authoritative type gate.
- First Modal validation attempt with `--apply-local-patch` was blocked by the
  approval reviewer because it would upload uncommitted patch contents to
  Modal. Next validation should commit and push this slice first, then run the
  same Modal tests against the pushed branch without `--apply-local-patch`.
- Commit completed:
  - `eca14441` — `[feat] add InterleaveThinker prompt-set eval`
  - Pushed to `origin/interleavethinker-fastvideo`.
- First pushed-branch Modal validation:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-fCNtScbJVYrewckzqMvWcg`
  - Commit tested: `eca1444101c2398cbeb9fce495487f4e91a6627d`.
  - Result: `14 passed`, `1 failed`.
  - Failure:
    `test_interleave_eval_cli_defers_nested_config_loading` saw
    `args.config == ""`.
  - Cause: `FlexibleArgumentParser._DEFER_CONFIG_SUBCOMMANDS` did not include
    `interleave-eval`, so the global parser consumed `--config` before argparse
    handled the subcommand.
  - Fix implemented locally: add `interleave-eval` to
    `_DEFER_CONFIG_SUBCOMMANDS` in `fastvideo/utils.py`.
- Commit completed:
  - `874e4e2f` — `[bugfix] defer interleave eval config loading`
  - Pushed to `origin/interleavethinker-fastvideo`.
- Second pushed-branch Modal validation:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-LMynCR2q4HKdR7TV3BxaEf`
  - Commit tested: `874e4e2f9cd24e521ff10d9cde75ba652d6cb804`.
  - Pytest result: `15 passed, 14 warnings in 14.55s`.
  - Pre-commit result: yapf, ruff, and codespell passed; mypy failed on
    `fastvideo/entrypoints/interleave/evaluation.py`.
  - Mypy failures:
    - `Name "results" already defined`;
    - `Call to untyped function (unknown) in typed context`.
  - Fix implemented locally: renamed the all-resumed branch result variable and
    replaced the untyped cleanup lambda with a typed `_noop_cleanup()`.
- Commit completed:
  - `022aedb0` — `[bugfix] clean up interleave eval mypy`
  - Pushed to `origin/interleavethinker-fastvideo`.
- Final pushed-branch Modal validation:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-eeQpAgNQvQGi2H8MB0kJCU`
  - Commit tested: `022aedb0e10492e64f30dc169eb852f51e6be4af`.
  - Command:
    `pytest tests/local_tests/test_interleave_run_cli.py
    tests/local_tests/test_interleave_model_providers.py -q && pre-commit
    run --files examples/interleave/README.md
    examples/interleave/eval_prompts.jsonl
    fastvideo/entrypoints/cli/interleave_eval.py
    fastvideo/entrypoints/cli/main.py
    fastvideo/entrypoints/interleave/__init__.py
    fastvideo/entrypoints/interleave/config.py
    fastvideo/entrypoints/interleave/evaluation.py fastvideo/utils.py
    tests/local_tests/test_interleave_run_cli.py`
  - Pytest result: `15 passed, 14 warnings in 19.96s`.
  - Pre-commit result: yapf, ruff, codespell, mypy, filename, and suggestion
    hooks passed; PyMarkdown/actionlint skipped with no files to check.
  - Modal result metadata:
    `local_patch_applied=false`, `install_extra=dev`, `build_kernel=false`.

Next recommended integration step:

- Add trace-level evaluation scripts for saved prompt-set outputs:
  `scripts/interleave_thinker/evaluate_traces.py` and/or a small HTML/contact
  sheet report. The prompt-set runner now produces the trace and summary
  artifacts those tools can consume.

## Stage 12 Execution: Trace Evaluation Reports

Status: completed and pushed.

Goal:

- Add a trace-level evaluator for the prompt-set output produced by
  `fastvideo interleave-eval`.
- Support both machine-readable aggregate metrics and a lightweight HTML report
  with final-image thumbnails.

Implemented changes:

- Added `fastvideo/entrypoints/interleave/trace_eval.py`.
  - Discovers traces from:
    - prompt-set output directories;
    - `summary.json` files;
    - explicit `trace.json` files.
  - Computes:
    - trace count, success count, success rate;
    - total/average attempts;
    - total/average retry attempts;
    - traces with final images;
    - total/average inference time when attempt timing is present;
    - failure-reason counts;
    - category-level success rates from `prompt_set_metadata.category`;
    - per-trace rows with sample id, prompt, final image, and failure fields.
  - Writes JSON metrics and an optional HTML report.
- Added `scripts/interleave_thinker/evaluate_traces.py`.
  - CLI shape:
    `python scripts/interleave_thinker/evaluate_traces.py
    <trace-or-summary-or-dir> [--output metrics.json] [--html-output
    report.html]`.
- Exported trace-evaluation helpers from `fastvideo.entrypoints.interleave`.
- Updated `examples/interleave/README.md` with post-run evaluation commands.
- Added `tests/local_tests/test_interleave_trace_eval.py` for:
  - evaluating traces discovered through `summary.json`;
  - directory discovery;
  - HTML report generation;
  - script-level JSON and HTML output.

Validation so far:

- Local `python -m py_compile` passed for changed Python files.
- Local `git diff --check` passed.
- Local pre-commit with `PRE_COMMIT_HOME=/tmp/fastvideo-pre-commit-cache`
  passed yapf, ruff, codespell, filename, and suggestion hooks.
- Local mypy still fails for the known hyphenated worktree path issue:
  `fastvideo-interleavethinker is not a valid Python package name`.
  Modal mypy remains the authoritative type gate.
- Commit completed:
  - `47335f09` — `[feat] add InterleaveThinker trace evaluation`
  - Pushed to `origin/interleavethinker-fastvideo`.
- Final pushed-branch Modal validation:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-s7ewT9rDZSTdPNhyYrEYO7`
  - Commit tested: `47335f09ee54ae178a93ea298754a10bffe23d6d`.
  - Command:
    `pytest tests/local_tests/test_interleave_run_cli.py
    tests/local_tests/test_interleave_model_providers.py
    tests/local_tests/test_interleave_trace_eval.py -q && pre-commit run
    --files examples/interleave/README.md
    fastvideo/entrypoints/interleave/__init__.py
    fastvideo/entrypoints/interleave/trace_eval.py
    scripts/interleave_thinker/evaluate_traces.py
    tests/local_tests/test_interleave_trace_eval.py`
  - Pytest result: `19 passed, 14 warnings in 39.53s`.
  - Pre-commit result: yapf, ruff, codespell, mypy, filename, and suggestion
    hooks passed; PyMarkdown/actionlint skipped with no files to check.
  - Modal result metadata:
    `local_patch_applied=false`, `install_extra=dev`, `build_kernel=false`.

Next recommended integration step:

- Prepare the review/decomposition package:
  - concise design doc for public Interleave surfaces;
  - validation matrix covering service, run CLI, prompt-set eval, trace eval,
    SFT, critic GRPO, planner GRPO, and real-checkpoint smokes;
  - PR split recommendation and unresolved risks.

## Stage 13 Execution: Review Package

Status: completed and pushed.

Goal:

- Add a stable review entrypoint outside the long exploration log.
- Summarize public surfaces, validation evidence, PR split, and remaining risks
  for reviewers.

Implemented changes:

- Added `docs/design/interleave_thinker.md`.
  - Public inference/orchestration surfaces.
  - Public training/model/reward surfaces.
  - Architecture boundaries.
  - Representative Modal validation matrix.
  - Recommended PR stack.
  - Remaining risks and review checklist.
- Added the page to the MkDocs Design nav in `mkdocs.yml`.

Validation completed:

- Local `git diff --check` passed.
- Local `pre-commit run --files docs/design/interleave_thinker.md mkdocs.yml`
  passed:
  - codespell;
  - PyMarkdown;
  - filename check;
  - suggestion hook.
  - yapf, ruff, mypy, and actionlint had no files to check.
- Local `python scripts/check_docs_links.py` failed on pre-existing generated
  examples links under `docs/examples/`, `docs/getting_started/`, `docs/inference/`,
  and `docs/training/`. No failures referenced the new
  `docs/design/interleave_thinker.md` page.
- Commit completed:
  - `6a6ebf1e` — `[docs] add InterleaveThinker review package`
  - Pushed to `origin/interleavethinker-fastvideo`.

## Stage 14 Execution: API/CLI Surface Cleanup

Status: completed in `/home/toolbox/FastVideo` on branch
`interleavethinker-fastvideo`.

User correction:

- Do not add standalone InterleaveThinker `fastvideo` subcommands such as
  `interleave-run`, `interleave-serve`, or `interleave-eval`.
- Do not add a separate InterleaveThinker HTTP API surface unless strictly
  necessary.
- Keep useful additions integrated into existing FastVideo library/training
  surfaces.
- Work in the current checkout; do not use the old `/tmp` worktree.

Implemented cleanup:

- Removed CLI registration for `interleave-run`, `interleave-serve`, and
  `interleave-eval` from `fastvideo/entrypoints/cli/main.py`.
- Restored `FlexibleArgumentParser._DEFER_CONFIG_SUBCOMMANDS` to the existing
  `{"generate", "serve"}` set.
- Deleted standalone CLI modules:
  - `fastvideo/entrypoints/cli/interleave_run.py`;
  - `fastvideo/entrypoints/cli/interleave_serve.py`;
  - `fastvideo/entrypoints/cli/interleave_eval.py`.
- Deleted the separate Interleave compatibility server:
  - `fastvideo/entrypoints/interleave/server.py`.
- Removed `build_app` from `fastvideo.entrypoints.interleave` exports.
- Deleted command/service-oriented examples and scripts:
  - `examples/interleave/flux2_klein_interleave_serve.yaml`;
  - `examples/interleave/interleave_run.yaml`;
  - `examples/interleave/eval_prompts.jsonl`;
  - `scripts/interleave_thinker/evaluate_traces.py`.
- Removed the `interleave-api` optional extra and updated Gemini/Nano Banana
  install guidance to use the existing `eval-judge` extra or direct
  `google-genai` installation.
- Rewrote `examples/interleave/README.md` and
  `docs/design/interleave_thinker.md` so they document library/training
  integration, not new FastVideo CLI/API commands.
- Kept the reusable Python helper layer and training integration:
  - `fastvideo.entrypoints.interleave` schema/generator/orchestrator/providers/
    runner/evaluation/trace helpers;
  - `fastvideo.train.models.interleave_thinker`;
  - `InterleaveThinkerSFTMethod`;
  - `InterleaveThinkerRLMethod`;
  - InterleaveThinker reward utilities and training YAMLs.

Validation completed locally:

- `python -m py_compile` passed for the touched Python files.
- `git diff --check` passed.
- `pre-commit run --files ...` passed for all surviving changed files:
  yapf, ruff, codespell, PyMarkdown, mypy, filename check, and suggestion hook.
- Focused local pytest initially failed collection because importing FastVideo
  loads `fastvideo_kernel`; Triton reports `0 active drivers` on this CPU-only
  machine.
- With a temporary local `/tmp` `fastvideo_kernel` import stub used only for
  CPU collection, all Interleave local tests passed:
  `PYTHONPATH=/tmp/fastvideo-test-stubs:$PYTHONPATH UV_CACHE_DIR=/tmp/uv-cache
  uv run pytest tests/local_tests/test_interleave_*.py -q`
  -> `62 passed, 16 warnings`.

Full-suite validation status:

- Attempted local broad suite:
  `PYTHONPATH=/tmp/fastvideo-test-stubs:$PYTHONPATH UV_CACHE_DIR=/tmp/uv-cache
  uv run pytest tests/ fastvideo/tests/ -q`.
- It failed during collection due environment/test-infra prerequisites, not a
  completed behavioral failure:
  - `flashinfer` missing for `tests/local_tests/test_nvfp4_fa4.py`;
  - several encoder/transformer/VAE tests attempted Hugging Face downloads, but
    DNS/network is blocked locally;
  - SSIM reference videos were not present locally and download failed;
  - eval tests hit missing `libxcb.so.1` for `cv2`;
  - local CPU-only import paths need a GPU or stubs for `fastvideo_kernel`.
- Attempted to launch the focused Modal L40S validation with
  `--apply-local-patch`, but the sandbox approval reviewer rejected uploading
  the unpublished local patch to Modal as external data exfiltration.
- To run the true full Modal suite for this cleanup, get explicit user approval
  to either:
  - upload this local patch to Modal with `--apply-local-patch`; or
  - commit/push the cleanup branch and run Modal against the pushed commit.

Follow-up after explicit user approval to upload unpublished local patches to
Modal:

- Focused Modal Interleave/pre-commit validation:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-LMa9TsYd1e0t54NWnNyX8A`.
  - Command:
    `pytest tests/local_tests/test_interleave_entrypoint.py
    tests/local_tests/test_interleave_model_providers.py
    tests/local_tests/test_interleave_run_cli.py
    tests/local_tests/test_interleave_trace_eval.py
    tests/local_tests/test_interleave_thinker_api_models.py -q &&
    pre-commit run --files ...`.
  - Result: `22 passed, 14 warnings`; pre-commit hooks passed.
  - Metadata: `local_patch_applied=true`, `install_extra=dev`,
    `build_kernel=false`.
- Broad combined Modal suite attempt:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-7CTQcRbDhvSyV4FdPZ2rJW`.
  - Command: `pytest tests/ fastvideo/tests/ -q`.
  - Result: failed during collection with 23 errors after SSIM refs downloaded.
    Main causes shown in output:
    - `tests/local_tests/test_nvfp4_fa4.py` requires missing `flashinfer`;
    - many subsequent errors were `ImportError: cannot import name
      'VideoGenerator' from 'fastvideo' (unknown location)` or similar
      `fastvideo.pipelines` import errors, consistent with collection-path /
      partial-import fallout during the combined run.
- Documented-suite Modal attempt with separate pytest processes:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-HnpvrSW5AX1VKolTKolBBI`.
  - Command:
    `pytest tests/ -q; pytest fastvideo/tests/ -q`.
  - Result:
    - `tests/` exited status `2` during collection because
      `tests/local_tests/test_nvfp4_fa4.py` imports `flashinfer`, which is not
      installed in the dev image.
    - `fastvideo/tests/` downloaded SSIM refs, ran for a long time, emitted
      multiple failures/errors, then exited status `131` (`Quit`) before pytest
      printed a final failure summary.
- Existing API/CLI regression tests after CLI cleanup:
  - App URL:
    `https://modal.com/apps/hao-ai-lab/main/ap-yl4NSHKpSL6ayDC8P1lPov`.
  - Command:
    `pytest fastvideo/tests/api/test_cli_translation.py
    fastvideo/tests/api/test_schema_parity_inventory.py
    fastvideo/tests/api/test_extra_overrides_routing.py -q`.
  - Result: `42 passed, 14 warnings`.

Commit completed:

- `555a600f` — `[bugfix] remove InterleaveThinker CLI surface`.
