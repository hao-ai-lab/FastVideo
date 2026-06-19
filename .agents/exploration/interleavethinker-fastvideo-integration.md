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
