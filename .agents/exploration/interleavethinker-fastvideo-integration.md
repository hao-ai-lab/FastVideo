# Exploration Log: InterleaveThinker FastVideo Integration

## Status

Draft handoff, shortened on 2026-06-21.

Current working location:

- Directory: `/home/toolbox/FastVideo`
- Branch: `interleavethinker-fastvideo`
- Latest observed branch head: `bb1e8935` (`[refactor] use existing workflow namespace for InterleaveThinker`)

This file is the canonical handoff for the InterleaveThinker integration work.
It intentionally summarizes older execution logs; use git history for the full
append-only detail if needed.

## Current Hard Instructions

- Work in `/home/toolbox/FastVideo` on the checked-out branch.
- Do not add standalone InterleaveThinker `fastvideo` subcommands such as
  `interleave-run`, `interleave-serve`, or `interleave-eval`.
- Do not add a separate InterleaveThinker HTTP API surface unless strictly
  necessary.
- Keep useful additions integrated into existing FastVideo library and training
  surfaces.
- Keep this handoff updated before context compaction, interruption, or a major
  direction change.
- Make focused commits as frequently as useful; push committed checkpoints when
  validation evidence should be durable.
- Do not run tests on the local machine. The local environment is not reliable
  for this work because both hardware and software prerequisites are missing.
- Run validation on Modal through `fastvideo/tests/modal/launch_l40s_job.py`.
  L40S is the normal target, but H100 or B200-class GPUs may be used when the
  task needs more memory or speed. Check Modal availability before relying on a
  specific larger GPU type.
- User approval is already granted for all Modal actions needed to finish this
  task set, including running jobs and uploading files or uncommitted patches
  from `/home/toolbox/FastVideo`.
- Prefer FastVideo's modular `fastvideo/train` stack for new training work.
  Do not migrate legacy `fastvideo/training` pipelines unless explicitly asked.
- Do not vendor InterleaveThinker, EasyR1, LLaMA-Factory, or their full training
  stacks into FastVideo.
- Planner and critic are Transformers Qwen3-VL `ModelBase` wrappers, not native
  FastVideo DiT components. A native Qwen3-VL port should happen only if
  checkpoint conversion, distribution, or performance requirements justify it.
- Boundaries:
  - VLM model details live in planner/critic model wrappers.
  - RL algorithms live in `fastvideo/train/methods/rl`.
  - Reward parsing/scoring lives under `fastvideo/train/methods/rl/rewards`.
  - Interleaved inference helpers live under
    `fastvideo/workflow/interleave_thinker`; use the pre-existing singular
    `fastvideo/workflow` namespace, not a parallel `fastvideo/workflows`
    package.

## Goal

Add a native FastVideo integration surface for InterleaveThinker-style workflows:

- run planner -> generator/edit -> critic loops through reusable Python helpers;
- train/fine-tune planner and critic Qwen3-VL actors through FastVideo YAML
  configs and the modular trainer;
- support InterleaveThinker SFT, critic GRPO, planner GRPO, reward parsing, and
  trace/evaluation utilities;
- keep tests deterministic with fake backends, and reserve real-checkpoint
  validation for Modal.

Out of scope unless explicitly re-opened:

- standalone FastVideo CLI commands dedicated to InterleaveThinker;
- a separate InterleaveThinker HTTP API/server surface;
- full-parameter 8B training as a default path;
- deterministic regression tests against live closed-source services.

## Architecture Snapshot

Implemented and retained surfaces:

- `fastvideo.workflow.interleave_thinker`
  - schema objects, generator backend translation, orchestrator, provider
    adapters, config/runner helpers, prompt-set evaluation, and trace metrics.
  - Standalone command registration and standalone server modules were removed.
- `fastvideo.train.models.interleave_thinker`
  - shared Qwen3-VL actor base;
  - planner wrapper for `InterleaveThinker/InterleaveThinker-Planner-8B`;
  - critic wrapper for `InterleaveThinker/Critic-SFT-8B`;
  - dataset normalization for planner SFT/RL and critic SFT/RL files.
- `fastvideo.train.methods.fine_tuning.interleave_thinker_sft`
  - response-token-only SFT for planner and critic.
- `fastvideo.train.methods.rl.interleave_thinker`
  - GRPO-style managed RL loop with grouped rollouts, old logprobs, optional
    frozen reference policy KL, and LoRA-first configs.
- `fastvideo.train.methods.rl.rewards.interleave_thinker`
  - critic reward parser/scorer and planner format/plan reward utilities.
- `examples/train/configs/interleave_thinker/`
  - planner and critic SFT LoRA configs.
- `examples/train/configs/rl/interleave_thinker/`
  - critic and planner GRPO LoRA configs.
- `docs/design/interleave_thinker.md`
  - review/design entrypoint that should stay shorter and more reviewer-facing
    than this exploration file.

Removed by the API/CLI cleanup:

- Interleave-specific `fastvideo` subcommand registration.
- Standalone Interleave compatibility server.
- command/service-oriented examples and scripts.
- `interleave-api` optional extra.

Namespace integration status:

- Completed correction. The reusable helper layer lives under the pre-existing
  singular `fastvideo/workflow/interleave_thinker` package, not under a new
  parallel `fastvideo/workflows` package.
- Internal imports, tests, examples, docs, and this handoff now use
  `fastvideo.workflow.interleave_thinker`.
- The old `fastvideo.entrypoints.interleave` package remains deleted rather
  than kept as a compatibility shim. This branch has not merged, so preserving
  the old public path is not required.
- Do not scatter the helper code into unrelated core modules unless a genuinely
  generic abstraction emerges. The planner -> generator/edit -> critic loop is
  InterleaveThinker-specific workflow code, not `VideoGenerator`, training
  method, or reward-parser core behavior.

## Condensed Execution History

- Initial service/orchestration slice added Interleave request/trace schema,
  generator request translation, fake-provider tests, and an early compatibility
  service. The later cleanup removed the standalone service/CLI surface but kept
  reusable Python helpers.
- Critic backend hardening added Gemini/Nano Banana-style API wrappers with lazy
  imports, fake-client tests, and no live API calls in CI-style tests.
- Real critic smoke loaded `InterleaveThinker/Critic-SFT-8B` on Modal L40S with
  `Qwen/Qwen3-VL-8B-Instruct` and produced a non-empty response.
- Shared actor/planner work added a shared Qwen3-VL actor base,
  `InterleaveThinkerPlannerModel`, planner parsing, and real planner/critic
  smokes. Commit: `3b9ecb34`.
- Provider adapters wired planner and critic model wrappers into the native
  `InterleaveOrchestrator`; real planner + fake generator + real critic smoke
  passed. Commit: `2d3bb7c7`.
- Native run/config helpers were added and validated with a real FastVideo
  FLUX.2-klein generator smoke. Later cleanup removed dedicated command
  registration while keeping reusable helper code. Commits included
  `ee4021e5` and `375b944b`.
- Dataset normalization added support for upstream planner SFT, critic SFT,
  critic RL, and planner RL formats with image path resolution and clear data
  errors. Commit: `dcd82f93`.
- Planner/critic SFT added response-token-only supervised fine-tuning and
  LoRA-first configs. Commit: `df88af31`.
- Critic GRPO upgraded from advantage-weighted NLL to response-token logprob
  policy loss with PPO/GRPO ratio, clipping, optional KL input, and metrics.
  Commit: `b7a923c0`.
- PEFT LoRA was added for Qwen actors after FastVideo's native DiT LoRA wrapper
  failed on HF Qwen modules. Real one-step critic RL smoke then passed.
  Commit: `0cc04784`.
- Optional frozen reference policy KL was added through `models.reference` and
  validated with a real one-step critic RL reference smoke. Commit: `42c2fe6a`.
- Planner GRPO added planner rollouts, planner rewards, `planner_rl` data, and
  real one-step planner RL smoke. Commit: `2cf9aa0d`.
- Prompt-set evaluation and trace metrics/report helpers were added as reusable
  Python/library surfaces. Commits included `eca14441`, `874e4e2f`,
  `022aedb0`, and `47335f09`.
- Review package added `docs/design/interleave_thinker.md` and MkDocs nav.
  Commit: `6a6ebf1e`.
- API/CLI cleanup removed standalone InterleaveThinker FastVideo commands and
  the separate server, restored normal parser behavior, and rewrote docs toward
  library/training integration. Commits: `555a600f`, `704e5667`.
- Handoff instructions were condensed and updated with standing Modal approval
  and the no-local-tests rule. Commit: `d2c53951`.
- Namespace integration first moved the helper package to
  `fastvideo.workflows.interleave_thinker`, renamed stale tests, updated docs
  and examples, and deleted the old entrypoints package. Commits: `11d55fb5`,
  `91d8fb85`.
- Follow-up correction requested by the user: move the helper package into the
  pre-existing singular `fastvideo.workflow.interleave_thinker` namespace and
  remove the parallel `fastvideo.workflows` package. Commit: `bb1e8935`.

## Validation Evidence

Representative Modal real-checkpoint or GPU-backed smokes:

- Critic SFT smoke:
  - model `InterleaveThinker/Critic-SFT-8B`;
  - processor `Qwen/Qwen3-VL-8B-Instruct`;
  - backend `Qwen3VLForConditionalGeneration`;
  - marker `SMOKE_OK`.
- Planner smoke:
  - model `InterleaveThinker/InterleaveThinker-Planner-8B`;
  - `max_new_tokens=2048` was needed for the tested prompt;
  - parsed `3` execution steps;
  - marker `PLANNER_SMOKE_OK`.
- Critic refactor smoke:
  - real critic through shared actor base;
  - marker `CRITIC_REFACTOR_SMOKE_OK`.
- Provider loop smoke:
  - real planner + fake deterministic image generator + real critic;
  - marker `INTERLEAVE_PROVIDER_REAL_LOOP_SMOKE_OK`.
- FastVideo generator smoke:
  - loaded `black-forest-labs/FLUX.2-klein-4B`;
  - generated an image and trace through the reusable interleave runner path.
  - The old command entrypoint used for this smoke has since been removed.
- Real critic RL smoke:
  - trainable LoRA critic student on `InterleaveThinker/Critic-SFT-8B`;
  - `ConstantInterleaveEditScorer`;
  - one GRPO update completed;
  - marker `INTERLEAVE_CRITIC_RL_SMOKE_OK`.
- Real critic RL reference smoke:
  - trainable LoRA critic student plus frozen critic reference;
  - old and reference response-token logprobs computed;
  - marker `INTERLEAVE_CRITIC_RL_REFERENCE_SMOKE_OK`.
- Real planner RL smoke:
  - trainable LoRA planner student plus frozen planner reference;
  - `InterleavePlannerRewardScorer`;
  - one GRPO update completed;
  - marker `INTERLEAVE_PLANNER_RL_SMOKE_OK`.

Latest cleanup validation:

- Local `python -m py_compile` passed for touched Python files.
- Local `git diff --check` passed.
- Local `pre-commit run --files ...` passed for surviving changed files:
  yapf, ruff, codespell, PyMarkdown, mypy, filename check, and suggestion hook.
- Focused local Interleave tests passed with a temporary CPU-only
  `fastvideo_kernel` import stub:
  `62 passed, 16 warnings`.
- Focused Modal Interleave/pre-commit validation passed:
  `22 passed, 14 warnings`; pre-commit hooks passed.
- Existing API/CLI regression tests after cleanup passed on Modal:
  `42 passed, 14 warnings`.
- Namespace migration validation passed on Modal L40S:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-APG1eoMxnajN1wpzPd0S4r`
  - Commit: `91d8fb85e6bb36bbeacde5e82aac8ccb22a2c9ee`
  - Pytest:
    `tests/local_tests/test_interleave_workflow_backend.py`,
    `tests/local_tests/test_interleave_model_providers.py`,
    `tests/local_tests/test_interleave_workflow_runner.py`,
    `tests/local_tests/test_interleave_trace_eval.py`, and
    `tests/local_tests/test_interleave_thinker_api_models.py`
    -> `22 passed, 14 warnings`.
  - Pre-commit on changed docs/examples/workflow/reward/test files passed:
    yapf, ruff, codespell, PyMarkdown, mypy, filename check, and suggestion.
  - `local_patch_applied=false`; validation used the pushed commit.
- Singular workflow namespace correction validation passed on Modal L40S:
  - App URL: `https://modal.com/apps/hao-ai-lab/main/ap-zAYZ80ExlxJbpvSDVWbtTu`
  - Commit: `bb1e8935ee37ea1e99896cf96fa1ea4139ff119e`
  - Pytest:
    `tests/local_tests/test_interleave_workflow_backend.py`,
    `tests/local_tests/test_interleave_model_providers.py`,
    `tests/local_tests/test_interleave_workflow_runner.py`,
    `tests/local_tests/test_interleave_trace_eval.py`, and
    `tests/local_tests/test_interleave_thinker_api_models.py`
    -> `22 passed, 14 warnings`.
  - Pre-commit on changed docs/examples/workflow/reward/test files passed:
    yapf, ruff, codespell, PyMarkdown, mypy, filename check, and suggestion.
  - `local_patch_applied=false`; validation used the pushed commit.

Broad-suite status:

- Local broad `pytest tests/ fastvideo/tests/ -q` is not a reliable signal on
  this machine due missing GPU/runtime dependencies, blocked Hugging Face
  downloads, missing SSIM references, missing `flashinfer`, and missing GUI
  libraries for `cv2`.
- Broad Modal attempts did not produce a clean full-suite result. Known blockers
  included `flashinfer` absence in the dev image and collection/import fallout
  during combined suite runs. Treat focused Modal suites plus targeted API/CLI
  regressions as the current evidence until the broad-suite environment is
  repaired.

## Current Risks And Decisions

- One-process memory residency for real planner + real critic + real generator
  is still not the recommended default. The validated approach separates heavy
  concerns or uses fake/lightweight providers for orchestration tests.
- Live Gemini/Nano Banana behavior can change and may incur cost or rate
  limits. Unit tests must use fake clients; live API runs should be recorded as
  smoke evidence only.
- HF model/dataset access may require tokens and may change over time. Keep
  tiny checked-in fixtures for parser, loader, and reward tests.
- Full-parameter 8B training is unvalidated. LoRA is the supported first path.
- Broad test validation needs a better Modal/dev image or a documented skip
  strategy for tests requiring unavailable packages and external downloads.
- Keep the standalone CLI/API cleanup intact unless the user explicitly reverses
  that product decision.

## Recommended Next Steps

1. For code work, continue from `/home/toolbox/FastVideo` on
   `interleavethinker-fastvideo` and inspect `git status --short --branch`
   before editing.
2. Read the relevant per-directory `AGENTS.md` before touching files under
   `fastvideo/`, `examples/`, `docs/`, `scripts/`, or tests.
3. The API cleanup and namespace correction are complete. Continue with review
   preparation, PR decomposition, reward-backend hardening, or broad-suite
   environment repair.
4. Validate only on Modal. Local syntax-only commands such as `git diff --check`
   are acceptable, but no local pytest or other local test execution should be
   used.
5. For the next implementation slice, either decompose the branch into
   reviewable PRs using `docs/design/interleave_thinker.md` as the reviewer
   entrypoint, or harden reward backends/cache/concurrency and the broad-suite
   validation environment.

## Useful Commands

```bash
git status --short --branch
git log --oneline -12
git diff --check
pre-commit run --files <changed paths>
```

Use Modal for all test execution and authoritative validation.
