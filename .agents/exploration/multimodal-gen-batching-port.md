# Exploration Log: Multimodal Generation Batching Port

## Status: implementation in progress

## Context
User requested a staged plan, approval before implementation, and then a port of
SGLang `python/sglang/multimodal_gen` batching into FastVideo on branch
`multimodal-gen-batching`.

Upstream reference inspected locally from a sparse clone at:
`/tmp/sglang-multimodal-gen/python/sglang/multimodal_gen`.

Primary upstream files inspected:
- `runtime/managers/scheduler.py`
- `runtime/managers/dynamic_batch_admission.py`
- `runtime/managers/gpu_worker.py`
- `runtime/pipelines_core/schedule_batch.py`
- `runtime/pipelines_core/composed_pipeline_base.py`
- `runtime/pipelines_core/executors/{pipeline_executor,sync_executor,parallel_executor}.py`
- `runtime/pipelines_core/stages/{base,dedup,latent_preparation,denoising}.py`
- `configs/sample/sampling_params.py`
- `configs/pipeline_configs/base.py`

FastVideo files inspected:
- `fastvideo/entrypoints/video_generator.py`
- `fastvideo/entrypoints/openai/{api_server,video_api}.py`
- `fastvideo/worker/{executor,multiproc_executor,gpu_worker,worker_base}.py`
- `fastvideo/pipelines/{pipeline_batch_info,composed_pipeline_base}.py`
- `fastvideo/pipelines/stages/{base,text_encoding,latent_preparation,timestep_preparation,denoising,decoding}.py`
- `fastvideo/api/schema.py`
- `fastvideo/configs/pipelines/base.py`
- `fastvideo/tests/modal/launch_l40s_job.py`

## Progress
- [x] Read repository onboarding, codebase map, relevant AGENTS guidance, and
      exploration template.
- [x] Sparse-cloned upstream SGLang multimodal generation source to `/tmp`.
- [x] Identified upstream batching mechanism and corresponding FastVideo gaps.
- [x] Drafted implementation plan for user approval.
- [x] User approved the staged implementation plan on 2026-05-30.
- [x] Stage 1 implementation: typed batching config, FastVideoArgs wiring,
      batching admission rules, compatibility signatures, and focused unit
      tests have been added locally.
- [x] Stage 1 remote validation passed on Modal L40S.
- [ ] Stage 1 commit.
- [ ] After approval, implement remaining stages with commits and remote Modal validation.
- [ ] Produce final Markdown write-up with test and benchmark results and commit it.

## Findings
Upstream SGLang batching is not just a larger prompt list. It has four pieces:

1. Queue/admission in `runtime/managers/scheduler.py`.
   Requests are received into a FIFO queue with enqueue timestamps. Compatible
   text-only generation requests are coalesced up to `batching_max_size` or
   after `batching_delay_ms`, with metrics and rejection reasons.

2. Compatibility and admission control.
   Compatibility is based on sampling-param signatures with selected fields
   excluded via `metadata={"batch_sig_exclude": True}`. Admission applies user
   max batch size plus optional JSON rules keyed by model/resolution/memory and
   a rough request cost from pipeline config.

3. Request merge/split.
   Compatible requests are deep-copied into one request with `prompt: list[str]`
   and per-request seeds/output paths stashed in `extra`. The worker runs one
   merged request and the scheduler splits tensor/list/path outputs back into
   one output per original request.

4. Pipeline support.
   SGLang added grouped pipeline execution, stage-level dedup hooks, request-local
   schedulers, grouped latent preparation that preserves per-request RNG streams,
   and denoising code that can process latent batch dimensions greater than one.

FastVideo currently has:
- `ForwardBatch.prompt` already accepts `str | list[str]`.
- Text encoding already handles list prompts.
- Latent preparation mostly handles batch sizes, but RNG equivalence must be
  preserved for grouped requests.
- Standard `DenoisingStage` has an explicit `assert latent_model_input.shape[0] == 1`;
  this is a hard blocker for true native batching.
- `VideoGenerator._generate_request_impl` expands prompt lists sequentially.
- OpenAI video API launches each request as an independent background thread
  against one global `VideoGenerator`; there is no scheduler queue/admission
  layer.

## Approved Plan
Approved by the user on 2026-05-30.

Stages:
1. Add batching config and pure admission/signature primitives.
2. Refactor generator/executor/pipeline surfaces for merge/split.
3. Enable true native batching for a conservative text-only path.
4. Integrate the OpenAI server queue scheduler.
5. Run all validation remotely through
   `fastvideo/tests/modal/launch_l40s_job.py`.
6. Run parity and before/after benchmarks.
7. Save a final Markdown report, commit it, and push the branch.

Scope guard: initial dynamic batching only supports compatible text-only
requests. Image/video/audio/refine/continuation/control inputs remain routed
to sequential execution until separately audited.

## Implementation Log

### Stage 1: Config And Pure Batching Primitives
Files changed/added:
- `fastvideo/api/schema.py`
  - Added `BatchingConfig` under `EngineConfig`.
- `fastvideo/fastvideo_args.py`
  - Added `batching_mode`, `batching_max_size`, `batching_delay_ms`,
    `batching_config`, and `enable_batching_metrics`.
  - Added CLI flags and basic validation.
- `fastvideo/api/compat.py`
  - Mapped legacy flat kwargs to `EngineConfig.batching`.
  - Emitted typed batching config back to `FastVideoArgs`.
- `fastvideo/entrypoints/video_generator.py`
  - Allowed batching kwargs through `from_pretrained` convenience handling.
- `fastvideo/configs/pipelines/base.py`
  - Added default `estimate_request_cost()` for admission budgets.
- `fastvideo/batching/`
  - Added `admission.py` with SGLang-style rule parsing and cap logic.
  - Added `signature.py` with request-local exclusions and safe text-only
    compatibility checks.
- `fastvideo/tests/batching/`
  - Added focused admission/signature tests.
- `fastvideo/tests/api/test_compat_translation.py`
  - Added batching config translation coverage.

Pending for Stage 1:
- Commit Stage 1 once remote validation passes.

Validation attempt:
- Modal L40S command:
  `pytest fastvideo/tests/batching fastvideo/tests/api/test_compat_translation.py -q && pre-commit run --files ...`
- Result:
  - Tests passed: `27 passed, 14 warnings`.
  - Pre-commit failed only on Ruff `SIM103` in the two new batching helper
    files.
- Fix applied locally:
  - Simplified the relevant boolean returns in `admission.py` and
    `signature.py`.

Validation rerun:
- Modal app: `ap-q9SRzkYvwXzoPA5oWoaM3i`
- Command:
  `pytest fastvideo/tests/batching fastvideo/tests/api/test_compat_translation.py -q && pre-commit run --files ...`
- Result:
  - Tests passed: `27 passed, 14 warnings`.
  - Pre-commit passed: yapf, ruff, codespell, mypy, filename spaces, and
    suggestion hooks.

## Validation Plan
All validation must run remotely through:
`python -m modal run fastvideo/tests/modal/launch_l40s_job.py ...`

Planned remote validation after implementation:
- CPU/light import and unit tests on Modal image, not local.
- Focused batching unit tests for compatibility/admission/merge/split.
- GPU parity tests on Modal L40S comparing sequential vs batched outputs with
  fixed seeds, prompt sets, model, resolution, steps, backend, and save disabled.
- Existing SSIM regression for affected model(s) if references are available.
- Before/after benchmark suite with dynamic batching disabled vs enabled and
  sequential prompt-file baseline vs dynamic server batching.

## Mistakes / Dead Ends
None yet.

## Proposed Standardization
If this port lands cleanly, create a runtime batching SOP covering:
- compatibility-signature design,
- deterministic grouped latent generation,
- per-stage grouped execution contracts,
- Modal-only parity and benchmark commands.

## Hand-Off
Current branch: `multimodal-gen-batching`.

Current local implementation state:
- Stage 1 code is present locally and remote validated.
- Stage 1 commit is the next action.

Important constraints:
- Do not edit unrelated untracked files already present in the worktree.
- User pre-approved tool/command use, commits to current branch, and push to
  origin after implementation stages.
- Use Modal L40S jobs for tests; local machine should not be treated as a valid
  test environment.

Next step:
Commit the Stage 1 slice, then begin Stage 2 generator/executor merge-split
work.
