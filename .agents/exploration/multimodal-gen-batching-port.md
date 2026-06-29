# Exploration Log: Multimodal Generation Batching Port

## Status: complete with documented dynamic parity limitation

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
- [x] Stage 1 commit: `3d3157fb` (`[feat]: add generation batching primitives`).
- [x] Stage 1 state commit: `ce758820`
      (`[misc]: record batching stage 1 state`).
- [x] Stage 2 local implementation:
      - added generator prepared-work-item merge/split helpers,
      - added `generate_video_batch()` for later server queue use,
      - enabled prompt-file dynamic batching,
      - preserved explicit per-request seeds for prompt-list batches,
      - repeated negative prompts for CFG with prompt lists,
      - removed the standard denoising stage batch-size-1 assertion,
      - added focused CPU-light tests with fake forward execution.
- [x] Stage 2 remote validation passed on Modal L40S.
- [x] Stage 2 commit: `fe5572f3`
      (`[feat]: add generator dynamic batching path`).
- [x] Stage 2 state commit: `39cb1d43`
      (`[misc]: record batching stage 2 state`).
- [x] Stage 5 local implementation:
      - added `VideoBatchScheduler` for OpenAI video requests,
      - stores the scheduler in OpenAI server state,
      - starts/stops the scheduler in API server lifespan,
      - routes `video_api._run_generation()` through the scheduler when
        dynamic batching is enabled,
      - added async scheduler tests for compatible grouping and incompatible
        fallback.
- [x] Stage 5 remote validation passed on Modal L40S.
- [x] Stage 5 commit: `4c6a9dc2`
      (`[feat]: add OpenAI video batching scheduler`).
- [x] Added GPU parity/benchmark helper script:
      `fastvideo/tests/batching/run_dynamic_batching_parity.py`.
- [x] Remote GPU parity run completed; dynamic batched denoising did not meet
      near-bit-identical tolerance and is documented as a limitation.
- [x] Remote benchmark runs completed for batch size 2, batch size 4, and an
      8-step batch size 2 workload.
- [x] Commit GPU helper, parity fixes, and benchmark state:
      `9c6eb355` (`[fix]: harden dynamic generation batching`).
- [x] Commit final Markdown write-up with test and benchmark results:
      `2304837e` (`[docs]: record multimodal batching validation report`).
- [x] Push branch to origin after final report commit.
- [x] Final changed-file validation passed on Modal L40S.
- [x] Full `pre-commit run --all-files` passed on Modal L40S at `71fe451b`
      using `fastvideo/tests/modal/launch_l40s_job.py` from `interleavethinker`
      (`ap-XLXARFbxkJa40Yom5VxxfX`).
- [x] Reviewed `docs/inference/support_matrix.md`; no update is required because
      this branch does not add a model ID or registry entry.

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

Stage 1 is complete.

### Stage 2: Generator Merge/Split And Batch-Safe Standard Stages
Files changed/added:
- `fastvideo/entrypoints/video_generator.py`
  - Added `_GenerationWorkItem`, forward execution helper, output
    postprocessing helper, output split helper, work-item merge/grouping logic,
    and `generate_video_batch()`.
  - `prompt_txt` / `SamplingParam.prompt_path` now uses dynamic batching when
    `batching_mode=dynamic` and `batching_max_size>1`; otherwise it keeps the
    prior sequential behavior.
- `fastvideo/pipelines/stages/input_validation.py`
  - Preserves `ForwardBatch.seeds` when already supplied by a merged batch.
  - Generates one seed per prompt for prompt-list batches.
- `fastvideo/pipelines/stages/text_encoding.py`
  - Expands a single negative prompt across a prompt list for CFG.
- `fastvideo/pipelines/stages/denoising.py`
  - Removed the explicit standard-path `shape[0] == 1` assertion.
- `fastvideo/tests/entrypoints/test_video_generator.py`
  - Added fake-forward tests for merged compatible requests and sequential
    fallback on incompatible requests.
- `fastvideo/tests/stages/test_input_validation_batching.py`
  - Added seed preservation and prompt-list seed fanout tests.

Stage 2 is complete.

Validation attempt:
- Modal app: `ap-vsU8ZgjMvfGgpRBWk4IfCV`
- Result:
  - Tests failed before pre-commit: `1 failed, 50 passed, 14 warnings`.
  - Failure was an existing entrypoint test using a `SimpleNamespace` test
    double without new batching fields.
- Fix applied locally:
  - `_dynamic_batching_enabled()` now defaults missing batching fields to
    disabled / max size 1.

Validation rerun:
- Modal app: `ap-Y6B6cRSzW9frq4IIU7Tyqj`
- Result:
  - Tests passed: `51 passed, 14 warnings`.
  - Pre-commit failed only on Ruff `F841` for an unused `sampling_param`
    local in `VideoGenerator._postprocess_generation_output()`.
- Fix applied locally:
  - Removed the unused local.

Validation clean rerun:
- Modal app: `ap-0LrnrxcMw4Ni3M9YJ6gtZv`
- Command:
  `pytest fastvideo/tests/batching fastvideo/tests/api/test_compat_translation.py fastvideo/tests/entrypoints/test_video_generator.py fastvideo/tests/stages/test_input_validation_batching.py -q && pre-commit run --files ...`
- Result:
  - Tests passed: `51 passed, 14 warnings`.
  - Pre-commit passed: yapf, ruff, codespell, mypy, filename spaces, and
    suggestion hooks.

### Stage 5: OpenAI Server Queue Integration
Files changed/added:
- `fastvideo/entrypoints/openai/batching.py`
  - Added `VideoBatchScheduler`, an async FIFO queue with batching delay,
    compatibility checks, background dispatch, and per-request futures.
- `fastvideo/entrypoints/openai/state.py`
  - Added global scheduler storage and accessor.
- `fastvideo/entrypoints/openai/api_server.py`
  - Starts the scheduler during lifespan when `batching_mode=dynamic` and
    `batching_max_size>1`; stops it before generator shutdown.
- `fastvideo/entrypoints/openai/video_api.py`
  - `_run_generation()` submits to the scheduler when enabled; otherwise keeps
    the prior direct executor-thread path.
- `fastvideo/tests/entrypoints/test_openai_api.py`
  - Added scheduler grouping and incompatible fallback tests using a fake
    generator.

Stage 5 is complete.

Validation attempt:
- Modal app: `ap-2PJ4b5eKPnxR9HTEb9x3UM`
- Command:
  `pytest fastvideo/tests/entrypoints/test_openai_api.py -q && pre-commit run --files ...`
- Result:
  - Tests passed: `61 passed, 14 warnings`.
  - Pre-commit failed only on mypy for assigning to an `exc` variable outside
    an `except` block in `openai/batching.py`.
- Fix applied locally:
  - Renamed that local variable to `error`.

Validation clean rerun:
- Modal app: `ap-sgV5gRsGJeHE4g9a1Dswk3`
- Command:
  `pytest fastvideo/tests/entrypoints/test_openai_api.py -q && pre-commit run --files ...`
- Result:
  - Tests passed: `61 passed, 14 warnings`.
  - Pre-commit passed: yapf, ruff, codespell, mypy, filename spaces, and
    suggestion hooks.

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

GPU validation helper:
- `fastvideo/tests/batching/run_dynamic_batching_parity.py`
- Supports:
  - `--mode parity`: sequential `generate_video()` for each request vs one
    `generate_video_batch()` call in the same checkout; compares latent tensors.
  - `--mode sequential`: benchmark current sequential behavior.
  - `--mode dynamic`: benchmark dynamic batching behavior.
- Defaults use Wan2.1 T2V 1.3B, latent output, 256x256, 9 frames, 2 steps,
  batch size 2. Final parity/benchmark runs may override these if the model
  requires a larger valid shape.

Benchmark run, batch size 2:
- Modal app: `ap-V0fkFaplHrGYto9hFWvRmc`
- Command:
  `python fastvideo/tests/batching/run_dynamic_batching_parity.py --mode sequential --warmup-runs 1 --measurement-runs 3 --output-json /tmp/fastvideo_dynamic_batching/sequential.json && python fastvideo/tests/batching/run_dynamic_batching_parity.py --mode dynamic --warmup-runs 1 --measurement-runs 3 --output-json /tmp/fastvideo_dynamic_batching/dynamic.json`
- Workload:
  - Wan2.1 T2V 1.3B, one L40S, latent output, 256x256, 9 frames, 2 denoise
    steps, two prompts, `guidance_scale=1.0`.
- Results:
  - Sequential baseline times: `[2.1736583650, 2.1744417920, 2.1720341440]`
  - Sequential average: `2.1733781003s`; throughput `0.9202264437 req/s`
  - Dynamic times: `[2.1407064020, 2.1398537520, 2.1382482180]`
  - Dynamic average: `2.1396027907s`; throughput `0.9347529405 req/s`
  - Throughput improvement: about `1.6%`.

Benchmark run, batch size 4:
- Modal app: `ap-K5tj5fWQKByZ0Sl9wn6OGC`
- Command:
  `python fastvideo/tests/batching/run_dynamic_batching_parity.py --mode sequential --batch-size 4 --warmup-runs 1 --measurement-runs 3 ... && python fastvideo/tests/batching/run_dynamic_batching_parity.py --mode dynamic --batch-size 4 --warmup-runs 1 --measurement-runs 3 ...`
- Workload:
  - Wan2.1 T2V 1.3B, one L40S, latent output, 256x256, 9 frames, 2 denoise
    steps, four prompts, `guidance_scale=1.0`.
- Results:
  - Sequential baseline times: `[4.3420497740, 4.3434355840, 4.3416078150]`
  - Sequential average: `4.3423643910s`; throughput `0.9211571485 req/s`
  - Dynamic times: `[4.2617743810, 4.4826704910, 4.2678945680]`
  - Dynamic average: `4.3374464800s`; throughput `0.9222015807 req/s`
  - Throughput improvement: about `0.1%`.
  - Interpretation: with exact per-prompt text encoding and only two denoise
    steps, this small synthetic benchmark is dominated by text/launch overhead,
    so dynamic denoising has little room to help.

Benchmark run, batch size 2, 8 denoise steps:
- Modal app: `ap-yPsbXeIc6YbCG7NvAueN4t`
- Command:
  `python fastvideo/tests/batching/run_dynamic_batching_parity.py --mode sequential --num-inference-steps 8 --warmup-runs 1 --measurement-runs 2 ... && python fastvideo/tests/batching/run_dynamic_batching_parity.py --mode dynamic --num-inference-steps 8 --warmup-runs 1 --measurement-runs 2 ...`
- Workload:
  - Wan2.1 T2V 1.3B, one L40S, latent output, 256x256, 9 frames, 8 denoise
    steps, two prompts, `guidance_scale=1.0`.
- Results:
  - Sequential baseline times: `[2.7835521810, 2.5315261900]`
  - Sequential average: `2.6575391855s`; throughput `0.7525759210 req/s`
  - Dynamic times: `[2.4052083240, 2.3961168300]`
  - Dynamic average: `2.4006625770s`; throughput `0.8331033354 req/s`
  - Throughput improvement: about `10.7%`.

Final report:
- Created `.agents/exploration/multimodal-gen-batching-final-report.md`.
- Includes implementation summary, commit list, remote validation results,
  dynamic parity results, benchmark tables, and limitations/follow-up.

### Stage 6: GPU Parity And Benchmark Validation
Parity attempt:
- Modal app: `ap-UTIijf9LTsX3ua9Czvm5dq`
- Command:
  `python fastvideo/tests/batching/run_dynamic_batching_parity.py --mode parity --output-json /tmp/fastvideo_dynamic_batching/parity.json`
- Result:
  - Failed before tensor comparison.
  - Sequential `generate_video()` accepted the legacy request kwarg
    `embedded_cfg_scale`, but `generate_video_batch()` tried to apply all
    request kwargs directly to `SamplingParam.update()` and rejected
    `embedded_cfg_scale`.
- Fix applied locally:
  - `VideoGenerator.generate_video_batch()` now routes each request through
    `legacy_generate_call_to_request()`, `request_to_sampling_param()`, and
    `request_to_pipeline_overrides()`, matching `generate_video()`.
  - Identical pipeline overrides reuse one resolved `FastVideoArgs` object so
    compatible requests can still merge; different overrides remain separated
    by the existing object-identity compatibility guard.
  - Added a unit test covering `generate_video_batch()` with
    `embedded_cfg_scale`.

Focused validation for the compat fix:
- Modal app: `ap-gmob40FWTO5knEPA39s3bd`
- Command:
  `pytest fastvideo/tests/entrypoints/test_video_generator.py -q && pre-commit run --files fastvideo/entrypoints/video_generator.py fastvideo/tests/entrypoints/test_video_generator.py fastvideo/tests/batching/run_dynamic_batching_parity.py .agents/exploration/multimodal-gen-batching-port.md`
- Result:
  - Tests passed: `23 passed, 14 warnings`.
  - Pre-commit passed: yapf, ruff, codespell, mypy, filename spaces, and
    suggestion hooks.

Parity rerun:
- Modal app: `ap-C8JZ6i4R7mv6NOBTBDCRTc`
- Command:
  `python fastvideo/tests/batching/run_dynamic_batching_parity.py --mode parity --output-json /tmp/fastvideo_dynamic_batching/parity.json`
- Result:
  - Failed in the batched forward path before tensor comparison.
  - The Wan tokenizer received a prompt list with variable token lengths and
    `return_tensors="pt"` but no padding, causing Hugging Face tokenization to
    reject non-rectangular `input_ids`.
- Fix applied locally:
  - `TextEncodingStage.encode_text()` now adds tokenizer `padding=True` when
    encoding multiple processed prompts and no explicit padding mode is already
    configured.
  - Added a unit test to cover default padding insertion for prompt-list text
    encoding.

Focused validation attempt for padding fix:
- Modal app: `ap-MFeCHJF61EuAgsZro5cAb4`
- Command:
  `pytest fastvideo/tests/entrypoints/test_video_generator.py fastvideo/tests/stages/test_text_encoding.py -q && pre-commit run --files ...`
- Result:
  - Product tests mostly passed, but the new test had a misplaced assertion
    referencing `out2` outside its original test.
- Fix applied locally:
  - Moved the prompt/negative attention-mask assertions back into
    `test_forward_integration_cfg_off_and_on()` and left the padding test
    scoped to tokenizer kwargs.

Focused validation clean rerun:
- Modal app: `ap-IJIJjdLogSGkzeN4sCP46E`
- Command:
  `pytest fastvideo/tests/entrypoints/test_video_generator.py fastvideo/tests/stages/test_text_encoding.py -q && pre-commit run --files ...`
- Result:
  - Tests passed: `28 passed, 14 warnings`.
  - Pre-commit passed: yapf, ruff, codespell, mypy, filename spaces, and
    suggestion hooks.

Parity rerun after padding fix:
- Modal app: `ap-MMEOgUNlqYzeoYaD2blJVh`
- Command:
  `python fastvideo/tests/batching/run_dynamic_batching_parity.py --mode parity --output-json /tmp/fastvideo_dynamic_batching/parity.json`
- Result:
  - Batched forward completed successfully.
  - Tensor parity was not close enough:
    - request 0 max abs diff `0.0625`, mean abs diff `0.0048387293`
    - request 1 max abs diff `0.1708983183`, mean abs diff `0.0173878949`
    - aggregate max abs diff `0.1708983183`, mean abs diff `0.0111133121`
    - `torch.allclose(..., atol=1e-4, rtol=1e-4)` failed.
- Follow-up fix applied locally:
  - `TextEncodingStage.forward()` now preserves the existing single-prompt text
    encoding path for prompt-list batches by encoding each prompt separately
    and concatenating postprocessed embeddings/masks. This keeps denoising
    batched while removing tokenizer padding/sequence-length drift from the
    parity path.
  - Added a unit test proving prompt-list `forward()` uses one tokenizer call
    per prompt, while direct `encode_text(list)` still supports padded batched
    tokenization.

Focused validation for single-text-encode parity fix:
- Modal app: `ap-DIIEE6Wy0I728fqsc63C6s`
- Command:
  `pytest fastvideo/tests/entrypoints/test_video_generator.py fastvideo/tests/stages/test_text_encoding.py -q && pre-commit run --files ...`
- Result:
  - Tests passed: `29 passed, 14 warnings`.
  - Pre-commit passed: yapf, ruff, codespell, mypy, filename spaces, and
    suggestion hooks.

Parity rerun after single-text-encode fix:
- Modal app: `ap-fYT25LrzvFbUZv50JhxmWC`
- Command:
  `python fastvideo/tests/batching/run_dynamic_batching_parity.py --mode parity --output-json /tmp/fastvideo_dynamic_batching/parity.json`
- Result:
  - Batched forward completed successfully.
  - Tensor parity improved only slightly and is still not near bit-identical:
    - request 0 max abs diff `0.0380859375`, mean abs diff
      `0.0047932155`
    - request 1 max abs diff `0.1457520127`, mean abs diff
      `0.0196863897`
    - aggregate max abs diff `0.1457520127`, mean abs diff
      `0.0122398026`
    - `torch.allclose(..., atol=1e-4, rtol=1e-4)` failed.
- Interpretation:
  - Since the prompt-list path now reuses the same single-prompt text encoding
    calls, the remaining drift is likely from batched Wan denoising/model math
    rather than tokenization.
  - Keep this as a documented limitation in the final report unless a later
    exact-denoising mode is added.

## Mistakes / Dead Ends
- First GPU parity attempt found the `generate_video_batch()` legacy-compat
  gap described in Stage 6. This was a useful pre-parity functional bug, not a
  numerical mismatch.
- Second GPU parity attempt found that prompt-list tokenizer calls need padding
  for variable-length Wan prompts.
- Third GPU parity attempt completed but exposed non-negligible numerical drift.
  The next hypothesis is text-encoder sequence/padding drift; the local fix now
  preserves the single-prompt text-encoding path for merged requests.
- Fourth GPU parity attempt still showed non-negligible drift, so dynamic
  batched denoising is not near bit-identical to sequential denoising for this
  Wan latent test.

## Proposed Standardization
If this port lands cleanly, create a runtime batching SOP covering:
- compatibility-signature design,
- deterministic grouped latent generation,
- per-stage grouped execution contracts,
- Modal-only parity and benchmark commands.

## Hand-Off
Current branch: `multimodal-gen-batching`.

Current local implementation state:
- Stage 1 code is committed as `3d3157fb`; state commit is `ce758820`.
- Stage 2 code is committed as `fe5572f3`.
- Stage 2 state commit is `39cb1d43`; branch is pushed to origin through
  Stage 2.
- Stage 5 code is committed as `4c6a9dc2`.
- Stage 5 state commit is `1bea1dee`.
- GPU helper, parity hardening fixes, and benchmark state are committed as
  `9c6eb355`.
- Final report is committed as `2304837e`.
- Branch was pushed to origin through `2304837e`.
- Final changed-file validation passed on Modal app
  `ap-1mFqrE5eCwPkEKnffQcQou` from commit
  `2304837ed1bc0e1cd733d61f864d6cb1e7682b26`:
  `119 passed, 14 warnings`, and pre-commit passed.
- Post-report Wan T2V SSIM validation:
  - H100 attempt `ap-KaJr2loSTefvmj8ijYwWOK` generated videos but failed
    before comparison because H100 reference folders are missing.
  - L40S run `ap-iWP6PA1IyZbXHDKtIE1LQH` passed:
    `2 passed, 6 warnings`; mean SSIM `0.9786614696` for `FLASH_ATTN`
    and `0.9743387236` for `TORCH_SDPA`.
- `pre-commit run --all-files` was attempted on Modal app
  `ap-r20n8jCBwqQnh8I5Us1yTN`, but failed because yapf/ruff rewrote a
  large set of pre-existing repository files. This was not committed because
  it would introduce unrelated formatting churn.
- No code changes remain after the final validation report/state update.

Important constraints:
- Do not edit unrelated untracked files already present in the worktree.
- User pre-approved tool/command use, commits to current branch, and push to
  origin after implementation stages.
- Use Modal L40S jobs for tests; local machine should not be treated as a valid
  test environment.

Next step:
Push `multimodal-gen-batching` to origin if the final docs-only state update is
not already present there.
