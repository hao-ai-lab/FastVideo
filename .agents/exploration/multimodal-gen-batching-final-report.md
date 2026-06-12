# Multimodal Generation Batching Port Report

Date: 2026-05-30
Branch: `multimodal-gen-batching`

## Summary

This port adds an SGLang-style generation batching path to FastVideo for
compatible text-only generation requests. The implementation covers typed
batching config, request compatibility/admission rules, generator merge/split,
batch-safe standard pipeline stages, and an OpenAI server queue scheduler.

Initial scope is intentionally conservative: text-only compatible requests can
batch. Requests with image, video, audio, action, continuation, refine, or other
conditioning inputs are rejected by the compatibility signature and continue on
the sequential path.

## Commits

- `3d3157fb` - `[feat]: add generation batching primitives`
- `ce758820` - `[misc]: record batching stage 1 state`
- `fe5572f3` - `[feat]: add generator dynamic batching path`
- `39cb1d43` - `[misc]: record batching stage 2 state`
- `4c6a9dc2` - `[feat]: add OpenAI video batching scheduler`
- `1bea1dee` - `[misc]: record batching stage 5 state`
- `9c6eb355` - `[fix]: harden dynamic generation batching`
- `2304837e` - `[docs]: record multimodal batching validation report`

## Implementation

Added batching configuration:

- `fastvideo/api/schema.py`: `BatchingConfig` under `EngineConfig`.
- `fastvideo/fastvideo_args.py`: `batching_mode`, `batching_max_size`,
  `batching_delay_ms`, `batching_config`, and `enable_batching_metrics`.
- `fastvideo/api/compat.py`: legacy kwarg translation into typed batching
  config and back into `FastVideoArgs`.

Added batching primitives:

- `fastvideo/batching/admission.py`: max batch size and JSON-rule admission.
- `fastvideo/batching/signature.py`: request compatibility signatures and
  unsupported multimodal/conditioning fields.
- `PipelineConfig.estimate_request_cost()`: default hook for future admission
  cost rules.

Added generator batching:

- `VideoGenerator.generate_video_batch()` accepts legacy request kwargs.
- Compatible work items are merged into one `ForwardBatch` with prompt lists and
  per-request seeds, then split back into per-request result dictionaries.
- Incompatible adjacent requests fall back to existing single-request execution.
- Prompt-file generation uses dynamic batching when enabled.
- `generate_video_batch()` now routes each request through the same compatibility
  adapter as `generate_video()`, including pipeline overrides such as
  `embedded_cfg_scale`.

Made standard stages batch-aware:

- `InputValidationStage` preserves explicit per-request seeds and fans out
  prompt-list seeds.
- `TextEncodingStage` repeats a single negative prompt across prompt lists.
- `TextEncodingStage.encode_text()` adds tokenizer padding for direct prompt-list
  tokenization.
- `TextEncodingStage.forward()` preserves the existing single-prompt text
  encoding path for merged prompt lists, then concatenates postprocessed
  embeddings/masks. This reduces text-encoder-induced parity drift.
- `DenoisingStage` no longer asserts batch size 1 on the standard path.

Added OpenAI server batching:

- `fastvideo/entrypoints/openai/batching.py`: async FIFO `VideoBatchScheduler`.
- `fastvideo/entrypoints/openai/state.py`: scheduler lifecycle state.
- `fastvideo/entrypoints/openai/api_server.py`: scheduler start/stop in
  lifespan.
- `fastvideo/entrypoints/openai/video_api.py`: routes requests through the
  scheduler when dynamic batching is enabled.

Added validation helper:

- `fastvideo/tests/batching/run_dynamic_batching_parity.py`
- Modes:
  - `parity`: sequential `generate_video()` vs dynamic `generate_video_batch()`
    latent comparison.
  - `sequential`: disabled-batching benchmark baseline.
  - `dynamic`: dynamic batching benchmark.

## Remote Validation

All validation was run through
`fastvideo/tests/modal/launch_l40s_job.py` on Modal L40S.

Focused tests and hooks:

| Stage | Modal app | Command summary | Result |
| --- | --- | --- | --- |
| Config/admission/signature | `ap-q9SRzkYvwXzoPA5oWoaM3i` | `pytest fastvideo/tests/batching fastvideo/tests/api/test_compat_translation.py -q && pre-commit run --files ...` | `27 passed`, pre-commit passed |
| Generator batching | `ap-0LrnrxcMw4Ni3M9YJ6gtZv` | batching, API compat, generator, input validation tests plus pre-commit | `51 passed`, pre-commit passed |
| OpenAI scheduler | `ap-sgV5gRsGJeHE4g9a1Dswk3` | `pytest fastvideo/tests/entrypoints/test_openai_api.py -q && pre-commit run --files ...` | `61 passed`, pre-commit passed |
| Batch compat fix | `ap-gmob40FWTO5knEPA39s3bd` | `pytest fastvideo/tests/entrypoints/test_video_generator.py -q && pre-commit run --files ...` | `23 passed`, pre-commit passed |
| Text padding fix | `ap-IJIJjdLogSGkzeN4sCP46E` | entrypoint and text encoding tests plus pre-commit | `28 passed`, pre-commit passed |
| Single-text-encode fix | `ap-DIIEE6Wy0I728fqsc63C6s` | entrypoint and text encoding tests plus pre-commit | `29 passed`, pre-commit passed |
| Final changed-file suite | `ap-1mFqrE5eCwPkEKnffQcQou` | batching, generator, text encoding, OpenAI API, compat, and input-validation tests plus pre-commit | `119 passed`, pre-commit passed |

Post-report validation:

| Check | Modal app | Command summary | Result |
| --- | --- | --- | --- |
| Wan T2V SSIM on H100 | `ap-KaJr2loSTefvmj8ijYwWOK` | `FASTVIDEO_SSIM_MODEL_ID=Wan2.1-T2V-1.3B-Diffusers pytest fastvideo/tests/ssim/test_wan_t2v_similarity.py -vs` on `H100:2` | Generated both videos, but failed before SSIM comparison because `H100_reference_videos` are missing for both `FLASH_ATTN` and `TORCH_SDPA` |
| Wan T2V SSIM on L40S | `ap-iWP6PA1IyZbXHDKtIE1LQH` | same targeted Wan T2V SSIM command on `L40S:2`, `--install-extra none` | `2 passed`, `6 warnings`; mean SSIM `0.9786614696` for `FLASH_ATTN`, `0.9743387236` for `TORCH_SDPA` |
| Full pre-commit attempt | `ap-r20n8jCBwqQnh8I5Us1yTN` | `pre-commit run --all-files` on `L40S:1` | Failed because yapf/ruff rewrote a large set of pre-existing repository files; not taken as PR-local evidence |

## Parity

Parity workload:

- Model: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- GPU: one L40S
- Output: latent tensors, no video save
- Shape: `256x256`, 9 frames
- Steps: 2
- Batch size: 2
- Guidance: `guidance_scale=1.0`, `embedded_cfg_scale=6.0`

Final dynamic parity run:

- Modal app: `ap-fYT25LrzvFbUZv50JhxmWC`
- Sequential time: `2.8672916360s`
- Dynamic time: `2.1450136500s`
- Speedup in this run: `1.3367x`

Tensor comparison:

| Request | Shape | Max abs diff | Mean abs diff | allclose 1e-4 |
| --- | --- | ---: | ---: | --- |
| 0 | `[1, 16, 3, 32, 32]` | `0.0380859375` | `0.0047932155` | false |
| 1 | `[1, 16, 3, 32, 32]` | `0.1457520127` | `0.0196863897` | false |
| Aggregate | - | `0.1457520127` | `0.0122398026` | false |

Conclusion: dynamic batched denoising is not near bit-identical to sequential
denoising for this Wan latent test. The text-encoding path was adjusted to match
single-request encoding, so the remaining difference is likely from batched
Wan transformer/denoising math. The existing disabled/sequential path remains
available and is the default unless `batching_mode=dynamic` is explicitly set.

## Benchmarks

Benchmark environment:

- Modal L40S
- Wan2.1 T2V 1.3B
- Latent output, no video save
- `256x256`, 9 frames
- `guidance_scale=1.0`
- One warmup per mode

| Workload | Mode | Runs | Avg seconds | Throughput req/s |
| --- | --- | --- | ---: | ---: |
| batch 2, 2 steps | sequential | 3 | `2.1733781003` | `0.9202264437` |
| batch 2, 2 steps | dynamic | 3 | `2.1396027907` | `0.9347529405` |
| batch 4, 2 steps | sequential | 3 | `4.3423643910` | `0.9211571485` |
| batch 4, 2 steps | dynamic | 3 | `4.3374464800` | `0.9222015807` |
| batch 2, 8 steps | sequential | 2 | `2.6575391855` | `0.7525759210` |
| batch 2, 8 steps | dynamic | 2 | `2.4006625770` | `0.8331033354` |

Observed throughput changes:

- Batch 2, 2 steps: about `+1.6%`
- Batch 4, 2 steps: about `+0.1%`
- Batch 2, 8 steps: about `+10.7%`

Interpretation: with the exact per-prompt text-encoding safeguard, very short
2-step workloads are dominated by text encoding and launch overhead. Dynamic
batching shows clearer benefit once denoising compute is a larger fraction of
the request.

## Limitations And Follow-Up

- Dynamic batched Wan denoising is not near bit-identical to sequential Wan
  denoising in the current implementation.
- The implementation is text-only for dynamic batching. Multimodal and
  conditioning-heavy requests are intentionally routed to sequential execution.
- The benchmarks used small latent-output workloads to keep Modal iteration
  time reasonable. Larger production shapes and longer schedules should be
  benchmarked before enabling dynamic batching by default.
- A future exact-parity mode could preserve the scheduler/admission queue but
  execute denoising per request. That would satisfy strict numerical parity but
  would give up most batching speedup.
