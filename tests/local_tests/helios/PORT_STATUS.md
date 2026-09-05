# Helios Port Status

## Summary

- model_family: `helios`
- workload_types: `T2V`
- official_ref: Diffusers `0.39.0` `HeliosPyramidPipeline`; `PKU-YuanGroup/Helios@8f2a2faa`
- official_ref_dir: `../Helios`
- hf_weights_path: `BestWishYsh/Helios-Distilled`
- local_weights_dir: `official_weights/helios`
- source_layout: `diffusers`
- local_tests_readme: `tests/local_tests/helios/README.md`

## Current Phase

- phase: `final_verification`
- status: `complete`
- owner: `orchestrator`
- last_updated: `2026-08-26`

## Component Matrix

| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transformer | dit | ported | Diffusers `HeliosTransformer3DModel` | Exact HF transformer config; short/mid/long histories and frame indices | native Helios DiT/config | complete | identity 1101/1101 | non_skip_pass | none |
| scheduler | scheduler | ported | Diffusers `HeliosDMDScheduler` | Three stages, dynamic shift, stage-local re-noising | native Helios DMD scheduler | complete | not_needed | non_skip_pass, bit-exact | none |
| vae | vae | reused | Diffusers `AutoencoderKLWan` | Exact Helios VAE config/weights | native Wan VAE | complete | not_needed | non_skip_pass, decode exact | none |
| text_encoder | encoder | reused | Transformers `UMT5EncoderModel` | UMT5-XXL, max length 512 | native UMT5 | complete | not_needed | non_skip_pass FP32/BF16 | none |
| tokenizer | encoder | passthrough | `T5TokenizerFast` | Exact checkpoint assets | production tokenizer loader | complete | not_needed | exact IDs/masks | none |
| pipeline | pipeline | ported | Diffusers `HeliosPyramidPipeline` | 9-frame AR chunks, `[16,2,1]` history, 3 spatial stages | `HeliosPyramidPipeline` | complete | not_needed | non_skip_pass | T2V Distilled only |

## Conversion State

- conversion_script: `not_needed`
- converted_weights_dir: `not_needed`
- source_layout: `diffusers`
- strict_load_status: transformer 1101/1101; VAE strict; UMT5 all parameters loaded
- passthrough_components: tokenizer assets
- retry_history: none

## Parity Commands

| Scope | Command | Last Result | Notes |
| --- | --- | --- | --- |
| transformer | `pytest tests/local_tests/transformers/test_helios_transformer_parity.py -v -s` | current PR component evidence: non-skip PASS | strict load, tiny/full, SP=2, FA2/SDPA |
| scheduler | `pytest tests/local_tests/schedulers/test_helios_dmd_scheduler_parity.py -v -s` | 10 passed | registry plus bit-exact schedules/steps |
| VAE | `pytest tests/local_tests/vaes/test_helios_vae_parity.py -v -s` | 2 passed | decode diff max/mean 0/0 |
| UMT5/tokenizer | `pytest tests/local_tests/encoders/test_helios_umt5_parity.py -v -s` | 5 passed | exact tokenizer; FP32/BF16 encoder parity |
| pipeline math/stage/smoke | three `test_helios_pipeline_*` files | 29 passed | includes CUDA block noise and CPU-output regression |
| pipeline parity | `pytest tests/local_tests/pipelines/test_helios_pipeline_parity.py -v -s` | 1 passed in 82.85 s | cosine 0.979703, MAE 0.170017, RMSE 0.246508, drift 0.376% |
| typed API regression | smoke + parser/compat/config tests | 42 passed | registry, preset, class resolution, typed/CLI fields |
| typed example | `python examples/inference/basic/basic_helios_distilled_t2v.py` | PASS, generation 29.43 s | H.264 640×384, 33 frames, full decode |
| quality/container | `HELIOS_QUALITY_CANDIDATE=... pytest test_helios_quality_regression.py` | 1 passed, 1 skipped | reference comparison deferred pending upload approval |

## Open Questions

| ID | Question | Owner | Needed By Phase | Status | Resolution |
| --- | --- | --- | --- | --- | --- |
| Q001 | Is `transformer_ode` required for Distilled T2V? | orchestrator | Phase 1 | resolved | No model index or official T2V call loads it; excluded. |
| Q002 | Can the native Wan VAE be reused? | component:vae | Phase 3 | resolved | Exact config/weight strict load and decode parity are bit-exact. |
| Q003 | Can native UMT5/tokenizer be reused? | component:encoder | Phase 3 | resolved | Token IDs/masks exact; FP32 and BF16 output parity pass. |
| Q004 | Where do Helios DMD runtime fields live? | component:scheduler | Phase 1 | resolved | Scheduler owns sigma/timestep math; public per-call knobs live in SamplingParam/preset. |

## Issues And Blockers

| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I001 | prep | environment | medium | No project venv initially. | system Python lacked ML deps | prep | resolved | Python 3.12 project venv installed. |
| I002 | prep | repository | low | Early branch was behind main. | rev-list showed upstream commits | prep | resolved | Final integration branch rebased on `a159b63c`. |
| I003 | prep | weights | low | Snapshot includes unused 58 GB ODE transformer. | model indexes omit `transformer_ode` | orchestrator | resolved | Pinned inference assets downloaded without ODE directory. |
| I004 | parity | tokenizer | low | Direct loader import had a repository-root import cycle in the older workspace. | collection traceback | orchestrator | resolved_for_test | Production loader runs in a clean subprocess. |
| I005 | official_reference | CUDA accounting | low | Peak reset initially used an invalid device after offload hooks. | runtime error before inference | orchestrator | resolved | Set current device before hooks; parity completes. |
| I006 | transformer | RoPE buffers | medium | Meta construction left non-persistent RoPE buffers on meta. | meta/cuda einsum mismatch | component:transformer | resolved | Loader materializes exact buffers. |
| I007 | encoder | BF16 drift | low | Fused-QKV and three-GEMM paths differ slightly. | max 0.0234375, mean 0.00160135 | component:encoder | resolved | FP32 proves math; separate bounded BF16 gate passes. |
| I008 | pipeline | history geometry | high | Coarse-stage history RoPE used current-grid positions. | 294 history tokens versus 72 positions | component:transformer | resolved | Short-grid positions are center-downsampled for mid/long histories. |
| I009 | pipeline | guidance cross-attention | medium | History and current queries were combined before masking. | activation trace diverged | component:transformer | resolved | Current/history split occurs before cross-attention. |
| I010 | tests | repository package root | low | A root `__init__.py` makes invalid worktree basenames fail mypy. | `helios-pr... is not a valid Python package name` | orchestrator | resolved | Worktree basename changed to `helios_pr1670_full_pipeline`. |
| I011 | registry | variant safety | high | Broad Helios detection would route Base/Mid to Distilled. | negative metadata probes | orchestrator | resolved | Require pipeline class, `is_distilled=true`, and Helios DMD scheduler. |
| I012 | quality | tiny smoke | medium | 128×192 is not a visual-quality target. | poor-detail local smoke | orchestrator | resolved_as_scope | Public example and integrity gate use 384×640. |
| I013 | distributed | nightly coverage | medium | SP/FSDP/repeated-run coverage is not a committed CI lane. | earlier local SP/FSDP smoke passed | orchestrator | open_nightly | Keep as follow-up; no absent runner is referenced by this PR. |
| I014 | production_validation | multiprocessing output | high | Returning GPU output through CUDA IPC OOMed while the 14.31B DiT remained resident. | v1 completed stages then failed in `_new_shared_cuda` | pipeline | resolved | Decode/latent outputs move to CPU inside worker; regression is GREEN and typed example v2 passes. |
| I015 | final_verification | upstream main | high | Integration base `6388db81` had eight unrelated unit-lane failures. | 909 passed and 8 failed before upstream CI/schema fixes. | upstream | resolved | Rebased through `b2062556` onto `a159b63c`; the current shared unit script passes all 1047 tests. |

## Escape Hatches

No escape hatch is open. CI SSIM reference upload remains deliberately
unauthorized and is recorded as a quality-regression deferral, not a blocker to
local pipeline parity.

## Decisions

| Date | Decision | Rationale | Impact |
| --- | --- | --- | --- |
| 2026-07-11 | First public scope is Helios-Distilled T2V. | Keep a reviewable, verifiable variant. | Base/Mid, ODE, training and conditioned modes remain out of scope. |
| 2026-07-11 | Use Diffusers 0.39.0 as executable parity reference. | It matches the published Diffusers-layout checkpoint. | No conflicting Helios research requirements are installed. |
| 2026-07-11 | Preserve 1101 transformer keys directly. | Official and native key surfaces match. | No conversion script. |
| 2026-07-11 | Reuse Wan VAE and UMT5 only after exact-asset parity. | Architecture resemblance is insufficient. | Both reused components have non-skip evidence. |
| 2026-08-25 | Use a dedicated Helios pyramid stage inside `ComposedPipelineBase`. | Generic denoising cannot express AR history, three spatial levels and stage-local DMD. | One FastVideo architecture, model-specific stages only where required. |
| 2026-08-25 | Keep zero-init call fields but do not apply zero-star math for the pinned Distilled checkpoint. | Its model index declares `is_cfg_zero_star=false`; Diffusers also takes the standard CFG branch. | Signature stays compatible without claiming an inactive feature changes output. |
| 2026-08-25 | Move worker output to CPU before multiprocessing return. | Prevent CUDA IPC allocation after high-memory inference. | Public typed example is robust on 48 GB cards. |

## Handoff Notes

- Required components and pipeline parity are green on the integration branch.
- `quality_regression=deferred_with_reason`: local real-video integrity passes;
  publishing a CI reference needs separate approval.
- Repository-wide pre-commit is green. The current shared unit script passes
  all 1047 collected tests; `I015` remains resolved on `a159b63c`.
- No weights, generated media, reference clone, private report, token, push, or
  PR mutation is part of this state file.
