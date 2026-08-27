# Cosmos Predict2.5 Distilled Port Status

## Summary
- model_family: cosmos25_distilled
- workload_types: T2W (released support); V2W/rolling experimental and deferred
- official_ref: NVIDIA/Cosmos-Predict2.5@a2c298b0a3df3778b973fe65e9e58877b292d8a7
- official_ref_dir: `${COSMOS25_OFFICIAL_REF_DIR:-$PWD/cosmos-predict2.5}`
- hf_weights_path: `nvidia/Cosmos-Predict2.5-2B/base/distilled`
- local_weights_dir: not created
- source_layout: official monolithic checkpoint; FastVideo conversion/loading path pending
- local_tests_readme: `tests/local_tests/cosmos25/README.md`

## Current Phase
- phase: component parity
- status: in_progress
- owner: parity
- last_updated: 2026-08-27

## Component Matrix
| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| TrigFlow sampler | scheduler | port | `modules/denoiser_scaling.py`; `distill/models/video2world_model_distill_dmd2.py` | `generate_samples_from_batch` | `Cosmos25DistilledScheduler` | complete | n/a | non-skip pass | none |
| student DiT | transformer | reuse Cosmos25 architecture with distilled weights | `MinimalV1LVGDiT` through distillation model | `get_x0_fn_from_batch` / `denoise_edm` | `Cosmos25Transformer3DModel` | existing | pending | pending real-weight forward | I001 |
| Reason1 encoder | text encoder | reuse | Predict2.5 Video2World config | distilled inference CLI | existing Cosmos25 encoder | existing | pending packaged layout | pending production-loader parity | I002 |
| tokenizer VAE | VAE | reuse | Predict2.5 tokenizer | distilled inference CLI | existing Cosmos25 VAE | existing | pending packaged layout | pending production-loader parity | I002 |
| T2W pipeline | pipeline | extend after components pass | `generate_samples_from_batch` | distilled inference CLI | Cosmos2_5 staged pipeline | not started | blocked by component gates | not started | I003 |

## Conversion State
- conversion_script: `scripts/checkpoint_conversion/cosmos25_distilled_to_diffusers.py`
- converted_weights_dir: not created
- source_layout: official `base/distilled` checkpoint
- strict_load_status: not run
- passthrough_components: Reason1 encoder and tokenizer VAE are expected to reuse the existing Cosmos25 layout
- retry_history: synthetic direct/nested checkpoint extraction and output-layout contracts pass; released checkpoint not run

## Parity Commands
| Scope | Command | Last Result | Notes |
|---|---|---|---|
| scheduler unit | `pytest fastvideo/tests/schedulers/test_cosmos25_distilled_scheduler.py -q` | 7 passed | CPU-only; includes registry resolution; 2026-08-27 |
| official scheduler | `COSMOS25_OFFICIAL_REF_DIR=/path/to/Cosmos-Predict2.5 pytest tests/local_tests/cosmos25/test_cosmos25_distilled_scheduler_parity.py -v -s` | 2 passed, non-skip | CPU-only; pinned source; 2026-08-27 |
| conversion contracts | `pytest tests/local_tests/cosmos25/test_cosmos25_distilled_conversion.py -q` | 7 passed | Synthetic checkpoints/layout; 2026-08-27 |
| student DiT | not yet created | not run | requires CUDA and released weights |
| pipeline | not yet created | not run | forbidden until component parity passes |

## Open Questions
| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | What packaged Diffusers-style model ID should carry the distilled scheduler and converted student weights? | conversion | conversion | open | pending |
| Q002 | Does experimental distilled V2W retain acceptable quality after official T2W parity? | pipeline | post-parity experiment | open | intentionally outside initial support claim |

## Issues And Blockers
| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|
| I001 | parity | student DiT | high | No non-skip real-weight distilled forward comparison yet | Existing Spark runs used distilled weights with the base UniPC inference path | parity | open | pending CUDA run |
| I002 | conversion | packaged model | high | Released official checkpoint is not yet isolated in a FastVideo-loadable component layout | No `local_weights_dir` or strict-load record | conversion | open | pending |
| I003 | pipeline | T2W | high | Pipeline wiring is gated on component parity | add-model pipeline contract | pipeline | open | pending I001 and I002 |

## Escape Hatches
| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
|---|---|---|---|---|---|---|

## Decisions
| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-08-27 | Implement a Cosmos-specific sampler instead of reusing RCM | RCM uses different times and fresh per-step noise | Preserves TurboDiffusion behavior and official Cosmos equations |
| 2026-08-27 | Support the released distilled checkpoint as T2W first | NVIDIA documents the released distilled checkpoint for T2W | No V2W/rolling claim before experimental validation |
| 2026-08-27 | Keep the existing full-step Cosmos25 path unchanged | Distilled and post-trained checkpoints require different inference semantics | Avoids regression for current users |
| 2026-08-27 | Defer pipeline wiring until real-weight component parity | Required by the repository add-model workflow | Next GPU task is DiT parity, not generation |
| 2026-08-27 | Preserve native `net.*` student keys during conversion | Existing Cosmos25 loader owns the authoritative mapping | Converter only isolates student tensors and emits package metadata |

## Handoff Notes
- CPU scheduler unit and pinned-reference parity tests pass locally without skips.
- Next implementation is isolated distilled-weight conversion/loading plus a real student DiT parity test.
- Do not use the prior FlowUniPC/Karras Spark run as distilled parity evidence.
