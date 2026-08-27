# Cosmos Predict2.5 Distilled Port Status

## Summary
- model_family: cosmos25_distilled
- workload_types: T2W (released support); V2W/rolling experimental and deferred
- official_ref: NVIDIA/Cosmos-Predict2.5@a2c298b0a3df3778b973fe65e9e58877b292d8a7
- official_ref_dir: `${COSMOS25_OFFICIAL_REF_DIR:-$PWD/cosmos-predict2.5}`
- hf_weights_path: `nvidia/Cosmos-Predict2.5-2B/base/distilled`
- local_weights_dir: `~/models/Cosmos-Predict2.5-2B-Distilled-TrigFlow-FastVideo` (Spark validation host)
- source_layout: official monolithic checkpoint converted to a clean FastVideo-loadable package
- local_tests_readme: `tests/local_tests/cosmos25/README.md`

## Current Phase
- phase: initial T2W support
- status: complete
- owner: pipeline
- last_updated: 2026-08-27

## Component Matrix
| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| TrigFlow sampler | scheduler | port | `modules/denoiser_scaling.py`; `distill/models/video2world_model_distill_dmd2.py` | `generate_samples_from_batch` | `Cosmos25DistilledScheduler` | complete | n/a | non-skip pass | none |
| student DiT | transformer | reuse Cosmos25 architecture with distilled weights | `MinimalV1LVGDiT` through distillation model | `get_x0_fn_from_batch` / `denoise_edm` | `Cosmos25Transformer3DModel` | existing | complete | non-skip BF16 pass | none |
| Reason1 encoder | text encoder | reuse | Predict2.5 Video2World config | distilled inference CLI | existing Cosmos25 encoder | existing | packaged passthrough | production loader pass | none |
| tokenizer VAE | VAE | reuse | Predict2.5 tokenizer | distilled inference CLI | existing Cosmos25 VAE | existing | packaged passthrough | production loader pass | none |
| T2W pipeline | pipeline | isolated scheduler-selected route | `generate_samples_from_batch` | distilled inference CLI | Cosmos2_5 staged pipeline | complete | complete | full-resolution video and eye gate pass | none |

## Conversion State
- conversion_script: `scripts/checkpoint_conversion/cosmos25_distilled_to_diffusers.py`
- converted_weights_dir: `~/models/Cosmos-Predict2.5-2B-Distilled-TrigFlow-FastVideo`
- source_layout: official `base/distilled` checkpoint
- strict_load_status: pass; 685 student tensors, no training counters, production FastVideo loader pass
- passthrough_components: Reason1 encoder and tokenizer VAE are expected to reuse the existing Cosmos25 layout
- retry_history: synthetic contracts pass; released 3.9 GB checkpoint converted in 22.5 s to a 20 GB package

## Parity Commands
| Scope | Command | Last Result | Notes |
|---|---|---|---|
| scheduler unit | `pytest fastvideo/tests/schedulers/test_cosmos25_distilled_scheduler.py -q` | 7 passed | CPU-only; includes registry resolution; 2026-08-27 |
| official scheduler | `COSMOS25_OFFICIAL_REF_DIR=/path/to/Cosmos-Predict2.5 pytest tests/local_tests/cosmos25/test_cosmos25_distilled_scheduler_parity.py -v -s` | 2 passed, non-skip | CPU-only; pinned source; 2026-08-27 |
| conversion contracts | `pytest tests/local_tests/cosmos25/test_cosmos25_distilled_conversion.py -q` | 7 passed | Synthetic checkpoints/layout; 2026-08-27 |
| student DiT | `COSMOS25_OFFICIAL_REF_DIR=/path/to/Cosmos-Predict2.5 COSMOS25_DISTILLED_CHECKPOINT=/path/to/distilled.pt pytest tests/local_tests/cosmos25/test_cosmos25_distilled_transformer_parity.py -v -s` | passed, non-skip | Spark BF16: first-block relative mean 0.000655; final relative mean 0.038397; 2026-08-27 |
| pipeline contracts | `pytest tests/local_tests/cosmos25/test_cosmos25_distilled_pipeline.py -q` | 9 passed | Spark; CPU-only sampler/stage isolation checks |
| pipeline smoke | `python examples/inference/basic/basic_cosmos2_5_distilled_t2w.py --model /path/to/converted-model --steps 1 --frames 9 --height 256 --width 448` | passed | 2.19 s end-to-end after load; 2026-08-27 |
| full T2W quality | `python examples/inference/basic/basic_cosmos2_5_distilled_t2w.py --model /path/to/converted-model` | passed + eye gate | 704x1280x77, 4 steps; 143.53 s end-to-end after load; visually coherent |
| DreamVerse frames | example command with `--return-frames` | passed | 9 RGB frames; first shape `(256, 448, 3)`; 2026-08-27 |

## Open Questions
| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | What packaged Diffusers-style model ID should carry the distilled scheduler and converted student weights? | conversion | conversion | open | pending |
| Q002 | Does experimental distilled V2W retain acceptable quality after official T2W parity? | pipeline | post-parity experiment | open | intentionally outside initial support claim |

## Issues And Blockers
| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|
| I001 | parity | student DiT | high | No non-skip real-weight distilled forward comparison yet | Spark official-vs-FastVideo BF16 comparison | parity | closed | passed at final relative mean 0.038397 |
| I002 | conversion | packaged model | high | Released official checkpoint is not yet isolated in a FastVideo-loadable component layout | Converted package and strict production load | conversion | closed | 685 clean student tensors; load pass |
| I003 | pipeline | T2W | high | End-to-end distilled generation is not yet validated | small and full-resolution Spark runs plus visual inspection | pipeline | closed | full T2W quality gate passed |

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
| 2026-08-27 | Accept calibrated BF16 DiT parity | Preprocess is exact, first-block drift is 0.000655 relative, and drift grows smoothly to 0.038397 final relative | Clears the component gate without claiming bitwise equality |
| 2026-08-27 | Select distilled stages from the packaged scheduler class | The package already carries authoritative inference semantics | Existing full Cosmos2.5 packages remain on their unchanged path |
| 2026-08-27 | Accept the full-resolution T2W quality gate | The four-step 704x1280x77 run completed without runtime faults and passed visual inspection | Clears basic FastVideo T2W support; does not claim continuation or real-time latency |
| 2026-08-27 | Accept the decoded-frame return contract | The small Spark run returned 9 RGB frames with shape `(256, 448, 3)` without writing an MP4 | Clears the downstream frame-consumer contract without claiming DreamVerse integration |

## Handoff Notes
- CPU scheduler unit and pinned-reference parity tests pass locally without skips.
- Released checkpoint conversion, production strict load, and official-vs-FastVideo DiT parity pass on Spark.
- Small and full-resolution T2W generation pass on Spark; the full video passed visual inspection.
- The `save_video=False`, `return_frames=True` result contract passes on Spark.
- A public converted package ID remains open; until then, use the documented local conversion flow.
- Do not use the prior FlowUniPC/Karras Spark run as distilled parity evidence.
