# Cosmos Predict2.5 Distilled + DFD Port Status

## Summary
- model_family: cosmos25_distilled
- workload_types: T2W bootstrap plus one-frame-conditioned DFD V2W continuation
- official_ref: NVIDIA/Cosmos-Predict2.5@a2c298b0a3df3778b973fe65e9e58877b292d8a7; csy2077/data-forcing-distillation@de6416cac1e06562d29aaf96a13bb0ab99099cdf
- official_ref_dir: `${COSMOS25_OFFICIAL_REF_DIR:-$PWD/cosmos-predict2.5}`; `${COSMOS25_DFD_REF_DIR:-$PWD/DFDReference}`
- hf_weights_path: `nvidia/Cosmos-Predict2.5-2B/base/distilled`; `csusupergear/cosmos_i2v_checkpoints/cosmos_dfd_checkpoints/0000040.net_model.zip`
- local_weights_dir: `~/models/Cosmos-Predict2.5-2B-Distilled-TrigFlow-FastVideo` (Spark validation host)
- source_layout: official monolithic T2W checkpoint plus custom PyTorch DCP DFD checkpoint; both require conversion
- local_tests_readme: `tests/local_tests/cosmos25/README.md`

## Current Phase
- phase: DFD Phase 3 native pipeline integration
- status: in_progress
- owner: orchestrator
- last_updated: 2026-09-11

## Component Matrix
| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| TrigFlow sampler | scheduler | port | `modules/denoiser_scaling.py`; `distill/models/video2world_model_distill_dmd2.py` | `generate_samples_from_batch` | `Cosmos25DistilledScheduler` | complete | n/a | non-skip pass | none |
| student DiT | transformer | reuse Cosmos25 architecture with distilled weights | `MinimalV1LVGDiT` through distillation model | `get_x0_fn_from_batch` / `denoise_edm` | `Cosmos25Transformer3DModel` | existing | complete | non-skip BF16 pass | none |
| Reason1 encoder | text encoder | reuse | Predict2.5 Video2World config | distilled inference CLI | existing Cosmos25 encoder | existing | packaged passthrough | production loader pass | none |
| tokenizer VAE | VAE | reuse | Predict2.5 tokenizer | distilled inference CLI | existing Cosmos25 VAE | existing | packaged passthrough | production loader pass | none |
| T2W pipeline | pipeline | isolated scheduler-selected route | `generate_samples_from_batch` | distilled inference CLI | Cosmos2_5 staged pipeline | complete | complete | full-resolution video and eye gate pass | none |
| DFD ODE sampler | scheduler | port | `fastgen/methods/model.py::_student_sample_loop`; `fastgen/networks/noise_schedule.py::RFNoiseSchedule` | `config_dmd2.py` fixed `t_list` | `Cosmos25DFDScheduler` | complete | n/a | non-skip pass | none |
| DFD student DiT | transformer | reuse Cosmos25 architecture with DFD weights and FPS-modulated RoPE | `fastgen/networks/cosmos_predict2/network.py::CosmosPredict2` | `config_dmd2_v2w.py` | `Cosmos25Transformer3DModel` | reuse complete | converter complete | real-weight BF16 pass | converted-package strict load pending |
| DFD VAE | VAE | reuse | `CosmosPredict2.init_vae` / `WanVideoEncoder` | `prepare_i2v_condition` | existing Cosmos25 VAE | complete | packaged passthrough | isolated contract authored | encode only the single conditioning frame |
| DFD Reason1 encoder | text encoder | reuse | `CosmosPredict2.init_text_encoder` | VBench I2V inference | existing Cosmos25 encoder | complete | packaged passthrough | component production pass | none |
| DFD V2W pipeline | pipeline | scheduler-selected one-frame route | `video_model_inference_vbench.py` | four-step student sampling | Cosmos2_5 staged pipeline | complete | complete | four-step parity pending non-skip run | first decoded frame repeats conditioning frame by design |

## Conversion State
- conversion_script: `scripts/checkpoint_conversion/cosmos25_distilled_to_diffusers.py`
- converted_weights_dir: `~/models/Cosmos-Predict2.5-2B-Distilled-TrigFlow-FastVideo`
- source_layout: official `base/distilled` checkpoint
- strict_load_status: pass; 685 student tensors, no training counters, production FastVideo loader pass
- passthrough_components: Reason1 encoder and tokenizer VAE are expected to reuse the existing Cosmos25 layout
- retry_history: synthetic contracts pass; released 3.9 GB checkpoint converted in 22.5 s to a 20 GB package

DFD conversion target:

- conversion_script: `scripts/checkpoint_conversion/cosmos25_dfd_to_diffusers.py`
- source_layout: extracted PyTorch DCP directory (`0000040.net_model`)
- strict_load_status: not run
- passthrough_components: Cosmos25 VAE, Reason1 encoder, tokenizer, and safety checker from an existing FastVideo package
- retry_history: synthetic DCP contract authored; local execution environment lacks the repository runtime dependencies

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
| DFD scheduler parity | `COSMOS25_DFD_REF_DIR=$PWD/DFDReference pytest tests/local_tests/cosmos25/test_cosmos25_dfd_scheduler_parity.py -v -s` | 1 passed, non-skip | Exact fixed timesteps, initial-noise scaling, and four RF transitions; 2026-09-10 |
| DFD conversion contracts | `pytest tests/local_tests/cosmos25/test_cosmos25_dfd_conversion.py -q` | 4 passed | Synthetic DCP-normalized state dictionaries and package metadata; 2026-09-10 |
| DFD student DiT | `COSMOS25_DFD_REF_DIR=/path/to/data-forcing-distillation COSMOS25_DFD_CHECKPOINT_DIR=/path/to/0000040.net_model pytest tests/local_tests/cosmos25/test_cosmos25_dfd_transformer_parity.py -v -s` | 1 passed, non-skip | max 0.15625, mean 0.01315392, relative mean 0.01986194; 2026-09-11 |
| DFD pipeline contracts | `pytest tests/local_tests/cosmos25/test_cosmos25_dfd_pipeline.py -q` | not run | CPU-isolated image encode, mask, timestep, preservation, and validation contracts |
| DFD pipeline parity | `COSMOS25_DFD_REF_DIR=/path/to/data-forcing-distillation COSMOS25_DFD_CHECKPOINT_DIR=/path/to/0000040.net_model pytest tests/local_tests/cosmos25/test_cosmos25_dfd_pipeline_parity.py -v -s` | not run | Compare the complete four-step latent rollout with identical image latent, text embeddings, and noise |

## Open Questions
| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | What packaged Diffusers-style model ID should carry the distilled scheduler and converted student weights? | conversion | conversion | open | pending |
| Q002 | Does experimental distilled V2W retain acceptable quality after official T2W parity? | pipeline | post-parity experiment | closed | DFD checkpoint selected instead; hybrid T2W-to-DFD and DFD-to-DFD visual gates passed on 2026-09-10 |
| Q003 | Can the DFD DCP state dict be converted without instantiating the upstream teacher/training graph? | conversion | conversion | closed | `dcp_to_torch_save` materializes the student state for key normalization without building any training-only network |
| Q004 | What public FastVideo package IDs should carry the T2W and DFD students? | conversion | release | open | pending |

## Issues And Blockers
| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|
| I001 | parity | student DiT | high | No non-skip real-weight distilled forward comparison yet | Spark official-vs-FastVideo BF16 comparison | parity | closed | passed at final relative mean 0.038397 |
| I002 | conversion | packaged model | high | Released official checkpoint is not yet isolated in a FastVideo-loadable component layout | Converted package and strict production load | conversion | closed | 685 clean student tensors; load pass |
| I003 | pipeline | T2W | high | End-to-end distilled generation is not yet validated | small and full-resolution Spark runs plus visual inspection | pipeline | closed | full T2W quality gate passed |
| I004 | prep | DFD checkpoint | medium | Public release has no `model_index.json` and stores the student as zipped PyTorch DCP shards | HF metadata inspection; seven-file custom layout | conversion | open | converter is complete; real converted-package strict load remains |
| I005 | pipeline | DFD V2W | high | Native four-step rollout parity has not yet been established | scheduler and DiT parity pass; dedicated pipeline route is authored | parity | open | run isolated contracts, full rollout parity, then converted-package smoke |

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
| 2026-09-10 | Retain the distilled T2W student as the DreamVerse bootstrap model | The published DFD checkpoint requires an input image | Segment 1 uses T2W; later segments use DFD V2W |
| 2026-09-10 | Port DFD as a distinct scheduler/checkpoint profile | DFD uses rectified-flow ODE steps, not the fixed-noise TrigFlow rollout | Existing T2W behavior stays isolated and unchanged |
| 2026-09-10 | Standardize the hybrid runtime at 24 FPS | Both Cosmos networks use 24 FPS temporal encoding; the DFD CLI's `--fps` only controls student output container metadata | DreamVerse should stream at 24 FPS and trim the repeated DFD frame |
| 2026-09-10 | Treat the reference hybrid eye gate as model evidence, not native-port parity | T2W-to-DFD and DFD-to-DFD boundaries were seamless and the changed prompt produced the requested pivot | Clears model selection while retaining native parity requirements |
| 2026-09-10 | Rebuild PR 1768 as a dedicated `GenerationBackend` | Merged FastH3 established the model-owned lifecycle and continuation boundary | Cosmos will not add conditionals to the LTX2 backend or revive the removed shared generation implementation |
| 2026-09-10 | Keep two explicit Cosmos package roles in the backend | The public DFD model cannot create an unconditioned first segment, while the released TrigFlow student cannot continue a frame | Segment 1 uses the bootstrap package unless the user supplies an initial image; all conditioned segments use the DFD package |
| 2026-09-10 | Synthesize silent 24 kHz audio in the Cosmos backend | DreamVerse's fMP4 streamer currently requires an audio track | Video-only Cosmos output remains streamable without changing the shared AV contract |
| 2026-09-11 | Accept DFD real-weight DiT parity and begin pipeline wiring | Exact preprocessing and bounded 1.986% aggregate BF16 drift show the reused transformer implements the DFD student | Clears the add-model component gate; does not yet clear the four-step pipeline gate |
| 2026-09-11 | Use a dedicated scheduler-selected DFD V2W route | The DFD checkpoint has fixed geometry, FPS, timestep, mask, and first-frame semantics distinct from both base Cosmos and TrigFlow T2W | Existing Cosmos paths remain unchanged; DFD validates 704x1280x81 at 24 FPS and four steps |

## Handoff Notes
- Existing T2W CPU scheduler unit and pinned-reference parity tests pass without skips; the new DFD tests are authored but not executable in this local environment because its optional FastVideo/DFD dependencies are absent.
- Released checkpoint conversion, production strict load, and official-vs-FastVideo DiT parity pass on Spark.
- Small and full-resolution T2W generation pass on Spark; the full video passed visual inspection.
- The `save_video=False`, `return_frames=True` result contract passes on Spark.
- A public converted package ID remains open; until then, use the documented local conversion flow.
- Do not use the prior FlowUniPC/Karras Spark run as distilled parity evidence.
- DFD reference commit is staged under ignored `DFDReference/`; no weights or dependencies were downloaded locally.
- Public DFD reference inference passed one T2W-to-DFD and one DFD-to-DFD continuation with boundary MAE 2.694 and 2.459; the unrelated control was 39.142.
- DreamVerse integration must follow the merged `GenerationBackend` boundary from PR #1800 rather than modifying the former shared `video_generation.py` implementation.
- The 1768 rewrite should own two generators or an equivalent safe weight-switching lifecycle, retain a copied RGB terminal frame, trim one video frame plus one frame-equivalent of synthesized silence on DFD continuations, and warm both bootstrap and continuation paths before readiness.
