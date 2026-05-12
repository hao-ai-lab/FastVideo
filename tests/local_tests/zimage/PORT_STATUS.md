# Z-Image Port Status

## Summary
- model_family: zimage
- workload_types: T2I
- official_ref: Tongyi-MAI / Z-Image (Qwen3-based T2I)
- official_ref_dir: `<repo_root>/Z-Image/src` (cloned manually; SHA not yet pinned)
- hf_weights_path: `<TODO: pin published HF id>`
- local_weights_dir: `<repo_root>/official_weights/Z-Image/` (subfolders: `text_encoder/`, `tokenizer/`, `vae/`, `scheduler/`)
- source_layout: diffusers
- local_tests_readme: `tests/local_tests/zimage/README.md`

## Current Phase
- phase: Phase 4 (component parity)
- status: in_progress
- owner: parity
- last_updated: 2026-05-12

## Component Matrix
| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| Text encoder (Qwen3) | text_encoder | ported | `zimage.Qwen3Model` (HF-Qwen3 layout) | `Qwen3ForCausalLM` checkpoint at `<weights>/text_encoder/` | `fastvideo/models/encoders/qwen3.py` + `fastvideo/configs/models/encoders/qwen3.py` | DONE | not_needed (raw safetensors load) | PASS (fp32 + bf16) | I001, I003 |
| Tokenizer | tokenizer | reused | `AutoTokenizer` | `<weights>/tokenizer/` | `fastvideo/models/loader/component_loader.py::TokenizerLoader` | DONE | not_needed | PASS | none |
| VAE | vae | reused | `zimage.AutoencoderKL` (Diffusers-compatible) | `<weights>/vae/` | `fastvideo/models/vaes/autoencoder_kl.py` | DONE | not_needed | PASS (decode only) | encode-path parity deferred to pipeline |
| Scheduler | scheduler | reused (with extension) | `zimage.FlowMatchEulerDiscreteScheduler` | `<weights>/scheduler/scheduler_config.json` | `fastvideo/models/schedulers/scheduling_flow_match_euler_discrete.py` (added `use_reference_discrete_timesteps`) | DONE | not_needed | PASS | I002 |
| Transformer (ZImageTransformer2DModel) | dit | ported | `zimage.ZImageTransformer2DModel` | `<weights>/transformer/` | `<TODO>` | NOT_STARTED | TBD | NOT_STARTED | I004 |
| Pipeline | pipeline | new | `zimage.ZImagePipeline` | `<weights>/model_index.json` | `<TODO>` | NOT_STARTED | TBD | NOT_STARTED | I005 |

## Conversion State
- conversion_script: `<none — encoder loads raw safetensors via tests/local_tests/zimage/test_zimage_encoder_parity.py helpers; VAE/tokenizer/scheduler use HF subfolder loaders>`
- converted_weights_dir: `<none>`
- source_layout: diffusers
- strict_load_status: `pass_with_documented_exclusions` (Qwen3Model.ALLOWED_UNEXPECTED_KEYS = {"lm_head.weight"})
- passthrough_components: VAE config, scheduler config, tokenizer assets
- retry_history: `<none>`

## Parity Commands
| Scope | Command | Last Result | Notes |
|---|---|---|---|
| Scheduler | `pytest tests/local_tests/zimage/test_zimage_scheduler_parity.py -v -s` | PASS | full `scheduler_config.json` now forwarded (was 3 keys) |
| Tokenizer | `pytest tests/local_tests/zimage/test_zimage_tokenizer_parity.py -v -s` | PASS | `apply_chat_template` parity included |
| VAE decode | `pytest tests/local_tests/zimage/test_zimage_vae_parity.py -v -s` | PASS | encode-path deferred |
| Text encoder | `pytest tests/local_tests/zimage/test_zimage_encoder_parity.py -v -s` | PASS (fp32 + bf16) | bf16 uses `atol=0.05` + abs-mean drift check + per-batch diagnostics |

## Open Questions
| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | Pin a Z-Image reference clone SHA in the README before handoff | prep | Phase 1 | open | |
| Q002 | Final HF id for published Z-Image weights | prep | Phase 1 | open | |
| Q003 | Does Z-Image use Qwen3 chat-template tokenization at pipeline runtime? Currently `Qwen3Config.is_chat_model=False` | pipeline | Phase 6 | open | |

## Issues And Blockers
| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|
| I001 | Phase 4 | text_encoder | medium | `Qwen3ForCausalLM` checkpoints ship `lm_head.weight`; encoder-only `Qwen3Model` does not own an LM head. Strict-load must allowlist this key. | `fastvideo/models/encoders/qwen3.py::Qwen3Model.ALLOWED_UNEXPECTED_KEYS`; encoder parity test asserts the unexpected-key set ⊆ allowlist | parity | resolved | Allowlist landed in this PR; loader raises on any other unexpected key. |
| I002 | Phase 5/6 | scheduler | high | `scheduler_config.json` at `<weights>/Z-Image/scheduler/` does not pin `use_reference_discrete_timesteps=True`. Stock loaders will silently fall back to default timestep mode (numerically different — parity tests prove the divergence). | `tests/local_tests/zimage/test_zimage_scheduler_parity.py` sets the flag programmatically | pipeline | open | Pin the flag in `scheduler_config.json` when wiring the pipeline preset. |
| I003 | Phase 6 | text_encoder | low | `Qwen3ArchConfig.text_len=512` → `tokenizer_kwargs.max_length=512`, but parity tests tokenize at 96/128. Pipeline preset must reconcile. | `fastvideo/configs/models/encoders/base.py::TextEncoderArchConfig.__post_init__` | pipeline | open | Set the correct `text_len` from the Z-Image preset config when adding the pipeline preset. |
| I004 | Phase 4 | transformer | high | `ZImageTransformer2DModel` not yet ported | PR body, in-progress | port | open | Future PR. |
| I005 | Phase 6 | pipeline | high | No FastVideo pipeline class, registry entry, preset, or example yet | n/a | pipeline | open | Future PR. |

## Escape Hatches
| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
|---|---|---|---|---|---|---|

## Decisions
| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-05-12 | bf16 encoder parity uses `atol=0.05` + abs-mean drift < 5e-3 + per-batch max/mean/median/p99 diagnostics. | 24-layer Qwen3 in bf16 accumulates per-GEMM epsilon into a long tail; median-near-zero + abs-mean-below-threshold catches real bugs (mean drift signature), per add-model-02-parity calibration block. | Replaces the silent fp32 downgrade. |
| 2026-05-12 | Scheduler parity forwards the full `scheduler_config.json` dict (minus Diffusers loader keys), not 3 hand-picked keys. | Future on-disk fields (`time_shift_type`, `invert_sigmas`, etc.) would have been silently dropped. | Makes parity reflect the actual on-disk config. |

## Handoff Notes
- Next agent should resolve Q001 (pin Z-Image SHA), then move to transformer port (I004).
- Loader-side strictness is now contract-asserted in the encoder parity test; do not relax `ALLOWED_UNEXPECTED_KEYS` without updating the test.
- Pipeline preset wiring must also pin `use_reference_discrete_timesteps=True` in `scheduler_config.json` (I002).
