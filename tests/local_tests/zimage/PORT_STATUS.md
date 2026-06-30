# Z-Image Port Status

## Summary
- model_family: zimage
- workload_types: T2I
- official_ref: Tongyi-MAI / Z-Image (Qwen3-based S3-DiT T2I, Apache-2.0)
- official_ref_dir: `<repo_root>/Z-Image/src` (clone `github.com/Tongyi-MAI/Z-Image`, pinned to `26f23eda626ffadda020b04ff79488e1d72004cd`)
- hf_weights_path: `Tongyi-MAI/Z-Image-Turbo` (also `Tongyi-MAI/Z-Image` for the full model — components are interchangeable for parity)
- local_weights_dir: `<repo_root>/official_weights/Z-Image/` (subfolders: `text_encoder/`, `tokenizer/`, `vae/`, `scheduler/`)
- source_layout: diffusers
- local_tests_readme: `tests/local_tests/zimage/README.md`

## Current Phase
- phase: Phase 4 (component parity)
- status: in_progress
- owner: parity
- last_updated: 2026-06-21

## Component Matrix
| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| Text encoder (Qwen3) | text_encoder | reused | `zimage.Qwen3Model` (HF-Qwen3 layout) | `Qwen3ForCausalLM` checkpoint at `<weights>/text_encoder/` | shared `fastvideo/models/encoders/qwen3.py::Qwen3ForCausalLM` (+ `Qwen3TextConfig`), added for Flux2 Klein; Z-Image's `Qwen3Model` arch string routes to it in the registry | DONE | not_needed (raw safetensors load) | PASS (fp32 bit-exact + bf16, L40S 2026-06-21) | I001, I003 |
| Tokenizer | tokenizer | reused | `AutoTokenizer` | `<weights>/tokenizer/` | `fastvideo/models/loader/component_loader.py::TokenizerLoader` | DONE | not_needed | PASS | none |
| VAE | vae | reused | `zimage.AutoencoderKL` (Diffusers-compatible) | `<weights>/vae/` | `fastvideo/models/vaes/autoencoder_kl.py` | DONE | not_needed | PASS (decode only) | encode-path parity deferred to pipeline |
| Scheduler | scheduler | reused (with extension) | `zimage.FlowMatchEulerDiscreteScheduler` | `<weights>/scheduler/scheduler_config.json` | `fastvideo/models/schedulers/scheduling_flow_match_euler_discrete.py` (added `use_reference_discrete_timesteps`) | DONE | not_needed | PASS | I002 |
| Transformer (ZImageTransformer2DModel) | dit | ported | `zimage.ZImageTransformer2DModel` | `<weights>/transformer/` | `<TODO>` | NOT_STARTED | TBD | NOT_STARTED | I004 |
| Pipeline | pipeline | new | `zimage.ZImagePipeline` | `<weights>/model_index.json` | `<TODO>` | NOT_STARTED | TBD | NOT_STARTED | I005 |

## Conversion State
- conversion_script: `<none — encoder loads raw safetensors via tests/local_tests/zimage/test_zimage_encoder_parity.py helpers; VAE/tokenizer/scheduler use HF subfolder loaders>`
- converted_weights_dir: `<none>`
- source_layout: diffusers
- strict_load_status: `pass_with_documented_exclusions` (allowlist `{"lm_head.weight"}` asserted in the encoder parity test; the shared encoder is body-only and owns no LM head)
- passthrough_components: VAE config, scheduler config, tokenizer assets
- retry_history: `<none>`

## Parity Commands
| Scope | Command | Last Result | Notes |
|---|---|---|---|
| Scheduler | `pytest tests/local_tests/zimage/test_zimage_scheduler_parity.py -v -s` | PASS (2/2) on Z-Image-Turbo, A40, 2026-05-12 | full `scheduler_config.json` now forwarded (was 3 keys) |
| Tokenizer | `pytest tests/local_tests/zimage/test_zimage_tokenizer_parity.py -v -s` | PASS (2/2) on Z-Image-Turbo, A40, 2026-05-12 | tokenizer resolves to `Qwen2TokenizerFast`; `apply_chat_template` parity included |
| VAE decode | `pytest tests/local_tests/zimage/test_zimage_vae_parity.py -v -s` | PASS (1/1) on Z-Image-Turbo, A40, 2026-05-12 | encode-path deferred |
| Text encoder fp32 | `pytest tests/local_tests/zimage/test_zimage_encoder_parity.py::test_zimage_qwen3_encoder_parity_forward[fp32]` | PASS on Z-Image-Turbo, L40S, 2026-06-21 | Bit-exact against the shared `Qwen3ForCausalLM` (both batch elements): `last_hidden_state` max=0.0000 mean=0.0000; `hidden_states[-2]` max=0.0010 mean=0.0000. Confirms the batched-RoPE `position_ids` fix. |
| Text encoder bf16 | `pytest tests/local_tests/zimage/test_zimage_encoder_parity.py::test_zimage_qwen3_encoder_parity_forward[bf16]` | PASS on Z-Image-Turbo, L40S, 2026-06-21 | `last_hidden_state` mean=0.0163/0.0164 median=0.0127/0.0137 (thresholds 0.025 / 0.020); `hidden_states[-2]` mean=0.0804/0.0717 median=0.0625 (thresholds 0.120 / 0.100). In line with the prior bespoke profile — existing thresholds hold on L40S, no recalibration needed. |
| Per-layer bf16 diag | `pytest tests/local_tests/zimage/test_zimage_encoder_parity.py::test_zimage_qwen3_encoder_per_layer_bf16_diagnostic -v -s` | PASS (informational) on Z-Image-Turbo, L40S, 2026-06-21 | 37 hidden states (embedding + 35 layers + post-norm); smooth monotonic drift, post-norm mean=0.0154. `hidden_states` length/indexing matches HF. |

## Open Questions
| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | Pin a Z-Image reference clone SHA in the README before handoff | prep | Phase 1 | resolved | Pinned `Tongyi-MAI/Z-Image@26f23eda626ffadda020b04ff79488e1d72004cd` (2026-05-12) |
| Q002 | Final HF id for published Z-Image weights | prep | Phase 1 | resolved | `Tongyi-MAI/Z-Image-Turbo` (6B, 8 NFE, fits 16 GB) and `Tongyi-MAI/Z-Image` (full, 32.9 GB). Both Apache-2.0 |
| Q003 | Does Z-Image use Qwen3 chat-template tokenization at pipeline runtime? The shared `Qwen3TextConfig` (from Flux2 Klein) defaults `is_chat_model=True`; Z-Image's removed bespoke config had `False`. Reconcile when wiring the pipeline preset. | pipeline | Phase 6 | open | |

## Issues And Blockers
| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|
| I001 | Phase 4 | text_encoder | medium | Z-Image-Turbo ships a full Qwen3 checkpoint with `lm_head.weight`; the shared encoder (`Qwen3ForCausalLM`) is body-only and owns no LM head, so that key goes unmatched. Must be allowlisted; anything else is a real silent drop. | encoder parity test asserts the unexpected-key set ⊆ `_ALLOWED_UNEXPECTED_KEYS = {"lm_head.weight"}` | parity | resolved | Allowlist enforced in the parity test (the shared encoder's loader is intentionally lenient as it serves multiple models, so the contract lives test-side). |
| I002 | Phase 5/6 | scheduler | high | `scheduler_config.json` at `<weights>/Z-Image/scheduler/` does not pin `use_reference_discrete_timesteps=True`. Stock loaders will silently fall back to default timestep mode (numerically different — parity tests prove the divergence). | `tests/local_tests/zimage/test_zimage_scheduler_parity.py` sets the flag programmatically | pipeline | open | Pin the flag in `scheduler_config.json` when wiring the pipeline preset. |
| I003 | Phase 6 | text_encoder | low | The shared `Qwen3TextArchConfig.text_len=512` → `tokenizer_kwargs.max_length=512`, but parity tests tokenize at 96/128. Pipeline preset must reconcile. | `fastvideo/configs/models/encoders/base.py::TextEncoderArchConfig.__post_init__` | pipeline | open | Set the correct `text_len` from the Z-Image preset config when adding the pipeline preset. |
| I004 | Phase 4 | transformer | high | `ZImageTransformer2DModel` not yet ported | PR body, in-progress | port | open | Future PR. |
| I005 | Phase 6 | pipeline | high | No FastVideo pipeline class, registry entry, preset, or example yet | n/a | pipeline | open | Future PR. |

## Escape Hatches
| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
|---|---|---|---|---|---|---|

## Decisions
| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-06-21 | Fix the shared Qwen3 encoder's RoPE positions for batched inputs: expand `position_ids` to `[batch_size, seq_len]`. | L40S re-validation caught a parity divergence that was present **only at batch>1** (batch=1 matched; batch=2 fp32 mean_diff 0.22). Root cause: the rotary layer flattens `position_ids` to `num_tokens` and reshapes q/k to `(num_tokens, -1, head_dim)`; the encoder built `position_ids` as `[1, seq_len]`, so for batch>1 `num_tokens` stayed `seq_len` and the batch dim got folded into the head dim, misaligning RoPE. (Two earlier hypotheses — causal mask, and `sdpa_mask` mask construction — were both falsified by re-runs: bidirectional was *worse*, and `sdpa_mask` gave byte-identical results since Z-Image is right-padded.) | One-line fix in `Qwen3ForCausalLM.forward` (`.expand(batch_size, -1)`); batch=1 is byte-identical so Flux2 Klein is unaffected. This is also a latent batch bug in the shared encoder. Needs the L40S re-run to confirm. |
| 2026-06-21 | Reuse the shared `Qwen3ForCausalLM` encoder (+ `Qwen3TextConfig`) instead of the bespoke `Qwen3Model` this PR originally added. | `main` gained a config-driven Qwen3 text encoder via the Flux2 Klein port (#1349); it returns `BaseEncoderOutput(last_hidden_state, hidden_states)`, supports GQA + the same qkv/gate_up fusion, and `update_model_arch` populates Z-Image-Turbo's dims (2048/24/16) from its `config.json`. Keeping our parallel encoder would have been ~540 lines of duplicate maintenance. | Dropped `fastvideo/models/encoders/qwen3.py` + `fastvideo/configs/models/encoders/qwen3.py` from this PR; added a 1-line registry map (`"Qwen3Model" → Qwen3ForCausalLM`). Encoder parity needs re-validation on A40 against the shared class (see Parity Commands). |
| 2026-05-12 | bf16 encoder parity uses distribution checks (mean + median) instead of element-wise `assert_close`; thresholds calibrated to empirical Z-Image-Turbo numbers on A40 + 1.5–1.6x headroom. | Z-Image-Turbo's Qwen3 text encoder is 35 layers (not the 24 originally assumed). Cross-kernel bf16 (FastVideo's fused QKVParallel + MergedColumnParallel + SiluAndMul vs HF's unfused equivalents) accumulates into a long max tail (~4.0 at layer 34) but median stays low (0.06). Per-layer diagnostic test confirmed growth is smooth and monotonic with no single-layer spike — textbook bf16-tail signature, fp32 is bit-exact. Element-wise `assert_close` is meaningless on this profile; mean + median + the per-layer diag together detect real bugs (which push mean ≫ atol AND median > 0.01). | Final assertion shape: `last_hidden_state` mean < 0.025, median < 0.020; `hidden_states[-2]` mean < 0.120, median < 0.100. Validated on NVIDIA A40 (driver 565.57.01, 46068 MiB) 2026-05-12. |
| 2026-05-12 | `AutoModel.from_pretrained` uses `dtype=` (not `torch_dtype=`). | transformers 4.57.3 emits `torch_dtype is deprecated! Use dtype instead!` warning. Mrinaald's original `dtype=` kwarg was correct; the temporary switch to `torch_dtype=` (in response to a Copilot review comment) was reverted. | – |
| 2026-05-12 | Scheduler parity forwards the full `scheduler_config.json` dict (minus Diffusers loader keys), not 3 hand-picked keys. | Future on-disk fields (`time_shift_type`, `invert_sigmas`, etc.) would have been silently dropped. | Makes parity reflect the actual on-disk config. |

## Handoff Notes
- Component parity (scheduler / tokenizer / VAE on A40 2026-05-12; text encoder fp32 + bf16 on the shared `Qwen3ForCausalLM`, L40S 2026-06-21) is **fully validated** on `Tongyi-MAI/Z-Image-Turbo`. fp32 is bit-exact; bf16 holds the existing thresholds.
- Encoder reuse required one shared-encoder fix: batched RoPE `position_ids` (`.expand(batch_size, -1)`) — see Decisions. batch=1 is byte-identical so Flux2 Klein is unaffected.
- Loader-side strictness is contract-asserted in the encoder parity test via `_ALLOWED_UNEXPECTED_KEYS`; do not relax it without updating the test.
- Next port-stack steps (separate PR, not in #1339 scope): `ZImageTransformer2DModel` port (I004), pipeline preset including `use_reference_discrete_timesteps=True` pinned in `scheduler_config.json` (I002), conversion-or-direct-load story, SSIM media regression (blocked on PR #1321's `media_extension` helper landing for T2I `.png` output).
