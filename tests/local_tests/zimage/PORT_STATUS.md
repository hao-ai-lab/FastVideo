# Z-Image Port Status

## Summary

- model_family: `zimage`
- variant: `Z-Image-Turbo`
- workload_types: `T2I`
- PR scope: components plus native transformer prototype/parity
- official_ref: `https://github.com/Tongyi-MAI/Z-Image`
- official_ref_dir: `<repo_root>/Z-Image/src`
- official_ref_commit: `26f23eda626ffadda020b04ff79488e1d72004cd`
- hf_weights_path: `Tongyi-MAI/Z-Image-Turbo@f332072aa78be7aecdf3ee76d5c247082da564a6`
- local_weights_dir: `<repo_root>/official_weights/Z-Image/`
- source_layout: `diffusers`
- local_tests_readme: `tests/local_tests/zimage/README.md`

The pinned text encoder has 36 decoder blocks, hidden size 2560,
intermediate size 9728, 32 attention heads, 8 KV heads, and head dimension
128. FastVideo obtains those values from the pinned `config.json`.

## Current Phase

- phase: Phase 6 (transformer parity activation and component revalidation)
- status: in_progress
- owner: dit/parity
- last_updated: 2026-07-13

## Component Matrix

| Component | Type | Reuse/Port | Official Definition / Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|
| Text encoder (Qwen3 body) | text_encoder | reused production body + native implementation | Pinned `Qwen3ForCausalLM` checkpoint; Transformers `AutoModel` body; official pipeline consumes `hidden_states[-2]` | `TextEncoderLoader` -> `Qwen3ForCausalLM.from_pretrained_local` -> independent body-only `AutoModel`, plus native `fastvideo.models.encoders.qwen3.Qwen3ForCausalLM` | DONE | not_needed for current HF subfolder | REVALIDATION REQUIRED (historical unpinned native fp32/bf16 PASS only) | I001, I006, I007 |
| Tokenizer | tokenizer | reused | `AutoTokenizer` from `<weights>/tokenizer/`; official thinking template and length 512 | `TokenizerLoader` | DONE | not_needed | REVALIDATION REQUIRED (historical unpinned PASS) | I003, I010 |
| VAE | vae | reused with accepted wrapper exception | Pinned `zimage.AutoencoderKL`; official decode uses `(latents / scaling_factor) + shift_factor` | Existing Diffusers-backed `fastvideo.models.vaes.autoencoder_kl.AutoencoderKL` through production `VAELoader`, plus direct implementation coverage | DONE | not_needed | REVALIDATION REQUIRED (historical unpinned direct-decode PASS only) | I008 |
| Scheduler | scheduler | reused with Z-Image extension | Pinned `zimage.FlowMatchEulerDiscreteScheduler`; official pipeline sets `sigma_min=0.0` | `FlowMatchEulerDiscreteScheduler(use_reference_discrete_timesteps=True, sigma_min=0.0)` | DONE | not_needed | REVALIDATION REQUIRED; asset-free regressions remain separately runnable | I002, I009 |
| Transformer (`ZImageTransformer2DModel`) | dit | native port | Pinned `src/zimage/transformer.py`; published Diffusers transformer subfolder | `fastvideo.models.dits.zimage.ZImageTransformer2DModel` through `TransformerLoader` | DONE (production meta + tiny CPU forward) | not_needed; exact 521-key surface | TINY NON-SKIP PASS; production real-weight forward PENDING | I011 |
| Pipeline | pipeline | planned | Pinned `ZImagePipeline` | future PR | NOT_STARTED | depends_on_transformer | NOT_STARTED | I005 |

## Conversion State

- conversion_script: none for the reused components in this component-only PR
- converted_weights_dir: none
- source_layout: diffusers
- transformer_conversion: not_needed; official, HF index, and native names align exactly
- transformer_key_surface: 521 tensors; sorted-key SHA256 `3a9216f208c1873b2cf06394411a53e1e95e10fae3b01dca0f7223556e47c354`
- strict_load_status: transformer real-weight strict load pending; other components revalidation_required
- production_passthrough: body-only Qwen3 `AutoModel`, tokenizer assets
- direct_load_components: native Qwen3 parity target, VAE, scheduler
- native_qwen_exclusion: `lm_head.weight` only; the native encoder owns the body, not an LM head
- native_qwen_fused_contract: every Q/K/V and gate/up shard is required unless the exact fused destination is present
- native_qwen_auxiliary_contract: mapped quantization auxiliaries, including scale parameters, must load and may not be silently discarded
- retry_history: none

## Parity Commands

| Scope | Command | Last Result | Notes |
|---|---|---|---|
| Scheduler | `pytest tests/local_tests/zimage/test_zimage_scheduler_parity.py -v -s` | RERUN REQUIRED; historical unpinned 2/2 PASS on A40, 2026-05-12 | Now enforces reference SHA/import origin and exact zero endpoint; includes asset-free positional/default regressions. |
| Tokenizer | `pytest tests/local_tests/zimage/test_zimage_tokenizer_parity.py -v -s` | RERUN REQUIRED; historical unpinned 2/2 PASS on A40, 2026-05-12 | Now uses `enable_thinking=True`, `max_length=512`, and fails if chat templating is unavailable. |
| VAE | `pytest tests/local_tests/zimage/test_zimage_vae_parity.py -v -s` | RERUN REQUIRED; historical unpinned 1/1 direct-decode PASS on A40, 2026-05-12 | Scope is now `both`: direct decode plus `VAELoader` and official latent normalization. |
| Text encoder fp32 | `pytest tests/local_tests/zimage/test_zimage_encoder_parity.py::test_zimage_qwen3_encoder_parity_forward[fp32] -v -s` | RERUN REQUIRED; historical unpinned native PASS on L40S, 2026-06-21 | New run must cover independent reference, body-only production path, and native implementation. |
| Text encoder bf16 | `pytest tests/local_tests/zimage/test_zimage_encoder_parity.py::test_zimage_qwen3_encoder_parity_forward[bf16] -v -s` | RERUN REQUIRED; historical unpinned native PASS on L40S, 2026-06-21 | Historical thresholds retained pending pinned-snapshot validation. |
| Per-layer bf16 diagnostic | `pytest tests/local_tests/zimage/test_zimage_encoder_parity.py::test_zimage_qwen3_encoder_per_layer_bf16_diagnostic -v -s` | RERUN REQUIRED; historical unpinned informational run on L40S, 2026-06-21 | Pinned architecture has 36 blocks and 37 hidden-state entries under the tested contract. |
| Transformer meta/tiny parity | `ZIMAGE_OFFICIAL_REF_DIR=<pinned-clone> pytest tests/local_tests/zimage/test_zimage_transformer_parity.py -v -s` | 4 PASSED, 1 SKIPPED on CPU, 2026-07-13 | Production 521-key/shape surface and pinned HF key digest match; additive-mask attention and tiny variable-length full-forward parity are exact; SP>1 guard passes. Real-weight test skipped because the 24.6 GB transformer subfolder/CUDA were absent. |

## Open Questions

| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | Which exact Z-Image source revision is the parity oracle? | prep | Phase 1 | resolved | Pinned `Tongyi-MAI/Z-Image@26f23eda626ffadda020b04ff79488e1d72004cd`; scheduler/VAE tests fail on a wrong HEAD or import origin. |
| Q002 | Which immutable HF snapshot supplies component weights? | prep | Phase 1 | resolved | `Tongyi-MAI/Z-Image-Turbo@f332072aa78be7aecdf3ee76d5c247082da564a6`. |
| Q003 | What prompt/tokenization/hidden-state contract does the official pipeline use? | pipeline | Phase 6 | resolved | `apply_chat_template(tokenize=False, add_generation_prompt=True, enable_thinking=True)`, tokenization length 512, and valid-token `hidden_states[-2]`. |
| Q004 | Does the native transformer require conversion? | conversion | Phase 5 | resolved | No. Pinned HF index, official production meta model, and FastVideo production meta model expose the same 521 keys; official/native key shapes match exactly. |

## Issues And Blockers

| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|
| I001 | Phase 6 | text_encoder | medium | The full checkpoint includes `lm_head.weight`, while the native FastVideo encoder owns only the body. | Native parity allowlists exactly `lm_head.weight`; production uses body-only Transformers `AutoModel`. | parity | in_progress | Implementation is complete; resolve only after the pinned fp32/bf16 rerun. |
| I002 | Phase 7 | scheduler/pipeline | high | Published `scheduler_config.json` omits the reference-schedule flag and official zero endpoint. | Pinned pipeline mutates `sigma_min=0.0`; parity requires both scheduler values. | pipeline | open | Future pipeline config must serialize `use_reference_discrete_timesteps=True` and `sigma_min=0.0`. |
| I003 | Phase 6 | tokenizer/text_encoder | medium | Prompt formatting and target length previously differed from the official call. | Pinned source uses thinking chat template, 512 tokens, and `hidden_states[-2]`. | parity | in_progress | Test/config behavior corrected; pinned asset rerun required before resolving. |
| I004 | Phase 4 | transformer | high | `ZImageTransformer2DModel` was not ported. | Native config/model/registry and deterministic parity test now exist. | port | resolved | Production meta surface matches all 521 pinned HF keys; tiny full-forward parity passes exactly. |
| I005 | Phase 7 | pipeline | high | No FastVideo pipeline/config/preset/registry/example or quality regression exists. | Component-only PR scope. | pipeline | open | Separate future PR after component gates pass. |
| I006 | Phase 6 | text_encoder | high | Production loading previously used a causal-LM object and could override CPU placement, while parity exercised only direct native construction. | Production path is now body-only `AutoModel`; loader records `_fastvideo_input_device`; stage moves tokens to that device; parity uses a distinct reference instance. | encoder | in_progress | Implementation complete; pinned production-loader rerun required. |
| I007 | Phase 6 | text_encoder | high | Lenient native loading could miss one fused source shard or an auxiliary quantization scale. | Loader completeness now covers every destination, required split shard, exact fused destination, and mapped scale parameter. | encoder | in_progress | Asset-free strictness regressions and pinned checkpoint load must pass before resolving. |
| I008 | Phase 6 | vae | medium | Direct decode alone did not cover production registry/config/device/strict-load behavior; the shared target is a runtime Diffusers wrapper. | VAE parity scope is now `both` and includes `VAELoader` plus official normalization. | parity | in_progress | Existing Diffusers `AutoencoderKL` wrapper exception is accepted only for this component-only PR; pinned rerun required. |
| I009 | Phase 2 | reference assets | high | Adding a nonexistent clone path to `sys.path` could import another installed `zimage` package and yield false parity. | Tests now verify exact clone HEAD and module paths under `Z-Image/src`; `/Z-Image/` is ignored. | prep | resolved | Wrong SHA/path/import is a failure; absent local assets may skip. |
| I010 | Phase 6 | tokenizer | medium | Missing `apply_chat_template` previously skipped and could hide an incompatible tokenizer. | Tokenizer parity now asserts the API and exact official kwargs. | parity | resolved | Missing chat-template behavior fails. Asset-backed numerical rerun remains I003. |
| I011 | Phase 6 | transformer | high | Production real-weight strict load and full-forward parity are not verified. | The pinned transformer is 24.6 GB and is not present locally; the CUDA test scaffold therefore skipped. | parity | open | Run `test_zimage_transformer_production_loader_forward_parity` non-skip with the pinned HF transformer subfolder on CUDA and record drift. |

## Escape Hatches

| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
|---|---|---|---|---|---|---|

## Decisions

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-07-13 | Preserve the published transformer key surface and use raw masked SDPA at SP=1. | The checkpoint already matches the official class, while padded variable-length attention needs a key mask unavailable in current distributed wrappers. | No converter is needed; runtime rejects SP>1 until a mask-aware distributed path has parity evidence. |
| 2026-07-04 | Pin both external sources and treat all earlier GPU results as historical until rerun. | Earlier evidence did not record an immutable HF revision, and source imports were not origin-checked. | README has executable clone/download commands; tests enforce the source SHA; all asset-backed parity rows are revalidation-required. |
| 2026-07-04 | Use a body-only Transformers `AutoModel` for production and a separate `AutoModel` object as the parity oracle. | Every FastVideo consumer needs hidden states, not full-vocabulary logits or an LM head; comparing an object with itself is not independent evidence. | Production avoids the LM head while parity covers production loading and native implementation separately. |
| 2026-07-04 | Make fused-source and quantization-scale completeness part of native Qwen strict loading. | A loaded destination name alone did not prove that every required Q/K/V or gate/up shard and auxiliary scale arrived. | Missing split shards or mapped scale parameters fail; an exact fused destination remains valid. |
| 2026-07-04 | Accept the existing Diffusers-backed AutoencoderKL wrapper only for this component-only PR. | The target is established shared code and this PR adds production-loader evidence rather than a new VAE implementation. | Exception is limited to the VAE here and is not precedent for the transformer or future full pipeline. |
| 2026-07-04 | Treat absent chat-template behavior as incompatibility, not an optional test environment. | Thinking-format prompt construction is part of the official numerical contract. | Tokenizer parity fails when `apply_chat_template` is unavailable. |
| 2026-07-04 | Record the pinned Qwen architecture as 36 blocks, hidden size 2560, intermediate size 9728, 32 attention heads, and 8 KV heads. | Earlier notes described a different architecture and could mis-size the native target. | Config/review evidence now matches `f332072aa78be7aecdf3ee76d5c247082da564a6`; expected hidden-state length is 37. |
| 2026-06-21 | Expand generated Qwen `position_ids` to `[batch_size, seq_len]`. | The rotary layer flattens positions; `[1, seq_len]` misaligned RoPE for batch sizes greater than one. | Batch-one behavior is unchanged; pinned batch-two parity still requires rerun. |
| 2026-06-21 | Reuse the shared native `Qwen3ForCausalLM` implementation rather than maintain a second Z-Image encoder. | The shared implementation supports the same config-driven GQA, fused QKV/gate-up, and hidden-state output contract. | Registry/native parity uses the shared class; architecture values come from the pinned config. |
| 2026-05-12 | Use mean and median drift checks for deep bf16 encoder parity. | Cross-kernel fused/unfused bf16 arithmetic produces a long max-error tail; fp32 and per-layer diagnostics distinguish this from structural divergence. | Existing thresholds remain provisional until the pinned 36-block snapshot is rerun. |
| 2026-05-12 | Forward the complete scheduler config except loader metadata. | Hand-picking three fields could silently omit future scheduler behavior. | Scheduler parity consumes all on-disk constructor fields and explicitly supplies Z-Image runtime overrides. |

## Handoff Notes

- No asset-backed component currently has pinned-snapshot `non_skip_pass`
  evidence. Do not promote a historical row to PASS without recording the exact
  command and non-skip result against both immutable pins.
- Text-encoder revalidation must cover the independent `AutoModel` oracle,
  body-only production loader, native implementation, requested input device,
  fused/split source completeness, and auxiliary quantization scales.
- VAE revalidation must cover both direct implementation decode and production
  `VAELoader` behavior with official scaling/shift normalization.
- Transformer production key/shape and tiny output parity pass. The real-weight
  production-loader/full-forward CUDA test remains I011 and must pass non-skip.
- Next work after I011: pipeline/config/preset/registry/example, pipeline parity,
  and image-quality regression.
