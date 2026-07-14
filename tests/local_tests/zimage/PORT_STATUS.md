# Z-Image Port Status

## Summary

- model_family: `zimage`
- variant: `Z-Image-Turbo`
- workload_types: `T2I`
- PR scope: required components plus same-PR FastVideo-native pipeline, example/parity, SSIM test, and L40S seed workflow
- official_ref: `https://github.com/Tongyi-MAI/Z-Image`
- official_ref_dir: `<repo_root>/Z-Image/src`
- official_ref_commit: `26f23eda626ffadda020b04ff79488e1d72004cd`
- hf_weights_path: `Tongyi-MAI/Z-Image-Turbo@f332072aa78be7aecdf3ee76d5c247082da564a6`
- local_weights_dir: `<repo_root>/official_weights/Z-Image/`
- source_layout: official repo-native `src/zimage/`; HF component subfolders provide assets only
- weight_layout: HF per-component subfolders; these are assets, not the reference implementation
- local_tests_readme: `tests/local_tests/zimage/README.md`

The pinned text encoder has 36 decoder blocks, hidden size 2560,
intermediate size 9728, 32 attention heads, 8 KV heads, and head dimension
128. FastVideo obtains those values from the pinned `config.json`.

## Current Phase

- phase: Phase 7 (pipeline definition and verification), with Phase 6 exact-head closure I012 still open
- status: in_progress
- owner: pipeline/parity
- last_updated: 2026-07-14

## Pinned Official Pipeline Context

- official_pipeline: `Tongyi-MAI/Z-Image@26f23eda626ffadda020b04ff79488e1d72004cd`,
  `src/zimage/pipeline.py::generate` and `src/config/inference.py`
- oracle_policy: official Z-Image source only; Diffusers is not the implementation oracle
- workload_and_output: `T2I`; prompt plus optional negative prompt to PIL images, or latents when requested
- model_index_contract: `_class_name=ZImagePipeline`; required modules are `transformer`, `vae`,
  `text_encoder`, `tokenizer`, and `scheduler`
- official_defaults: `height=1024`, `width=1024`, `num_inference_steps=8`,
  `guidance_scale=0.0`, `cfg_truncation=1.0`, `max_sequence_length=512`
- reproducible_rng: `torch.Generator("cuda").manual_seed(42)`; the API accepts a generator and has no fixed seed default
- scheduler_contract: `use_reference_discrete_timesteps=True`, runtime `sigma_min=0.0`, endpoint-preserving
  `num_steps + 1` schedule, and no DiT call for the final zero timestep
- decode_contract: apply `(latents / 0.3611) + 0.1159` before official VAE decode
- implementation_status: native pipeline/config/preset/registry/example and the 8-step 1024x1024 PNG SSIM test are
  implemented; hardware-free pipeline surface and stage-contract checks pass; pinned native full GPU pipeline parity
  and L40S reference generation/upload have not run

## Component Matrix

| Component | Type | Reuse/Port | Official Definition / Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|
| Production text encoder (Qwen3 body) | text_encoder | reused production passthrough | Official `src/utils/loader.py` uses Transformers with typed load then device-only move; official pipeline consumes `hidden_states[-2]` | `TextEncoderLoader` -> `Qwen3ForCausalLM.from_pretrained_local` -> body-only Transformers `AutoModel` | DONE | not_needed for current HF subfolder | PINNED GB200 PASS; reference/production outputs exact | none |
| Native Qwen quality | implementation_subcomponent | shared native implementation | Same pinned text-encoder weights and official `hidden_states[-2]` boundary; not loaded by the pipeline | direct `fastvideo.models.encoders.qwen3.Qwen3ForCausalLM` construction | DONE | not_needed | BF16 PASS; FP32 PRE-NORM QUALITY DECISION PENDING | I001 |
| Tokenizer | tokenizer | reused | Official loader's `AutoTokenizer` from `<weights>/tokenizer/`; official thinking template and length 512 | `TokenizerLoader` | DONE | not_needed | PINNED GB200 PASS (2 passed) | none |
| VAE | vae | reused with accepted wrapper exception | Pinned official `src/zimage/autoencoder.py`; raw decode is the component boundary | Existing Diffusers-backed `fastvideo.models.vaes.autoencoder_kl.AutoencoderKL` through production `VAELoader`, plus direct raw-decode coverage | DONE | not_needed | PINNED GB200 PASS (2 passed) | I002 (pipeline-only normalization) |
| Scheduler | scheduler | reused with Z-Image extension | Pinned official `src/zimage/scheduler.py`; official pipeline sets `sigma_min=0.0` | `FlowMatchEulerDiscreteScheduler(use_reference_discrete_timesteps=True, sigma_min=0.0)` | DONE | not_needed | PINNED GB200 PASS (5 passed) | I002 |
| Transformer (`ZImageTransformer2DModel`) | dit | native port | Pinned official `src/zimage/transformer.py`; HF `transformer/` supplies weights only | `fastvideo.models.dits.zimage.ZImageTransformer2DModel` through `TransformerLoader` | DONE | not_needed; exact 521-key surface | PINNED GB200 PASS (5 passed; real-weight full forward exact) | none |
| Pipeline | pipeline | native same-PR implementation | Pinned official `src/zimage/pipeline.py::generate` | FastVideo `ZImagePipeline` plus config/preset/registry/example/parity | DONE | not_needed | HARDWARE-FREE SURFACE/STAGE PASS; FULL GPU PARITY NOT RUN | I005 |

## Conversion State

- conversion_script: none for the reused components or native transformer
- converted_weights_dir: none
- source_layout: official repo-native Python (`Tongyi-MAI/Z-Image`)
- hf_weight_layout: per-component subfolders; not a Diffusers implementation oracle
- transformer_conversion: not_needed; official, HF index, and native names align exactly
- transformer_key_surface: 521 tensors; sorted-key SHA256 `3a9216f208c1873b2cf06394411a53e1e95e10fae3b01dca0f7223556e47c354`
- strict_load_status: transformer real-weight strict load passed; native Qwen strictness regressions passed
- passthrough_components: body-only Qwen3 `AutoModel`, tokenizer assets
- direct_load_components: native Qwen3 parity target, VAE, scheduler
- native_qwen_exclusion: `lm_head.weight` only; the native encoder owns the body, not an LM head
- native_qwen_fused_contract: every Q/K/V and gate/up shard is required unless the exact fused destination is present
- native_qwen_auxiliary_contract: mapped quantization auxiliaries, including scale parameters, must load and may not be silently discarded
- retry_history: none

## Parity Commands

| Scope | Command | Last Result | Notes |
|---|---|---|---|
| Scheduler | `pytest tests/local_tests/zimage/test_zimage_scheduler_parity.py -v -s` | 5 PASSED, no skips, GB200, 2026-07-13 | Enforces official-source SHA/import origin, fp32 scalar contract, exact zero endpoint, and positional/default regressions. |
| Tokenizer | `pytest tests/local_tests/zimage/test_zimage_tokenizer_parity.py -v -s` | 2 PASSED as part of the pinned full GB200 run, 2026-07-13 | Uses `enable_thinking=True`, `max_length=512`, and fails if chat templating is unavailable. |
| VAE | `pytest tests/local_tests/zimage/test_zimage_vae_parity.py -v -s` | 2 PASSED, no skips, GB200, 2026-07-13 | Direct raw decode plus production `VAELoader`; config values are checked, while latent scale/shift application is Phase 7 pipeline behavior. |
| Production text encoder | `pytest tests/local_tests/zimage/test_zimage_encoder_parity.py::test_zimage_qwen3_encoder_parity_forward[fp32] -v -s` | REFERENCE/PRODUCTION CHECKS EXACT, GB200, 2026-07-13 | The required pipeline path passes before the same test continues into the optional native-Qwen comparison; exact-head required-component closure is still I012. |
| Native Qwen fp32 quality | `pytest tests/local_tests/zimage/test_zimage_encoder_parity.py::test_zimage_qwen3_encoder_parity_forward[fp32] -v -s` | FINAL OUTPUT PASS; PRE-NORM QUALITY DECISION PENDING, GB200, 2026-07-13 | Native pre-norm has 11/48,640 batch-0 misses at `atol=rtol=1e-4`; largest failing absolute difference is `1.686811e-4`. See `model/r10`. |
| Native Qwen bf16 quality | `pytest tests/local_tests/zimage/test_zimage_encoder_parity.py::test_zimage_qwen3_encoder_parity_forward[bf16] -v -s` | PASS, GB200, 2026-07-13 | Existing mean/median distribution thresholds pass unchanged against an independent materialization of the official loader contract. |
| Native Qwen per-layer bf16 diagnostic | `pytest tests/local_tests/zimage/test_zimage_encoder_parity.py::test_zimage_qwen3_encoder_per_layer_bf16_diagnostic -v -s` | PASS, GB200, 2026-07-13 | Pinned architecture has 36 blocks and 37 hidden-state entries under the tested contract. |
| Transformer parity | `ZIMAGE_OFFICIAL_REF_DIR=<pinned-clone> pytest tests/local_tests/zimage/test_zimage_transformer_parity.py -v -s` | 5 PASSED, no skips, GB200, 2026-07-13 | Production 521-key/shape surface, tiny CPU parity, mask behavior, SP guard, real-weight strict load, and 6.15B full forward pass; real-weight output was exact (`max=0`, `mean=0`). |
| Pipeline typed surface | `pytest tests/local_tests/pipelines/test_zimage_pipeline_smoke.py::test_zimage_typed_surface_preflight -v` | PASS, hardware-free, 2026-07-14 | Covers exact `EntryClass`, config, preset, registry, model detection, and official defaults. |
| Native stage contract | `pytest tests/local_tests/pipelines/test_zimage_pipeline_parity.py::test_zimage_native_default_stage_math -v` | PASS, hardware-free, 2026-07-14 | Covers conditioning trim, endpoint-preserving timestep setup, denoising sign/timestep math, and VAE scale/shift. |
| Full pipeline parity | `ZIMAGE_REFERENCE_REPO=<pinned-clone> ZIMAGE_MODEL_DIR=<pinned-weights> DISABLE_SP=1 pytest tests/local_tests/pipelines/test_zimage_pipeline_parity.py::test_zimage_pipeline_latents_match_pinned_native_repo -v -s` | NOT RUN | Must compare real FastVideo latents with the pinned native Tongyi repository; Diffusers is never the oracle. |
| PNG SSIM quality | `pytest fastvideo/tests/ssim/test_zimage_similarity.py -v -s` | TEST IMPLEMENTED; L40S REFERENCE NOT GENERATED OR UPLOADED | Uses the official 8-step, 1024x1024, seed-42 Turbo recipe and outputs `.png`. |

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
| I001 | Phase 6 | native_qwen_quality | medium | The optional native fp32 `hidden_states[-2]` tail narrowly exceeds the current `atol=1e-4` after smooth cross-kernel accumulation. | Exact layer trace: embeddings exact, layer-0 max `3.815e-6`, first misses at layer 33, largest failing boundary difference `1.686811e-4`; final RMSNorm passes. The required production `AutoModel` path is exact. | parity | author_gate_non_pipeline | `model/r10` asks to change only Z-Image `_FP32_ATOL` to `2e-4`; no shared-Qwen rewrite or silent tolerance change. This row does not gate the production pipeline loader. |
| I002 | Phase 7 | scheduler/pipeline | high | Published `scheduler_config.json` omits the reference-schedule flag and official zero endpoint. | The native stage applies both values, and the hardware-free stage contract passes; full GPU parity has not run. | pipeline | implemented_gpu_verification_pending | The Z-Image pipeline applies `use_reference_discrete_timesteps=True` and `sigma_min=0.0`; close after pinned full GPU parity. |
| I003 | Phase 6 | tokenizer/text_encoder | medium | Prompt formatting and target length previously differed from the official call. | Pinned source uses thinking chat template, 512 tokens, and `hidden_states[-2]`. | parity | resolved | Corrected behavior passed the pinned GB200 run. |
| I004 | Phase 4 | transformer | high | `ZImageTransformer2DModel` was not ported. | Native config/model/registry and deterministic parity test now exist. | port | resolved | Production meta surface matches all 521 pinned HF keys; tiny full-forward parity passes exactly. |
| I005 | Phase 7 | pipeline/quality | high | Pinned native full GPU pipeline parity and the L40S PNG reference are outstanding. | Native pipeline/config/preset/registry/example, parity tests, and the 8-step 1024x1024 PNG SSIM test are implemented; hardware-free surface and stage-contract checks pass. | pipeline | open | Run I012 component closure and pinned full GPU parity/basic example at the exact implementation head, then generate and show the L40S PNG before any HF upload. |
| I006 | Phase 6 | text_encoder | high | HF passthrough loading inherited an ambient target-device context, and the earlier test harness incorrectly cast float32 RoPE buffers during bf16 placement. | HF loading now occurs outside the ambient context; typed load is followed by device-only movement, matching official `src/utils/loader.py`; production/independent outputs match exactly. | encoder | resolved | Hardware-free placement/default-dtype tests and the pinned GB200 production path pass. |
| I007 | Phase 6 | text_encoder | high | Lenient native loading could miss one fused source shard or an auxiliary quantization scale. | Loader completeness covers every destination, required split shard, exact fused destination, and mapped scale parameter. | encoder | resolved | Strictness regressions and pinned checkpoint loading pass. |
| I008 | Phase 6 | vae | medium | Direct decode alone did not cover production registry/config/device/strict-load behavior; the shared target is a runtime Diffusers wrapper. | VAE parity now covers direct raw decode and production `VAELoader` against official `src/zimage/autoencoder.py`. | parity | resolved | Pinned GB200 subset passed 2/2. The wrapper is an implementation exception, not the oracle; official latent normalization remains Phase 7 pipeline scope. |
| I009 | Phase 2 | reference assets | high | Adding a nonexistent clone path to `sys.path` could import another installed `zimage` package and yield false parity. | Tests now verify exact clone HEAD and module paths under `Z-Image/src`; `/Z-Image/` is ignored. | prep | resolved | Wrong SHA/path/import is a failure; absent local assets may skip. |
| I010 | Phase 6 | tokenizer | medium | Missing `apply_chat_template` previously skipped and could hide an incompatible tokenizer. | Tokenizer parity now asserts the API and exact official kwargs. | parity | resolved | Missing chat-template behavior fails. Asset-backed numerical rerun remains I003. |
| I011 | Phase 6 | transformer | high | Production real-weight strict load and full-forward parity were not verified. | Pinned 24.6 GB weights and official source were exercised on GB200. | parity | resolved | Strict load and full forward passed exactly (`max=0`, `mean=0`). |
| I012 | Phase 6 | required_components | high | A complete non-skip run of every required production component row has not been recorded at the exact current PR head after the loader/oracle corrections. | The earlier full run was at `c823ca775`; later targeted results validate individual corrections but do not constitute the exact-head closure run. | parity | open | Run every required production row non-skip in a clean worktree at the exact head before claiming Phase 7 parity closure. The optional native-Qwen r10 row is tracked separately. |

## Escape Hatches

| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
|---|---|---|---|---|---|---|
| E001 | Phase 6 | cost | Approve one standard 8-hour 4×GB200 hold to run the complete pinned component and real-weight transformer parity suite at PR #1339? | Approve `model/r9`; reuse a suitable existing allocation if available. | resolved | Author approved. Existing job `1545932` was reused, so no new hold was created. Excludes Modal, SSIM seeding/upload, and merge. |
| E002 | Phase 6 | numerical | Accept the traced Z-Image-only native encoder fp32 tail without weakening shared Qwen or bf16 gates? | Approve `model/r10`: change only `_FP32_ATOL` from `1e-4` to `2e-4`, retain `_FP32_RTOL=1e-4`, then rerun native parity at the exact resulting head. | pending | Awaiting author response for the optional native-Qwen quality row; the required production `AutoModel` path is exact and tracked independently. |

## Decisions

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-07-14 | Carry the native FastVideo pipeline, example/parity, SSIM test, and L40S seed workflow on PR #1339. | The author explicitly requested the pipeline and quality regression on the same PR rather than a follow-up. | Implementation is present; I012 and pinned full GPU parity remain open, and the generated reference must be attached for eyeballing before any HF upload. |
| 2026-07-14 | Treat the exact production `AutoModel` loader as the required pipeline text encoder and direct native-Qwen parity as a separate quality row. | `TextEncoderLoader` calls `Qwen3ForCausalLM.from_pretrained_local`, which returns body-only Transformers `AutoModel`; the pipeline does not direct-construct FastVideo-native Qwen. | The production component is exact. `model/r10` remains pending without blocking the production loader or silently changing tolerance. |
| 2026-07-13 | Use pinned `Tongyi-MAI/Z-Image@26f23eda...` as the implementation oracle; treat HF subfolders only as weight/config assets. | The official repo defines transformer, VAE, scheduler, pipeline, and loader semantics. A runtime component may inherit Diffusers without making Diffusers the reference. | Tests verify official clone SHA/import origin; docs distinguish the oracle from storage/runtime implementation details. |
| 2026-07-13 | Keep official latent scale/shift application out of the VAE component assertion. | `(latents / 0.3611) + 0.1159` is pipeline-stage behavior, while VAE component parity compares raw decode and production loading. | The Z-Image stage implements the exact formula; pinned full GPU pipeline parity remains pending. |
| 2026-07-13 | Preserve the published transformer key surface and use raw masked SDPA at SP=1. | The checkpoint already matches the official class, while padded variable-length attention needs a key mask unavailable in current distributed wrappers. | No converter is needed; runtime rejects SP>1 until a mask-aware distributed path has parity evidence. |
| 2026-07-04 | Pin both external sources and treat all earlier GPU results as historical until rerun. | Earlier evidence did not record an immutable HF revision, and source imports were not origin-checked. | README has executable clone/download commands; tests enforce the source SHA; all asset-backed parity rows are revalidation-required. |
| 2026-07-04 | Use a body-only Transformers `AutoModel` for production and a separate `AutoModel` object to materialize the official loader contract independently. | The pinned official Z-Image source is the oracle. Every FastVideo consumer needs hidden states, not full-vocabulary logits or an LM head, and comparing an object with itself is not independent evidence. | Production avoids the LM head while parity covers production loading and native implementation separately. |
| 2026-07-04 | Make fused-source and quantization-scale completeness part of native Qwen strict loading. | A loaded destination name alone did not prove that every required Q/K/V or gate/up shard and auxiliary scale arrived. | Missing split shards or mapped scale parameters fail; an exact fused destination remains valid. |
| 2026-07-04 | Reuse the existing Diffusers-backed AutoencoderKL wrapper for the Z-Image VAE component. | The target is established shared code with production-loader parity against pinned official `src/zimage/autoencoder.py`. | The wrapper remains an implementation detail for this VAE only; official Z-Image source defines pipeline behavior and numerical parity. |
| 2026-07-04 | Treat absent chat-template behavior as incompatibility, not an optional test environment. | Thinking-format prompt construction is part of the official numerical contract. | Tokenizer parity fails when `apply_chat_template` is unavailable. |
| 2026-07-04 | Record the pinned Qwen architecture as 36 blocks, hidden size 2560, intermediate size 9728, 32 attention heads, and 8 KV heads. | Earlier notes described a different architecture and could mis-size the native target. | Config/review evidence now matches `f332072aa78be7aecdf3ee76d5c247082da564a6`; expected hidden-state length is 37. |
| 2026-06-21 | Expand generated Qwen `position_ids` to `[batch_size, seq_len]`. | The rotary layer flattens positions; `[1, seq_len]` misaligned RoPE for batch sizes greater than one. | Batch-one behavior is unchanged; pinned batch-two parity still requires rerun. |
| 2026-06-21 | Reuse the shared native `Qwen3ForCausalLM` implementation rather than maintain a second Z-Image encoder. | The shared implementation supports the same config-driven GQA, fused QKV/gate-up, and hidden-state output contract. | Registry/native parity uses the shared class; architecture values come from the pinned config. |
| 2026-05-12 | Use mean and median drift checks for deep bf16 encoder parity. | Cross-kernel fused/unfused bf16 arithmetic produces a long max-error tail; fp32 and per-layer diagnostics distinguish this from structural divergence. | Existing thresholds remain provisional until the pinned 36-block snapshot is rerun. |
| 2026-05-12 | Forward the complete scheduler config except loader metadata. | Hand-picking three fields could silently omit future scheduler behavior. | Scheduler parity consumes all on-disk constructor fields and explicitly supplies Z-Image runtime overrides. |

## Handoff Notes

- The implementation oracle is the pinned official `Tongyi-MAI/Z-Image` clone,
  never Diffusers. Transformers is used for text components only because the
  official loader does so; HF component subfolders provide immutable assets.
- The required production text encoder is the exact body-only `AutoModel` path.
  `model/r10` governs only the separate native-Qwen quality row; do not change
  its tolerance or shared Qwen math without that verdict.
- Close I012 in a clean worktree before claiming Phase 7 parity closure. No
  exact-head required-component closure run is claimed by this update.
- Native pipeline/config/preset/registry/example, parity tests, and the 8-step
  1024x1024 PNG SSIM test are implemented. Hardware-free surface/stage checks
  pass; no pinned full GPU pipeline parity or L40S seed result is claimed.
- Attach the generated L40S reference for author review before any HF upload.
