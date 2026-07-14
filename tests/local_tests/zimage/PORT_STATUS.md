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
- reference_source_layout: official repo-native `src/zimage/`
- weight_layout: HF per-component subfolders; these are assets, not the reference implementation
- local_tests_readme: `tests/local_tests/zimage/README.md`

The pinned text encoder has 36 decoder blocks, hidden size 2560,
intermediate size 9728, 32 attention heads, 8 KV heads, and head dimension
128. FastVideo obtains those values from the pinned `config.json`.

## Current Phase

- phase: Phase 6 (component parity closure)
- status: awaiting_author_decision (`model/r10`), then exact-head full-suite rerun
- owner: parity
- last_updated: 2026-07-13

## Component Matrix

| Component | Type | Reuse/Port | Official Definition / Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|
| Text encoder (Qwen3 body) | text_encoder | reused production body + native implementation | Official `src/utils/loader.py` uses Transformers with typed load then device-only move; official pipeline consumes `hidden_states[-2]` | `TextEncoderLoader` -> `Qwen3ForCausalLM.from_pretrained_local` -> independent body-only `AutoModel`, plus native `fastvideo.models.encoders.qwen3.Qwen3ForCausalLM` | DONE | not_needed for current HF subfolder | PRODUCTION EXACT + BF16 PASS; FP32 PRE-NORM TOLERANCE DECISION PENDING | I001 |
| Tokenizer | tokenizer | reused | Official loader's `AutoTokenizer` from `<weights>/tokenizer/`; official thinking template and length 512 | `TokenizerLoader` | DONE | not_needed | PINNED GB200 PASS (2 passed) | none |
| VAE | vae | reused with accepted wrapper exception | Pinned official `src/zimage/autoencoder.py`; raw decode is the component boundary | Existing Diffusers-backed `fastvideo.models.vaes.autoencoder_kl.AutoencoderKL` through production `VAELoader`, plus direct raw-decode coverage | DONE | not_needed | PINNED GB200 PASS (2 passed) | I002 (pipeline-only normalization) |
| Scheduler | scheduler | reused with Z-Image extension | Pinned official `src/zimage/scheduler.py`; official pipeline sets `sigma_min=0.0` | `FlowMatchEulerDiscreteScheduler(use_reference_discrete_timesteps=True, sigma_min=0.0)` | DONE | not_needed | PINNED GB200 PASS (5 passed) | I002 |
| Transformer (`ZImageTransformer2DModel`) | dit | native port | Pinned official `src/zimage/transformer.py`; HF `transformer/` supplies weights only | `fastvideo.models.dits.zimage.ZImageTransformer2DModel` through `TransformerLoader` | DONE | not_needed; exact 521-key surface | PINNED GB200 PASS (5 passed; real-weight full forward exact) | none |
| Pipeline | pipeline | planned | Pinned `ZImagePipeline` | future PR | NOT_STARTED | depends_on_transformer | NOT_STARTED | I005 |

## Conversion State

- conversion_script: none for the reused components in this component-only PR
- converted_weights_dir: none
- reference_source_layout: official repo-native Python (`Tongyi-MAI/Z-Image`)
- hf_weight_layout: per-component subfolders; not a Diffusers implementation oracle
- transformer_conversion: not_needed; official, HF index, and native names align exactly
- transformer_key_surface: 521 tensors; sorted-key SHA256 `3a9216f208c1873b2cf06394411a53e1e95e10fae3b01dca0f7223556e47c354`
- strict_load_status: transformer real-weight strict load passed; native Qwen strictness regressions passed
- production_passthrough: body-only Qwen3 `AutoModel`, tokenizer assets
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
| Text encoder fp32 | `pytest tests/local_tests/zimage/test_zimage_encoder_parity.py::test_zimage_qwen3_encoder_parity_forward[fp32] -v -s` | FINAL OUTPUT PASS; `hidden_states[-2]` TOLERANCE DECISION PENDING, GB200, 2026-07-13 | Production wrapper/reference equality is exact. Native pre-norm has 11/48,640 batch-0 misses at `atol=rtol=1e-4`; largest failing absolute difference is `1.686811e-4`. See `model/r10`. |
| Text encoder bf16 | `pytest tests/local_tests/zimage/test_zimage_encoder_parity.py::test_zimage_qwen3_encoder_parity_forward[bf16] -v -s` | PASS, GB200, 2026-07-13 | Existing mean/median distribution thresholds pass unchanged against an independent materialization of the official loader contract. |
| Per-layer bf16 diagnostic | `pytest tests/local_tests/zimage/test_zimage_encoder_parity.py::test_zimage_qwen3_encoder_per_layer_bf16_diagnostic -v -s` | PASS, GB200, 2026-07-13 | Pinned architecture has 36 blocks and 37 hidden-state entries under the tested contract. |
| Transformer parity | `ZIMAGE_OFFICIAL_REF_DIR=<pinned-clone> pytest tests/local_tests/zimage/test_zimage_transformer_parity.py -v -s` | 5 PASSED, no skips, GB200, 2026-07-13 | Production 521-key/shape surface, tiny CPU parity, mask behavior, SP guard, real-weight strict load, and 6.15B full forward pass; real-weight output was exact (`max=0`, `mean=0`). |

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
| I001 | Phase 6 | text_encoder | medium | The native fp32 `hidden_states[-2]` tail narrowly exceeds the current `atol=1e-4` after smooth cross-kernel accumulation. | Exact layer trace: embeddings exact, layer-0 max `3.815e-6`, first misses at layer 33, largest failing boundary difference `1.686811e-4`; final RMSNorm passes. | parity | author_gate | `model/r10` asks to change only Z-Image `_FP32_ATOL` to `2e-4`; no shared-Qwen rewrite or silent tolerance change. |
| I002 | Phase 7 | scheduler/pipeline | high | Published `scheduler_config.json` omits the reference-schedule flag and official zero endpoint. | Pinned pipeline mutates `sigma_min=0.0`; parity requires both scheduler values. | pipeline | open | Future pipeline config must serialize `use_reference_discrete_timesteps=True` and `sigma_min=0.0`. |
| I003 | Phase 6 | tokenizer/text_encoder | medium | Prompt formatting and target length previously differed from the official call. | Pinned source uses thinking chat template, 512 tokens, and `hidden_states[-2]`. | parity | resolved | Corrected behavior passed the pinned GB200 run. |
| I004 | Phase 4 | transformer | high | `ZImageTransformer2DModel` was not ported. | Native config/model/registry and deterministic parity test now exist. | port | resolved | Production meta surface matches all 521 pinned HF keys; tiny full-forward parity passes exactly. |
| I005 | Phase 7 | pipeline | high | No FastVideo pipeline/config/preset/registry/example or quality regression exists. | Component-only PR scope. | pipeline | open | Separate future PR after component gates pass. |
| I006 | Phase 6 | text_encoder | high | HF passthrough loading inherited an ambient target-device context, and the earlier test harness incorrectly cast float32 RoPE buffers during bf16 placement. | HF loading now occurs outside the ambient context; typed load is followed by device-only movement, matching official `src/utils/loader.py`; production/independent outputs match exactly. | encoder | resolved | Hardware-free placement/default-dtype tests and the pinned GB200 production path pass. |
| I007 | Phase 6 | text_encoder | high | Lenient native loading could miss one fused source shard or an auxiliary quantization scale. | Loader completeness covers every destination, required split shard, exact fused destination, and mapped scale parameter. | encoder | resolved | Strictness regressions and pinned checkpoint loading pass. |
| I008 | Phase 6 | vae | medium | Direct decode alone did not cover production registry/config/device/strict-load behavior; the shared target is a runtime Diffusers wrapper. | VAE parity now covers direct raw decode and production `VAELoader` against official `src/zimage/autoencoder.py`. | parity | resolved | Pinned GB200 subset passed 2/2. The wrapper is an implementation exception, not the oracle; official latent normalization remains Phase 7 pipeline scope. |
| I009 | Phase 2 | reference assets | high | Adding a nonexistent clone path to `sys.path` could import another installed `zimage` package and yield false parity. | Tests now verify exact clone HEAD and module paths under `Z-Image/src`; `/Z-Image/` is ignored. | prep | resolved | Wrong SHA/path/import is a failure; absent local assets may skip. |
| I010 | Phase 6 | tokenizer | medium | Missing `apply_chat_template` previously skipped and could hide an incompatible tokenizer. | Tokenizer parity now asserts the API and exact official kwargs. | parity | resolved | Missing chat-template behavior fails. Asset-backed numerical rerun remains I003. |
| I011 | Phase 6 | transformer | high | Production real-weight strict load and full-forward parity were not verified. | Pinned 24.6 GB weights and official source were exercised on GB200. | parity | resolved | Strict load and full forward passed exactly (`max=0`, `mean=0`). |

## Escape Hatches

| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
|---|---|---|---|---|---|---|
| E001 | Phase 6 | cost | Approve one standard 8-hour 4×GB200 hold to run the complete pinned component and real-weight transformer parity suite at PR #1339? | Approve `model/r9`; reuse a suitable existing allocation if available. | resolved | Author approved. Existing job `1545932` was reused, so no new hold was created. Excludes Modal, SSIM seeding/upload, and merge. |
| E002 | Phase 6 | numerical | Accept the traced Z-Image-only native encoder fp32 tail without weakening shared Qwen or bf16 gates? | Approve `model/r10`: change only `_FP32_ATOL` from `1e-4` to `2e-4`, retain `_FP32_RTOL=1e-4`, then rerun the full suite at the exact resulting head. | pending | Awaiting author response; rejection leaves Phase 6 open. |

## Decisions

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-07-13 | Use pinned `Tongyi-MAI/Z-Image@26f23eda...` as the implementation oracle; treat HF subfolders only as weight/config assets. | The official repo defines transformer, VAE, scheduler, pipeline, and loader semantics. A runtime component may inherit Diffusers without making Diffusers the reference. | Tests verify official clone SHA/import origin; docs distinguish the oracle from storage/runtime implementation details. |
| 2026-07-13 | Keep official latent scale/shift application out of the component-only VAE assertion. | `(latents / 0.3611) + 0.1159` is pipeline-stage behavior, while VAE component parity compares raw decode and production loading. | Phase 7 must implement and parity-test the exact formula in a Z-Image-specific stage. |
| 2026-07-13 | Preserve the published transformer key surface and use raw masked SDPA at SP=1. | The checkpoint already matches the official class, while padded variable-length attention needs a key mask unavailable in current distributed wrappers. | No converter is needed; runtime rejects SP>1 until a mask-aware distributed path has parity evidence. |
| 2026-07-04 | Pin both external sources and treat all earlier GPU results as historical until rerun. | Earlier evidence did not record an immutable HF revision, and source imports were not origin-checked. | README has executable clone/download commands; tests enforce the source SHA; all asset-backed parity rows are revalidation-required. |
| 2026-07-04 | Use a body-only Transformers `AutoModel` for production and a separate `AutoModel` object to materialize the official loader contract independently. | The pinned official Z-Image source is the oracle. Every FastVideo consumer needs hidden states, not full-vocabulary logits or an LM head, and comparing an object with itself is not independent evidence. | Production avoids the LM head while parity covers production loading and native implementation separately. |
| 2026-07-04 | Make fused-source and quantization-scale completeness part of native Qwen strict loading. | A loaded destination name alone did not prove that every required Q/K/V or gate/up shard and auxiliary scale arrived. | Missing split shards or mapped scale parameters fail; an exact fused destination remains valid. |
| 2026-07-04 | Accept the existing Diffusers-backed AutoencoderKL wrapper only for this component-only PR. | The target is established shared code and this PR adds production-loader evidence rather than a new VAE implementation. | Exception is limited to the VAE here and is not precedent for the transformer or future full pipeline. |
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
- Wait for `model/r10`; do not change the tolerance or shared Qwen math without
  that verdict.
- After the verdict is implemented (or rejected), create a clean worktree at the
  resulting exact PR head and run the entire component suite non-skip. The prior
  full run was at `c823ca775`; later targeted subsets validate the corrections
  but do not replace an exact-head full run.
- Component parity has not closed until that full run passes. Only then start
  pipeline/config/preset/registry/example work, including the official latent
  scale/shift formula, pipeline parity, and image-quality regression.
