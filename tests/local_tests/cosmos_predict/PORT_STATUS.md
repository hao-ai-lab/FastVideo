# cosmos_predict Port Status

## Summary

- model_family: `cosmos_predict`
- workload_types: `Text2World`
- official_ref: `https://github.com/NVIDIA/Cosmos`
- official_ref_dir: `Cosmos`
- hf_weights_path: `nvidia/Cosmos-Predict2.5-2B`
- local_weights_dir: `official_weights/cosmos_predict`
- source_layout: `raw_official`
- local_tests_readme: `tests/local_tests/cosmos_predict/README.md`

## Progress Checklist
- [x] Phase 1: Preparation (Official code staged, test framework scaffolded)
- [x] Phase 2: Parity Scaffold (Component test skeletons added)
- [x] Phase 3: Component Parity - VAE (Completed. Re-wrote architecture to use GroupNorm)
- [ ] Phase 3: Component Parity - Text Encoder
- [ ] Phase 3: Component Parity - Transformer (DiT)
- [ ] Phase 4: Full Pipeline Integration
- [ ] Phase 5: Checkpoint Conversion Script
- [ ] Phase 6: SSIM / End-to-End Evaluation

## Current Phase

- phase: `parity`
- status: `in_progress`
- owner: `parity`
- last_updated: `2026-08-27`

## Component Matrix

| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| `vae` | `vae` | `unknown` | `diffusers.models.autoencoders.autoencoder_kl_cosmos_video` | `CosmosAutoencoder()` | `fastvideo.models.vaes.cosmos_predict_vae` | `not_started` | `not_started` | `completed` | `none` |
| `text_encoder` | `encoder` | `unknown` | `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl` | `Qwen2_5_VLForConditionalGeneration()` | `fastvideo.models.encoders.cosmos_predict_text_encoder` | `not_started` | `not_started` | `completed` | `none` |
| `transformer` | `dit` | `unknown` | `diffusers.models.transformers.transformer_cosmos` | `CosmosTransformer3DModel()` | `fastvideo.models.dits.cosmos2_5.cosmos2_5_transformer` | `not_started` | `not_started` | `completed` | `none` |

## Conversion State

- conversion_script: `scripts/checkpoint_conversion/cosmos_predict_to_diffusers.py`
- converted_weights_dir: `converted_weights/cosmos_predict`
- source_layout: `raw_official`
- strict_load_status: `not_run`
- passthrough_components: `<none or list>`
- retry_history: `<none>`

## Parity Commands

| Scope | Component | Unit Parity | E2E Parity | Notes |
| :--- | :---: | :---: | :--- |
| Text Encoder | ✅ PASS | ❌ | Wrapped Qwen2.5-VL hidden states extraction |
| VAE | ✅ PASS | ❌ | Restored official component, fixed architectural differences |
| DiT | ✅ PASS | ❌ | Adjusted cross-attention, state dict keys, and condition_mask |
| Pipeline | ✅ PASS | ❌ | CosmosPredictPipeline constructed, smoke tests passing. Waiting on HF weights download for E2E parity. |

## Open Questions

| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | `How will we handle the missing HF Token for weight downloads?` | `user` | `prep` | `resolved` | `User provided the HF Token.` |

## Issues And Blockers

| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|
| I001 | `prep` | `all` | `blocker` | `nvidia/Cosmos-Predict2.5-2B is gated and requires HF authentication` | `download_hf_weights.py failed with 403 Client Error` | `user` | `resolved` | `User accepted the NVIDIA model agreement and provided an authorized token` |

## Escape Hatches

| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
|---|---|---|---|---|---|---|
| E001 | `<phase>` | `<scope/dependency/auth/cost/destructive/ambiguity/blocker>` | `<one precise question>` | `<safe recommended option>` | `<open/resolved>` | `<resolution or blank>` |

## Decisions

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| `<YYYY-MM-DD>` | `<decision>` | `<why>` | `<affected components/phases>` |

## Handoff Notes

- `<short notes for the next agent>`
