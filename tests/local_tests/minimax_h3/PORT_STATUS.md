# MiniMax H3 Port Status

## Summary

- model_family: `minimax_h3`
- workload_types: T2V/I2V-compatible joint audio/video output; pipeline deferred to Stage 2
- official_ref: `https://github.com/huggingface/diffusers/pull/14355`
- official_ref_dir: `DiffusersMiniMaxH3/`
- official_ref_commit: `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`
- hf_weights_path: `MiniMaxAI/MiniMax-H3` as described by the draft; unavailable locally
- local_weights_dir: unavailable
- source_layout: `raw_official`
- local_tests_readme: `tests/local_tests/minimax_h3/README.md`

## Current phase

- phase: engineering Stage 1, contracts/native components/synthetic parity
- status: `complete_synthetic`
- owner: `orchestrator`
- last_updated: `2026-08-02`

## Component matrix

| Component | Type | Reuse/Port | Official definition | Official instantiation | FastVideo target | Prototype | Conversion | Parity | Open issues |
|---|---|---|---|---|---|---|---|---|---|
| Transformer | dit | port | `src/diffusers/models/transformers/transformer_minimax_h3.py` | upstream tiny transformer model test | `fastvideo/models/dits/minimax_h3.py`; matching arch config | complete | synthetic CLI pass | non_skip_pass | I001, I002 |
| Video VAE | vae | port | `src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py` | upstream tiny video-VAE model test | `fastvideo/models/vaes/minimax_h3_video.py`; matching arch config | complete | synthetic CLI pass | non_skip_pass | I001, I002 |
| Audio VAE | vae | port | `src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py` | upstream tiny audio-VAE model test | `fastvideo/models/vaes/minimax_h3_audio.py`; matching arch config | complete | identity FP32 synthetic CLI pass | non_skip_pass | I001, I002 |
| Scheduler | scheduler | port | `src/diffusers/schedulers/scheduling_minimax_h3.py` | `scheduler` shift 12; `audio_scheduler` shift 3 | `fastvideo/models/schedulers/scheduling_minimax_h3.py` | complete | stateless role configs pass | non_skip_pass | I002 |
| FL2VA packer | generic | port | `src/diffusers/modular_pipelines/minimax_h3/packing.py` | prepare-layout/latents/denoise blocks | `fastvideo/pipelines/basic/minimax_h3/packing.py` | complete | not_applicable | non_skip_pass | I002 |
| Converter | generic | port | `scripts/convert_minimax_h3_to_diffusers.py` | raw component directory layout | `scripts/checkpoint_conversion/convert_minimax_h3_to_diffusers.py` | complete | synthetic three-component CLI pass | non_skip_pass | I001, I002 |
| Component loaders | loader | extend | current FastVideo registry/meta loader | `transformer`, `vae`, `audio_vae`, `scheduler`, `audio_scheduler` | `fastvideo/models/loader/component_loader.py` | complete | not_applicable | non_skip_pass | I001 |
| Request bridge | API | minimal extension | current FastVideo typed request path | `GenerationRequest -> SamplingParam -> ForwardBatch` | `fastvideo/api/*`; `pipeline_batch_info.py` | complete | not_applicable | non_skip_pass | none |

## Conversion state

- conversion_script: `scripts/checkpoint_conversion/convert_minimax_h3_to_diffusers.py`
- converted_weights_dir: `converted_weights/minimax_h3`
- source_layout: `raw_official`
- strict_load_status: `synthetic_transformer_video_audio_pass`; real weights not run
- passthrough_components: Qwen3-VL/tokenizer/processor deferred to Stage 2
- retry_history: none

## Parity commands

| Scope | Command | Last result | Notes |
|---|---|---|---|
| Transformer | `pytest tests/local_tests/transformers/test_minimax_h3_transformer_parity.py -q` | 4 passed | synthetic tiny random weights; activation and both output heads |
| Video VAE | `pytest tests/local_tests/vaes/test_minimax_h3_video_vae_parity.py -q` | 3 passed | activation, encode/decode, normalization, tiling |
| Audio VAE | `pytest tests/local_tests/vaes/test_minimax_h3_audio_vae_parity.py -q` | 1 passed | activation, posterior, direct decode, round trip |
| Scheduler | `pytest tests/local_tests/minimax_h3/test_minimax_h3_scheduler_parity.py -q` | 5 passed | CPU pinned-reference parity |
| Packing/conversion/API/loaders | `pytest tests/local_tests/minimax_h3 -q` | 25 passed | CPU contracts, CLI fixtures, actual loader paths |
| Complete Stage 1 | command in `README.md` | 33 passed | all non-skip CPU tests |
| Pipeline | not created | not_started | Stage 2 |

## Open questions

| ID | Question | Owner | Needed by phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | Does the eventual accessible checkpoint preserve the draft component layout, configs, and mixed dtypes? | upstream | Stage 4 | open | Re-audit the exact authorized revision before real conversion. |
| Q002 | Are FL2VA and Ref2VA shared components byte-identical in the accessible release? | upstream | Stage 3/4 | open | Compare component hashes before packaging workflow snapshots. |

## Issues and blockers

| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|
| I001 | Stage 1/4 | all stateful components | blocker for Stage 4 only | No usable real checkpoint is available in this workspace. | Current design constraint; `official_weights/minimax_h3` absent. | upstream | open | Stage 1 uses non-skip synthetic parity; never promote it to a real-weight claim. |
| I002 | Stage 1 | all | medium | Implementation reference is an unmerged draft and may change. | Diffusers PR #14355 pinned at `abc5e9bf…`. | orchestrator | open | Pin every synthetic test and re-audit before Stage 4. |

## Escape hatches

| ID | Phase | Decision type | Question | Recommended option | Status | Resolution |
|---|---|---|---|---|---|---|
| none | Stage 1 | none | No user decision required for synthetic implementation. | Continue within the current document. | resolved | User explicitly selected current-document Stage 1. |

## Decisions

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-08-02 | Treat the edited design document as the implementation source of truth. | Explicit user instruction. | One branch, four engineering stages; stop after Stage 1 in this turn. |
| 2026-08-02 | Accept non-skip synthetic parity for Stage 1 while keeping real-weight parity blocked. | Real weights are unavailable and the document explicitly defines this evidence tier. | No quality, memory, speed, or real loader claim. |
| 2026-08-02 | Keep Ref2VA packing, Qwen3-VL conditioning, pipelines, and registry activation out of Stage 1. | They are assigned to later stages in the current document. | Stage 1 remains component/contract-only. |
| 2026-08-02 | Preserve the video/audio scheduler shifts from their own configs. | A generic global flow shift cannot represent H3's simultaneous 12/3 roles. | `SchedulerLoader` bypasses its global override for `MiniMaxH3Scheduler`. |
| 2026-08-02 | Rebuild analytic RoPE state after meta construction. | Non-persistent buffers are absent from checkpoints but cannot remain on the meta device. | The actual mixed-dtype loader fixture materializes FP32 `rope.inv_freq`. |

## Handoff notes

- Stage 1 synthetic acceptance is complete; real-weight, CUDA, SP>1, FSDP inference, media, quality, and performance
  evidence remains explicitly deferred to Stage 4.
- Do not commit or push in this turn; leave the finished diff staged for user review.
- Preserve independent reference/FastVideo packing code paths.
