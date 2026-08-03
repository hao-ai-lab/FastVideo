# MiniMax H3 Port Status

## Summary

- model_family: `minimax_h3`
- workloads: T2VA, FL2VA, and Ref2VA joint video/stereo-audio generation
- implementation_reference: `https://github.com/huggingface/diffusers/pull/14355`
- checkpoint: `MiniMaxAI/MiniMax-H3`; not available in this workspace
- loading_boundary: Diffusers component folders
- current_phase: Stage 2 T2VA/FL2VA pipeline and synthetic parity
- status: `in_review`

## Component matrix

| Component | FastVideo target | Diffusers folder | State | Evidence | Open issues |
|---|---|---|---|---|---|
| Transformer | `fastvideo/models/dits/minimax_h3.py` | `transformer/` or `transformer_ref/` | FastVideo-style review | tiny-weight parity | I001, I002 |
| Video VAE | `fastvideo/models/vaes/minimax_h3_video.py` | `vae/` | implemented | synthetic parity | I001 |
| Audio VAE | `fastvideo/models/vaes/minimax_h3_audio.py` | `audio_vae/` | implemented | synthetic parity | I001 |
| Scheduler | `fastvideo/models/schedulers/scheduling_minimax_h3.py` | `scheduler/`, `audio_scheduler/` | implemented | independent 12/3 schedule parity | none |
| FL2VA packer | `fastvideo/pipelines/basic/minimax_h3/packing.py` | reference packing code | implemented | independent CPU contracts | none |
| Component loaders | `fastvideo/models/loader/component_loader.py` | direct component folders | implemented | strict synthetic loading | I001 |
| Request bridge | `fastvideo/api/*`; `pipeline_batch_info.py` | request fields | implemented | `last_image`, `references`, and `audio_latents` round trip | none |
| Qwen3-VL encoder | `fastvideo/models/encoders/minimax_h3_qwen3_vl.py` | `text_encoder/`, `tokenizer/`, `processor/` | isolated Transformers base-model passthrough | synthetic standard-forward contract | I001 |
| T2VA/FL2VA pipeline | `fastvideo/pipelines/basic/minimax_h3/` | `transformer/` partition | implemented, internal | private factory, offload, and four-path contracts | I001 |
| Joint AV result | existing `GenerationResult` and save path | decoded video and stereo audio | wired | tiny decoded output through typed result and real MP4 mux | I001 |
| Ref2VA references | `fastvideo/pipelines/basic/minimax_h3/types.py` | reference media schema | carrier implemented | request round trip | media processing starts in Stage 3 |

## Validation commands

| Scope | Command | Required evidence |
|---|---|---|
| Transformer | `PYTHONPATH=DiffusersMiniMaxH3/src pytest tests/local_tests/transformers/test_minimax_h3_transformer_parity.py -q` | activation and both output heads |
| Video VAE | `PYTHONPATH=DiffusersMiniMaxH3/src pytest tests/local_tests/vaes/test_minimax_h3_video_vae_parity.py -q` | activation, encode, decode, normalization, and tiling |
| Audio VAE | `PYTHONPATH=DiffusersMiniMaxH3/src pytest tests/local_tests/vaes/test_minimax_h3_audio_vae_parity.py -q` | activation, posterior, decode, and round trip |
| Scheduler | `PYTHONPATH=DiffusersMiniMaxH3/src pytest tests/local_tests/minimax_h3/test_minimax_h3_scheduler_parity.py -q` | independent video/audio schedules |
| Stage 1 | command in `README.md` | all synthetic CPU tests pass without skip |
| Conditioner | `pytest tests/local_tests/minimax_h3/test_minimax_h3_conditioner.py -q` | `BaseEncoderOutput` plus stage-owned picture presentation, tags, and layer-50 selection |
| Stage 2 pipeline | `pytest tests/local_tests/minimax_h3/test_minimax_h3_pipeline.py -q` | private factory, offload, four paths, dual denoise, decode, and mux |

## Issues and blockers

| ID | Scope | Issue | Resolution |
|---|---|---|---|
| I001 | real-checkpoint parity | The checkpoint is unavailable in this workspace. | Keep all claims synthetic until strict loading and E2E finish with real weights. |
| I002 | upstream stability | The Diffusers implementation is still a draft. | Re-audit component configs before real-checkpoint acceptance. |

## Decisions

| Decision | Rationale | Impact |
|---|---|---|
| Load the published Diffusers component folders directly. | They are the checkpoint boundary understood by FastVideo loaders. | T2VA/FL2VA selects `transformer/`; Ref2VA selects `transformer_ref/`. |
| Keep H3-specific DiT components family-local. | LTX follows the same pattern; no second model shares these contracts. | Shared modules contain only genuinely reusable primitives. |
| Preserve video/audio scheduler shifts from their own configs. | One global `flow_shift` cannot represent `12/3`. | H3 schedulers bypass the generic override. |
| Keep `last_image`, `references`, and `audio_latents` in the request bridge. | H3 needs all three inputs and later stages own media validation. | All fields survive to `ForwardBatch`. |
| Rebuild analytic RoPE state after meta initialization. | Non-persistent buffers are absent from the checkpoint. | `rope.inv_freq` is materialized in FP32. |
| Keep H3 state in `batch.extra["minimax_h3"]`. | Packed joint state is family-specific. | Stages share one typed source, then decoding removes it before executor return. |
| Reuse `GenerationResult` and the existing mux path. | H3 only needs to expose decoded video plus a 2D stereo waveform. | No family-specific result or container writer is introduced. |
| Keep the Stage 2 pipeline direct-import only. | Public registry activation belongs to Stage 4 acceptance. | No `EntryClass`, preset, or detector is added. |
| Isolate Qwen3-VL as a Transformers base-model passthrough adapter. | The encoder should expose the standard FastVideo `forward` contract only. | It returns `BaseEncoderOutput`; H3 picture presentation, tags, and layer-50 selection stay in the conditioning stage. |
| Follow FastVideo CPU-offload lifecycle. | Qwen and both VAEs are CPU-parked by default. | Each component moves only for its forward and then returns to CPU. |

## Evidence boundary

Stage 2 evidence is synthetic CPU parity plus a tiny generated MP4 container contract. Real-checkpoint, CUDA,
sequence-parallel, FSDP, model-generated media, quality, memory, and performance remain unverified.
