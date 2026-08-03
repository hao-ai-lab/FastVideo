# MiniMax H3 Port Status

## Summary

- model_family: `minimax_h3`
- workloads: T2VA, FL2VA, and Ref2VA joint video/stereo-audio generation
- implementation_reference: `https://github.com/huggingface/diffusers/pull/14355`
- checkpoint: `MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da`
- loading_boundary: Diffusers component folders
- current_phase: Stage 4 public registration and real-weight distributed acceptance
- status: `in_review`

## Component matrix

| Component | FastVideo target | Diffusers folder | State | Evidence | Open issues |
|---|---|---|---|---|---|
| Transformer | `fastvideo/models/dits/minimax_h3.py` | `transformer/` or `transformer_ref/` | FastVideo-style review | tiny parity; both real 33.12B partitions strict-load on 4 GPUs | I001, I002 |
| Video VAE | `fastvideo/models/vaes/minimax_h3_video.py` | `vae/` | implemented | synthetic parity; real T2VA, FL2VA, and Ref2VA decode | I001 |
| Audio VAE | `fastvideo/models/vaes/minimax_h3_audio.py` | `audio_vae/` | implemented | synthetic parity; three real stereo decodes | I001 |
| Scheduler | `fastvideo/models/schedulers/scheduling_minimax_h3.py` | `scheduler/`, `audio_scheduler/` | implemented | independent 12/3 schedule parity | none |
| FL2VA packer | `fastvideo/pipelines/basic/minimax_h3/packing.py` | reference packing code | implemented | independent CPU contracts | none |
| Component loaders | `fastvideo/models/loader/component_loader.py` | direct component folders | implemented | strict synthetic loading | I001 |
| Request bridge | `fastvideo/api/*`; `pipeline_batch_info.py` | request fields | implemented | `last_image`, `references`, and `audio_latents` round trip | none |
| Qwen3-VL encoder | `fastvideo/models/encoders/minimax_h3_qwen3_vl.py` | `text_encoder/`, `tokenizer/`, `processor/` | strict full-checkpoint load; base-model inference | synthetic contract and released-weight T2VA | I001 |
| T2VA/FL2VA pipeline | `fastvideo/pipelines/basic/minimax_h3/` | `transformer/` partition | public | synthetic four-path contracts; real 4-GPU T2VA and official FL2VA | I001 |
| Joint AV result | existing `GenerationResult` and save path | decoded video and stereo audio | wired | three real H.264/AAC outputs inspected | I001 |
| Ref2VA media | `fastvideo/pipelines/basic/minimax_h3/` | ordered reference schema and media | implemented | synthetic contracts; official video, soundtrack, and 44.1 kHz voice input | I001 |
| Ref2VA packer | `fastvideo/pipelines/basic/minimax_h3/packing_ref2va.py` | reference packing code | implemented | handwritten oracle and independent exact parity | none |
| Ref2VA pipeline | `fastvideo/pipelines/basic/minimax_h3/` | `transformer_ref/` partition | public | synthetic contracts plus real official Ref2VA joint AV output | I001 |
| Registry and presets | `fastvideo/registry.py`; `presets.py` | official modular manifest | public | exact class resolution, three presets, modular-manifest smoke | none |

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
| Stage 3 pipeline | command in `README.md` | ordered media, Ref2VA packing, partition isolation, condition encoding, and joint AV output |
| Stage 4 T2VA | command in `README.md` | official revision, strict component loading, FSDP/SP=4, model-generated video and stereo audio |
| Stage 4 FL2VA/Ref2VA | command in `README.md` | official media/prompts, both Transformer partitions, synchronized model-generated video and audio |

## Issues and blockers

| ID | Scope | Issue | Resolution |
|---|---|---|---|
| I001 | remaining real-checkpoint parity | T2VA, FL2VA, and Ref2VA pass; component-level Diffusers numerics and SP=1 versus SP=4 remain open. | Keep these as follow-up parity gates; they no longer block public pipeline discovery. |
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
| Register manifest-compatible public subclasses. | Official `_class_name` must resolve exactly while Ref2VA needs an explicit alternate class. | `MiniMaxH3ModularPipeline` is the default; Ref2VA selects `MiniMaxH3Ref2VAModularPipeline`. |
| Accept both Diffusers manifest filenames. | H3 publishes `modular_model_index.json`, not legacy `model_index.json`. | Local and Hub discovery validate either form without downloading weights. |
| Strict-load the released full Qwen3-VL model, then retain its base model. | Released keys use the `model.*` prefix even though H3 never calls the LM head. | Partial loads fail closed; the adapter still returns `BaseEncoderOutput`. |
| Follow FastVideo CPU-offload lifecycle. | Qwen and both VAEs are CPU-parked by default. | Each component moves only for its forward and then returns to CPU. |
| Keep Ref2VA order semantic. | Order controls both presentation labels and the shared rotary clock. | Media preparation, modality rows, and layout preserve request order. |
| Resolve `transformer_ref/` as logical `transformer`. | The two Transformer partitions are alternative workloads. | Ref2VA never loads the FL2VA partition. |
| Fall back to polyphase audio resampling. | Matching torchaudio wheels are absent from some NVIDIA development images. | 44.1 kHz reference audio still reaches the 32 kHz audio VAE deterministically. |

## Evidence boundary

Stage 4 covers T2VA plus the official FL2VA and Ref2VA requests with the pinned revision: 4x GB200, BF16 DiT,
FP32 VAEs, FSDP, SP=4, FlashAttention-2, and visually inspected joint MP4s. It establishes both Transformer
partitions and public discovery, but not component-level real-weight Diffusers numerical parity, SP=1 versus SP=4
parity, or a formal memory benchmark.
