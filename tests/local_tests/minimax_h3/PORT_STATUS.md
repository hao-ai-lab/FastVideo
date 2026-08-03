# MiniMax H3 Port Status

## Summary

- model_family: `minimax_h3`
- workloads: T2VA, FL2VA, and Ref2VA joint video/stereo-audio generation
- implementation_reference: `https://github.com/huggingface/diffusers/pull/14355`
- checkpoint: `MiniMaxAI/MiniMax-H3`; not available in this workspace
- loading_boundary: Diffusers component folders
- current_phase: Stage 1 native components and synthetic parity
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
| Ref2VA references | `fastvideo/pipelines/basic/minimax_h3/types.py` | reference media schema | carrier implemented | request round trip | media processing starts in Stage 3 |

## Validation commands

| Scope | Command | Required evidence |
|---|---|---|
| Transformer | `PYTHONPATH=DiffusersMiniMaxH3/src pytest tests/local_tests/transformers/test_minimax_h3_transformer_parity.py -q` | activation and both output heads |
| Video VAE | `PYTHONPATH=DiffusersMiniMaxH3/src pytest tests/local_tests/vaes/test_minimax_h3_video_vae_parity.py -q` | activation, encode, decode, normalization, and tiling |
| Audio VAE | `PYTHONPATH=DiffusersMiniMaxH3/src pytest tests/local_tests/vaes/test_minimax_h3_audio_vae_parity.py -q` | activation, posterior, decode, and round trip |
| Scheduler | `PYTHONPATH=DiffusersMiniMaxH3/src pytest tests/local_tests/minimax_h3/test_minimax_h3_scheduler_parity.py -q` | independent video/audio schedules |
| Stage 1 | command in `README.md` | all synthetic CPU tests pass without skip |

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

## Evidence boundary

Stage 1 evidence is synthetic CPU parity. Real-checkpoint, CUDA, sequence-parallel, FSDP, generated-media, quality,
memory, and performance claims remain unverified.
