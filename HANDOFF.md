# MiniMax H3 FastVideo port: full handoff

## Mission

Port MiniMax H3 into FastVideo from the architecture and behavior exposed by [Diffusers PR #14355](https://github.com/huggingface/diffusers/pull/14355). The requested workloads are T2VA, FL2VA, and Ref2VA joint video/stereo-audio generation. The implementation must be native to FastVideo's architecture and style rather than a copy of the reference pipeline structure.

The current branch is `feat/kaiqin/minimax-h3`, tracking the matching `h1yori233` remote branch. Verify live Git state before acting.

## Authoritative artifacts

The user edited the design documentation during development. The current workspace versions are authoritative; do not restore older drafts from conversation history.

- Frontier interpretation and integration phases: `<workspace>/docs/design/minimax_h3_frontier_and_integration_plan.md`
- Current phase, component matrix, decisions, blockers, and evidence boundary: `<workspace>/tests/local_tests/minimax_h3/PORT_STATUS.md`
- Test inventory and commands for Stages 1-3: `<workspace>/tests/local_tests/minimax_h3/README.md`
- Deliberately concise development pitfalls: `<workspace>/docs/design/minimax_h3_porting_pitfalls.md`
- Reference implementation: <https://github.com/huggingface/diffusers/pull/14355>

Read these instead of recreating the prior research, model comparison, architecture plan, or phase breakdown.

## Work completed so far

### Research and planning

- The repository was updated from the user's FastVideo remote and the feature branch was created.
- Diffusers PR #14355 was studied in detail, including H3's shared packed audio/video Transformer, dual schedulers, Qwen3-VL conditioning, two Transformer weight partitions, VAE geometry, RNG order, and Ref2VA media clock.
- The frontier document explains how H3 differs from Wan and LTX and defines the staged FastVideo integration plan.
- The planning commit is `d1e7fb7a`; use the document, not this handoff, for technical detail.

### Stage 1: native components and contracts

Stage 1 is implemented. It introduced the native H3 Transformer, video VAE, audio VAE, scheduler, FL2VA packer, direct component-folder loading, API/request fields, and synthetic parity tests.

Important settled contracts:

- Video and audio scheduler shifts remain independently `12/3`; FastVideo's global `flow_shift` must not override them.
- DiT FP32 islands and both FP32 VAEs are intentional.
- H3-specific DiT components remain family-local, following the LTX precedent; only genuinely reusable primitives belong in shared modules.
- `last_image`, `audio_latents`, and `references: list[Any]` remain in the typed request path.
- The loading boundary is the published Diffusers component folders.

### Stage 2: private T2VA/FL2VA pipeline

Stage 2 is implemented.

It added the private composed T2VA/FL2VA pipeline, a thin Qwen3-VL base-model adapter, H3-owned conditioning/presentation, first/last-frame VAE conditioning, dual-modality denoising, decoding, existing `GenerationResult` integration, stereo output, mux coverage, and FastVideo CPU-offload lifecycles.

The repository's other Qwen-family components were reviewed. H3 keeps a small isolated adapter because its required Qwen3-VL base-model boundary is not provided by an existing FastVideo encoder. The adapter returns the standard `BaseEncoderOutput`; H3 picture presentation, tags, multimodal positions, and layer-50 selection stay in the conditioning stage.

Stage 2 remains direct-import only. It adds no public registry entry, detector, or preset.

### Stage 3: private Ref2VA pipeline

Stage 3 is implemented.

It adds:

- immutable typed references plus deferred preparation for image, video, standalone audio, and video soundtracks;
- ordered Qwen3-VL presentation and reference video/audio VAE conditioning;
- an independent Ref2VA packer with `[text | ordered references | target audio | target video]` layout;
- sequential reference-time accumulation, paired soundtrack/video origins, row tags, indices, and rotary positions;
- a separate private `MiniMaxH3RefPipeline` that resolves physical `transformer_ref/` weights as logical `transformer` and never loads both Transformer partitions;
- shared target latent/timestep/denoise/decode stages without merging FL2VA and Ref2VA packing semantics;
- complete DiT CPU-offload and layerwise-manager release behavior;
- synthetic media, packer, loader-isolation, RNG, conditioning, and end-to-end Ref2VA tests;
- concise additions to the status, test, and pitfalls documents.

Two shared changes in the current diff are deliberate and covered by focused tests:

- an explicit `PipelineConfig` instance is authoritative before registry detection;
- `_extra_config_module_map` consistently maps a logical component role to its physical component folder and loads it under the logical role. This also preserves the existing MatrixGame3 `vae -> light_vae` intent.

Inspect `git diff` for exact implementation details; do not duplicate the diff into another document.

## User-directed design guardrails

- Prioritize FastVideo style and architecture over mirroring Diffusers code organization.
- Keep the diff clean and minimal; do not introduce speculative abstractions.
- Preserve `last_image`, `audio_latents`, and `references` now rather than postponing them.
- Keep media I/O and validation in pipeline stages, not in immutable request carriers.
- Keep FL2VA and Ref2VA packers independent because their clocks differ.
- Keep the pitfalls document short: one sentence per durable failure mode, without hashes, dates, or low-level journals.
- Do not publicly activate H3 before the real-weight and distributed acceptance phase.
- Do not present synthetic parity, startup, or a tiny forward as real-checkpoint or CUDA acceptance.

## Current Git and workspace safety
 
- Do not stage, commit, push, rebase, or start Stage 4 without a new user request.

Always begin with `git status --short --branch` and inspect both staged and unstaged diffs. The working tree is intentionally dirty.
