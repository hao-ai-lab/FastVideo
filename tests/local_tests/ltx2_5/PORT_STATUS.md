# LTX-2.5 Port Status

## Summary

- model_family: `ltx2_5`
- workload_types: `T2AV, I2AV` through the existing T2V/I2V compatibility surface
- official_ref: `https://github.com/Lightricks/LTX-2`
- official_ref_dir: `LTX-2-Reference/`
- hf_weights_path: `Lightricks/LTX-2.5`
- local_weights_dir: `Modal-backed Hugging Face cache`
- source_layout: `separate_components`
- local_tests_readme: `tests/local_tests/ltx2_5/README.md`

## Current Phase

- phase: `launch_validation`
- status: `complete`
- owner: `orchestrator`
- last_updated: `2026-08-12`

## Component Matrix

| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| LTX-2.5 transformer | dit | extend | `packages/ltx-core/src/ltx_core/model/transformer/` | `LTXModelConfigurator.from_metadata` | `fastvideo/models/dits/ltx2.py`, `fastvideo/configs/models/dits/ltx2.py` | implemented | real_split_pass | real_strict_pass | — |
| Gemma 4 unified encoder/connector | encoder | port | `packages/ltx-core/src/ltx_core/text_encoders/gemma/` | Gemma encoder configurator + prompt encoder block | `fastvideo/models/encoders/gemma.py`, `fastvideo/configs/models/encoders/gemma.py` | implemented | real_split_pass | real_pipeline_pass | — |
| Convolutional video VAE | vae | extend | `packages/ltx-core/src/ltx_core/model/video_vae/` | `VideoEncoderConfigurator` / `VideoDecoderConfigurator` from checkpoint metadata | `fastvideo/models/vaes/ltx2vae.py`, `fastvideo/configs/models/vaes/ltx2vae.py` | implemented | real_split_pass | real_pipeline_pass | — |
| Audio VAE/vocoder | vae | reuse | `packages/ltx-core/src/ltx_core/model/audio_vae/` | audio VAE/vocoder configurators from checkpoint metadata | existing LTX-2 audio decoder/vocoder | reused | real_split_pass | real_pipeline_pass | — |
| Euler ancestral sampling | generic | port | official diffusion step and distilled pipeline selection | distilled preset | `fastvideo/pipelines/basic/ltx2/stages/ltx2_denoising.py` | implemented | stateless | unit_pass | — |
| T2AV/I2AV pipeline | pipeline | extend | official one-stage and distilled pipelines | split component paths + official defaults | `fastvideo/pipelines/basic/ltx2/` | implemented | real_split_pass | real_t2av_pass | — |

## Conversion State

- conversion_script: `scripts/checkpoint_conversion/convert_ltx2_weights.py`
- converted_weights_dir: `converted_weights/ltx2_5`
- source_layout: `separate_components`
- strict_load_status: `dev_and_distilled_pass`
- passthrough_components: `packed Gemma tokenizer/assets are unpacked into text_encoder/gemma and tokenizer`
- retry_history: `I002 resolved before conversion: emitted config allowlist now preserves the 2.3/2.5 gates`

## Parity Commands

| Scope | Command | Last Result | Notes |
|---|---|---|---|
| lightweight launch suite | `pytest -q tests/local_tests/ltx2_5 tests/local_tests/transformers/test_ltx2_5_transformer_parity.py -k 'not real_weights'` | 14 passed, 7 skipped, 2 deselected | Modal L40S before official source clone; skips were reference-dependent |
| transformer source parity | `pytest -q -rs tests/local_tests/transformers/test_ltx2_5_transformer_parity.py -k 'not real_weights'` | 8 passed, 2 deselected | Same live Modal L40S after cloning official v1.2.0 source |
| expanded launch/legacy suite | `pytest -q tests/local_tests/ltx2_5 tests/local_tests/transformers/test_ltx2_5_transformer_parity.py fastvideo/tests/api/test_presets.py::TestLtx2Presets::test_ltx2_presets_registered tests/local_tests/ltx2/test_ltx2_pipeline_smoke.py::test_ltx2_typed_surface_preflight -k 'not real_weights'` | 24 passed, 2 deselected | Same live Modal L40S; includes first-frame keyframe-mask wiring and legacy preset regression checks |
| real dev strict load | `pytest -q -rs 'tests/local_tests/transformers/test_ltx2_5_transformer_parity.py::test_real_weights_load_strictly_through_production_loader[dev]'` | 1 passed | Modal L40S; official builder and production `TransformerLoader`; 28.76s |
| real distilled strict load | `pytest -q -rs 'tests/local_tests/transformers/test_ltx2_5_transformer_parity.py::test_real_weights_load_strictly_through_production_loader[distilled]'` | 1 passed | Same live Modal L40S; official builder and production `TransformerLoader`; 27.06s |
| split conversion | `pytest -q tests/local_tests/ltx2_5/test_ltx2_5_conversion.py` | pass in launch suite | Synthetic files validate component routing, packed assets, and metadata |
| Gemma/VAE/sampler/registry | `pytest -q tests/local_tests/ltx2_5 -k 'not conversion'` | pass in launch suite | Focused structural and stateless behavior coverage |
| real distilled one-stage T2AV | `basic_ltx2_5_t2av.py --variant distilled ... --steps 1 --num-gpus 2` | pass | 2xH100; joint H.264/AAC output, 9 video frames and 17 audio frames; generation 11.77s |
| real distilled two-stage T2AV | `basic_ltx2_5_t2av.py --variant distilled ... --steps 1 --num-gpus 2` | pass | 2xB200; base ancestral denoise, 2x spatial upsample, 1,632-layer LoRA, three-step refine, joint H.264/AAC output; generation 76.17s |

## Open Questions

| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | Which exact 2.5 transformer metadata fields and state-dict deltas are required beyond 2.3? | component:transformer | prototype | resolved | `use_prompt_adaln_single`, `ff_bias`, `audio_ff_bias`, and `use_keyframes_abs_pos_embedding`; defaults are True, True, True, False. |
| Q002 | Does the published convolutional video VAE instantiate the existing native VAE without architecture changes? | component:video_vae | prototype | resolved | Existing graph is reused with checkpoint-derived `decoder_base_channels=128`. |
| Q003 | Does the published audio VAE/BWE stack instantiate the existing native audio path unchanged? | component:audio_vae | prototype | resolved | Official v1.2.0 retains the LTX-2.3 audio VAE/vocoder/BWE topology; real joint video-audio pipeline generation passes. |
| Q004 | Can ancestral sampling reuse an existing FastVideo scheduler while preserving official per-step RNG? | component:scheduler | parity | resolved | Implemented the official rectified-flow Euler ancestral step in the LTX denoising stage, including seed offset and video-before-audio RNG order. |
| Q005 | Which official dev/distilled T2AV and I2AV defaults should become public FastVideo presets? | pipeline | pipeline | resolved | Dev uses 30-step CFG/STG guidance at 512x768; distilled uses eight-step ancestral denoising and optional three-step 2x refine. |

## Issues And Blockers

| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|
| I001 | parity | all | high | Real LTX-2.5 artifact loading and end-to-end generation were blocked by gated Hugging Face access. | Pinned revision `ce298b1259d61ce6c87e05154b9ad339b16f32a0` now downloads successfully in the HAO AI Lab workspace. | repository owners | resolved | Repository access was approved; both BF16 transformer variants now pass official and FastVideo strict loading. |
| I002 | conversion | transformer | high | Transformer config filtering dropped the 2.3 and 2.5 architecture gates, so converted metadata could instantiate the legacy parameter surface. | `scripts/checkpoint_conversion/convert_ltx2_weights.py::_filter_transformer_config` allowlist review. | component:conversion | resolved | Allowlist now preserves `cross_attention_adaln`, `caption_proj_before_connector`, `apply_gated_attention`, `use_prompt_adaln_single`, `ff_bias`, `audio_ff_bias`, and `use_keyframes_abs_pos_embedding`. |

## Escape Hatches

| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
|---|---|---|---|---|---|---|

## Decisions

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-08-12 | Build a complete launch inference port even if the PR is H3-sized. | Users should receive meaningful FastVideo acceleration immediately after model launch. | Dev/distilled T2AV/I2AV, native components, examples, quality coverage, and multi-GPU verification belong in the first PR. |
| 2026-08-12 | Use the official convolutional video VAE in the launch PR. | It provides authentic 2.5 inference without coupling launch support to the separate iterative DiffVAE subsystem. | DiffVAE/NATTEN/CuTe work follows independently. |
| 2026-08-12 | Keep work local until a coherent tested checkpoint, then open a non-draft PR on the fork. | CodeRabbit should review real progress while incomplete experimental fragments stay private. | Every later push is preceded by review-feedback triage. |
| 2026-08-12 | Extend the native LTX-2 transformer behind metadata-compatible gates. | Official 2.5 preserves the LTX-2.3 graph and changes only prompt AdaLN, FFN biases, and keyframe token embedding surfaces. | Existing LTX-2/2.3 configs retain byte-compatible parameter surfaces by default. |

## Handoff Notes

- Branch: `aryan/ltx-2-5-initial-support`, based on upstream `8208536cd`.
- Official source: `v1.2.0` at `d151147788a9284cca791edc6ce898007e727fe6`.
- Active Modal profile: `ai-lab`; workspace verified as `hao-ai-lab`.
- Modal policy: reuse one warm GPU-backed shell for iterative checks; do not
  create a fresh app per test command.
- Transformer prototype targets the pinned official `ltx-core` v1.2.0 graph and
  preserves FastVideo sequence-parallel and compile block boundaries.
- Official source/random transformer parity and both real BF16 transformer
  strict loads pass in Modal. Real one-stage generation passes on 2xH100, and
  the full distilled upsample/refine path passes on 2xB200 with verified H.264
  video and AAC audio streams. The tracked launch run is
  `https://wandb.ai/aryan5v-san-jose-state-university/fastvideo-ltx2-5/runs/o9ixwewk`.
