# Stable Audio Open 1.0 Port Status

## Summary

- model_family: `stable_audio`
- workload_types: `T2A` (text-to-audio + audio-to-audio + inpaint)
- official_ref: `Stability-AI/stable-audio-tools` (local clone)
- official_ref_dir: `stable-audio-tools/`
- hf_weights_path: `stabilityai/stable-audio-open-1.0`, `FastVideo/stable-audio-open-1.0-Diffusers`
- local_weights_dir: HF cache (no in-repo copy)
- source_layout: `monolithic` (split via FastVideo conversion into Diffusers layout)
- local_tests_readme: `tests/local_tests/stable_audio/README.md`

## Current Phase

- phase: `<TODO>`
- status: `<TODO>`
- owner: `<TODO>`
- last_updated: `<TODO: YYYY-MM-DD>`

## Component Matrix

| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| `vae` (Oobleck) | `vae` | `port` | `stable_audio_tools.models.autoencoders.{OobleckEncoder,OobleckDecoder}` | `stable_audio_tools.models.AudioAutoencoder` | `fastvideo.models.vaes.OobleckVAE` | `pass` | `pass` | `non_skip_pass` (bit-identical) | `<TODO>` |
| `transformer/DiT` | `dit` | `port` | `stable_audio_tools.models.{dit,transformer}` | `stable_audio_tools.models.DiffusionTransformer` | `fastvideo.models.dits.stable_audio.StableAudioDiT` | `pass` | `pass` | `non_skip_pass` (diff=0 on shared latents) | `<TODO>` |
| `conditioner` | `encoder` | `port` | `stable_audio_tools.models.conditioners` | `<TODO>` | `fastvideo.models.encoders.stable_audio_conditioner.StableAudioMultiConditioner` | `pass` | `pass` | `<TODO>` | `<TODO>` |
| `sampler` | `generic` | `reuse` | `k_diffusion.sampling.sample_dpmpp_3m_sde` | `<TODO>` | reuse `k_diffusion` | `pass` | `n/a` | `n/a` | `<TODO>` |
| `pipeline` | `pipeline` | `port` | `stable_audio_tools.inference.generation.generate_diffusion_cond` | `<TODO>` | `fastvideo.pipelines.basic.stable_audio.StableAudioPipeline` | `pass` | `n/a` | `non_skip_pass` (~0.015% drift) | `<TODO>` |

## Conversion State

- conversion_script: `<TODO>`
- converted_weights_dir: `FastVideo/stable-audio-open-1.0-Diffusers` (HF-published)
- source_layout: `monolithic` (single upstream `model.safetensors` split into VAE/DiT/conditioner)
- strict_load_status: `<TODO: not_run | pass | pass_with_documented_exclusions | blocked>`
- passthrough_components: `<TODO>`
- retry_history: `<TODO>`

## Parity Commands

| Scope | Command | Last Result | Notes |
|---|---|---|---|
| oobleck vae (real weights) | `pytest tests/local_tests/stable_audio/test_oobleck_vae_parity.py -v -s` | `<TODO>` | bit-identical (fp32, diff=0) |
| oobleck vae (random init structure) | `pytest tests/local_tests/stable_audio/test_oobleck_vae_official_parity.py -v -s` | `<TODO>` | architectural parity |
| pipeline (T2A) | `pytest tests/local_tests/stable_audio/test_stable_audio_pipeline_parity.py -v -s` | `<TODO>` | drift bound: < 1% / 0.05 element-wise |
| pipeline (A2A) | `pytest tests/local_tests/stable_audio/test_stable_audio_a2a_parity.py -v -s` | `<TODO>` | drift bound: < 2% / 0.05 mean |
| pipeline (inpaint, self-consistency) | `pytest tests/local_tests/stable_audio/test_stable_audio_inpaint_parity.py -v -s` | `<TODO>` | RePaint blending; SA Open 1.0 isn't inpaint-trained |
| pipeline (smoke, CPU-only) | `pytest tests/local_tests/stable_audio/test_stable_audio_pipeline_smoke.py -v` | `<TODO>` | imports + registry + preset wiring |

## Open Questions

| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO: open/resolved>` | `<TODO>` |

## Issues And Blockers

| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|
| I001 | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |

## Escape Hatches

| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
|---|---|---|---|---|---|---|
| E001 | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |

## Decisions

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| `<TODO: YYYY-MM-DD>` | `<TODO>` | `<TODO>` | `<TODO>` |

## Handoff Notes

- `<TODO: short notes for the next agent>`
