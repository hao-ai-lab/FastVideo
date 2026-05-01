# LTX-2 Port Status

## Summary

- model_family: `ltx2`
- workload_types: `T2V` (video + optional audio modality)
- official_ref: `Lightricks/LTX-2` (local clone)
- official_ref_dir: `LTX-2/`
- hf_weights_path: `Lightricks/LTX-2`, `FastVideo/LTX2-base`, `FastVideo/LTX2-Distilled-Diffusers`
- local_weights_dir: `<TODO>`
- source_layout: `<TODO>`
- local_tests_readme: `tests/local_tests/ltx2/README.md`

## Current Phase

- phase: `<TODO>`
- status: `<TODO>`
- owner: `<TODO>`
- last_updated: `<TODO: YYYY-MM-DD>`

## Component Matrix

| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| `gemma_text_encoder` | `encoder` | `<TODO>` | `<TODO>` | `<TODO>` | `fastvideo.models.encoders.gemma.LTX2GemmaTextEncoderModel` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |
| `transformer` | `dit` | `<TODO>` | `<TODO>` | `<TODO>` | `fastvideo.configs.models.dits.LTX2VideoConfig` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |
| `vae_video` | `vae` | `<TODO>` | `<TODO>` | `<TODO>` | `fastvideo.models.vaes.ltx2vae.{LTX2VideoEncoder,LTX2VideoDecoder}` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |
| `vae_audio` | `vae` | `<TODO>` | `<TODO>` | `<TODO>` | `fastvideo.models.audio.ltx2_audio_vae.{LTX2AudioEncoder,LTX2AudioDecoder,LTX2Vocoder}` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |
| `pipeline` | `pipeline` | `<TODO>` | `<TODO>` | `<TODO>` | `fastvideo.pipelines.basic.ltx2.*` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |

## Conversion State

- conversion_script: `<TODO: scripts/checkpoint_conversion/ltx2_to_diffusers.py>`
- converted_weights_dir: `<TODO: converted_weights/ltx2>`
- source_layout: `<TODO>`
- strict_load_status: `<TODO: not_run | pass | pass_with_documented_exclusions | blocked>`
- passthrough_components: `<TODO>`
- retry_history: `<TODO>`

## Parity Commands

| Scope | Command | Last Result | Notes |
|---|---|---|---|
| gemma encoder | `pytest tests/local_tests/ltx2/test_ltx2_gemma_parity.py -v -s` | `<TODO>` | `<TODO>` |
| gemma connector only | `pytest tests/local_tests/ltx2/test_ltx2_gemma_encoder.py -v -s` | `<TODO>` | `<TODO>` |
| dit (video) | `pytest tests/local_tests/ltx2/test_ltx2.py -v -s` | `<TODO>` | `<TODO>` |
| dit (audio modality) | `pytest tests/local_tests/ltx2/test_ltx2_audio.py -v -s` | `<TODO>` | `<TODO>` |
| vae video | `pytest tests/local_tests/ltx2/test_ltx2_vae.py -v -s` | `<TODO>` | `<TODO>` |
| vae video (official path) | `pytest tests/local_tests/ltx2/test_ltx2_vae_official.py -v -s` | `<TODO>` | `<TODO>` |
| vae audio | `pytest tests/local_tests/ltx2/test_ltx2_audio_vae.py -v -s` | `<TODO>` | `<TODO>` |
| pipeline (smoke) | `pytest tests/local_tests/ltx2/test_ltx2_pipeline_smoke.py -v -s` | `<TODO>` | `<TODO>` |
| registry | `pytest tests/local_tests/ltx2/test_ltx2_registry.py -v` | `<TODO>` | `<TODO>` |

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
