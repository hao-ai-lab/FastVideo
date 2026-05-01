# Stable Diffusion 3.5 Port Status

## Summary

- model_family: `sd35`
- workload_types: `T2I`
- official_ref: `diffusers.SD3Transformer2DModel` + `diffusers.AutoencoderKL` + `transformers` text encoders
- official_ref_dir: `none`
- hf_weights_path: `stabilityai/stable-diffusion-3.5-medium`
- local_weights_dir: `official_weights/stabilityai__stable-diffusion-3.5-medium`
- source_layout: `diffusers`
- local_tests_readme: `tests/local_tests/sd35/README.md`

## Current Phase

- phase: `<TODO>`
- status: `<TODO>`
- owner: `<TODO>`
- last_updated: `<TODO: YYYY-MM-DD>`

## Component Matrix

| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| `clip_text_encoder` | `encoder` | `<TODO>` | `transformers.CLIPTextModelWithProjection` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |
| `t5_text_encoder` | `encoder` | `<TODO>` | `transformers.T5EncoderModel` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |
| `transformer` | `dit` | `<TODO>` | `diffusers.SD3Transformer2DModel` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |
| `vae` | `vae` | `<TODO>` | `diffusers.AutoencoderKL` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |
| `scheduler` | `generic` | `<TODO>` | `diffusers.FlowMatchEulerDiscreteScheduler` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |

## Conversion State

- conversion_script: `<TODO>`
- converted_weights_dir: `<TODO>`
- source_layout: `diffusers`
- strict_load_status: `<TODO: not_run | pass | pass_with_documented_exclusions | blocked>`
- passthrough_components: `<TODO>`
- retry_history: `<TODO>`

## Parity Commands

| Scope | Command | Last Result | Notes |
|---|---|---|---|
| component (omnibus) | `pytest tests/local_tests/sd35/test_sd35_component_parity.py -v -s` | `<TODO>` | covers text encoders, transformer, VAE, scheduler |

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
