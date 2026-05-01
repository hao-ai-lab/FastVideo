# Hunyuan GameCraft Port Status

## Summary

- model_family: `gamecraft`
- workload_types: `T2V`, `I2V`
- official_ref: `Hunyuan-GameCraft-1.0` (local clone)
- official_ref_dir: `Hunyuan-GameCraft-1.0/`
- hf_weights_path: `<TODO>`
- local_weights_dir: `Hunyuan-GameCraft-1.0/weights/`
- source_layout: `<TODO>`
- local_tests_readme: `tests/local_tests/gamecraft/README.md`

## Current Phase

- phase: `<TODO>`
- status: `<TODO>`
- owner: `<TODO>`
- last_updated: `<TODO: YYYY-MM-DD>`

## Component Matrix

| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| `llama_text_encoder` | `encoder` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |
| `clip_text_encoder` | `encoder` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |
| `transformer` | `dit` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |
| `vae` | `vae` | `<TODO>` | `Hunyuan-GameCraft-1.0/hymm_sp/vae` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |
| `pipeline` | `pipeline` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |

## Conversion State

- conversion_script: `<TODO: scripts/checkpoint_conversion/gamecraft_to_diffusers.py>`
- converted_weights_dir: `<TODO: converted_weights/gamecraft>`
- source_layout: `<TODO>`
- strict_load_status: `<TODO: not_run | pass | pass_with_documented_exclusions | blocked>`
- passthrough_components: `<TODO>`
- retry_history: `<TODO>`

## Parity Commands

| Scope | Command | Last Result | Notes |
|---|---|---|---|
| encoders | `DISABLE_SP=1 pytest tests/local_tests/gamecraft/test_gamecraft_encoders_parity.py -v -s` | `<TODO>` | `<TODO>` |
| transformer | `pytest tests/local_tests/gamecraft/test_gamecraft_parity.py -v -s` | `<TODO>` | `<TODO>` |
| vae | `DISABLE_SP=1 pytest tests/local_tests/gamecraft/test_gamecraft_vae_parity.py -v -s` | `<TODO>` | `<TODO>` |
| pipeline | `DISABLE_SP=1 pytest tests/local_tests/gamecraft/test_gamecraft_pipeline_parity.py -v -s` | `<TODO>` | `<TODO>` |

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
