# <Model Family> Port Status

## Summary

- model_family: `<model_family>`
- workload_types: `<T2V/I2V/V2V/T2I/or compatibility shim with rationale>`
- official_ref: `<url or import path>`
- official_ref_dir: `<ReferenceDir or none>`
- hf_weights_path: `<HF repo id/url or local path>`
- local_weights_dir: `<official_weights/model_family or local path>`
- source_layout: `<diffusers/raw_official/monolithic/separate_components/mixed/custom/unknown>`
- local_tests_readme: `tests/local_tests/<model_family>/README.md`

## Current Phase

- phase: `prep`
- status: `in_progress`
- owner: `prep`
- last_updated: `<YYYY-MM-DD>`

## Component Matrix

| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| `<component>` | `<dit/vae/encoder/generic>` | `<unknown/reuse/port>` | `<path + symbols>` | `<path + args>` | `<target files>` | `<not_started/in_progress/pass/blocked>` | `<not_started/pass/blocked>` | `<not_started/scaffold_skip/debug_red/non_skip_pass/blocked>` | `<none or IDs>` |

## Conversion State

- conversion_script: `scripts/checkpoint_conversion/<model_family>_to_diffusers.py`
- converted_weights_dir: `converted_weights/<model_family>`
- source_layout: `<diffusers/separate_components/monolithic/mixed/custom/unknown>`
- strict_load_status: `not_run`
- passthrough_components: `<none or list>`
- retry_history: `<none>`

## Parity Commands

| Scope | Command | Last Result | Notes |
|---|---|---|---|
| component | `pytest tests/local_tests/<bucket>/test_<model_family>_<component>_parity.py -v -s` | `not_run` | `<notes>` |
| pipeline | `pytest tests/local_tests/pipelines/test_<model_family>_pipeline_parity.py -v -s` | `not_run` | `<notes>` |

## Open Questions

| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | `<question>` | `<owner>` | `<phase>` | `<open/resolved>` | `<resolution or blank>` |

## Issues And Blockers

| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|
| I001 | `<phase>` | `<component or all>` | `<low/medium/high/blocker>` | `<issue>` | `<logs/paths/commands>` | `<owner>` | `<open/resolved>` | `<resolution or blank>` |

## Escape Hatches

| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
|---|---|---|---|---|---|---|
| E001 | `<phase>` | `<scope/dependency/auth/cost/destructive/ambiguity/blocker>` | `<one precise question>` | `<safe recommended option>` | `<open/resolved>` | `<resolution or blank>` |

## Decisions

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| `<YYYY-MM-DD>` | `<decision>` | `<why>` | `<affected components/phases>` |

## Handoff Notes

- `<short notes for the next agent>`
