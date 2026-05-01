# Kandinsky-5 Port Status

## Summary

- model_family: `kandinsky5`
- workload_types: `T2V`
- official_ref: `diffusers.Kandinsky5Transformer3DModel`
- official_ref_dir: `none`
- hf_weights_path: `kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers`
- local_weights_dir: `official_weights/kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers`
- source_layout: `diffusers`
- local_tests_readme: `tests/local_tests/kandinsky5/README.md`

## Current Phase

- phase: `<TODO>`
- status: `<TODO>`
- owner: `<TODO>`
- last_updated: `<TODO: YYYY-MM-DD>`

## Component Matrix

| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| `transformer` | `dit` | `<TODO>` | `diffusers.Kandinsky5Transformer3DModel` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |

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
| transformer | `pytest tests/local_tests/kandinsky5/test_kandinsky5_lite_transformer_parity.py -v` | `<TODO>` | `<TODO>` |

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
