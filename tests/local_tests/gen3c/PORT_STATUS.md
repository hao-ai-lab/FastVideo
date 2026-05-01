# GEN3C Port Status

## Summary

- model_family: `gen3c`
- workload_types: `T2V`
- official_ref: `GEN3C` (local clone)
- official_ref_dir: `GEN3C/`
- hf_weights_path: `<TODO>`
- local_weights_dir: `<TODO>`
- source_layout: `<TODO>`
- local_tests_readme: `tests/local_tests/gen3c/README.md`

## Current Phase

- phase: `<TODO>`
- status: `<TODO>`
- owner: `<TODO>`
- last_updated: `<TODO: YYYY-MM-DD>`

## Component Matrix

| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| `transformer` | `dit` | `<TODO>` | `<TODO>` | `<TODO>` | `fastvideo.models.dits.gen3c.Gen3CTransformer3DModel` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |
| `pipeline` | `pipeline` | `<TODO>` | `<TODO>` | `<TODO>` | `fastvideo.pipelines.basic.gen3c.gen3c_pipeline.Gen3CPipeline` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |

## Conversion State

- conversion_script: `<TODO: scripts/checkpoint_conversion/gen3c_to_diffusers.py>`
- converted_weights_dir: `<TODO: converted_weights/gen3c>`
- source_layout: `<TODO>`
- strict_load_status: `<TODO: not_run | pass | pass_with_documented_exclusions | blocked>`
- passthrough_components: `<TODO>`
- retry_history: `<TODO>`

## Parity Commands

| Scope | Command | Last Result | Notes |
|---|---|---|---|
| transformer | `pytest tests/local_tests/gen3c/test_gen3c.py -v` | `<TODO>` | `<TODO>` |
| pipeline (smoke) | `pytest tests/local_tests/gen3c/test_gen3c_pipeline_smoke.py -v` | `<TODO>` | random-weight smoke; full mode requires `GEN3C_DIFFUSERS_PATH` |

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
