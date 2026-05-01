# Wan2.2 Port Status

## Summary

- model_family: `wan22`
- workload_types: `I2V`
- official_ref: `<TODO>`
- official_ref_dir: `<TODO>`
- hf_weights_path: `<TODO>`
- local_weights_dir: `<TODO>`
- source_layout: `<TODO>`
- local_tests_readme: `tests/local_tests/wan22/README.md`

## Current Phase

- phase: `<TODO>`
- status: `<TODO>`
- owner: `<TODO>`
- last_updated: `<TODO: YYYY-MM-DD>`

## Component Matrix

| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| `i2v_record_schema` | `generic` | `port` | `n/a` | `n/a` | `fastvideo.dataset.dataloader.record_schema.i2v_record_creator` | `pass` | `n/a` | `non_skip_pass` | `<TODO>` |
| `transformer` | `dit` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |
| `vae` | `vae` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |
| `pipeline` | `pipeline` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |

## Conversion State

- conversion_script: `<TODO>`
- converted_weights_dir: `<TODO>`
- source_layout: `<TODO>`
- strict_load_status: `<TODO: not_run | pass | pass_with_documented_exclusions | blocked>`
- passthrough_components: `<TODO>`
- retry_history: `<TODO>`

## Parity Commands

| Scope | Command | Last Result | Notes |
|---|---|---|---|
| record schema | `pytest tests/local_tests/wan22/test_i2v_record_no_clip.py -v` | `<TODO>` | CPU-only; no weights required |

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
