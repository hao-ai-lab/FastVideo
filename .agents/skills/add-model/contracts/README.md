# Add Model Contracts

Canonical handoff schemas for the `/add-model` workflow. When a skill needs to
send or receive structured context, use these files instead of inventing a local
schema.

| Contract | Use |
|---|---|
| `prep_handoff.md` | `add-model-01-prep` output and `/add-model` Phase 0 input. |
| `port_state.md` | Per-port `PORT_STATUS.md` file tracking progress, open questions, and issues. |
| `escape_hatch.md` | Shared pause-and-ask schema for user decisions the workflow cannot safely choose. |
| `component_context.md` | Per-component packet passed to parity, prototype, conversion, and parity-debug subagents. |
| `parity_status.md` | `add-model-02-parity` scaffold/activation status returned to `/add-model`. |
| `conversion_request.md` | Phase 5 conversion input and Phase 6 conversion retry request. |
| `conversion_handoff.md` | `add-model-07-conversion` output back to `/add-model` and component subagents. |
| `component_skill_handoff.md` | Component porting skill output in prototype or parity-debug mode. |
| `pipeline_context.md` | Phase 7 packet passed to `add-model-09-pipeline` after component parity is green. |
| `pipeline_handoff.md` | `add-model-09-pipeline` output back to `/add-model` after pipeline definition or parity-debug. |
| `final_handoff.md` | Final `/add-model` pre-handoff checklist summary. |

Rules:

- Do not omit required fields. Use `unknown` plus the search already performed
  when the value is not known yet.
- Do not include raw token values. Use env var names only.
- Keep model-specific mapping details in conversion scripts and the local tests
  README/status notes, not in generic skill docs.
- Use `next_step=ask_user` only with an `escape_hatch` block matching
  `escape_hatch.md`.
