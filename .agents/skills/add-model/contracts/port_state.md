# Port State Contract

Canonical per-port state file created during prep and updated by every
`/add-model` phase.

Path:

```text
tests/local_tests/<model_family>/PORT_STATUS.md
```

Purpose:

- Single source of truth for resumable port progress.
- Tracks component status, conversion status, parity status, open questions,
  blockers, escape hatches, and issue history.
- Lets review agents run the same setup/tests without reconstructing handoffs
  from conversation history.

Required sections:

```text
# <Model Family> Port Status

## Summary
- model_family:
- workload_types:
- official_ref:
- official_ref_dir:
- hf_weights_path:
- local_weights_dir:
- source_layout:
- local_tests_readme:

## Current Phase
- phase:
- status: not_started | in_progress | blocked | complete
- owner: orchestrator | prep | parity | conversion | component:<name> | pipeline
- last_updated:

## Component Matrix
| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|

## Conversion State
- conversion_script:
- converted_weights_dir:
- source_layout:
- strict_load_status:
- passthrough_components:
- retry_history:

## Parity Commands
| Scope | Command | Last Result | Notes |
|---|---|---|---|

## Open Questions
| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|

## Issues And Blockers
| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|

## Escape Hatches
| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
|---|---|---|---|---|---|---|

## Decisions
| Date | Decision | Rationale | Impact |
|---|---|---|---|

## Handoff Notes
- <short notes for the next agent>
```

Rules:

- Update this file whenever a phase starts, blocks, resolves an issue, or hands
  off to another skill.
- Record open questions and issues immediately. Do not leave blockers only in
  chat history or subagent responses.
- Use stable IDs: `Q001`, `Q002`, `I001`, `I002`, etc.
- Use stable escape-hatch IDs: `E001`, `E002`, etc. Link them from handoff
  `escape_hatch.state_snapshot.evidence` when returning `next_step=ask_user`.
- Do not include raw token values, machine-local cache internals, or large output
  dumps. Use repo-relative paths when possible.
- If a question or issue is resolved, keep the row and fill `Resolution` instead
  of deleting it.
