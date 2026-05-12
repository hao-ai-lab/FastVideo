# Component Skill Handoff Contract

Returned by `add-model-03-port-dit`, `add-model-04-port-vae`,
`add-model-05-port-encoder`, and `add-model-06-port-generic`.

```text
component: <name>
mode: prototype | parity-debug
files_changed: <model/config/export/test/readme paths>
official_files_used: <definition files, instantiation files>
prototype_key_dumps: <official path, fastvideo path, or none>
port_state_file: tests/local_tests/<model_family>/PORT_STATUS.md
concerns_or_unknowns: <remaining or newly discovered concerns>
parity_test: <path>
parity_status: scaffold_skip | debug_red | non_skip_pass | blocked
production_loader_strictness: strict | non_strict_with_allowed_keys | stateless
strict_load: pass | pass_with_documented_exclusions | blocked | not_run
pytest_output: <command + short result>
blocker: <none or exact missing dependency/weights/numeric mismatch>
conversion_retry_request: <none or failing keys/shapes/prefixes/evidence for add-model-07-conversion>
readme_updated: yes | no
next_step: phase_5_conversion | phase_5_conversion_retry | phase_6_continue | ask_user | blocked
escape_hatch: <none or block matching contracts/escape_hatch.md>
```

Rules:

- In `mode=prototype`, parity may be `scaffold_skip` or `blocked`; key dumps are
  the required artifact. Successful prototype handoff should use
  `next_step=phase_5_conversion`.
- In `mode=parity-debug`, final success requires `parity_status=non_skip_pass`.
- If conversion is implicated, return `conversion_retry_request` and do not edit
  conversion scripts or converted weights directly.
- If production loading is non-strict, list allowed missing/unexpected keys in
  the parity test or handoff and mark `strict_load=pass_with_documented_exclusions`.
- Update `port_state_file` before returning: component row, open questions,
  issues/blockers, decisions, and handoff notes.
- Return an `escape_hatch` only for user decisions, not for normal component
  implementation or parity-debug failures.
- Use `next_step=ask_user` only with an `escape_hatch` block and a matching
  `PORT_STATUS.md` row.
