# Final Handoff Contract

Completed by `/add-model` before handing work back to the user or opening a PR.

```text
final_handoff:
  prep_handoff_complete: yes | no
  conversion_status: not_needed | pass | blocked
  components:
    - name: <component>
      reuse_or_port: reused | ported
      parity_test: <path>
      parity_status: non_skip_pass | blocked
      concerns_or_unknowns: <none or list>
  pipeline_smoke: pass | blocked | not_run
  pipeline_parity: pass | blocked | not_run
  example_status: pass | blocked | not_run
  quality_regression: added | deferred_with_reason | not_applicable
  local_tests_readme: tests/local_tests/<model_family>/README.md
  port_state_file: tests/local_tests/<model_family>/PORT_STATUS.md
  token_values_committed: no
  runtime_third_party_model_imports: none | listed_with_rationale
  blockers: <none or list>
  escape_hatch: <none or block matching contracts/escape_hatch.md>
```

Required before handoff:

- Every required component, reused or ported, has non-skip local parity PASS.
- Pipeline smoke and pipeline parity are non-skip PASS, or a blocker is explicit.
- Basic example runs and writes a non-corrupt output.
- `local_tests_readme` lists every component parity command/status/blocker.
- `port_state_file` has no unresolved blocker that is omitted from the final
  response or PR notes.
- No raw HF token values, credentials, `.env`, reference clone, or staged weight
  blobs are committed.
- If final handoff is blocked on user input, include an `escape_hatch` block and
  matching `PORT_STATUS.md` row.
