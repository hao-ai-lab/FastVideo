# Pipeline Handoff Contract

Returned by `add-model-09-pipeline` to `/add-model` after pipeline definition or
pipeline parity-debug work.

```text
pipeline_handoff:
  model_family: <snake_case>
  mode: pipeline-definition | pipeline-parity-debug
  files_changed:
    - <pipeline/config/preset/registry/stage/example/test/readme/status paths>
  official_files_used:
    - <definition/call/default source paths>
  required_config_modules:
    emitted: <list from pipeline class>
    model_index: <list from converted or source model_index.json>
    status: match | mismatch | blocked
  pipeline_class_resolution:
    model_index_class_name: <_class_name>
    entry_class_names: <registered EntryClass.__name__ values>
    status: exact_match | alias_added | blocked
  sampling_param:
    fields_added: <none or list>
    cli_fields_added: <none or list>
    unknown_kwargs_checked: yes | no | blocked
  stage_chain:
    - <stage names in execution order>
  pipeline_config:
    file: <path>
    classes: <class names>
    official_defaults_checked: yes | no | blocked
  presets:
    file: <path>
    names: <preset names>
    status: pass | blocked | not_run
  registry:
    status: pass | blocked | not_run
    detectors: <HF paths and model_index _class_name strings covered>
  smoke_test:
    path: tests/local_tests/pipelines/test_<family>_pipeline_smoke.py
    status: non_skip_pass | blocked | not_run
    pytest_output: <command + short result>
  pipeline_parity:
    path: tests/local_tests/pipelines/test_<family>_pipeline_parity.py
    status: scaffold_skip | debug_red | non_skip_pass | blocked
    pytest_output: <command + short result>
    comparison_target: <latents|decoded video|audio|joint outputs>
  example:
    path: examples/inference/basic/basic_<family>.py
    status: pass | blocked | not_run
    output: <path or none>
  readme_updated: yes | no
  port_state_updated: yes | no
  blockers: <none or exact blocker list>
  next_step: phase_10_quality_regression | return_to_phase_6 | ask_user
  escape_hatch: <none or block matching contracts/escape_hatch.md>
```

Rules:

- `pipeline-definition` may return with parity still `scaffold_skip` only if the
  exact missing dependency, weight, or call-path blocker is recorded.
- Final `/add-model` handoff requires `smoke_test.status=non_skip_pass` and
  `pipeline_parity.status=non_skip_pass`, unless the user explicitly accepts a
  documented blocker.
- If parity failure traces to a component, conversion, or strict-load issue,
  return `next_step=return_to_phase_6` and include the exact failing evidence.
- Keep `local_tests_readme` and `port_state_file` synchronized with this
  handoff before returning.
- Use `next_step=ask_user` only with an `escape_hatch` block and a matching
  `PORT_STATUS.md` row.
