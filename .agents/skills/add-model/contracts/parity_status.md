# Parity Status Contract

Returned by `../add-model-02-parity/SKILL.md` to `/add-model` and later updated by
component parity-debug subagents.

```text
component_parity:
  - component: <name>
    test: tests/local_tests/<bucket>/test_<family>_<component>_parity.py
    status: scaffold_skip | debug_red | non_skip_pass | blocked
    missing: <none | fastvideo_class | converted_weights | official_import | ...>
    coverage_scope: production_loader | implementation_subcomponent | both
    official_definition_files: <paths>
    official_instantiation_files: <paths>
    concerns_or_unknowns: <short list>
pipeline_parity:
  test: <path or not-created>
  status: not_started | scaffold_skip | debug_red | non_skip_pass | blocked
local_tests_readme: tests/local_tests/<model_family>/README.md
port_state_file: tests/local_tests/<model_family>/PORT_STATUS.md
notes: <short list>
escape_hatch: <none or block matching contracts/escape_hatch.md>
```

Status meanings:

- `scaffold_skip`: test is present but skips for a specific missing dependency,
  FastVideo class, or weights.
- `debug_red`: both sides load and the test fails numerically.
- `non_skip_pass`: required before final handoff for every required component,
  including reused components.
- `blocked`: a precise missing dependency, weight, official call path, or
  component/conversion regression prevents local activation.

Coverage meanings:

- `production_loader`: FastVideo side loads through the same loader/path used by
  a pipeline.
- `implementation_subcomponent`: FastVideo side constructs classes or remaps
  tensors directly to isolate implementation behavior.
- `both`: the test covers both production loading and implementation behavior.

Use `escape_hatch` only when blocked status requires a user decision. Normal
skips or red parity should be debugged by the workflow without asking.
