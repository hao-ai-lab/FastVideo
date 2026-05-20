# Component Context Contract

Canonical per-component packet passed from `/add-model` to parity, prototype,
conversion, and parity-debug subagents.

```text
component_context:
  model_family: <snake_case>
  component: <name>
  component_type: <dit|vae|encoder|scheduler|conditioner|upsampler|vocoder|generic>
  mode: parity-scaffold | prototype | parity-debug
  official_ref_dir: <path or import path>
  official_definition_files:
    - path: <repo-relative or absolute path in official repo>
      symbols: <class/function names>
      notes: <layer graph, output contract, state-dict owner>
  official_instantiation_files:
    - path: <repo-relative or absolute path in official repo>
      symbols: <factory/pipeline/config names>
      args: <constructor args, config values, runtime flags>
  official_weight_source: <checkpoint file, subfolder, prefix, or passthrough source>
  fastvideo_target_files:
    - fastvideo/models/<bucket>/<file>.py
    - fastvideo/configs/models/<bucket>/<file>.py
  local_tests_readme: tests/local_tests/<model_family>/README.md
  port_state_file: tests/local_tests/<model_family>/PORT_STATUS.md
  parity_test: tests/local_tests/<bucket>/test_<family>_<component>_parity.py
  prototype_key_dumps:
    official: converted_weights/<family>/_mapping/<component>_official_keys.json | planned | unknown
    fastvideo: converted_weights/<family>/_mapping/<component>_fastvideo_keys.json | planned | unknown
  conversion:
    script: scripts/checkpoint_conversion/<family>_to_diffusers.py | not_created | not_needed | unknown
    converted_component_dir: converted_weights/<family>/<component> | not_created | not_needed | unknown
    model_index_library: <diffusers|transformers|fastvideo|fastvideo.*|unknown|none>
    config_file: <config.json|scheduler_config.json|none|unknown>
    mapping_notes: <key prefixes, split/fuse concerns, skipped keys, not_created, not_needed, or unknown>
    production_loader_strictness: <strict|non_strict_with_allowed_keys|stateless|unknown>
    strict_load: <not_run | pass | pass_with_documented_exclusions | blocked>
  concerns_or_unknowns:
    - <prototype mismatch, ambiguous arg, missing op, dtype concern, output head, etc.>
```

Rules:

- If any required path is unknown, pass `unknown` plus the exact search already
  performed.
- Do not silently omit ambiguous official files, instantiation args, or prototype
  concerns.
- For reused components, still fill every field and set `fastvideo_target_files`
  to the reused class/config.
- In `mode=parity-scaffold`, prototype and conversion fields may be `planned`,
  `not_created`, `not_needed`, or `unknown`; do not invent paths or statuses that
  do not exist yet.
- Update `port_state_file` when concerns, issues, conversion status, or parity
  status change.
