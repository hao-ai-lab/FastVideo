# Conversion Handoff Contract

Returned by `../add-model-07-conversion/SKILL.md` to `/add-model` and component
parity-debug subagents.

```text
conversion_script: scripts/checkpoint_conversion/<family>_to_diffusers.py
source_layout: <diffusers|raw_official|separate_components|monolithic|mixed|custom>
converted_weights_dir: converted_weights/<model_family>
port_state_file: tests/local_tests/<model_family>/PORT_STATUS.md
components_written: <list>
passthrough_components: <list>
strict_load: pass | pass_with_documented_exclusions | blocked
component_context_updates:
  - component: <name>
    converted_component_dir: <path>
    model_index_library: <diffusers|transformers|fastvideo|fastvideo.*>
    config_file: <path or none>
    config_validation: pass | blocked | not_applicable
    mapping_notes: <prefixes, split/fuse ops, skipped keys>
    production_loader_strictness: strict | non_strict_with_allowed_keys | stateless
    strict_load: pass | pass_with_documented_exclusions | blocked | not_run
    retry_resolved: <yes | no | not_a_retry>
    concerns_or_unknowns: <remaining list>
blocked_on: <none or exact blocker>
next_step: phase_6_component_parity_debug | ask_user
escape_hatch: <none or block matching contracts/escape_hatch.md>
```

Rules:

- Include strict-load evidence for every stateful converted component.
- If a component intentionally loads non-strictly, list the exact missing or
  unexpected keys and why they are safe.
- Include the actual `model_index.json` library token, config filename, and config
  validation result for every emitted component.
- Preserve retry evidence so the requesting component subagent can resume with
  updated context.
- Keep `port_state_file` synchronized with `component_context_updates`.
- Use `next_step=ask_user` only with an `escape_hatch` block and a matching
  `PORT_STATUS.md` row.
