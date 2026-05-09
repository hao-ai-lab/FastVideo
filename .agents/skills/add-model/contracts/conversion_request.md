# Conversion Request Contract

Consumed by `../add-model-07-conversion/SKILL.md` in Phase 5 and during Phase 6
conversion retries.

Initial conversion request:

```text
model_family: <snake_case>
source_layout: diffusers | raw_official | monolithic | separate_components | mixed | custom
official_weights: <HF repo, local dir, or checkpoint file>
hf_revision: <revision | default | none>
converted_weights_dir: converted_weights/<model_family>
local_tests_readme: tests/local_tests/<model_family>/README.md
port_state_file: tests/local_tests/<model_family>/PORT_STATUS.md
components:
  - name: <transformer|vae|text_encoder|conditioner|scheduler|...>
    component_type: <dit|vae|encoder|scheduler|conditioner|upsampler|vocoder|generic>
    official_definition_files: <paths + symbols>
    official_instantiation_files: <paths + call sites + args>
    official_weight_source: <checkpoint file, prefix, subfolder, or passthrough source>
    official_keys: <path to official key/shape dump>
    fastvideo_keys: <path to FastVideo prototype key/shape dump>
    fastvideo_class: <class name>
    model_index_library: <diffusers|transformers|fastvideo|fastvideo.*>
    config_filename: <config.json|scheduler_config.json|none>
    production_loader_strictness: <strict|non_strict_with_allowed_keys|stateless>
    source_prefix_or_path: <prefix or path>
    parity_test: <component parity test path>
    prototype_concerns_or_unknowns: <short list>
```

Retry request from a component skill:

```text
conversion_retry_request:
  component: <name>
  parity_test: <path>
  failing_keys: <official and FastVideo keys, if known>
  expected_actual_shapes: <expected vs actual shapes, if known>
  source_prefix_or_path: <prefix/path implicated by the failure>
  evidence: <strict-load error, first divergent tensor, parity log excerpt>
  suspected_fix: <rename | split | fuse | skip | component bucket | config | unknown>
```

Rules:

- Phase 5 conversion requires Phase 4 official/FastVideo key dumps.
- Component skills must use the retry request instead of editing conversion
  scripts or converted weights directly.
- Conversion must update `port_state_file` with conversion status, retry history,
  strict-load status, new issues, and resolved issues.
- Conversion must validate emitted config keys through the production config
  update path and record the config filename expected by each loader.
