# Prep Handoff Contract

Produced by `../add-model-01-prep/SKILL.md` and consumed by `/add-model` Phase 0.

```text
model_family: <snake_case>
workload_types: <T2V/I2V/V2V/T2I/or compatibility shim with rationale>
official_ref: <url or import path>
official_ref_dir: <ReferenceDir or none>
official_ref_commit: <sha or unknown>
hf_weights_path: <HF id or local path>
hf_revision: <revision or default>
local_weights_dir: official_weights/<model_family> or <local path>
source_layout: diffusers | raw_official | monolithic | separate_components | mixed | custom | unknown
model_index_class: <_class_name or none>
components_seen: <components>
needs_conversion: yes | no | unknown
hf_token_env: <env var name only>
dependency_changes: none | installed no-deps editable | installed official deps in current env | blocked on user
official_env_status: imports_ok | private_deps_need_stubs | blocked
local_tests_readme: tests/local_tests/<model_family>/README.md
port_state_file: tests/local_tests/<model_family>/PORT_STATUS.md
gitignore_entries_added: <list>
next_step: add-model | ask_user
open_questions: <short list>
escape_hatch: <none or block matching contracts/escape_hatch.md>
```

Validation:

- `official_env_status` must be `imports_ok` or `private_deps_need_stubs` before
  component parity scaffolding.
- `local_tests_readme` must exist and describe official setup, HF weights,
  dependency changes, planned parity commands, and review notes.
- `port_state_file` must exist and follow `contracts/port_state.md`.
- Prep does not go directly to conversion; `/add-model` must run component
  prototype/key-dump Phase 4 before Phase 5 conversion.
- `T2A`, `A2A`, and `AV` may be used only after `WorkloadType` supports them;
  otherwise record the compatibility shim and rationale explicitly.
- Never include HF token values.
- Use `next_step=ask_user` only with an `escape_hatch` block and a matching
  `PORT_STATUS.md` row.
