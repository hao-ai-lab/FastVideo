---
name: env-var-conventions
description: Add, read, rename, or remove an environment variable in FastVideo, or change the environment-variable policy. Use before touching fastvideo/envs.py, os.environ, os.getenv, or monkeypatch.setenv in fastvideo/, and when fastvideo/tests/contract/test_env_policy.py fails.
---

# Environment Variable Conventions

## Purpose

FastVideo registers its environment variables as typed fields in
`fastvideo/envs.py`. The policy that governs them is
`docs/contributing/env_vars.md`, and the contract test
`fastvideo/tests/contract/test_env_policy.py` enforces the policy in the unit
CI lane. This skill routes an environment-variable change through that policy.
The policy doc is the single source of the rules; read it instead of relying
on a summary here.

## Prerequisites

- Read `docs/contributing/env_vars.md` in full.
- Decide whether the setting belongs in an environment variable or an argument
  (rule 5 in the policy doc). Settings that users change per deployment are
  arguments; add them through `fastvideo/fastvideo_args.py` instead.

## Inputs

| Parameter  | Required | Description                                                    |
| ---------- | -------- | -------------------------------------------------------------- |
| `change`   | Yes      | Add, read, rename, or remove a variable, or change the policy. |
| `variable` | Yes      | The variable name, with the `FASTVIDEO_` prefix.               |

## Steps

1. **Declare or edit the variable in `fastvideo/envs.py`.**
   - Pick the field type and category that the policy doc lists.
   - Write a description that states what the variable does and its units.
   - To rename, keep the old name in `deprecated_names`. To remove, add the
     name to `DEPRECATED_VARIABLES`. Update the uses in `examples/`,
     `scripts/`, `docs/`, `apps/`, and the tests.
2. **Read the variable with `envs.NAME.get()` inside a function.**
   - In tests, change the value with `envs.NAME.override(value)`, and a variable
     outside the registry with `envs.override_external(name, value)`; the
     `env_overrides` fixture keeps either until the end of the test.
   - Name a variable that only tests read `FASTVIDEO_TEST_*`.
   - Do not call `os.environ`, `os.getenv`, or `monkeypatch.setenv` for a
     FastVideo variable.
   - To set a variable that another tool reads, call `envs.set_external`,
     `envs.setdefault_external`, or `envs.unset_external`.
3. **Regenerate the table in the policy doc.**
   - Run `python fastvideo/tests/contract/test_env_policy.py`.
4. **Run the contract test.**
   - Run `pytest fastvideo/tests/contract/test_env_policy.py`.
   - When the test reports a fixed known violation, delete or lower its entry
     in `KNOWN_VIOLATIONS`. Never add an entry to `KNOWN_VIOLATIONS`.
5. **When the policy itself changes, update the policy doc and the contract
   test in the same pull request.**
   - The rules in `docs/contributing/env_vars.md`, the checks and allowlist in
     `fastvideo/tests/contract/test_env_policy.py`, and this skill must agree.

## Outputs

- A registry entry in `fastvideo/envs.py` and call sites that use
  `envs.NAME.get()`.
- A regenerated table in `docs/contributing/env_vars.md`.
- A passing `fastvideo/tests/contract/test_env_policy.py`.

## Example Usage

```
Add a FASTVIDEO_DEBUG_MY_STAGE switch that logs MyStage inputs.
```

## References

- `docs/contributing/env_vars.md`: the policy, the field types, and the
  violation kinds that the contract test reports.
- `fastvideo/envs.py`: the registry.
- `fastvideo/tests/contract/test_env_policy.py`: the contract test,
  `EXTERNAL_ALLOWLIST`, and `KNOWN_VIOLATIONS`.
