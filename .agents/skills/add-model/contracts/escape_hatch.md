# Escape Hatch Contract

Canonical pause-and-ask schema for `/add-model` skills. Use this when the next
action requires user input instead of autonomous debugging.

```text
escape_hatch:
  needs_user_input: yes | no
  decision_type: scope | dependency | auth | cost | destructive | ambiguity | blocker
  question: <one precise question>
  recommended_option: <safe recommended choice>
  options:
    - <option + consequence>
  safe_default: <what the agent will do after approval, or none>
  blocked_until_answered: yes | no
  state_snapshot:
    phase: <phase or skill mode>
    files_changed:
      - <paths>
    command_or_test: <last relevant command, or not_run>
    evidence: <short logs, paths, error text, or blocker ID>
```

Use `needs_user_input=no` when the handoff is green or the next step is already
specified by the workflow.

Ask the user only for decisions the workflow cannot safely choose:

- product or PR scope changes, including dropping a modality, output head, or
  variant;
- core dependency changes, version pin changes, or installing untrusted/private
  dependencies;
- auth setup for gated repos, using env var names only and never token values;
- large downloads, publishing weights, SSIM/reference uploads, or GPU-heavy work
  where cost/runtime approval is needed;
- destructive file/git operations, overwriting existing clones/weights, or
  deleting staged assets;
- ambiguous official sources of truth with incompatible behavior;
- accepting a known blocker, loosening parity/quality tolerances, or shipping
  without required non-skip parity.

Do not ask for normal recoverable failures:

- missing imports, missing local paths, skipped tests, failing parity, conversion
  mapping errors, strict-load failures, format/lint failures, or implementation
  bugs covered by the skill workflow.

Before returning `next_step=ask_user`, update
`tests/local_tests/<model_family>/PORT_STATUS.md` with the blocker/question ID,
include the exact evidence, and provide one recommended option plus at most three
alternatives.
