# Shared Add-Model Rules

These rules apply to every `add-model` related skill: prep, parity, conversion,
component porting, pipeline, and the main `/add-model` orchestrator.

## Token And Auth Safety

- Never accept, print, echo, log, hard-code, or commit raw HF token values.
- Refer only to token environment variable names: `HF_TOKEN`,
  `HUGGINGFACE_HUB_TOKEN`, or `HF_API_KEY`.
- Scripts may read those environment variables but must not print their values.
- Ask for auth setup only by env var name. Do not ask the user to paste a token.
- Read scope is needed for gated repos during conversion/load. Write scope is
  needed for publishing converted weights or seeding generated references.

## Shared State Files

- `tests/local_tests/<model_family>/README.md` is the reviewer-facing setup and
  verification log. Keep it current with setup commands, dependency blockers,
  parity commands, conversion commands, and pass/blocker status.
- `tests/local_tests/<model_family>/PORT_STATUS.md` is the per-port state file.
  It must follow `../contracts/port_state.md` and keep stable `Q###`, `I###`, and
  `E###` IDs.
- Keep resolved questions/issues in `PORT_STATUS.md` with the resolution instead
  of deleting them.
- Before returning a handoff, update both state files when the skill changed
  setup, tests, conversion, parity status, blockers, or decisions.
- Do not include raw tokens, non-reproducible absolute cache paths, large
  generated outputs, `.env`, credentials, or anything matching `*secret*`.

## Escape Hatches

Continue autonomously for recoverable setup, implementation, conversion,
strict-load, smoke, parity-debug, lint, or test failures. Stop and ask the user
only when the next action requires a product, cost, safety, auth, dependency, or
scope decision the workflow cannot safely choose.

Use `../contracts/escape_hatch.md` whenever returning `next_step=ask_user`.

Ask for user input only for:

- scope changes, such as dropping a modality, output head, variant, component, or
  public mode;
- core dependency changes, untrusted/private dependency installs, or version pin
  changes;
- auth setup for gated repos, using env var names only;
- large downloads, publishing weights, SSIM/reference uploads, or GPU-heavy work
  where cost/runtime approval is needed;
- destructive file/git operations, overwriting existing clones/weights, or
  deleting staged assets;
- incompatible official sources of truth where no reference can be chosen from
  published weights and docs;
- accepting a blocker, loosening tolerances, using shape-only substitutes, or
  shipping without required non-skip parity.

Do not ask for normal recoverable failures: missing imports, missing local paths,
skipped tests, failing parity, conversion mapping bugs, strict-load errors,
format/lint failures, smoke failures, registry import issues, example failures,
or implementation bugs covered by the phase workflow.

Before asking:

- update `PORT_STATUS.md` with an `E###` escape-hatch row plus any linked `Q###`
  or `I###` row;
- include exact evidence: command, path, short error text, parity or strict-load
  excerpt, or blocker ID;
- provide one recommended option and at most three alternatives;
- set the relevant handoff `next_step=ask_user` and include the `escape_hatch`
  block.

Skill-specific escape-hatch sections may add extra examples, but they must not
weaken these shared rules.

## Production Boundary

- No runtime `from diffusers import <model class>` or
  `from transformers import <model class>` in `fastvideo/` production code.
- Components that own weights or numerical behavior must be FastVideo-native
  unless the user explicitly accepts a documented lazy-wrapper exception.
- Allowed third-party runtime exceptions are tokenizers and pure data utilities
  when they match existing project patterns.
- Tests may import diffusers/transformers as parity references.
- Production comments explain why, not what or provenance. Avoid narrative
  comments like `vendored from`, `matches upstream`, `REVIEW`, or session-history
  commentary.

## Verification Semantics

- A committed local test may skip when clones, weights, or private deps are absent
  so CI and other contributors are not blocked.
- On the porter's machine, a skip is not a pass. Fix the missing import, weights,
  or path before claiming verification.
- New ports require local non-skip parity for required components and pipeline
  parity when a pipeline is in scope.
- Smoke tests prove loadability only. They are not a substitute for numerical
  component or pipeline parity.
