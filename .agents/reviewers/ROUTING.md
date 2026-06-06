# Reviewer Routing

How the dispatcher decides which reviewer(s) to run for a PR. The table below
mirrors the authoritative label-to-path map in `.github/mergify.yml` — when
mergify changes, keep this in sync.

## Routing rules

A PR activates a reviewer if **any** of its matchers fire. Multiple reviewers
can (and usually do) activate for a single PR.

### `model` reviewer
**Activates when any of:**
- PR has label `scope: model` or `type: new-model`
- PR title starts with `[new-model]`
- Changed paths match any of:
  - `fastvideo/models/**`
  - `fastvideo/layers/**`
  - `fastvideo/configs/models/**`
  - `fastvideo/pipelines/basic/**`  *(per-model pipeline wiring)*
  - `fastvideo/registry.py`
  - `fastvideo/tests/ssim/**`
  - `fastvideo/tests/encoders/**`
  - `fastvideo/tests/transformers/**`
  - `examples/inference/basic/**`

### `kernel` reviewer
**Activates when any of:**
- PR has label `scope: kernel` or `scope: attention`
- PR title starts with `[kernel]` or mentions `triton` / `cuda` / `fused`
- Changed paths match any of:
  - `fastvideo-kernel/**`
  - `csrc/**`
  - `fastvideo/attention/**`
  - `fastvideo-kernel/benchmarks/**`
  - `fastvideo-kernel/tests/**`

### `training` reviewer
**Activates when any of:**
- PR has label `scope: training`, `scope: data`, or `scope: distributed`
- Changed paths match any of:
  - `fastvideo/train/**`
  - `fastvideo/training/**`
  - `fastvideo/distillation/**`
  - `fastvideo/dataset/**`
  - `fastvideo/distributed/**`
  - `fastvideo/pipelines/preprocess/**`
  - `examples/train/**`
  - `examples/training/**`
  - `examples/distill/**`
  - `examples/preprocessing/**`
  - `scripts/distill/**`
  - `scripts/finetune/**`
  - `scripts/preprocess/**`
  - `fastvideo/tests/training/**`

### `general` reviewer
**Activates when any of:**
- **Always activates as a fallback** if no other reviewer matched.
- Additionally activates on its own when any of:
  - PR has label `scope: inference`, `scope: infra`, `scope: docs`, `scope: ui`, `type: ci`, `type: docs`, `type: misc`, `type: refactor`
  - Changed paths match any of:
    - `fastvideo/entrypoints/**`
    - `fastvideo/api/**`
    - `fastvideo/worker/**`
    - `fastvideo/pipelines/stages/**`
    - `fastvideo/pipelines/samplers/**`
    - `fastvideo/tests/modal/**`
    - `.github/**`
    - `.buildkite/**`
    - `.agents/**`
    - `docker/**`
    - `docs/**`
    - `ui/**`
    - `pyproject.toml`
    - `mkdocs.yml`
    - `comfyui/**`

## Edge cases

- **Multi-scope PRs** (e.g. #1087 "Nvtx tracer" touches 6 scopes): run every
  matched reviewer in parallel. Consolidation happens in the dispatcher's
  summary step.
- **`needs-rebase` PRs**: the dispatcher should refuse to review — the diff is
  stale and comments would land on moving code. Emit a single note and exit.
- **Draft PRs**: review if the author explicitly asks (via `gh pr comment`) or
  if the dispatcher is invoked with `--include-drafts`. Otherwise skip.
- **CI-only PRs** (`type: ci`): route to `general` only, even if they touch
  `fastvideo/tests/` (which would normally hit `scope: infra`).
- **Revert PRs**: skip automated review and flag for human — diffs are often
  mechanical and review focus should be "why are we reverting".

## Source of truth

The label → path map is defined in
[`.github/mergify.yml`](../../.github/mergify.yml). If you add a new scope
label there, update this file too. Consider writing a test that diffs the two.
