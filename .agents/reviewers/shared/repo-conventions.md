# Repo conventions every reviewer should know

Conventions that cut across scopes. If a PR violates one of these, it's fair
game regardless of which reviewer is running.

## Build & dev

- Editable install: `uv pip install -e .[dev]`.
- Hooks: `pre-commit install --hook-type pre-commit --hook-type commit-msg`.
- Full lint/format/type/spelling: `pre-commit run --all-files` (yapf, ruff,
  mypy, codespell). **Reviewers do not duplicate lint feedback** — pre-commit
  runs in CI and the failing PR is already marked.
- Line length target: **80** (see `pyproject.toml`). One exception: `[feat] pre-commit support 120 col num` (#1167) expanded the cap — verify current config before flagging.

## Commit / PR style

- PR title **must** match `(?i)^\[(feat|feature|bugfix|fix|refactor|perf|ci|doc|docs|misc|chore|kernel|new.?model)\]` — mergify enforces this.
- PR body should follow `.github/PULL_REQUEST_TEMPLATE.md` (Purpose / Changes / Test Plan / Test Results / Checklist).
- The PR template has a checkbox: **"For model/pipeline changes, also check I verified SSIM regression tests pass"** — reviewers should hold model PRs to this.

## Python style

- 3.10+, 4-space indents.
- `snake_case` functions/files, `PascalCase` classes, `UPPER_SNAKE_CASE`
  constants.
- Keep imports explicit. No star imports.
- Prefer typed signatures; `mypy` is part of pre-commit.

## Testing layout

- Package-level tests: `fastvideo/tests/` (domain-specific subdirs:
  `ssim/`, `encoders/`, `training/`, `transformers/`, `workflow/`).
- Top-level: `tests/local_tests/` for component/local checks.
- Kernel tests: `fastvideo-kernel/tests/`.
- SSIM tests are GPU-heavy and run via `pytest fastvideo/tests/ssim/ -vs`.
  References live on HF: `FastVideo/ssim-reference-videos`. Seeding a new test
  is automated by the `seed-ssim-references` skill.
- Reviewers verifying a claim locally: **don't actually run** the SSIM/training
  suite — it's expensive. Read the PR's posted test output instead and push
  back if it's missing.

## CI model

- Required checks for merge (from `.github/mergify.yml`):
  - `pre-commit` — lint/format/type.
  - `fastcheck-passed` — fast smoke tests.
  - `full-suite-passed` — full test suite, gated by `/merge` or the `ready` label.
- On-demand commands (write-access contributors only): `/test full`, `/test
  ssim`, `/test training`, `/test kernel`, etc. See
  `docs/contributing/pull_requests.md`.
- Reviewers may *suggest* which `/test <scope>` to run before merge.

## Domain structure (mental map)

```
fastvideo/
  models/     ← DiT / VAE / encoder / scheduler / upsampler / audio, loader
  configs/    ← arch configs (models/) + pipeline wiring (pipelines/) + sample (sample/)
  pipelines/  ← basic/<model>/ per-model; stages/ reusable; samplers
  train/      ← NEW YAML-driven training framework (preferred)
  training/   ← legacy training (being phased out)
  dataset/    ← dataset + dataloader
  distributed/← SP / TP / FSDP utilities
  attention/  ← attention backends (FLASH_ATTN, VSA, VMOBA, STA, SDPA...)
  layers/     ← TP layers, rotary_embedding, layernorm, etc.
  entrypoints/← CLI, CLI dispatch
  api/        ← public API (sampling params, serving)
  worker/     ← worker subprocess entry
  tests/      ← package tests (ssim/, encoders/, training/, etc.)
  registry.py ← unified config registry
fastvideo-kernel/ ← CUDA/C++/Triton kernels (separate build)
examples/     ← runnable examples (inference/, train/, distill/, preprocessing/)
docs/         ← MkDocs source (design/, training/, contributing/, ...)
```

## Add-a-model flow (cross-cutting — all reviewers may see pieces of this)

A canonical new-model PR touches:

1. `fastvideo/models/dits/<model>.py` — DiT implementation (inherits from `models/dits/base.py`).
2. `fastvideo/configs/models/dits/<model>.py` — arch config + `param_names_mapping` (HF keys → FastVideo keys).
3. `fastvideo/pipelines/basic/<model>/` — pipeline composed from `pipelines/stages/`.
4. `fastvideo/models/loader/` — safetensors loader (if HF format differs).
5. `fastvideo/registry.py` — register pipeline + config.
6. `fastvideo/tests/ssim/test_<model>_similarity.py` — SSIM regression test.
7. `examples/inference/basic/<model>.py` — minimal demo script.
8. (optional) `fastvideo/train/models/<model>/` — training wrapper.
9. (optional) docs support-matrix row.

See `docs/contributing/coding_agents.md` for the full "add a model" guide.

## SSIM reference videos

- Hosted on HF: `FastVideo/ssim-reference-videos`.
- When a reviewer flags "needs SSIM test", the author should add the test
  file and then invoke the `seed-ssim-references` skill to upload baseline
  outputs.
- CI pulls references during test (`fastvideo/tests/ssim/reference_utils.py`).

## Distributed defaults

- Sequence parallel (SP) is the default for most pipelines; code that
  broadcasts or reduces across groups must use the correct SP group (see
  `fastvideo/distributed/communication_op.py`).
- Known failure mode: deadlocks when a rank early-exits (#1178). Any code path
  with conditional `return` inside an SP region is suspect.

## FP precision

- Text/image encoders often run in BF16.
- LayerNorm / RMSNorm cache stats in FP32 where needed (#1245 added explicit
  FP32 cache for `FP32LayerNorm`). Kernel PRs claiming a speedup must preserve
  this.
- VAE decode is typically FP16/BF16 depending on model.

## Review gotchas (things that hide in diffs)

### Whitespace / line-ending churn masking real logic changes

If a file shows a +N/-N diff where most changed lines have identical content
(e.g. `-import pytest` / `+import pytest`), it's a CRLF/LF flip or
whitespace-only reformat. Real logic changes embedded in the same commit
become almost invisible — `git show --word-diff-regex='[^[:space:]]+'` is the
escape hatch.

**Rule:** flag as MAJOR and ask the author to split the reformat into a
separate `[misc]` PR. Then re-examine the logic-only diff. This is how
silent threshold drops / constant changes land unnoticed.

### Silent constant changes

Things that are easy to drop and hard to notice:

- `min_acceptable_ssim` thresholds in `fastvideo/tests/ssim/`.
- Default args in `@dataclass` configs under `fastvideo/configs/`.
- `guidance_scale`, `flow_shift`, `seed` defaults in example configs.
- Learning rate / warmup / max-steps in YAML training configs.
- `atol` / `rtol` in kernel correctness tests.

Any of these changing without a PR-body note is a MAJOR (often BLOCKER if
the direction is loosening — lower SSIM threshold, looser tolerance, bigger
LR).

### `/merge` mechanics

`/merge` is the Mergify auto-merge slash command (see `.github/mergify.yml`,
`docs/contributing/pull_requests.md`). It enters the merge queue, which
enforces `merge_protections.success_conditions`:

- `title~=(?i)^\[(feat|feature|bugfix|fix|refactor|perf|ci|doc|docs|misc|chore|kernel|new.?model)\]`
- `#approved-reviews-by>=1`
- `check-success~=pre-commit`
- `check-success=fastcheck-passed`
- `check-success=full-suite-passed`

**A merged PR with any of these red is anomalous** — it merged via direct
admin merge, not `/merge`. If you see this pattern, call it out; it usually
means someone bypassed the queue.

### "Expected failure" test-first workflow

Some PRs ship a test that is **known** to fail CI until a follow-up patch
seeds references or lands a companion change. Before flagging CI red as a
merge anomaly, grep the PR body for phrases like:

- "expected to fail", "first run will fail"
- "no reference video yet", "references will be seeded in a follow-up"
- "blocked on #NNNN" where the referenced PR provides the missing piece

If the body explicitly acknowledges the expected failure, **do not flag**
CI red as a process concern — mention it in context and move on. This is a
legitimate pattern in this repo (example: PR #1240).
