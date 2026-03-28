# CI Architecture

## Overview

FastVideo uses a three-tier CI pipeline designed to keep feedback fast on every push while
protecting `main` through a full GPU regression suite before any merge.

```
PR push
  │
  ├─► Tier 1: Pre-commit (~2 min)
  │     GitHub Actions / ubuntu-latest
  │     yapf, ruff, mypy, codespell, pymarkdown, actionlint, check-filenames
  │
  └─► Tier 2: Fastcheck (~10-20 min, path-filtered)
        Buildkite / Modal GPU instances
        Only runs tests for paths you changed

              │ (maintainer adds 'ready' label or comments /merge)
              ▼
        Merge Queue
          Mergify creates mergify/merge-queue/<branch>
              │
              ▼
        Tier 3: Full Suite (~60-90 min)
          Buildkite / Modal GPU instances
          All integration, SSIM, training, and performance tests
              │
          pass ──► squash-merge to main, branch deleted
          fail ──► PR ejected from queue; fix and re-queue
```

---

## CI Tiers

### Tier 1: Pre-commit (every PR push)

| Attribute | Value |
|-----------|-------|
| Triggered by | Every push to any PR branch, plus pushes to `main` |
| Runs on | GitHub Actions, `ubuntu-latest` |
| Duration | ~2 minutes |

**Checks run** (from `.pre-commit-config.yaml`, stage: `manual`):

| Hook | What it checks |
|------|---------------|
| `yapf` | Python code formatting |
| `ruff` | Python linting and auto-fixable style issues |
| `codespell` | Spelling errors in code and docs |
| `pymarkdown` | Markdown formatting |
| `actionlint` | GitHub Actions workflow syntax |
| `mypy` | Static type checking (Python 3.10 target) |
| `check-filenames` | No spaces in tracked filenames |

A failure here means code style or type issues. Run `pre-commit run --all-files` locally to
replicate CI results before pushing.

---

### Tier 2: Fastcheck (path-filtered, every PR push)

| Attribute | Value |
|-----------|-------|
| Triggered by | Every push; the monorepo-diff plugin skips tests for unchanged paths |
| Runs on | Buildkite, Modal GPU instances |
| Duration | ~10-20 minutes per test (run in parallel) |

**Tests and their path triggers:**

| Buildkite label | `TEST_TYPE` | Triggers when you change |
|-----------------|-------------|--------------------------|
| Encoder Tests | `encoder` | `fastvideo/models/encoders/**`, `fastvideo/models/loader/**`, `fastvideo/tests/encoders/**`, `pyproject.toml`, `docker/Dockerfile.python3.12` |
| VAE Tests | `vae` | `fastvideo/models/vaes/**`, `fastvideo/models/loader/**`, `fastvideo/tests/vaes/**`, `pyproject.toml`, `docker/Dockerfile.python3.12` |
| Transformer Tests | `transformer` | `fastvideo/models/dits/**`, `fastvideo/models/loader/**`, `fastvideo/tests/transformers/**`, `fastvideo/layers/**`, `fastvideo/attention/**`, `pyproject.toml`, `docker/Dockerfile.python3.12` |
| Kernel Tests | `kernel_tests` | `fastvideo-kernel/**`, `pyproject.toml`, `docker/Dockerfile.python3.12` |
| Unit Tests | `unit_test` | `fastvideo/**`, `.buildkite/**`, `.github/**`, `pyproject.toml`, `docker/Dockerfile.python3.12` |

A Fastcheck failure means a component-level regression. Check the Buildkite build log for the
failing test's output.

---

### Tier 3: Full Test Suite (Merge Queue only)

| Attribute | Value |
|-----------|-------|
| Triggered by | Mergify creating a `mergify/merge-queue/*` branch, OR a `/test full` command |
| Runs on | Buildkite, Modal GPU instances |
| Duration | 60-90 minutes total (tests run in parallel, path-filtered) |

**Tests:**

| Buildkite label | `TEST_TYPE` | Timeout |
|-----------------|-------------|---------|
| SSIM Tests | `ssim` | 90 min |
| LoRA Inference Tests | `inference_lora` | 20 min |
| Training Tests | `training` | 15 min |
| Distillation DMD Tests | `distillation_dmd` | 15 min |
| Self-Forcing Tests | `self_forcing` | 15 min |
| LoRA Training Tests | `training_lora` | 15 min |
| Training Tests VSA | `training_vsa` | 15 min |
| Inference Tests VMoBA | `inference_vmoba` | 15 min |
| Performance Tests | `performance` | 30 min |
| API Server Tests | `api_server` | 30 min |

A Full Suite failure means the PR is ejected from the Merge Queue. A Mergify comment will
link to the Buildkite build. Fix the regression, push, and re-queue.

---

## Merge Queue

The Merge Queue prevents untested code from landing on `main`. Mergify manages the queue;
Buildkite validates each entry.

**How it works:**

1. A developer comments `/merge` on an approved PR (or a maintainer adds the `ready` label).
2. The `ready` label triggers the Mergify rule `enter merge queue when ready and approved`.
3. Mergify checks the **queue conditions** before accepting:
   - `pre-commit` check must be green
   - At least 1 approved review (`#approved-reviews-by>=1`)
   - PR is not a draft
   - No merge conflicts
   - Not closed
   - No `do-not-merge` label
4. If conditions pass, Mergify creates a temporary branch: `mergify/merge-queue/<branch-name>`
5. Buildkite Full Suite runs on that branch (triggered by the branch name pattern).
6. Once all Full Suite checks pass (`check-success~=buildkite/ci`), Mergify squash-merges
   to `main` with the commit message: `<title> (#<number>)` followed by the PR body.
7. If any Full Suite test fails, the PR is ejected from the queue and Mergify posts a comment.
   The developer fixes the issue, pushes, and comments `/merge` again.

**Queue conditions summary:**

| Condition | Meaning |
|-----------|---------|
| `check-success~=pre-commit` | Tier 1 pre-commit must be green |
| `#approved-reviews-by>=1` | At least one approved review |
| `-draft` | Not a draft PR |
| `-conflict` | No merge conflicts with base branch |
| `-closed` | PR is still open |
| `label!=do-not-merge` | No `do-not-merge` label present |

---

## Label System

Labels are applied automatically. You don't need to set them manually.

### Type Labels (from PR title prefix)

Applied by Mergify based on the `[tag]` at the start of the PR title.

| Label | Matched title prefix | Meaning |
|-------|---------------------|---------|
| `type: feat` | `[feat]` or `[feature]` | New feature or capability |
| `type: bugfix` | `[bugfix]` or `[fix]` | Bug fix |
| `type: refactor` | `[refactor]` | Code restructuring, no behavior change |
| `type: perf` | `[perf]` | Performance improvement |
| `type: ci` | `[ci]` | CI/CD or tooling changes |
| `type: docs` | `[doc]` or `[docs]` | Documentation only |
| `type: misc` | `[misc]` or `[chore]` | Housekeeping, dependency bumps |
| `type: new-model` | `[new-model]` | Adding a new model |

### Scope Labels (from changed files)

Applied by Mergify based on which paths you modified. Multiple scope labels can be added.

| Label | File paths that trigger it |
|-------|---------------------------|
| `scope: training` | `fastvideo/train/`, `fastvideo/training/`, `fastvideo/distillation/`, `examples/train/`, `examples/training/`, `examples/distill/` |
| `scope: inference` | `fastvideo/pipelines/basic/`, `fastvideo/pipelines/stages/`, `fastvideo/pipelines/samplers/`, `fastvideo/entrypoints/`, `fastvideo/worker/`, `fastvideo/configs/sample/`, `fastvideo/configs/pipelines/`, `examples/inference/` |
| `scope: attention` | `fastvideo/attention/` |
| `scope: kernel` | `fastvideo-kernel/`, `csrc/` |
| `scope: data` | `fastvideo/dataset/`, `fastvideo/pipelines/preprocess/`, `examples/preprocessing/` |
| `scope: infra` | `.github/`, `.buildkite/`, `fastvideo/tests/`, `docker/` |
| `scope: distributed` | `fastvideo/distributed/` |
| `scope: docs` | `docs/` |
| `scope: ui` | `ui/` |
| `scope: model` | `fastvideo/models/`, `fastvideo/layers/`, `fastvideo/configs/models/` |

### Process Labels

| Label | Who sets it | Meaning |
|-------|-------------|---------|
| `ready` | Developer (`/merge` command) or maintainer | Enters the Merge Queue |
| `needs-rebase` | Mergify (automatic) | PR has merge conflicts; rebase needed |
| `do-not-merge` | Maintainer | Blocks queue entry regardless of other conditions |

---

## PR Title Format

All PR titles targeting `main` must start with a bracketed type tag. This is enforced by a
Mergify merge protection rule and is required before a PR can be squash-merged.

**Format:**

```
[type] Short description
```

**Valid type tags:**

`feat`, `feature`, `bugfix`, `fix`, `refactor`, `perf`, `ci`, `doc`, `docs`, `misc`, `chore`,
`kernel`, `new-model`

**Valid examples:**

```
[feat] Add causal Wan pipeline
[bugfix] Fix VAE temporal tiling corruption
[refactor] Restructure training framework
[perf] Optimize FlashAttention kernel dispatch
[docs] Add inference guide for LoRA
[new-model] Port HunyuanVideo 1.5 to FastVideo
```

**Invalid examples (will block merge):**

```
Add causal Wan pipeline          ← missing type tag
FEAT: Add pipeline               ← wrong format
feat: Add pipeline               ← square brackets required
```

If your title is invalid, Mergify posts a comment explaining the required format and the
merge protection check will remain failed until you update the title.

---

## Slash Commands

Slash commands let contributors and maintainers trigger CI actions directly from PR comments.
**Write permission to the repository is required.**

The command is recognized within a few seconds. The workflow reacts with a 🚀 emoji to confirm.

### `/merge`

```
/merge
```

Adds the `ready` label to the PR, which triggers Merge Queue entry (subject to queue
conditions being met).

### `/test <name>`

```
/test <name>
```

Triggers a specific Buildkite test or suite on the current PR branch.

| Command | Runs | Maps to `TEST_TYPE` |
|---------|------|---------------------|
| `/test encoder` | Encoder Tests (Fastcheck) | `encoder` |
| `/test vae` | VAE Tests (Fastcheck) | `vae` |
| `/test transformer` | Transformer Tests (Fastcheck) | `transformer` |
| `/test kernel` | Kernel Tests (Fastcheck) | `kernel_tests` |
| `/test unit` | Unit Tests (Fastcheck) | `unit_test` |
| `/test ssim` | SSIM regression tests | `ssim` |
| `/test training` | Training pipeline tests | `training` |
| `/test lora-inference` | LoRA inference tests | `inference_lora` |
| `/test lora-training` | LoRA training tests | `training_lora` |
| `/test distillation` | DMD distillation tests | `distillation_dmd` |
| `/test self-forcing` | Self-Forcing tests | `self_forcing` |
| `/test vsa` | VSA training tests | `training_vsa` |
| `/test vmoba` | VMoBA inference tests | `inference_vmoba` |
| `/test performance` | Performance benchmarks | `performance` |
| `/test api` | API server integration tests | `api_server` |
| `/test full` | Entire Full Suite | all (with `FULL_SUITE=true`) |
| `/test fastcheck` | Entire Fastcheck suite | fastcheck (with `FULL_SUITE=false`) |

---

## Auto Branch Cleanup

After a PR is squash-merged to `main`, Mergify automatically deletes the head branch.
Protected branches (`main`, `master`, `release/*`) are never deleted.

---

## Workflow File Reference

| Filename | Trigger | What it does |
|----------|---------|-------------|
| `ci-precommit.yml` | Every push / PR against `main` | Runs pre-commit hooks (yapf, ruff, mypy, codespell, pymarkdown, actionlint, check-filenames) |
| `ci-slash-commands.yml` | PR comment starting with `/merge` or `/test` | Handles slash commands; adds `ready` label or triggers Buildkite |
| `community-issue-labeler.yml` | Issue opened or edited | Auto-labels issues by keyword matching against title and body |
| `community-welcome.yml` | First contribution | Posts a welcome comment for first-time contributors |
| `community-stale.yml` | Scheduled | Marks and closes stale issues and PRs |
| `infra-build-image.yml` | Manual (`workflow_dispatch`) | Builds Docker images for CI |
| `infra-docs.yml` | Changes to `docs/` merged to `main` | Builds and deploys documentation to GitHub Pages |
| `publish-fastvideo.yml` | Version bump | Publishes `fastvideo` package to PyPI |
| `publish-kernel.yml` | Version bump | Publishes `fastvideo-kernel` package to PyPI |
| `publish-comfyui.yml` | Version bump | Publishes ComfyUI node package to PyPI |
