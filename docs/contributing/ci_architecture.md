# CI/CD Architecture

This is the canonical reference for FastVideo's CI/CD system. Contributor-facing
PR steps live in [Pull Requests](pull_requests.md), and test-authoring guidance
lives in [Testing](testing.md).

## Overview

FastVideo splits validation across GitHub Actions, Buildkite, and Modal:

```text
PR opened or updated
  |
  |-- Tier 1: pre-commit
  |     GitHub Actions on ubuntu-latest
  |     style, lint, type, spelling, Markdown, workflow syntax, filenames
  |
  |-- Tier 2: Fastcheck
  |     Buildkite orchestrates Modal GPU jobs
  |     path-filtered component and unit checks
  |
  |-- /merge, /test full, or ready label
        |
        `-- Tier 3: Full Suite
              Buildkite orchestrates Modal GPU jobs
              path-filtered integration, SSIM, training, eval, and performance checks
              |
              pass -> Mergify squash-merges when all merge conditions pass
              fail -> fix, push, and re-run
```

CI is not one monolithic job:

- GitHub Actions owns pre-commit, slash-command handling, aggregate status
  updates, docs deployment, image builds, package publishing, and community
  automations.
- Buildkite owns the GPU test pipeline and path filtering.
- Modal owns the actual GPU execution environment for test jobs.
- Mergify owns merge protection, labeling, and the final squash merge.

## CI Tiers

### Tier 1: Pre-commit

| Attribute | Value |
|---|---|
| Triggered by | Pull requests targeting `main`, plus `/test pre-commit` through `workflow_call` |
| Runner | GitHub Actions, `ubuntu-latest` |
| Workflow | `.github/workflows/ci-precommit.yml` |
| Local command | `pre-commit run --all-files` |

The workflow runs `.pre-commit-config.yaml` with `--hook-stage manual`.

| Hook | Checks |
|---|---|
| `yapf` | Python formatting |
| `ruff` | Python linting and auto-fixes |
| `codespell` | Spelling in code and docs |
| `pymarkdown` | Markdown formatting |
| `actionlint` | GitHub Actions workflow syntax |
| `mypy` | Static typing |
| `check-filenames` | Spaces in tracked filenames |

Run the pre-commit command above to reproduce failures locally. Do not bypass
the project hook chain by calling individual tools directly unless you are
debugging a hook implementation.

### Tier 2: Fastcheck

| Attribute | Value |
|---|---|
| Triggered by | Buildkite PR builds with `TEST_SCOPE=fastcheck` or unset |
| Runner | Buildkite agent that launches Modal GPU jobs |
| Definition | `.buildkite/pipeline.yml` |
| Entrypoint | `.buildkite/scripts/pr_test.sh` -> `fastvideo/tests/modal/pr_test.py` |

Fastcheck uses Buildkite's `monorepo-diff` plugin. Jobs whose watched paths did
not change are skipped and do not block the aggregate `fastcheck-passed`
status.

| Buildkite label | `TEST_TYPE` | Main watched paths |
|---|---|---|
| Encoder Tests | `encoder` | `fastvideo/models/encoders/**`, `fastvideo/models/loader/**`, `fastvideo/tests/encoders/**`, `pyproject.toml`, `docker/Dockerfile.python3.12` |
| VAE Tests | `vae` | `fastvideo/models/vaes/**`, `fastvideo/models/loader/**`, `fastvideo/tests/vaes/**`, `pyproject.toml`, `docker/Dockerfile.python3.12` |
| Transformer Tests | `transformer` | `fastvideo/models/dits/**`, `fastvideo/models/loader/**`, `fastvideo/tests/transformers/**`, `fastvideo/layers/**`, `fastvideo/attention/**`, `pyproject.toml`, `docker/Dockerfile.python3.12` |
| Kernel Tests | `kernel_tests` | `fastvideo-kernel/**`, `pyproject.toml`, `docker/Dockerfile.python3.12` |
| Unit Tests | `unit_test` | `fastvideo/**`, `.buildkite/**`, `.github/**`, `pyproject.toml`, `docker/Dockerfile.python3.12` |
| DreamVerse App Tests | `dreamverse_app` | `apps/dreamverse/**`, `pyproject.toml` |

### Tier 3: Full Suite

| Attribute | Value |
|---|---|
| Triggered by | `/merge`, adding `ready`, `/test full`, or a new push to a PR that already has `ready` |
| Runner | Buildkite agent that launches Modal GPU jobs |
| Definition | `.buildkite/pipeline.yml` |
| Entrypoint | `.buildkite/scripts/pr_test.sh` -> `fastvideo/tests/modal/pr_test.py` |

Full Suite is also path-filtered. It validates broader behavior before Mergify
can merge a PR.

| Buildkite label | `TEST_TYPE` | Main watched paths |
|---|---|---|
| SSIM Tests | `ssim` | `fastvideo/**/*.py`, `pyproject.toml`, `docker/Dockerfile.python3.12` |
| LoRA Inference Tests | `inference_lora` | LoRA tests, loader, transformer tests, pipelines, LoRA layers |
| Training Tests | `training` | `fastvideo/**`, `pyproject.toml`, `docker/Dockerfile.python3.12` |
| Distillation DMD Tests | `distillation_dmd` | `fastvideo/training/*distillation_pipeline.py` |
| Self-Forcing Tests | `self_forcing` | self-forcing distillation pipeline and tests |
| LoRA Training Tests | `training_lora` | `fastvideo/**`, `pyproject.toml`, `docker/Dockerfile.python3.12` |
| Training Tests VSA | `training_vsa` | `fastvideo/**`, `fastvideo-kernel/**`, `pyproject.toml`, `docker/Dockerfile.python3.12` |
| Inference Tests VMoBA | `inference_vmoba` | `fastvideo-kernel/**`, `fastvideo/attention/backends/vmoba.py` |
| Performance Tests | `performance` | DiTs, pipelines, attention, layers, worker, entrypoints, performance tests/configs |
| API Server Tests | `api_server` | OpenAI entrypoints, serve CLI, OpenAI API integration test |
| Train Framework Tests | `train_framework` | `fastvideo/train/**`, train model/method tests, model loader, DiTs |
| Eval Metrics Tests | `eval` | `fastvideo/eval/**`, `fastvideo/tests/eval/**`, `pyproject.toml`, `docker/Dockerfile.python3.12` |

See [Performance Benchmarks](performance_benchmarks.md) for the performance
lane's thresholds, rolling baseline, artifacts, and reseeding process.

## Slash Commands

Slash commands are handled by `.github/workflows/ci-slash-commands.yml`.
Repository write permission is required.

| Command | Effect |
|---|---|
| `/merge` | Adds `ready` and triggers Full Suite for the PR head branch. |
| `/test full` | Runs the whole Full Suite with `TEST_SCOPE=full`. |
| `/test fastcheck` | Runs the whole Fastcheck suite with `TEST_SCOPE=fastcheck`. |
| `/test pre-commit` | Re-runs the pre-commit workflow on the PR merge ref. |
| `/test <name>` | Runs one Buildkite test with `TEST_SCOPE=direct`. |

Valid direct test names:

| Command | `TEST_TYPE` |
|---|---|
| `/test encoder` | `encoder` |
| `/test vae` | `vae` |
| `/test transformer` | `transformer` |
| `/test kernel` | `kernel_tests` |
| `/test unit` | `unit_test` |
| `/test dreamverse` | `dreamverse_app` |
| `/test ssim` | `ssim` |
| `/test training` | `training` |
| `/test lora-inference` | `inference_lora` |
| `/test lora-training` | `training_lora` |
| `/test distillation` | `distillation_dmd` |
| `/test self-forcing` | `self_forcing` |
| `/test vsa` | `training_vsa` |
| `/test vmoba` | `inference_vmoba` |
| `/test performance` | `performance` |
| `/test api` | `api_server` |
| `/test train-framework` | `train_framework` |
| `/test eval` | `eval` |

When a direct test completes successfully, Buildkite posts
`direct-test-completed`. `.github/workflows/ci-aggregate-status.yml` then reads
the latest Buildkite statuses for the commit and updates `fastcheck-passed` or
`full-suite-passed` if all jobs in that group are green.

Skipped path-filtered jobs have no status entry and do not block the aggregate.

## Merge Protection

Mergify enforces these conditions before it squash-merges to `main`:

| Condition | Meaning |
|---|---|
| `check-success~=pre-commit` | Tier 1 passed. |
| `check-success=fastcheck-passed` | All triggered Fastcheck jobs passed. |
| `check-success=full-suite-passed` | All triggered Full Suite jobs passed. |
| `#approved-reviews-by>=1` | At least one approving review. |
| Valid title regex | PR title starts with an accepted `[type]` tag. |
| `label=ready` | The PR has entered the merge flow. |
| `-draft` | PR is not a draft. |
| `-conflict` | PR has no merge conflicts. |
| `-closed` | PR is still open. |

The final merge action is a squash merge. Mergify also labels conflicting PRs
with `needs-rebase` and removes that label after conflicts are resolved.

## Title Tags And Labels

PR title tags are the source for `type:*` labels and are required by merge
protection.

| Tag | Label | Use for |
|---|---|---|
| `[feat]`, `[feature]` | `type: feat` | New feature or capability |
| `[bugfix]`, `[fix]` | `type: bugfix` | Bug fix |
| `[refactor]` | `type: refactor` | Code restructuring without behavior change |
| `[perf]` | `type: perf` | Performance improvement |
| `[ci]` | `type: ci` | CI/CD or build tooling changes |
| `[infra]` | `type: infra` | Repo infrastructure, agent tooling, debug hooks, conversion scripts, dev infra |
| `[doc]`, `[docs]` | `type: docs` | Documentation only |
| `[misc]`, `[chore]` | `type: misc` | Housekeeping, dependency bumps, cleanup |
| `[kernel]` | No dedicated type label currently | CUDA kernel changes in `fastvideo-kernel/` |
| `[new-model]` | `type: new-model` | Adding a new model or pipeline |
| `[skill]`, `[skills]` | `type: skill` | Agent skills under `.agents/skills/` or `.claude/skills/` |

Scope labels are inferred from changed files.

| Label | File paths that trigger it |
|---|---|
| `scope: training` | `fastvideo/train/`, `fastvideo/training/`, `fastvideo/distillation/`, `examples/train/`, `examples/training/`, `examples/distill/` |
| `scope: inference` | `fastvideo/pipelines/basic/`, `fastvideo/pipelines/stages/`, `fastvideo/pipelines/samplers/`, `fastvideo/entrypoints/`, `fastvideo/worker/`, `fastvideo/api/sampling_param.py`, `fastvideo/configs/pipelines/`, `examples/inference/` |
| `scope: attention` | `fastvideo/attention/` |
| `scope: kernel` | `fastvideo-kernel/`, `csrc/` |
| `scope: data` | `fastvideo/dataset/`, `fastvideo/pipelines/preprocess/`, `examples/preprocessing/` |
| `scope: infra` | `.github/`, `.buildkite/`, `fastvideo/tests/`, `docker/` |
| `scope: distributed` | `fastvideo/distributed/` |
| `scope: docs` | `docs/` |
| `scope: studio` | `apps/fastvideo_studio/` |
| `scope: model` | `fastvideo/models/`, `fastvideo/layers/`, `fastvideo/configs/models/` |

Process labels:

| Label | Who sets it | Meaning |
|---|---|---|
| `ready` | `/merge` or maintainer action | Triggers/keeps Full Suite active and enables auto-merge. |
| `needs-rebase` | Mergify | PR has merge conflicts. |
| `do-not-merge` | Maintainer | Blocks merge regardless of CI status. |

## Modal Test Entrypoints

All Buildkite test jobs go through `.buildkite/scripts/pr_test.sh`, which:

1. Reads Buildkite secrets for Modal, Hugging Face, and W&B when needed.
2. Selects a Modal function based on `TEST_TYPE`.
3. Passes Buildkite metadata into the Modal container.
4. Runs the selected test command from `fastvideo/tests/modal/pr_test.py` or
   `fastvideo/tests/modal/ssim_test.py`.
5. Uploads performance artifacts for `TEST_TYPE=performance`.

If you add a new CI test category:

1. Add the Modal function in `fastvideo/tests/modal/pr_test.py` or a focused
   companion module.
2. Add the `TEST_TYPE` case in `.buildkite/scripts/pr_test.sh`.
3. Add the Buildkite direct-test step and any Fastcheck/Full Suite path filters
   in `.buildkite/pipeline.yml`.
4. Add or update the `/test` mapping in `.github/workflows/ci-slash-commands.yml`.
5. Document the lane here and link any domain-specific authoring guide.

## CD And Release Workflows

### Documentation

`.github/workflows/infra-docs.yml` builds documentation for PRs that touch
`docs/**`, `mkdocs.yml`, `requirements-mkdocs.txt`, or the workflow itself. On
pushes to `main`, it also deploys the built site to GitHub Pages.

The docs job:

1. Installs the pinned MkDocs dependencies.
2. Runs `mkdocs build`; the native MkDocs hook generates example pages before the build.
3. Runs `python scripts/check_docs_links.py` against the generated documentation.
4. Uploads the Pages artifact and deploys only from `main`.

### Docker Images

`.github/workflows/infra-build-image.yml` is a manual `workflow_dispatch`
workflow. Maintainers choose which image families to build, including Python
development images and DreamVerse CUDA 12.9 images.

The reusable implementation lives in
`.github/workflows/_template-build-image.yml`.

### Package Publishing

| Workflow | Trigger | Publishes |
|---|---|---|
| `publish-fastvideo.yml` | `pyproject.toml` changes on `main`, or manual dispatch | `fastvideo` package to PyPI when the version changes |
| `publish-kernel.yml` | `fastvideo-kernel/pyproject.toml` changes on `main`, or manual dispatch | `fastvideo-kernel` wheels to PyPI when the version changes |
| `publish-comfyui.yml` | `pyproject.toml` changes on `main`/`master`, or manual dispatch | ComfyUI custom node to the Comfy registry |

## Community Automations

| Workflow | Trigger | Purpose |
|---|---|---|
| `community-issue-labeler.yml` | Issue opened or edited | Adds scope/platform labels from issue keywords. |
| `community-welcome.yml` | First contribution | Posts a welcome comment for first-time contributors. |
| `community-stale.yml` | Daily schedule | Marks and closes stale issues and PRs, with exemption labels. |

## File Reference

| File | Owner area |
|---|---|
| `.github/mergify.yml` | Merge protection, PR title validation, PR labels, conflict labels, auto-merge |
| `.github/workflows/ci-precommit.yml` | Tier 1 pre-commit |
| `.github/workflows/ci-slash-commands.yml` | `/merge` and `/test` handling |
| `.github/workflows/ci-trigger-full-suite.yml` | Full Suite trigger for `ready` PRs and new pushes to ready PRs |
| `.github/workflows/ci-aggregate-status.yml` | Aggregate Fastcheck/Full Suite commit statuses |
| `.buildkite/pipeline.yml` | Buildkite test graph and path filters |
| `.buildkite/scripts/pr_test.sh` | Buildkite-to-Modal test dispatcher |
| `fastvideo/tests/modal/pr_test.py` | Modal functions for most GPU CI lanes |
| `fastvideo/tests/modal/ssim_test.py` | Modal functions and partitioning for SSIM |
| `.buildkite/performance-benchmarks/tests/*.json` | Performance benchmark configs and thresholds |
| `.github/workflows/infra-docs.yml` | Docs build and GitHub Pages deploy |
| `.github/workflows/infra-build-image.yml` | Manual Docker image builds |
| `.github/workflows/publish-fastvideo.yml` | FastVideo PyPI publishing |
| `.github/workflows/publish-kernel.yml` | FastVideo kernel PyPI publishing |
| `.github/workflows/publish-comfyui.yml` | ComfyUI registry publishing |
