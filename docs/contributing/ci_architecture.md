# CI/CD Architecture

This is the canonical reference for FastVideo's CI/CD system. Contributor-facing
PR steps live in [Pull Requests](pull_requests.md), and test-authoring guidance
lives in [Testing](testing.md).

## Overview

FastVideo splits validation across GitHub Actions, Buildkite, Slinky Slurm,
and Mergify:

```text
PR opened or updated
  |
  |-- Tier 1: pre-commit
  |     GitHub Actions on ubuntu-latest
  |     style, lint, type, spelling, Markdown, workflow syntax, filenames
  |
  |-- Tier 2: Fastcheck
  |     Buildkite schedules six lanes on the Slinky Slurm cluster
  |     encoder, VAE, transformer, kernel, unit, DreamVerse
  |
  |-- /merge or ready label
        |
        `-- Tier 3: Change-aware merge gate
              trusted base-branch planner classifies the complete PR diff
              Buildkite adds only relevant integration, quality, training,
              API, or performance lanes on Slinky Slurm
              |
              pass -> Mergify squash-merges when all merge conditions pass
              fail -> fix, push, and re-run

  |-- /test full
        `-- Explicit all-20-lane diagnostic run

  `-- weekly schedule on main
        `-- Complete four-GPU SSIM matrix
```

CI is not one monolithic job:

- GitHub Actions owns pre-commit, slash-command handling, aggregate status
  updates, docs deployment, image builds, package publishing, and community
  automations.
- Buildkite owns the GPU test graph, statuses, and trusted dispatch control
  plane. Its agent runs on the Slurm login plane; it does not execute test
  payloads.
- Slinky Slurm is the only active CI compute backend. A host-owned dispatcher
  leases GPUs from a persistent four-GPU allocation and runs each lane in an
  isolated Enroot container at the immutable PR SHA.
- Mergify owns merge protection, labeling, and the final squash merge.

The old files under `fastvideo/tests/modal/` are retained as dormant manual
rollback code. `.buildkite/scripts/pr_test.sh` rejects Buildkite invocations,
and no pipeline or slash-command route calls Modal.

Three Buildkite entry pipelines share the validated graph:

| Pipeline | Trigger | Scope |
|---|---|---|
| `pr-fastcheck` | Automatic pull-request webhook | Six Fastcheck lanes |
| `ci` | `/merge`, `ready`, schedules, and `/test` API builds | Change-aware merge gates, scheduled SSIM, explicit Full Suite, Fastcheck reruns, or one direct lane |
| `fastvideo-performance-lane` | Weekly scheduler | Direct performance lane |

Each entry pipeline starts with the same trusted `pipeline-upload` job on the
`ci-runner` queue. The `ci` pipeline's incoming GitHub webhook is disabled;
otherwise it would duplicate the automatic `pr-fastcheck` build. API and
scheduled builds continue to work with webhook processing disabled.

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
| Compute | Slinky Slurm (`ci-runner` queue) |
| Definition | `.buildkite/pipeline.yml` |
| Entrypoint | Trusted host driver -> `.buildkite/scripts/unit_test.sh` or `.buildkite/scripts/lanes/*.sh` |

Fastcheck always schedules these six lanes: encoder, VAE, transformer, custom
kernels, unit tests, and DreamVerse. Static steps replace the former
host-side path-filter plugin: the login plane never checks out or executes PR
code.

### Tier 3: Change-Aware Merge Gate

| Attribute | Value |
|---|---|
| Triggered by | `/merge`, adding `ready`, or a new push to a PR that already has `ready` |
| Compute | Slinky Slurm only (`ci-runner` queue) |
| Definition | `.buildkite/pipeline.yml` |
| Entrypoint | `/opt/fastvideo-ci-runner/run-ci` (`run-unit` is a compatibility wrapper) |

Fastcheck is the universal six-lane baseline. The merge gate does not repeat
those jobs: it classifies every changed path and adds only the relevant lanes
from the fourteen-lane integration set below. Selected jobs are hard gates;
there are no soft-fail hardware lanes. A documentation-only PR can therefore
finish its merge build after the trusted uploader, while model-family changes
typically add focused golden-gate and SSIM files and a training-only change
adds only its owning training lane.

`.github/scripts/plan_merge_ci.py` is the canonical path policy. It runs from
the immutable base SHA under `pull_request_target`; PR code is never executed
on the GitHub runner. The changed-file list includes both sides of renames. An
API failure, truncated response, empty list, unknown build input, or unknown
source path fails closed to all fourteen integration lanes.

A `ready`-labeled PR does not hit Buildkite immediately:
`ci-trigger-full-suite.yml` first runs `.github/scripts/gate_full_suite.sh`,
which waits for the cheap Tier-1 checks (pre-commit, docs build) on the PR
head. A red cheap check blocks the suite (fail closed; the next push re-arms
it), while a GitHub outage or a >25 min wait lets it run anyway (fail open).
`/test full` bypasses the gate.

The complete static graph remains available through `/test full`; path
selection never deletes or dynamically invents a Buildkite step.

| Lane | Public `TEST_TYPE` | GPUs | Typical merge trigger |
|---|---|---:|---|
| Encoder | `encoder` | 1 | Universal Fastcheck |
| VAE | `vae` | 1 | Universal Fastcheck |
| Transformer | `transformer` | 1 | Universal Fastcheck |
| Kernel | `kernel_tests` | 1 | Universal Fastcheck |
| Unit | `unit_test` | 1 | Universal Fastcheck |
| DreamVerse | `dreamverse_app` | 1 | Universal Fastcheck |
| Golden gate | `golden_gate` | 1 | Model, pipeline, attention, layer, or output changes |
| SSIM | `ssim` | 4 | Matching model/SSIM paths; focused files when possible |
| LoRA inference | `inference_lora` | 1 | LoRA inference/shared LoRA paths |
| LoRA extraction | `lora_extraction` | 1 | LoRA extraction/shared LoRA paths |
| Vanilla training | `training` | 4 | Legacy vanilla/shared training paths |
| DMD distillation | `distillation_dmd` | 2 | DMD/shared training paths |
| Self-forcing | `self_forcing` | 2 | Self-forcing/shared training paths |
| LoRA training | `training_lora` | 2 | LoRA/shared training paths |
| VSA training | `training_vsa` | 2 | VSA/shared training paths |
| VMoBA inference | `inference_vmoba` | 1 | VMoBA backend/config paths |
| Performance | `performance` | 2 | Performance tests or benchmark policy |
| API server | `api_server` | 1 | API, worker, or server entrypoint paths |
| Modular train framework | `train_framework` | 1 | `fastvideo/train/` and its tests |
| Eval metrics | `eval` | 1 | `fastvideo/eval/` and its tests |

Golden-gate and SSIM selections are basenames, not arbitrary pytest arguments.
The private host checks the comma-separated allowlist before staging, and the
container checks it again before invoking pytest. Shared quality-harness
changes still run the complete owning lane. The full SSIM matrix also runs on
`main` every Sunday at 05:00 UTC through `ci-scheduled-ssim.yml`; `/test ssim`
and `/test full` remain available for deliberate complete runs.

The four Buildkite workers may accept multiple jobs concurrently. The
agent-owned lease broker packs their requested GPU counts onto one persistent
four-GPU Slurm allocation and waits when capacity is full. A four-GPU lane
such as SSIM or vanilla training owns the whole tray; two two-GPU lanes or up
to four one-GPU lanes can overlap without sharing devices. SSIM and vanilla
training also share the `fastvideo/slinky/whole-tray` Buildkite concurrency
group. That keeps the second whole-tray lane in Buildkite instead of consuming
an agent and its command timeout while the first lane waits for four free GPUs.
Because packed Enroot containers share the node network namespace, each GPU
lease receives its own 100-port rendezvous range. Tests preserve the
runner-assigned `MASTER_PORT`, and parallel SSIM tasks use distinct offsets
inside that range.

See [Performance Benchmarks](performance_benchmarks.md) for the performance
lane's thresholds, rolling baseline, artifacts, and reseeding process.

## `/merge` Request Flow

The Buildkite agent and Slurm have deliberately separate responsibilities:

```text
/merge PR comment
  -> GitHub verifies write permission and refreshes the `ready` label
  -> base-branch `ci-trigger-full-suite` workflow fetches the PR file list,
     computes MERGE_TEST_PLAN plus focused golden/SSIM basenames, and gates on
     cheap checks
  -> sends the PR SHA to pipeline `ci` with TEST_SCOPE=merge and FULL_SUITE=true
  -> trusted `pipeline-upload` job on queue `ci-runner`
       fetch exact-SHA .buildkite/pipeline.yml
       normalize + validate the complete static 20-lane policy and conditions
       upload static Buildkite steps
  -> each accepted step reaches the host policy hook
       validate org/repo/SHA/ref/step key/command/scope/timeout
       skip checkout on the login plane
       stage a mode-0600 request on Lustre
       lease 1-4 GPUs and attach an `srun` step to the Slinky tray
  -> Enroot worker container
       clone and verify the exact PR SHA
       install the lane's project extras and cached kernel
       run the repository-owned lane script
       write numeric exit status and approved artifacts
  -> trusted host returns that status to Buildkite
  -> Buildkite publishes `full-suite-passed` only when every selected lane passes
```

The pipeline uploader and dispatcher run on the Slurm login plane, but those
are control-plane operations only. Python tests, model loading, CUDA kernels,
Node/Playwright checks, inference, training, SSIM generation, and performance
benchmarks all execute in Slurm allocations.

PR-controlled values never become host commands. The host policy accepts only
the pinned pipeline uploader or a known lane tuple. It rejects plugins,
artifact globs, shell injection variables, non-immutable commits, and unknown
commands before checkout. Hugging Face credentials are added only for lanes
that declare them, passed through a mode-0600 request file, and removed before
the PR payload starts. Active training lanes keep W&B offline and do not stage
a W&B credential. The ARM64 image includes the pinned FA4 CuTe overlay validated
on GB200. SSIM opts into FA4 to preserve its reference-video numerics; lanes
with FA2 baselines keep `FASTVIDEO_FA4=0`. Performance artifacts are relayed
afterward by the trusted host from an allowlisted directory and extension set.

## Slash Commands

Slash commands are handled by `.github/workflows/ci-slash-commands.yml`.
Repository write permission is required.

| Command | Effect |
|---|---|
| `/merge` | Adds `ready` and triggers the path-aware merge gate for the PR head. |
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
| `/test golden-gate` | `golden_gate` |
| `/test training` | `training` |
| `/test lora-inference` | `inference_lora` |
| `/test lora-training` | `training_lora` |
| `/test lora-extraction` | `lora_extraction` |
| `/test distillation` | `distillation_dmd` |
| `/test self-forcing` | `self_forcing` |
| `/test vsa` | `training_vsa` |
| `/test vmoba` | `inference_vmoba` |
| `/test performance` | `performance` |
| `/test api` | `api_server` |
| `/test train-framework` | `train_framework` |
| `/test eval` | `eval` |

The temporary `<name>-ci` spellings remain accepted as compatibility aliases;
they select the same Slurm lane and do not identify a second backend.

When a direct test completes successfully, Buildkite posts
`direct-test-completed`. `.github/workflows/ci-aggregate-status.yml` then reads
the latest Buildkite statuses for the commit and updates `fastcheck-passed` or
`full-suite-passed` if all jobs in that group are green.

Buildkite label emojis define the status namespace used by that aggregation:
`:microscope:` is reserved for the six Fastcheck lanes, while Full-Suite-only
lanes use `:test_tube:` or `:bar_chart:`. Each active lane has exactly one
label and therefore one status context.

## Merge Protection

Mergify enforces these conditions before it squash-merges to `main`:

| Condition | Meaning |
|---|---|
| `check-success~=pre-commit` | Tier 1 passed. |
| `check-success=fastcheck-passed` | All triggered Fastcheck jobs passed. |
| `check-success=full-suite-passed` | The selected merge gate or explicit Full Suite passed. |
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
| `ready` | `/merge` or maintainer action | Triggers/keeps the change-aware merge gate active and enables auto-merge. |
| `needs-rebase` | Mergify | PR has merge conflicts. |
| `do-not-merge` | Maintainer | Blocks merge regardless of CI status. |

## Slurm Lane Entrypoints

Every active test selection lives in `.buildkite/scripts/unit_test.sh` or a
focused `.buildkite/scripts/lanes/<lane>.sh`. The private, agent-owned lane
table binds each internal `*_ci` type to that script, its GPU count, wall-clock
limit, dependency extras, kernel-build policy, secrets, and artifacts. The
internal suffix is an implementation detail; there is only one active backend.

SSIM uses `fastvideo/tests/ssim/ci_runner.py` inside a single four-GPU lease.
It discovers `REQUIRED_GPUS` and `*_MODEL_TO_PARAMS` with AST parsing, then
packs independent pytest subprocesses across the visible GPUs with fail-fast
termination. Performance writes reports to a host-mounted artifact directory;
the host uploads only `.md`, `.html`, `.json`, and `.csv` files after the
container exits.

The Modal launchers remain in the repository for manual rollback archaeology,
but they are not CI entrypoints. `pr_test.sh` rejects Buildkite calls and needs
`FASTVIDEO_ENABLE_LEGACY_MODAL_CI=1` even for a local manual invocation.

If you add a new CI test category:

1. Add an executable `.buildkite/scripts/lanes/<lane>.sh` containing the test
   payload only.
2. Add the static Buildkite step in `.buildkite/pipeline.yml`, its source/test
   ownership in `.github/scripts/plan_merge_ci.py`, and the `/test` mapping in
   `.github/workflows/ci-slash-commands.yml`.
3. Extend `fastvideo/tests/contract/test_ci_test_collection.py`,
   `test_merge_ci_plan.py`, and the trusted private runner's lane table plus
   uploader policy in the same rollout.
4. Validate on the target GB200 hardware before making the lane a merge gate.
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

`.github/workflows/infra-build-image.yml` supports manual `workflow_dispatch`
runs and automatically rebuilds the CUDA matrix when a repository-controlled
image input changes on `main` in the canonical repository. Those inputs include
the CUDA Dockerfile and reusable workflow, dependency metadata, Docker context
policy, `fastvideo-kernel/**`, and the kernel artifact metadata/key helper.
Manual runs let maintainers choose which image families to build. The
`fastvideo-dev` matrix builds Python 3.12 images for CUDA 12.6 and CUDA 13 on
native `amd64` and `arm64` runners, then publishes one multi-platform manifest
per CUDA version. CUDA 12.6 owns the `py3.12-latest` and global `latest` tags, as
well as the explicit `py3.12-cuda12.6.3-latest` alias. CUDA 13 is published under
the explicit `py3.12-cuda13.0.0-latest` tag. This publication policy does not
change the unparameterized `docker/Dockerfile` build defaults, which remain CUDA
13 and `cu130`.

Published development images carry architecture-specific kernel wheels under
`/opt/fastvideo-kernel-prebuilt`. The Slurm worker selects the exact source,
ABI, and GPU-architecture match, so normal lanes reuse the trusted artifact
while kernel-changing PRs still build locally. Once a kernel or artifact-key
change reaches `main`, the image workflow republishes the matching artifact
before later jobs consume the updated image pin.

The same workflow publishes a single-architecture ARM64, CUDA 13, SM100 image
for the self-hosted CI runner under the
`py3.12-cuda13.0.0-sm100-{latest,sha-*}` tags. It carries the matching prebuilt
kernel wheel so runner jobs can validate and install the exact source and ABI
match instead of recompiling it in every lane.

The optional Dreamverse matrix builds backend and UI images for CUDA 12.6 and
CUDA 13 on `amd64`. Dreamverse remains `amd64`-only because its FA4 dependency
stack is not yet validated on ARM64.

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
| `.github/workflows/ci-trigger-full-suite.yml` | Change-aware merge-gate trigger for `ready` PRs and new pushes |
| `.github/workflows/ci-scheduled-ssim.yml` | Weekly complete SSIM trigger on `main` |
| `.github/scripts/plan_merge_ci.py` | Trusted changed-path to integration-lane planner |
| `.github/workflows/ci-aggregate-status.yml` | Aggregate Fastcheck and explicit Full Suite direct-rerun statuses |
| `.buildkite/pipeline.yml` | Static 20-lane Slurm Buildkite graph |
| `.buildkite/scripts/unit_test.sh`, `.buildkite/scripts/lanes/*.sh` | Active Slurm lane payloads |
| `fastvideo/tests/ssim/ci_runner.py` | Four-GPU Slurm SSIM scheduler |
| `.buildkite/scripts/pr_test.sh`, `fastvideo/tests/modal/*.py` | Dormant manual Modal rollback path (disabled in Buildkite) |
| `.buildkite/performance-benchmarks/tests/*.json` | Performance benchmark configs and thresholds |
| `.github/workflows/infra-docs.yml` | Docs build and GitHub Pages deploy |
| `.github/workflows/infra-build-image.yml` | CUDA matrix, CI runner image, and manual Docker image builds |
| `.github/workflows/publish-fastvideo.yml` | FastVideo PyPI publishing |
| `.github/workflows/publish-kernel.yml` | FastVideo kernel PyPI publishing |
| `.github/workflows/publish-comfyui.yml` | ComfyUI registry publishing |
