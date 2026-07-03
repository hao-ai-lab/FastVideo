# Contributing Via Pull Requests

This page is the contributor workflow for opening, validating, and merging a
FastVideo PR. For the full CI/CD implementation details, see
[CI/CD Architecture](ci_architecture.md).

## PR Title

Every PR targeting `main` must start with a bracketed type tag:

```text
[type] Short description of the change
```

Common examples:

```text
[feat] Add causal Wan 2.2 I2V pipeline
[bugfix] Fix VAE temporal tiling corruption on H100
[refactor] Restructure distributed attention dispatch
[docs] Add LoRA finetuning guide
[new-model] Port HunyuanVideo 1.5 to FastVideo
[infra] Add activation trace hooks for pipeline debugging
[skill] Add add-model agent skill
```

Accepted tags are `feat`, `feature`, `bugfix`, `fix`, `refactor`, `perf`,
`ci`, `infra`, `doc`, `docs`, `misc`, `chore`, `kernel`, `new-model`, `skill`,
and `skills`. Mergify checks this before merge. The full tag-to-label mapping
is in [CI/CD Architecture](ci_architecture.md#title-tags-and-labels).

## Labels

Most labels are automatic:

- Type labels come from the PR title tag.
- Scope labels come from the files you changed.
- `needs-rebase` is added and removed by Mergify when conflicts appear or are
  resolved.

The important process labels are:

| Label | Meaning |
|---|---|
| `ready` | The PR is ready for Full Suite and auto-merge consideration. |
| `needs-rebase` | The PR has merge conflicts with `main`. |
| `do-not-merge` | A maintainer has blocked merge. |

## CI Summary

FastVideo has three validation tiers:

| Tier | Runs when | What it does |
|---|---|---|
| Pre-commit | Pull requests and `/test pre-commit` | Formatting, linting, typing, spelling, Markdown, workflow syntax, filename checks |
| Fastcheck | PR Buildkite builds | Path-filtered component and unit checks on Modal GPU runners |
| Full Suite | `/merge`, `ready`, `/test full`, or new pushes to ready PRs | Path-filtered integration, SSIM, training, eval, API, and performance checks |

See [CI/CD Architecture](ci_architecture.md#ci-tiers) for exact jobs, path
filters, and workflow files.

## Merge Flow

1. Open a PR with a valid `[type]` title.
2. Push your changes. Pre-commit and Fastcheck run for the PR.
3. Fix pre-commit failures locally with `pre-commit run --all-files`.
4. Wait for at least one approving review.
5. When the PR is approved and ready, comment `/merge`.
6. `/merge` adds `ready` and triggers the Full Suite for the PR branch.
7. If all required checks pass, Mergify squash-merges the PR to `main`.
8. If Full Suite fails, fix the regression, push again, and re-run `/merge` or
   the failed test.

Only contributors with repository write permission can use slash commands. If
you are an external contributor, ask a maintainer to run `/merge` or add
`ready` after review.

## Running Tests On Demand

Use PR comments to re-run specific checks:

```text
/test pre-commit
/test fastcheck
/test full
/test ssim
/test performance
/test train-framework
```

All supported `/test` names and their `TEST_TYPE` mappings are listed in
[CI/CD Architecture](ci_architecture.md#slash-commands).

When a direct test succeeds, the aggregate `fastcheck-passed` or
`full-suite-passed` status is refreshed automatically if all jobs in that tier
are now green.

## Troubleshooting

### Pre-commit Fails

Run the project hook chain locally:

```bash
uv pip install pre-commit
pre-commit install --hook-type pre-commit --hook-type commit-msg
pre-commit run --all-files
```

Commit any fixes made by the hooks, then push again.

### PR Title Check Fails

Update the PR title so it starts with an accepted bracketed tag, for example
`[bugfix] Fix tensor-parallel shape guard`. Mergify re-evaluates after the
title changes.

### The PR Has `needs-rebase`

Rebase against `main` and force-push safely:

```bash
git fetch origin main
git rebase origin/main
# Resolve conflicts, then:
git push --force-with-lease
```

Mergify removes `needs-rebase` after conflicts are resolved.

### Full Suite Fails

The failing Buildkite step is the source of truth. Common causes are:

- a real regression from the PR,
- missing dependencies or secrets,
- GPU memory pressure or wrong hardware assumptions,
- kernel build failures after `fastvideo-kernel/` changes,
- stale SSIM or performance baselines after an intentional behavior change.

After fixing the issue, push again and use `/merge` or a targeted `/test`
command.
