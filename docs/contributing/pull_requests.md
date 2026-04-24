# Contributing via Pull Requests

This guide walks through the PR workflow: title format, labels, CI pipeline, and getting
your changes merged.

---

## PR Title Format (Required)

Every PR targeting `main` must start with a type tag in square brackets. This is checked by
Mergify before any merge is allowed.

**Format:**

```
[type] Short description of the change
```

**Valid type tags:**

| Tag | When to use |
|-----|-------------|
| `[feat]` or `[feature]` | New feature or capability |
| `[bugfix]` or `[fix]` | Bug fix |
| `[refactor]` | Code restructuring with no behavior change |
| `[perf]` | Performance improvement |
| `[ci]` | CI/CD or build tooling changes |
| `[doc]` or `[docs]` | Documentation only |
| `[misc]` or `[chore]` | Housekeeping, dependency bumps, minor cleanup |
| `[kernel]` | CUDA kernel changes in `fastvideo-kernel/` |
| `[new-model]` | Adding a new model or pipeline |

**Examples:**

```
[feat] Add causal Wan 2.2 I2V pipeline
[bugfix] Fix VAE temporal tiling corruption on H100
[refactor] Restructure distributed attention dispatch
[docs] Add LoRA finetuning guide
[new-model] Port HunyuanVideo 1.5 to FastVideo
```

If your title is missing the tag, Mergify will post a comment listing the valid formats.
Update the title and the check will re-evaluate automatically.

---

## Labels

Labels are applied automatically based on your PR title and the files you changed. You don't
need to set them manually.

**Type label** — set from the `[tag]` in your title:
`type: feat`, `type: bugfix`, `type: refactor`, `type: perf`, `type: ci`, `type: docs`,
`type: misc`, `type: new-model`

**Scope labels** — set from which files you modified (multiple labels can apply):
`scope: training`, `scope: inference`, `scope: attention`, `scope: kernel`, `scope: data`,
`scope: infra`, `scope: distributed`, `scope: docs`, `scope: ui`, `scope: model`

**Process labels** — set during review and merge:

| Label | Who sets it | Meaning |
|-------|-------------|---------|
| `ready` | You (`/merge` comment) or a maintainer | Triggers Full Suite and enables auto-merge |
| `needs-rebase` | Mergify (automatic) | Your PR has conflicts; rebase against `main` |
| `do-not-merge` | Maintainer | Blocks merge regardless of CI status |

---

## CI Pipeline

Three tiers run automatically on every PR.

**Tier 1: Pre-commit (~2 min) — runs on every push**

GitHub Actions checks formatting, linting, type correctness, and spelling using pre-commit
hooks: yapf, ruff, mypy, codespell, pymarkdown, actionlint, and check-filenames.

**Tier 2: Fastcheck (~10-20 min) — runs on every push, path-filtered**

Buildkite runs GPU tests only for the components you changed. If you only modified
`fastvideo/models/vaes/`, only VAE Tests run. Tests run in parallel.

**Tier 3: Full Suite (~60-90 min) — triggered by the `ready` label**

When you comment `/merge` (or a maintainer adds the `ready` label), Buildkite runs the
complete test suite on your PR branch: SSIM regression, LoRA inference and training,
distillation, self-forcing, VSA, VMoBA, performance benchmarks, and API server tests.

---

## Getting Your PR Merged

**Step-by-step:**

1. Open a PR with a title that starts with a valid `[type]` tag.
2. Push your changes. Pre-commit and Fastcheck run automatically.
3. Fix any pre-commit failures locally (`pre-commit run --all-files`) and push again.
4. Wait for at least one approving review.
5. Once approved and pre-commit is green, comment `/merge` on the PR.
6. The `ready` label is added, which triggers the Full Suite on your PR branch.
7. Mergify also auto-rebases your branch against `main` if it is behind and conflict-free.
8. If all Full Suite tests pass and all merge conditions are met (approval, valid title,
   pre-commit green, fastcheck green, no draft, no conflicts), Mergify squash-merges to
   `main` automatically. Your branch is deleted.
9. If a Full Suite test fails, check the Buildkite build log for the failing step. Fix the
   issue, push, and comment `/merge` again. You can also re-run individual failed tests
   with `/test <name>` — see below.

!!! note
    Only contributors with write permission to the repository can trigger slash commands.
    If you're an external contributor, ask a maintainer to run `/merge` or add the `ready`
    label for you.

---

## Running Tests On Demand

Comment on your PR to trigger specific tests independently of the auto-merge flow.

**Trigger the entire Full Suite:**

```
/test full
```

**Trigger the Fastcheck suite:**

```
/test fastcheck
```

**Trigger individual tests:**

```
/test encoder          # Encoder component tests
/test vae              # VAE component tests
/test transformer      # Transformer / DiT tests
/test kernel           # CUDA kernel tests
/test unit             # Unit tests

/test ssim             # SSIM regression tests
/test training         # Training pipeline tests
/test lora-inference   # LoRA inference tests
/test lora-training    # LoRA training tests
/test distillation     # DMD distillation tests
/test self-forcing     # Self-Forcing distillation tests
/test vsa              # VSA training tests
/test vmoba            # VMoBA inference tests
/test performance      # Performance benchmarks
/test api              # API server integration tests
/test pre-commit       # Pre-commit checks on PR code
```

The workflow reacts with a 🚀 emoji to confirm the command was received.

When you re-run an individual test with `/test <name>`, the new result overwrites the
original failed check (same Buildkite check name). Once all tests in a tier pass, the
`fastcheck-passed` or `full-suite-passed` status is automatically updated.

---

## Troubleshooting

### Pre-commit fails

Run locally to reproduce and auto-fix:

```bash
# Install pre-commit if needed
uv pip install pre-commit
pre-commit install

# Run all checks on all files
pre-commit run --all-files
```

Common quick fixes:

- **yapf**: `yapf -i <file>` (Python formatting)
- **ruff**: `ruff check --fix <file>` (linting)
- **codespell**: `codespell --write-changes <file>` (spelling)

### PR title format check fails

Update your title to start with a valid type tag. The Mergify merge protection check
re-evaluates automatically after you save the title.

Valid tags: `feat`, `feature`, `bugfix`, `fix`, `refactor`, `perf`, `ci`, `doc`, `docs`,
`misc`, `chore`, `kernel`, `new-model`

### My PR has merge conflicts (`needs-rebase` label)

Rebase against `main` and force-push:

```bash
git fetch origin main
git rebase origin/main
# Resolve any conflicts, then:
git push --force-with-lease
```

Mergify removes the `needs-rebase` label automatically once conflicts are resolved.

### Full Suite failed after `/merge`

The Full Suite found a regression. Check the failing Buildkite step's output for assertion
errors or tracebacks.

Common causes:

- Test failures caused by your code changes
- Missing dependency in `pyproject.toml`
- GPU memory issue (some tests require specific hardware like L40S or H100)
- Kernel build failure (if you changed `fastvideo-kernel/`)

After fixing, push and comment `/merge` again.

### I'm an external contributor without write permission

You can't use slash commands directly. After your PR is approved, ask a maintainer to
comment `/merge` or add the `ready` label.
