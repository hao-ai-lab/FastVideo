---
name: decompose-pipeline-pr
description: Decompose an oversized FastVideo pipeline PR into a stack of independently-reviewable PRs. Tiers the diff by blast radius (invisible / dead code / cross-cutting infra / activation), produces a branch graph and worktree bootstrap, drafts the AGENTS.md manifest, flags missing tests on cross-cutting infra changes, and extracts lessons from the PR body.
---

# Decompose Pipeline PR

## Purpose

When a PR adds a new pipeline (or first-class component port) and crosses
~3,000 LOC, single-shot review converges to rubber-stamping. This skill
decomposes such a PR into a stack of independently-reviewable PRs without
disturbing `main`.

It is the inverse of `add-model`: where `add-model` walks adding a new
pipeline as a fresh PR, this skill walks decomposing an existing oversized
pipeline PR.

**Worked example:** PR #1280 (daVinci-MagiHuman, 9,812 LOC, 56 files) →
2 prerequisite PRs off main + 8-PR stack:
- #1293 `will/activation-trace` (prerequisite)
- #1294 `will/loader-infra` (prerequisite)
- #1295 (1/8) housekeeping
- #1296 (2/8) t5gemma encoder
- #1297 (3/8) DiT
- #1298 (4/8) pipeline stages
- #1299 (5/8) pipeline orchestrator
- #1300 (6/8) provenance (AGENTS.md, JOURNAL.md, lessons)
- #1301 (7/8) conversion scripts
- #1302 (8/8) registry activation

## Prerequisites

- Open PR number on `hao-ai-lab/FastVideo` (or any FastVideo fork)
- `gh` CLI authenticated against the target remote
- Local git worktree support (`git worktree`)
- Git config `user.name` / `user.email` set
- Pre-commit installed (`pre-commit install --hook-type pre-commit --hook-type commit-msg`)
- The target PR's branch fetched locally as `origin/<feature-branch>`

## Inputs

| Parameter | Required | Description |
|-----------|----------|-------------|
| PR number or URL | Yes | E.g. `1280` or `https://github.com/hao-ai-lab/FastVideo/pull/1280` |
| Max desired PR size | No | Defaults to ~2,500 LOC of code per stack PR (excluding generated/journal files) |
| Output dir | No | Defaults to `.agents/exploration/decompose-<pr-number>.md` |

## Steps

### 1. Verify ground truth (do not trust `gh pr diff --name-only`)

`gh pr diff <N> --name-only` has been observed to emit phantom file entries.
Always cross-check against the authoritative `git diff`:

```bash
git fetch origin pull/<N>/head:<feature-branch>
git diff origin/main..origin/<feature-branch> --name-status > /tmp/pr-<N>-files.txt
git diff origin/main..origin/<feature-branch> --stat
```

Use the `--name-status` output as the authoritative file list. If it
disagrees with `gh pr diff --name-only`, trust the git diff.

### 2. Tier the diff by blast radius

Classify every changed file into one of four tiers:

| Tier | Description | Examples |
|---|---|---|
| **Tier 0 — Invisible** | Lint/style/CI configs that don't affect runtime | `.gitignore`, `pyproject.toml` (codespell only), `.agents/skills/index.jsonl` stubs |
| **Tier 1 — Dead code** | New files in their own dirs; aggregator one-liners | `fastvideo/models/dits/<new>/`, `fastvideo/pipelines/basic/<new>/`, `examples/inference/basic/basic_<new>*.py`, `tests/local_tests/<new>/`, `__init__.py` exports |
| **Tier 2 — Cross-cutting infra** | Modifications to files used by every pipeline | See protected-paths list below |
| **Tier 3 — Activation switch** | `register_configs(...)` calls + the example scripts that demo them | `fastvideo/registry.py` |

**FastVideo Tier 2 protected paths:**
```
fastvideo/utils.py
fastvideo/pipelines/composed_pipeline_base.py
fastvideo/models/loader/component_loader.py
fastvideo/configs/models/dits/__init__.py
fastvideo/configs/models/encoders/__init__.py
fastvideo/configs/models/vaes/__init__.py
fastvideo/envs.py
fastvideo/fastvideo_args.py
fastvideo/distributed/**
fastvideo/layers/**
fastvideo/attention/**
fastvideo/registry.py    # treat as Tier 3 if change is the activation
```

Tier 3 detection (mechanical):
```bash
git diff origin/main..origin/<feature-branch> -- fastvideo/registry.py | \
  grep -E "^\+.*register_configs\("
```

If `registry.py` only contains `register_configs` additions, treat it as
Tier 3. If it modifies existing behavior, treat it as Tier 2 (rare).

### 3. Identify reusable Tier-1 components

Within Tier 1, look for sub-trees that are **not** model-specific and could
land separately:

- Encoders matching a known multi-model base (T5/T5-Gemma/Llama/Gemma/CLIP variants)
- New stage classes that subclass shared bases without referencing the new model
- Hook/profiler/debug infra under `fastvideo/hooks/`
- New helpers that have no model-specific dependencies

These get split into their own PRs (e.g. PR 4 `t5gemma-encoder` in the
MagiHuman example).

### 4. Hunt for missing test coverage on Tier 2 changes

For every Tier-2 file modified, check whether the original PR added unit
tests for the new behavior:

```bash
for f in <list-of-tier-2-files>; do
  echo "=== Tests for $f ==="
  git diff origin/main..origin/<feature-branch> -- \
    "$(echo $f | sed 's|fastvideo/|fastvideo/tests/|; s|\.py|*|')"
done
```

If a Tier-2 PR has no accompanying tests, **emit a "must-add tests" list**
with a sketch of the case grid. Tier-2 PRs do not ship without those tests.

The MagiHuman example required this for PR-B (`utils.py`): the original PR
shipped no `test_utils_loader.py`, so the decomposition added 9 unit-test
cases covering the umbrella-detector boundary, the optional-component-dirs
relaxation, and regression coverage on every existing 2-segment HF id.

### 5. Build the dependency DAG and topo-sort

Edges:
- Tier 2 infra → Tier 1 code that imports it
- Reusable Tier 1 components → model-specific Tier 1 code that uses them
  (encoder before DiT before pipeline)
- Tier 1 → Tier 3 (activation always last)
- Tier 0 has no dependents (lands first as a freebie)

Topo-sort produces the stack ordering. Pull Tier-2 PRs **out of the stack**
when they have no model-specific dependency — they should land off main
with their own focused review, not buried in a model port.

Render as a tree (markdown):

```
main
 ├─ <prereq-A>
 │   └─ <prereq-B>
 │       ├─ <stack-01-housekeeping>
 │       │   └─ <stack-02-encoder>
 │       │       └─ <stack-03-dit>
 │       │           └─ ...
 │       │               └─ <stack-N-activate>
 │       └─ (parallel) <skill-pr> off main
```

### 6. Detect mis-shelved docs and debug scratch

Two categories to flag:

- **Mis-shelved docs**: Markdown files under `tests/local_tests/` are
  journals, not tests. Flag for relocation to the package dir as
  `JOURNAL.md`.
- **Debug scratch**: files starting with `_debug_`, `_scratch_`, or
  `_explore_`. Flag for drop (do not carry into any output PR).

For MagiHuman: `tests/local_tests/magi-human.md` → relocate. Two
`_debug_magi_human_*.py` files → drop.

### 7. Author the AGENTS.md manifest skeleton

For the new pipeline package, generate a 6-section `AGENTS.md` scaffold
with the file table pre-populated from the diff:

1. **Manifest** — file table by role
2. **Parity invariants** — load-bearing rules with one-paragraph each + lesson refs
3. **Cross-refs** — "If you change X, re-run Y" matrix
4. **Run book** — single pytest command + prereqs (HF tokens, GPU, wall-time)
5. **Open questions** — known issues (e.g. tolerance carve-outs)
6. **Provenance** — PR table with branch names and source SHA

The provenance section is filled incrementally during stack execution and
finalized in the activation PR.

### 8. Extract lessons from the PR body

Scan the PR body for sections titled "Key implementation work", "Bug hunt",
"Lessons", or sentences with patterns like "took N waves to localize",
"silent regression", "investigation revealed". Each becomes a candidate
`.agents/lessons/<YYYY-MM-DD>_<slug>.md` draft.

Lessons MUST follow the existing template in
`.agents/lessons/README.md`:
- YAML frontmatter: `date`, `experiment`, `category`, `severity`
- Sections: What Happened, Root Cause, Fix / Workaround, Prevention
- Filename: `<YYYY-MM-DD>_<short-slug>.md`

Lessons co-locate with the code they concern: a conversion-script lesson
lands in the same PR as the conversion script, not in the docs PR.

### 9. Emit the commit-footer convention

Every commit in the stack ends with:

```
<Feature>-Stack: N/M
```

E.g. `Magi-Stack: 5/8`. Use the package directory name as the feature key.
After all PRs squash-merge, `git log --grep='^<Feature>-Stack:'` reconstructs
the lineage even if PR numbers later get renumbered.

### 10. Produce the worktree bootstrap

Generate a runnable bash script:

```bash
#!/bin/bash
set -euo pipefail

REPO=/home/<user>/FastVideo
WORKTREE=/home/<user>/FastVideoMagi   # NB: directory name must be a valid
                                       # Python identifier (no hyphens) so
                                       # mypy doesn't choke
SOURCE_PR=<N>
SOURCE_BRANCH=will/<feature>
SOURCE_SHA=$(git -C "$REPO" rev-parse "origin/$SOURCE_BRANCH")

git -C "$REPO" fetch origin main:main
git -C "$REPO" fetch "origin/$SOURCE_BRANCH"
git -C "$REPO" worktree add "$WORKTREE" origin/main

# Capture baseline for provenance
mkdir -p "$REPO/.agents/exploration"
cat > "$REPO/.agents/exploration/<feature>-baseline-${SOURCE_SHA:0:8}.txt" <<EOF
Source PR: <repo>#$SOURCE_PR
Source SHA: $SOURCE_SHA
Authoritative file count: $(git -C "$REPO" diff origin/main..origin/$SOURCE_BRANCH --name-only | wc -l)
Date captured: $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
```

### 11. Author preserve via `git checkout`, not `cherry-pick`

For each stack PR:

```bash
git -C "$WORKTREE" switch -c <new-branch> <base-branch>
git -C "$WORKTREE" checkout origin/<source-branch> -- <file1> <file2> ...
git -C "$WORKTREE" commit -m "[<scope>]: <subject>

<body>

<Feature>-Stack: N/M"
git -C "$WORKTREE" push -u origin <new-branch>
gh pr create --base <base-branch> --head <new-branch> --title "..." --body "$(cat <<EOF ... EOF)"
```

Notes:
- `git checkout origin/<source> -- <files>` extracts only the named files,
  preserving the diff. The original PR's author is **not** preserved on the
  new commit (it's authored by whoever runs the script). Reference the
  original PR + source SHA in every commit body and PR description for
  authorship attribution.
- **Never use `git cherry-pick`** for this workflow — cherry-pick applies
  whole commits, which mixes concerns across PR boundaries.

## Outputs

The skill produces:

1. A markdown decomposition plan (`.agents/exploration/decompose-<pr>.md`)
2. A proposed branch graph
3. A worktree-bootstrap script
4. Per-PR file allocation lists (under `/tmp/<feature>-stack/`)
5. AGENTS.md scaffolds for any new pipeline packages
6. Draft lesson files (placed alongside the PR that owns the code they concern)
7. A finalized provenance table for the package AGENTS.md

## Anti-Patterns

The skill should warn against:

- **"Just rebase the megaPR into smaller commits."** Doesn't help review;
  reviewer still sees one PR.
- **Co-locating tests under the new package.** FastVideo's convention is
  by-kind under `fastvideo/tests/` and `tests/local_tests/<family>/`. Don't
  invent a new layout per pipeline.
- **Splitting Tier 2 changes into "one file per PR."** Tier 2 PRs are
  about semantic units (e.g., "loader umbrella + optional component dirs"
  together because they jointly define the new diffusers-format contract),
  not file-count.
- **Landing the activation switch first** ("just register, the code can
  be empty"). The skill enforces activation-last so every intermediate
  state is dead code, not broken code.
- **Trusting `gh pr diff --name-only`.** Cross-check against
  `git diff origin/main..origin/<feature-branch> --name-status` —
  `gh`'s output has been observed to include phantom entries.
- **Worktree dir names with hyphens.** mypy interprets them as invalid
  Python package names and refuses to run. Use CamelCase or underscores.
- **Skipping the lesson-extraction step.** PR bodies contain the most
  expensive learnings of the original implementation. Losing them to a
  squash-merge is the silent decay of institutional knowledge.

## Example Usage

```
User: split PR 1280
Agent: [invokes decompose-pipeline-pr]
       → produces .agents/exploration/decompose-1280.md with:
         - tiered file table (56 files: 3 tier-0, 35 tier-1, 9 tier-2,
           9 tier-3)
         - branch graph (PR-A + PR-B + 8-PR stack)
         - worktree bootstrap script
         - per-PR file lists
         - AGENTS.md scaffold for fastvideo/pipelines/basic/magi_human/
         - 3 draft lessons extracted from the PR body
       → asks user to confirm before opening branches
```

## References

- The MagiHuman decomposition (worked example):
  `fastvideo/pipelines/basic/magi_human/AGENTS.md` (after PR #1302 merges)
- Existing skill: `.agents/skills/add-model/SKILL.md` (the inverse — adding
  a new pipeline as a fresh PR)
- Lesson template: `.agents/lessons/README.md`
- Skill template: `.agents/skills/SKILL_TEMPLATE.md`
