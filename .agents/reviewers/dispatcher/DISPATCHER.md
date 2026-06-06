---
name: pr-review-dispatcher
description: Route a PR to the correct scoped reviewer(s) and consolidate their output into one report
---

# PR Review Dispatcher

Top-level orchestrator for the automatic PR reviewer system. Given a PR
number, it fetches context, selects one or more scoped reviewers, runs them,
and emits a consolidated report.

## Prerequisites

- `gh` CLI authenticated against `hao-ai-lab/FastVideo`.
- Working directory is the repo root.
- Repo-conventions doc read once per session: [`../shared/repo-conventions.md`](../shared/repo-conventions.md).

## Inputs

| Parameter | Required | Description |
|-----------|----------|-------------|
| `pr` | Yes | PR number (e.g. `1245`). |
| `include_drafts` | No | If true, review draft PRs. Default false. |
| `force_reviewers` | No | Comma-separated reviewer names to force (skip routing). |
| `skip_reviewers` | No | Comma-separated reviewer names to exclude. |

## Steps

### 1. Gate checks

- If the PR has label `needs-rebase`, emit:

    > **Dispatcher:** PR #`<N>` is marked `needs-rebase`. Skipping review — the
    > diff is stale. Re-invoke after rebase.

  and exit.
- If the PR is a draft and `include_drafts` is false, emit a one-line note and exit.
- If required CI check `pre-commit` failed, note it at the top of the report
  but continue (human will fix style; reviewer focuses on substance).
- If required CI checks `fastcheck-passed` / `full-suite-passed` are FAILURE
  (common on merged PRs being audited), **scan the PR body** for explicit
  acknowledgment phrases before flagging:
  - "expected to fail", "first run will fail"
  - "no reference video yet", "references will be seeded"
  - "blocked on #NNNN"

  If the body acknowledges the failure with a reason, record it as context
  ("PR body acknowledges expected CI failure: `<quote>`") and do **not** flag
  it in the consolidated report as a process concern. If the body does not
  acknowledge it, treat as a MAJOR anomaly in the general reviewer's scope.
  See `../shared/repo-conventions.md` → "Expected failure" workflow.
- Flag whether the PR is **open** or **merged** at the top of the report so
  reviewers calibrate severity (see `../shared/review-output.md` → Pre-merge
  vs post-merge semantics).

### 2. Fetch context

Follow [`../shared/pr-context.md`](../shared/pr-context.md). Fetch **once** and
pass the bundle to each reviewer (avoid re-hitting the GitHub API).

Record:
- `metadata` (number, title, author, labels, base, draft)
- `body` (PR description)
- `files` (list of path + additions + deletions)
- `diff` (unified diff — may be truncated for huge PRs)
- `checks` (pre-commit / fastcheck / full-suite status)
- `existing_review_comments`

### 3. Route

Apply the rules in [`../ROUTING.md`](../ROUTING.md). Produce a set of
activated reviewers. If no reviewer matches by path/label, always fall back
to `general`.

Minimum fan-out: 1 (general). Typical fan-out: 1–3. Max: 4 (all reviewers).

Examples:

| PR | Signal | Reviewers |
|----|--------|-----------|
| #1245 `[perf] fused RoPE Triton kernel + FP32LayerNorm` | `scope: kernel`, `scope: model` | `kernel`, `model` |
| #1244 `[feat] LongCat self-forcing training scaffold` | `scope: training` | `training` |
| #1236 `[new-model] Port Z-Image T2I` | `type: new-model`, `scope: model` | `model`, `general` *(touches examples/docs)* |
| #1230 `[CI] detect-changes + conditional Docker rebuild` | `type: ci`, `scope: infra` | `general` |
| #1225 `Attn-QAT Video Diffusion Code` | 9 scopes | `model`, `kernel`, `training`, `general` |

### 4. Run each reviewer

For each activated reviewer:

1. Load its `REVIEWER.md`, `checklist.md`, `references.md`.
2. Pass the shared PR context + the reviewer-scoped file list (narrow the diff
   to paths the reviewer owns — see `../shared/pr-context.md` "Scope reduction").
3. Produce output in the format defined by
   [`../shared/review-output.md`](../shared/review-output.md).

Run reviewers **in parallel** where the harness supports it (each reviewer is
an independent Agent call). They do not share state.

### 5. Consolidate

Produce a single top-level report with this structure:

```markdown
# PR #<N> Review — consolidated

**Title**: <title>
**Author**: <author>
**Reviewers run**: <list>
**Verdict (aggregate)**: <APPROVE if all reviewers approved, else REQUEST_CHANGES>
**Blocker count**: <N>  |  Major: <N>  |  Minor: <N>  |  Nit: <N>

---

## <reviewer-1> report
<verbatim output>

---

## <reviewer-2> report
<verbatim output>

...

## Unified action list
<Top blockers + majors, de-duplicated across reviewers, sorted by severity.>

## Untouched scopes
<Any scope that no reviewer covered. Dispatcher flags so a human can decide
whether to run that reviewer manually or treat as out of scope.>
```

### 6. Post (optional)

Posting the report as a PR comment is **not** part of the default flow. Do
not post automatically — this is a skeleton and output fidelity hasn't been
validated. If a human decides to post, they can run:

```bash
gh pr comment "$PR" --body-file <report>.md
```

## Output

- A single markdown report (console or file).
- Optional: a structured JSON sidecar with per-reviewer verdicts + finding
  counts. Not implemented yet.

## References

- [Routing](../ROUTING.md)
- [PR context](../shared/pr-context.md)
- [Review output format](../shared/review-output.md)
- [Repo conventions](../shared/repo-conventions.md)
- [`.github/mergify.yml`](../../../.github/mergify.yml) — authoritative label/path map

## Status

**Draft.** The dispatcher has not been tested end-to-end on a real PR.
Next steps: run against a recent merged PR (e.g. #1245) and compare the
consolidated report against the human review that actually landed.
