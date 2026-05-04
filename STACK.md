# PR Stack — `will/ltx2_sr_port` Split Tracker

**Status:** TEMPORARY — delete after the full stack lands on `main`.
**Last updated:** 2026-05-04
**Canonical source:** [`will/ltx2_sr_port`](https://github.com/hao-ai-lab/FastVideo/tree/will/ltx2_sr_port) — the unified branch holding all 41 commits in dependency order.

This file tracks the 10-PR split of `will/ltx2_sr_port`. Each PR is a slice
of the canonical branch, stacked on its predecessor.

For per-PR design rationale see [`.agents/memory/dreamverse-integration/pr-roadmap.md`](.agents/memory/dreamverse-integration/pr-roadmap.md).
For decisions log see [`.agents/memory/dreamverse-integration/decisions-log.md`](.agents/memory/dreamverse-integration/decisions-log.md).
For co-author attribution see [`CO-AUTHORS.md`](CO-AUTHORS.md).

---

## Stack diagram

```
origin/main
  └─ will/api_7.6        [open PR #1257]    GPU pool + worker subprocess (7 commits)
     └─ will/api_7.7     [open PR #1258]    Prompt enhancer + LLMProvider (3 commits)
        └─ will/api_7.8                     Streaming auxiliaries (2 commits)
           └─ will/api_7.9                  Router upstream (3 commits)
              └─ will/api_7.10              generate_async + Dynamo (3 commits)
                 └─ will/api_8              Server contract docs (3 commits)
                    └─ will/ltx2_sr_runtime         LTX-2 SR + i2v + harness (9 commits)
                       └─ will/ltx2_nvfp4           NVFP4 + per-component compile (6 commits)
                          └─ will/ltx2_post_fixes   Post-handoff parity fixes (2 commits)
                             └─ will/agents_cleanup .agents/ cleanup + STACK.md (3 commits)
```

Total: **41 commits across 10 PRs** (mean 4.1 per PR — reviewable size).

## Branch table

| # | PR branch | Open PR | Slice | Tip SHA | Commits | Concern |
|---|---|---|---|---|---|---|
| 1 | `will/api_7.6` | **#1257** | 1–7 | _SEE NOTES_ | 7 | GPU pool + worker subprocess + test coverage + 4 review-fixup commits |
| 2 | `will/api_7.7` | **#1258** | 8–10 | _SEE NOTES_ | 3 | Prompt enhancer + LLMProvider abstraction (cerebras, groq) |
| 3 | `will/api_7.8` | (not yet) | 11–12 | _SEE NOTES_ | 2 | Streaming auxiliaries (safety, rewrite, logger, mock) |
| 4 | `will/api_7.9` | (not yet) | 13–15 | _SEE NOTES_ | 3 | Router upstream + `fastvideo router-serve` CLI + test coverage |
| 5 | `will/api_7.10` | (not yet) | 16–18 | _SEE NOTES_ | 3 | `VideoEvent` hierarchy + `VideoGenerator.generate_async` + Dynamo health helper |
| 6 | `will/api_8` | (not yet) | 19–21 | _SEE NOTES_ | 3 | Server contract docs (OpenAI HTTP + Dynamo) + Dreamverse/Dynamo shape contract tests |
| 7 | `will/ltx2_sr_runtime` | (not yet) | 22–30 | _SEE NOTES_ | 9 | LTX-2 SR runtime port + i2v conditioning + alignment harness against FastVideo-internal |
| 8 | `will/ltx2_nvfp4` | (not yet) | 31–36 | _SEE NOTES_ | 6 | NVFP4 wire-up + per-component compile + typed `transformer_quant` flow |
| 9 | `will/ltx2_post_fixes` | (not yet) | 37–38 | _SEE NOTES_ | 2 | Post-handoff parity fixes (Gemma `to()` round-trip, list-of-generators) |
| 10 | `will/agents_cleanup` | (not yet) | 39–41 | _SEE NOTES_ | 3 | `.agents/` Phase 1 cleanup + dreamverse-integration memory + STACK/CO-AUTHORS docs |

**Tip SHAs**: see `git log --oneline --reverse origin/main..will/ltx2_sr_port` for the
authoritative sequence. SHAs change whenever `will/ltx2_sr_port` is rebased
or amended (e.g. when injecting new `Co-authored-by` trailers). Rather than
hardcode them here (which goes stale), use the live re-slicing commands
below.

## Re-slice commands (run when `will/ltx2_sr_port` SHAs change)

After any rebase or trailer injection on `will/ltx2_sr_port`, recompute the
boundary SHAs and reset all split branches:

```bash
# Show the 41 commits with their slice numbers
git log --oneline --reverse origin/main..will/ltx2_sr_port | nl -ba

# Capture boundary tips by index
SHA_7_6=$(git log --oneline --reverse origin/main..will/ltx2_sr_port | sed -n '7p'  | awk '{print $1}')
SHA_7_7=$(git log --oneline --reverse origin/main..will/ltx2_sr_port | sed -n '10p' | awk '{print $1}')
SHA_7_8=$(git log --oneline --reverse origin/main..will/ltx2_sr_port | sed -n '12p' | awk '{print $1}')
SHA_7_9=$(git log --oneline --reverse origin/main..will/ltx2_sr_port | sed -n '15p' | awk '{print $1}')
SHA_7_10=$(git log --oneline --reverse origin/main..will/ltx2_sr_port | sed -n '18p' | awk '{print $1}')
SHA_8=$(git log --oneline --reverse origin/main..will/ltx2_sr_port | sed -n '21p' | awk '{print $1}')
SHA_LTX2_SR=$(git log --oneline --reverse origin/main..will/ltx2_sr_port | sed -n '30p' | awk '{print $1}')
SHA_LTX2_NVFP4=$(git log --oneline --reverse origin/main..will/ltx2_sr_port | sed -n '36p' | awk '{print $1}')
SHA_LTX2_POST=$(git log --oneline --reverse origin/main..will/ltx2_sr_port | sed -n '38p' | awk '{print $1}')
SHA_AGENTS=$(git log --oneline --reverse origin/main..will/ltx2_sr_port | sed -n '41p' | awk '{print $1}')

# Reset all 10 branches to the new boundary tips
git branch -f will/api_7.6           "$SHA_7_6"
git branch -f will/api_7.7           "$SHA_7_7"
git branch -f will/api_7.8           "$SHA_7_8"
git branch -f will/api_7.9           "$SHA_7_9"
git branch -f will/api_7.10          "$SHA_7_10"
git branch -f will/api_8             "$SHA_8"
git branch -f will/ltx2_sr_runtime   "$SHA_LTX2_SR"
git branch -f will/ltx2_nvfp4        "$SHA_LTX2_NVFP4"
git branch -f will/ltx2_post_fixes   "$SHA_LTX2_POST"
git branch -f will/agents_cleanup    "$SHA_AGENTS"
```

After resetting, force-push the branches whose PRs are open:

```bash
git push --force-with-lease origin will/api_7.6
git push --force-with-lease origin will/api_7.7
git push --force-with-lease origin will/ltx2_sr_port
```

## Independence map

| PR | Hard-depends on | Could rebase to land independently? |
|---|---|---|
| 7.6 | `origin/main` only | ✅ already independent |
| 7.7 | 7.6 (uses `GpuPool` from 7.6) | ❌ |
| 7.8 | 7.7 (auxiliaries plug into prompt enhancer) | ❌ |
| 7.9 | `origin/main` (router is orthogonal to streaming) | ✅ — could rebase onto 7.6 directly if 7.7/7.8 stall |
| 7.10 | 7.6 (`GpuPool`) + on-main `ContinuationState` (PR 7) | ✅ — only needs 7.6, can fast-track over the 7.7/7.8 stack |
| 8 | 7.10 (consumes `generate_async` + `VideoEvent`) | ❌ |
| `ltx2_sr_runtime` | `origin/main` + (independent of 7.x streaming work) | ✅ — purely LTX-2-specific |
| `ltx2_nvfp4` | `ltx2_sr_runtime` + per-component compile (which is on `ltx2_sr_runtime`) | ❌ |
| `ltx2_post_fixes` | `ltx2_nvfp4` | ❌ |
| `agents_cleanup` | NOTHING — pure infra | ✅ — fully independent of every other PR |

## Landing strategy

The default landing path is bottom-up through the stack: 7.6 → 7.7 → 7.8 →
7.9 → 7.10 → 8 → ltx2_sr_runtime → ltx2_nvfp4 → ltx2_post_fixes →
agents_cleanup.

After a PR merges, the canonical branch and remaining splits should be
rebased onto the new `origin/main` (which now contains the just-merged PR),
then re-sliced via the commands above.

If review on a mid-stack PR stalls, **fast-track candidates** are:

- `will/agents_cleanup` (no dependency)
- `will/ltx2_sr_runtime` (no dependency on 7.x streaming work)
- `will/api_7.9` (router is orthogonal — rebase to land directly on 7.6)
- `will/api_7.10` (only needs 7.6 — rebase to land directly on 7.6)

## Verification per PR

For each PR's branch, verify the slice content with:

```bash
# Show commits + verify count
git log --oneline <PARENT>..<BRANCH> | wc -l   # expect the count from the table

# Confirm tree state at the tip matches the canonical branch's slice tip
git diff <slice-tip-SHA-on-canonical> <BRANCH>  # expect empty
```

## When to delete this file

When all 10 PRs have merged to `main`. Until then, keep this file fresh:
re-run the re-slice commands and update any commit-count cells whose
sequence shifts.
