# Authors — Dreamverse Integration

**Status:** PERMANENT — keep around as the source of truth for who collaborated
on the dreamverse-integration work, even after every PR in the integration
scope has merged.
**Last updated:** 2026-05-05 (strategy reversal — single mega-PR #1288 on `will/ltx2_sr_port` replaces planned 6-PR split; #1287 closed; per [decisions-log.md D-17](decisions-log.md#d-17))

This file documents the human co-authors credited on every commit in the
dreamverse-integration scope (FastVideo public-API refactor, streaming server
upstream, GPU pool, prompt enhancer, NVFP4 wire-up, LTX-2 SR port). The
4 collaborators below worked on the FastVideo-internal precursor of this code
and are credited as co-authors on every public-side upstream commit via Git's
standard
[`Co-authored-by`](https://docs.github.com/en/pull-requests/committing-changes-to-your-project/creating-and-editing-commits/creating-a-commit-with-multiple-authors)
trailer convention.

Scope-wise this is the dreamverse-integration-flavored mirror of
[co-authors.md](co-authors.md), which is scoped to the broader
`will/ltx2_sr_port` 10-PR stack. The roster is identical; both files now live
in this memory dir so the provenance docs stay self-contained and discoverable.

## Co-author roster

| GitHub user | Real name | GitHub ID | Trailer email |
|---|---|---|---|
| [`@Davids048`](https://github.com/Davids048) | Junda (David) Su | 90978028 | `90978028+Davids048@users.noreply.github.com` |
| [`@RandNMR73`](https://github.com/RandNMR73) | Matthew Noto | 99706358 | `99706358+RandNMR73@users.noreply.github.com` |
| [`@XOR-op`](https://github.com/XOR-op) | (unset) | 17672363 | `17672363+XOR-op@users.noreply.github.com` |
| [`@jzhang38`](https://github.com/jzhang38) | Zhang Peiyuan | 42993249 | `42993249+jzhang38@users.noreply.github.com` |

## Verification — where these trailers appear

Verified via `gh pr view <PR> --json commits --jq '.commits[].messageBody'`
across every PR in the integration scope:

| PR | Branch | Status | Trailers present on every commit |
|---|---|---|---|
| #1257 | `will/api_7.6` (GPU pool upstream) | ✅ merged 2026-05-04 | yes (4/4) |
| #1258 | `will/api_7.7` (prompt enhancer + LLMProvider) | ✅ merged 2026-05-04 | yes (3/3) |
| #1284 | `will/api_7.8` (streaming auxiliaries) | ✅ merged 2026-05-04 | yes (2/2) |
| #1286 | `will/api_7.9` (streaming router) | ✅ merged 2026-05-05 at `2aaeee2a` (squash) | yes on commits 1-3; commit `a152cb77` (`[fix] streaming: router polish`) was missing trailers but got squashed into the merge commit, so the merge commit on main inherits the trailers from the other 3. The trailerless cherry-pick partner (`40e265b8` on `will/ltx2_sr_port`) was dropped by the post-#1286 rebase — gap permanently resolved. |
| #1287 | `will/api_7.10` (`generate_async` + `VideoEvent`) | ❌ CLOSED 2026-05-05 — superseded by #1288 per [D-17](decisions-log.md#d-17) | yes on all 3 commits (now part of #1288's chain) |
| **#1288** | **`will/ltx2_sr_port`** (mega-PR — full stack: SR runtime + NVFP4 + generate_async + Dynamo contract + agents memory + integration-review) | 🟢 OPEN, MERGEABLE at `b36bdbc9`, 36 commits / 70 files / ~+13.0k LOC (post STACK.md removal) | yes on all 36 commits |

Aggregate count across `will/ltx2_sr_port` (top of stack) at the time of
writing: 32-33 commits per co-author, matching the 32 commits in the stack
on top of base `cfccd292`. Numbers stay consistent because the rebase
command (see "How the trailers were applied" below) walks every commit.

## Trailer block (copy-paste ready)

The trailers added to every commit on `will/ltx2_sr_port` and every
dreamverse-integration PR:

```
Co-authored-by: Junda (David) Su <90978028+Davids048@users.noreply.github.com>
Co-authored-by: Matthew Noto <99706358+RandNMR73@users.noreply.github.com>
Co-authored-by: XOR-op <17672363+XOR-op@users.noreply.github.com>
Co-authored-by: Zhang Peiyuan <42993249+jzhang38@users.noreply.github.com>
```

For one-off `git commit -m` invocations, use `--trailer` flags:

```bash
git commit -m "..." \
  --trailer "Co-authored-by: Junda (David) Su <90978028+Davids048@users.noreply.github.com>" \
  --trailer "Co-authored-by: Matthew Noto <99706358+RandNMR73@users.noreply.github.com>" \
  --trailer "Co-authored-by: XOR-op <17672363+XOR-op@users.noreply.github.com>" \
  --trailer "Co-authored-by: Zhang Peiyuan <42993249+jzhang38@users.noreply.github.com>"
```

`--trailer` is idempotent (dedupes by full `key: value`) so re-running is safe.

## Why no-reply emails

GitHub's `<id>+<username>@users.noreply.github.com` form is the most reliable
way to link a `Co-authored-by` trailer to a GitHub account. It:

- Always works regardless of whether the user has a public verified email
- Survives the user changing their primary email
- Doesn't expose anyone's personal email to git history
- Is the format GitHub itself produces when you click "Add co-author" in the
  web UI

(All 4 collaborators have this email already used in `FastVideo-internal`
git history, verified via `git log --all` on that repo.)

## How the trailers were applied (bulk rebase)

```bash
git rebase --exec '
  git commit --amend --no-edit \
    --trailer "Co-authored-by: Junda (David) Su <90978028+Davids048@users.noreply.github.com>" \
    --trailer "Co-authored-by: Matthew Noto <99706358+RandNMR73@users.noreply.github.com>" \
    --trailer "Co-authored-by: XOR-op <17672363+XOR-op@users.noreply.github.com>" \
    --trailer "Co-authored-by: Zhang Peiyuan <42993249+jzhang38@users.noreply.github.com>"
' origin/main will/ltx2_sr_port
```

After running, re-slice all 10 split branches per [`STACK.md`](../../../STACK.md)
and force-push the published branches (`will/api_7.9`, `will/ltx2_sr_port`).

## How to add a new co-author later

1. Add the user to the roster table above (and [co-authors.md](co-authors.md)
   — keep them in sync).
2. Append their `Co-authored-by` line to the trailer block above.
3. Re-run the bulk rebase command on `will/ltx2_sr_port` — git's trailer
   dedupe handles the existing 4; the new one gets appended.
4. Re-slice all split branches per [`STACK.md`](../../../STACK.md).
5. Force-push the published branches.

## What we do NOT add

Per the repo's top-level [`AGENTS.md`](../../../AGENTS.md):

> Never add any coding agent or models such as Claude (or Claude Code), GPT,
> Codex or others as a co-author in commits or PRs. Do not include
> `Co-Authored-By: Claude ...` trailers or "Generated with Claude Code" and
> other such lines.

So no `Co-authored-by: Claude <noreply@anthropic.com>`, no
`Generated with Claude Code` footer, no `Cursor <cursoragent@cursor.com>`
trailer (one such commit exists on `will/ltx2_sr_port` from a pre-policy
external contribution and stays grandfathered; new commits MUST NOT introduce
the pattern). Only human collaborators.

## Known gaps

**Resolved 2026-05-05 by the post-#1286 rebase.** The two trailerless
commits (`a152cb77` on `will/api_7.9` and `40e265b8` on
`will/ltx2_sr_port`) are no longer reachable from any active branch:

- `a152cb77` was absorbed into squash merge `2aaeee2a` on main, which
  inherits the trailers from the other 3 commits in the squash.
- `40e265b8` was dropped by the post-#1286 rebase of
  `will/ltx2_sr_port`.

Both still exist on the local backup `will/ltx2_sr_port-pre-1286-rebase`
for archeological reference. No further action needed.

## See also

- [co-authors.md](co-authors.md) — stack-scoped co-authors file (same roster,
  broader scope)
- [`../../../STACK.md`](../../../STACK.md) — 10-PR split layout for
  `will/ltx2_sr_port` (re-slice commands live here)
- [`pr-roadmap.md`](pr-roadmap.md) — per-PR status within the
  dreamverse-integration scope
