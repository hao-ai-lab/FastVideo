# Co-Authors — `will/ltx2_sr_port` Stack

**Status:** PERMANENT — keep around as the source of truth for who collaborated on this work, even after the stack merges.
**Last updated:** 2026-05-04

This file documents the human co-authors credited on every commit in the
`will/ltx2_sr_port` stack and its 10 split PRs. The 4 collaborators below
worked on the FastVideo-internal precursor of this code (LTX-2 streaming
server, NVFP4 wire-up, GPU pool, prompt enhancer, etc.) and are credited as
co-authors on the public-side upstream commits via Git's standard
[`Co-authored-by`](https://docs.github.com/en/pull-requests/committing-changes-to-your-project/creating-and-editing-commits/creating-a-commit-with-multiple-authors)
trailer convention.

The trailers are added to every commit on `will/ltx2_sr_port` (see
[`STACK.md`](STACK.md)), which means GitHub will:

- Show the 4 co-authors on every commit detail page
- Show them on the merge commit / squash commit summary
- Display their avatars in the PR's "Contributors" sidebar
- Surface them in [`/contributors`](https://github.com/hao-ai-lab/FastVideo/contributors) once the stack lands

## Co-author roster

| GitHub user | Real name | GitHub ID | Trailer email |
|---|---|---|---|
| [`@Davids048`](https://github.com/Davids048) | Junda (David) Su | 90978028 | `90978028+Davids048@users.noreply.github.com` |
| [`@RandNMR73`](https://github.com/RandNMR73) | Matthew Noto | 99706358 | `99706358+RandNMR73@users.noreply.github.com` |
| [`@XOR-op`](https://github.com/XOR-op) | (unset) | 17672363 | `17672363+XOR-op@users.noreply.github.com` |
| [`@jzhang38`](https://github.com/jzhang38) | Zhang Peiyuan | 42993249 | `42993249+jzhang38@users.noreply.github.com` |

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

## Trailer block (copy-paste ready)

The trailers added to every commit on `will/ltx2_sr_port`:

```
Co-authored-by: Junda (David) Su <90978028+Davids048@users.noreply.github.com>
Co-authored-by: Matthew Noto <99706358+RandNMR73@users.noreply.github.com>
Co-authored-by: XOR-op <17672363+XOR-op@users.noreply.github.com>
Co-authored-by: Zhang Peiyuan <42993249+jzhang38@users.noreply.github.com>
```

## How the trailers were applied

```bash
git rebase --exec '
  git commit --amend --no-edit \
    --trailer "Co-authored-by: Junda (David) Su <90978028+Davids048@users.noreply.github.com>" \
    --trailer "Co-authored-by: Matthew Noto <99706358+RandNMR73@users.noreply.github.com>" \
    --trailer "Co-authored-by: XOR-op <17672363+XOR-op@users.noreply.github.com>" \
    --trailer "Co-authored-by: Zhang Peiyuan <42993249+jzhang38@users.noreply.github.com>"
' origin/main will/ltx2_sr_port
```

Git's `--trailer` flag is idempotent (it dedupes by the full `key: value`
string), so re-running the rebase is safe and won't add duplicates.

## How to add a new co-author later

1. Add the user to the roster table above.
2. Append their `Co-authored-by` line to the trailer block.
3. Re-run the rebase command above on `will/ltx2_sr_port` — git's
   trailer dedupe handles the existing 4; the new one gets appended.
4. Re-slice all 10 split branches per [`STACK.md`](STACK.md).
5. Force-push `will/api_7.6`, `will/api_7.7`, and `will/ltx2_sr_port`.

## What we do NOT add

Per [`AGENTS.md`](AGENTS.md):

> Never add any coding agent or models such as Claude (or Claude Code), GPT,
> Codex or others as a co-author in commits or PRs.

So no `Co-authored-by: Claude <noreply@anthropic.com>` or similar. Only
human collaborators.
