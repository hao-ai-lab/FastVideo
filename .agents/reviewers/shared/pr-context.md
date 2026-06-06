# Fetching PR context

Every reviewer needs the same baseline context about a PR. This doc is the
single source of truth for how to gather it via `gh` CLI. Reviewers reference
this instead of duplicating the commands.

## Required context

Before producing any review, a reviewer must have:

1. **Metadata**: number, title, author, base branch, draft status, labels.
2. **Description**: PR body (purpose, changes, test plan, test results).
3. **Diff**: unified diff or per-file changes.
4. **File list**: paths + per-file additions/deletions, to decide scope.
5. **CI status**: whether required checks (`pre-commit`, `fastcheck-passed`,
   `full-suite-passed`) have run and passed.
6. **Existing review comments**: to avoid re-raising points already discussed.
7. **Linked issues**: anything referenced as `Fixes #NNNN` in the body.

## Commands

All commands assume `gh` is authenticated and the working dir is the repo.

### One-shot metadata bundle

```bash
gh pr view "$PR" --json \
  number,title,author,baseRefName,isDraft,labels,body,additions,deletions,\
commits,files,reviewDecision,mergeable,mergeStateStatus,statusCheckRollup
```

### Unified diff

```bash
gh pr diff "$PR"
```

For large diffs (> ~1000 lines), prefer fetching per-file diffs and
concentrating on the files relevant to the reviewer's scope — see ROUTING.md.

### Existing review comments (top-level + inline)

```bash
# Review-level comments (left via the "Review" UI)
gh pr view "$PR" --json reviews

# Inline review comments on specific lines
gh api "repos/:owner/:repo/pulls/$PR/comments"

# Issue-style comments (left via the conversation tab)
gh api "repos/:owner/:repo/issues/$PR/comments"
```

### CI status

```bash
gh pr checks "$PR"
```

Focus on **`pre-commit`**, **`fastcheck-passed`**, and **`full-suite-passed`**
(the three checks required by `.github/mergify.yml`'s `merge_protections`).

### Linked issues

Parse `body` for `Fixes #NNNN` / `Closes #NNNN`. For each:

```bash
gh issue view "$N" --json title,body,labels
```

## Pre-review sanity checks

Before running scope-specific logic, every reviewer should confirm:

- [ ] PR is not a draft (unless caller passed `--include-drafts`).
- [ ] PR is not labeled `needs-rebase`. If it is, emit a single note and exit.
- [ ] The changed file list includes at least one path in the reviewer's
      scope — if not, the dispatcher routed incorrectly; emit a note and exit.
- [ ] `pre-commit` check has run. If it failed, surface that first and proceed
      with the rest of the review (don't block — the human fixes style).

## Scope reduction

Reviewers must **narrow the diff to their scope** before commenting. A `model`
reviewer should not comment on CI workflow changes even if the PR touches
both. The dispatcher runs multiple reviewers precisely so each can stay
focused.

Concretely:

```bash
# Filter the file list to paths the reviewer owns
gh pr view "$PR" --json files -q '.files[].path' | grep -E '^(fastvideo/models|fastvideo/configs/models)/'
```

Then request the diff for just those files:

```bash
gh pr diff "$PR" -- <scoped-paths>
```

## Caching

If the dispatcher runs multiple reviewers on the same PR, fetch the bundle
**once** and pass it to each. The `gh` API has a rate limit and re-fetching
is wasteful.
