# Runbook — How to Do Work in This Scope

Operational how-to for the dreamverse-integration scope. Read after
[state.md](state.md) and [open-threads.md](open-threads.md).

For design rationale see [design.md](design.md). For who to credit see
[authors.md](authors.md). For PR status see [pr-roadmap.md](pr-roadmap.md).

**Last updated:** 2026-05-05 (strategy reversed to single mega-PR #1288 on `will/ltx2_sr_port`; #1287 closed; STACK.md split model deprecated per [decisions-log.md D-17](decisions-log.md#d-17)).

## Worktree contract

```
Repo:   /home/william5lin/FastVideo
Branch: will/ltx2_sr_port
```

Other agents and the user share this worktree concurrently. If `git status`
shows changes you don't recognize, they belong to **someone else's work** —
don't revert, don't `git stash drop`, don't `git checkout -- <file>`.
Switch to `will/ltx2_sr_port` cleanly with `git checkout will/ltx2_sr_port`
(safe if your own working tree is clean) and proceed.

If your task requires a different branch (e.g. cherry-pick to
`will/api_7.9` for PR #1286 propagation), return to `will/ltx2_sr_port`
when done — that is the assumed default.

## Branch topology (single mega-PR model)

The dreamverse-integration work now ships as one PR (#1288) off
`will/ltx2_sr_port`. The split-PR model documented in earlier revisions
of this runbook (and in top-level `STACK.md`) is **abandoned** —
see [decisions-log.md D-17](decisions-log.md#d-17).

```
origin/main
  ↓ [public-API refactor: PRs 0..7.9 merged on main, latest #1286 = 2aaeee2a]
will/ltx2_sr_port   (**PR #1288 head** — single mega-PR, 34 commits, 71 files, +13,074/-583)
```

| Branch | Role | Status |
|---|---|---|
| `will/ltx2_sr_port` | **PR #1288 head**, default working branch | OPEN, MERGEABLE |
| `will/api_7.10` / `will/api_8` / `will/ltx2_sr_runtime` / `will/ltx2_nvfp4` / `will/ltx2_post_fixes` / `will/agents_cleanup` | deprecated split-PR bookmarks | local-only historical references; safe to delete |
| `will/ltx2_sr_port-pre-1286-rebase` | safety backup | local-only; preserves the 4 commits dropped during the post-#1286 rebase |

**Sanity check:** `git merge-base --is-ancestor origin/main will/ltx2_sr_port`
should exit 0. If it doesn't, the branch is in an unexpected state — read
[state.md](state.md) before continuing.

## After PR #1288 merges

When the mega-PR squash-merges into `main`:

1. `git fetch origin main` to pull the merge commit.
2. The entire `will/ltx2_sr_port` content is now on main; the branch can
   be deleted (locally + on origin) once all consumers are notified.
3. Delete deprecated split bookmarks: `git branch -D will/api_7.10
   will/api_8 will/ltx2_sr_runtime will/ltx2_nvfp4 will/ltx2_post_fixes
   will/agents_cleanup` (local-only, no remote).
4. Optionally remove top-level `STACK.md` (now a historical artifact).
   Keep [co-authors.md](co-authors.md) — still the canonical roster reference.
5. Decide whether to keep `will/ltx2_sr_port-pre-1286-rebase` (safety
   backup of the pre-rebase chain) — recommend deleting once #1288 is
   merged and verified on main.
6. Update memory dir to reflect the post-merge state — bump
   `Last reconciled` headers, mark Item D resolved in
   [open-threads.md](open-threads.md), record the merge commit in
   [decisions-log.md](decisions-log.md).

## Historical: split-PR re-slice protocol (deprecated)

Prior revisions of this runbook documented a 10-step re-slice protocol
for the abandoned 6-PR split model. That protocol is now obsolete.
The post-#1286 rebase (2026-05-05) was the last execution of it; details
are preserved in [state.md](state.md) "Post-#1286 rebase summary" and
git history at commit `b34d9704`.

## Verification

### Lint (pre-commit)

```bash
pre-commit run --files <changed-paths...>
```

- Binary: `/home/william5lin/miniconda3/envs/fv-main/bin/pre-commit`.
  NOT `.venv/bin/pre-commit` — that doesn't exist in this worktree.
- Auto-applies yapf reformatting; re-stage modified files after.
- Hook chain: yapf → ruff → codespell → mypy → spaces-check.
- Memory dir (`.agents/memory/`) is yapf/ruff/mypy excluded — only
  "spaces" runs. Memory edits don't need lint, but DO use UTF-8 and
  consistent line endings.

### Tests

Router tests (PR #1286 scope):
```bash
.venv/bin/python -m pytest fastvideo/tests/entrypoints/streaming/test_router.py -v --no-header
```

Stack baseline (May 2 handoff suite — re-run when you change anything in
api/, contract/, or LTX-2 paths):
```bash
.venv/bin/python -m pytest \
    fastvideo/tests/api/ \
    fastvideo/tests/contract/ \
    fastvideo/tests/ops/quantization/test_nvfp4_*.py \
    tests/local_tests/pipelines/test_ltx2_pipeline_smoke.py \
    -q --no-header
```

Expected baselines:
- May 2 handoff (`156103b9`): 222 passed, 1 skipped.
- Post-D-16 (`a152cb77` / `09647a30`): +7 router tests pass on top.

### LSP

Use `lsp_diagnostics` on changed files BEFORE running build. Pre-existing
warnings to ignore (predate this work):

- `fastvideo/entrypoints/streaming/router/main.py:37` — `Task` generic.
- `fastvideo/entrypoints/cli/router_serve.py:55` — `_SubParsersAction` generic.

### gh CLI for PR status

```bash
# PR #1286 quick status
gh pr view 1286 --json headRefOid,mergeable,statusCheckRollup \
  --jq '{headRefOid, mergeable, checks: [.statusCheckRollup[] | {name, status, conclusion}]}'

# All commits in a PR + co-author check
gh pr view 1286 --json commits \
  --jq '.commits[] | {oid: .oid[0:8], msg: .messageHeadline, author: .authors[0].login}'
```

## Commit workflow

### Subject convention

`[type] <scope>: <imperative summary>` — keep ≤ 72 chars.

Types observed in this scope: `feat`, `fix`, `test`, `docs`, `chore`,
`refactor`. Scopes observed: `streaming`, `dreamverse-integration`,
`api`, `quant`, `ltx2`, `nvfp4`, etc.

Examples:
- `[fix] streaming: router polish — bridge cancel + state machine + deps`
- `[docs] dreamverse-integration: add authors.md + track D-16 router polish`

### Body convention

Bullet list, one bullet per file or concern. Why-before-what. Wrap at
~80 chars (yapf doesn't reformat commit messages; readability is on you).

### Co-author trailers (REQUIRED on every commit)

The 4 trailers in [authors.md](authors.md) MUST appear on every commit
in this scope. Use `--trailer` flags or write the body to a file with
`-F` — DO NOT use multiple `-m` blocks for the trailers (each `-m` is
its own paragraph and git's trailer parser only reads the LAST paragraph,
yielding 1 trailer parsed instead of 4).

**Inline `--trailer` form (preferred for short commits):**

```bash
git commit -m "subject" -m "body..." \
  --trailer "Co-authored-by: Junda (David) Su <90978028+Davids048@users.noreply.github.com>" \
  --trailer "Co-authored-by: Matthew Noto <99706358+RandNMR73@users.noreply.github.com>" \
  --trailer "Co-authored-by: XOR-op <17672363+XOR-op@users.noreply.github.com>" \
  --trailer "Co-authored-by: Zhang Peiyuan <42993249+jzhang38@users.noreply.github.com>"
```

**File form (preferred for multi-paragraph bodies):**

```bash
cat > /tmp/opencode/msg.txt <<'EOF'
[type] scope: subject

* Bullet one with rationale.
* Bullet two with rationale.

Co-authored-by: Junda (David) Su <90978028+Davids048@users.noreply.github.com>
Co-authored-by: Matthew Noto <99706358+RandNMR73@users.noreply.github.com>
Co-authored-by: XOR-op <17672363+XOR-op@users.noreply.github.com>
Co-authored-by: Zhang Peiyuan <42993249+jzhang38@users.noreply.github.com>
EOF
git commit -F /tmp/opencode/msg.txt
```

The trailers MUST be a single block at the end of the message with no
blank lines between them.

**Verify trailers parsed:**

```bash
git log -1 --format='%(trailers:key=Co-authored-by,valueonly)'
```

Should print 4 lines (one per author). If only 1 line, you have the
multi-`-m` bug — amend with `-F` to fix (allowed if commit is unpushed
and you authored it in this session per AGENTS.md amend rules).

### NEVER add to commits

Per [`AGENTS.md`](../../../AGENTS.md):

- AI co-authors (Claude, GPT, Codex, Cursor, etc.) — explicitly forbidden
- "Generated with Claude Code" footer — explicitly forbidden
- `--no-verify` to skip pre-commit — explicitly forbidden

## Push + PR propagation

### Pushing `will/ltx2_sr_port` (top of stack)

```bash
git push origin will/ltx2_sr_port  # fast-forward, no force needed
```

If git wants to force-push, you've rewritten history. STOP and verify:

```bash
git log origin/will/ltx2_sr_port..will/ltx2_sr_port  # local-only commits
git log will/ltx2_sr_port..origin/will/ltx2_sr_port  # remote-only commits
```

Force-push requires explicit user confirmation per `AGENTS.md`.

### Propagating fixes to PR #1286 (`will/api_7.9`)

When a fix is in router code (`fastvideo/entrypoints/streaming/router/`,
`cli/router_serve.py`, `tests/entrypoints/streaming/test_router.py`,
or `pyproject.toml` router-related), it must land on BOTH branches.
Cherry-pick avoids any force-push:

```bash
# 1. Commit on will/ltx2_sr_port first (working branch)
git add <files...>
git commit -F /tmp/opencode/msg.txt   # with trailers per above

# 2. Cherry-pick onto will/api_7.9 (creates a separate SHA, identical diff)
git checkout will/api_7.9
git cherry-pick <ltx2_sr_port-sha>
git push origin will/api_7.9          # fast-forward, no force

# 3. Return to working branch
git checkout will/ltx2_sr_port

# 4. Verify PR #1286 picked it up
gh pr view 1286 --json headRefOid --jq '.headRefOid'
```

Two SHAs for the same diff — they'll dedupe naturally on the next
bulk-rebase via the trailer-injection rebase command in
[authors.md](authors.md).

### When a fix is memory-dir-only

`.agents/memory/dreamverse-integration/` lives in the `agents_cleanup`
layer of the stack — it does NOT belong on `will/api_7.9`. Memory updates
stay on `will/ltx2_sr_port` only.

### When a fix is non-router code in the integration scope

Land on `will/ltx2_sr_port`. If that fix needs to ship as a separate PR
(e.g. extending PR 7.10 or starting PR 9), open a new branch off the
right base per [pr-roadmap.md](pr-roadmap.md).

## Memory dir maintenance

When state changes, update the memory dir BEFORE moving on. Every file
has a "Last updated" header — bump when you edit.

| Change | File to update |
|---|---|
| Branch tip moves | [state.md](state.md) "Branch tips" + "Last reconciled" |
| PR opens / merges | [pr-roadmap.md](pr-roadmap.md) status table |
| New decision made | [decisions-log.md](decisions-log.md) — add D-N entry, bump header |
| Open thread resolved | [open-threads.md](open-threads.md) — strikethrough + "Resolved" note |
| New open thread | [open-threads.md](open-threads.md) — priority overview + section |
| New collaborator credited | [authors.md](authors.md) roster + trailer block + bulk-rebase |
| Source doc archived | [source-archive/README.md](source-archive/README.md) + [README.md](README.md) sources table |
| Process / runbook detail changes | [runbook.md](runbook.md) (this file) |

Cross-link siblings via relative paths. Never duplicate content — link.

## Common pitfalls

### `pre-commit` not in `.venv/bin`

`pre-commit` lives at `/home/william5lin/miniconda3/envs/fv-main/bin/pre-commit`.
The `.venv` here is for the FastVideo package itself, not pre-commit.

### Trailers split across paragraphs

`git commit -m A -m B -m C` makes A, B, C separate paragraphs. Git's
trailer parser only reads the LAST paragraph — multiple `-m
"Co-authored-by: ..."` produces 1 trailer parsed, not 4. Use `--trailer`
flags or `-F` with the trailers in a single block at the end.

### Stash 0 on FastVideo IS NOT yours

`stash@{0}: WIP on main: 71bfc13d HunyuanVideo plugin` predates this work.
**DO NOT POP.** See [state.md](state.md) "Stashes — DO NOT POP".

### `AbsMaxFP8` test "failure" is pre-existing

`fastvideo/tests/ops/quantization/test_absmax_fp8.py::test_create_weights_rejects_invalid_dtype`
fails on `main` and on every branch in this scope. NOT introduced by
integration work. See [open-threads.md](open-threads.md) item #2.

### Untracked nested clones at repo root

`dynamo/`, `ray/`, `vllm-omni/` are untracked nested git clones at the
FastVideo repo root. Reference repos for cross-repo work. **Do not
`rm -rf`** — they're someone else's working state.

### Live services on 8009 / 5274

`dreamverse-server` runs on 8009 (warmed GPU worker), Next.js dev server
on 5274. Don't start new instances on those ports without checking
[state.md](state.md) "Live services" first.

### Branch may have been switched by another agent

Other agents share this worktree. If `git branch --show-current` returns
something other than `will/ltx2_sr_port`, switch back cleanly with
`git checkout will/ltx2_sr_port` — don't disturb their work, don't
discard their uncommitted changes.

### Force-push policy

Per `AGENTS.md`: never force-push without explicit user confirmation.
For trailer fixes on already-pushed commits, prefer the bulk-rebase
command in [authors.md](authors.md) — safe to re-run.

### Two trailerless commits in PR #1286

`a152cb77` (on `will/api_7.9`) and `40e265b8` (now-superseded ancestor
on `will/ltx2_sr_port`) lack the 4 co-author trailers. **Accepted gap**
per user decision — see [authors.md](authors.md) "Known gaps".

## Self-test (verify your context is loaded)

After reading the memory dir, you should be able to answer:

1. What branch should I be on? → `will/ltx2_sr_port`
2. What's the active open PR in this scope? → #1286 on `will/api_7.9`
3. Where does PR #1286 land in the stack? → Bottom; ancestor of `will/ltx2_sr_port`
4. Who do I credit on every commit? → 4 authors per [authors.md](authors.md)
5. Where do memory updates land? → `will/ltx2_sr_port` only (NOT api_7.9)
6. What's the next-priority open thread? → See [open-threads.md](open-threads.md) "Recommended pull order" — D-8 verify is current top
7. What pre-existing failure can I ignore? → AbsMaxFP8 test (item #2)
8. What's the bulk-rebase command for adding trailers across the stack? → See [authors.md](authors.md) "How the trailers were applied"

If you can't answer one of these from the memory dir alone, the dir has
a gap — file it as a new entry in [open-threads.md](open-threads.md)
before continuing.

## First 60 seconds — copy-paste orientation

```bash
# 1. Confirm branch
cd /home/william5lin/FastVideo
git branch --show-current   # should print: will/ltx2_sr_port
# If not, recover: git checkout will/ltx2_sr_port

# 2. Confirm worktree clean (untracked nested clones expected)
git status --short

# 3. Confirm PR #1286 head matches expected api_7.9 tip
gh pr view 1286 --json headRefOid --jq '.headRefOid'
git rev-parse will/api_7.9   # should match PR head

# 4. Confirm your context vs the memory dir
git log -1 --oneline
cat .agents/memory/dreamverse-integration/state.md | head -30

# 5. Confirm live services still running
curl -s http://localhost:8009/readyz | head -c 200
curl -s http://localhost:5274/ -o /dev/null -w "%{http_code}\n"
```

If any of those produce unexpected output, read [state.md](state.md)
before changing anything.
