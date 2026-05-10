# `.agents/` Cleanup Log — Phase 1 (Deletes Only)

**Status:** TEMPORARY — delete this file after the cleanup is reviewed/committed.
**Date:** 2026-05-04
**Branch:** `will/ltx2_sr_port`
**Scope:** Phase 1 of the `.agents/` cleanup plan (deletes only; no rewrites or additions).

For the full multi-phase plan, see the prior session analysis. This file tracks
exactly what got deleted, why, and what cross-references still point at deleted
content (to fix in a future phase).

---

## Deletions executed

### Files deleted

| Path | Size | Reason |
|---|---|---|
| `.agents/STATUS.md` | 3.85 KB | Stale dashboard, last synced 2026-03-02. Counts wrong (claimed 8 skills/4 workflows/4 memory; actual 9/5/5). References old snake_case filenames (`codebase_map.md`/`experiment_journal.md`) that don't exist. Hand-maintained derivative of `.agents/{memory,skills}/index.jsonl` — strictly redundant. |
| `.agents/exploration/pr-link-review.md` | 1.11 KB | Status: "promoted" to `.agents/skills/review-pr-link/`. Per `.agents/exploration/README.md` lifecycle, promoted exploration logs should not linger after the skill exists. |
| `.agents/workflows/sync-dashboard.md` | 1.87 KB | SOP for maintaining `STATUS.md` (which is also deleted). Contained obsolete file paths (`.agents/skills/launch-experiment.md` flat layout vs. actual `<skill>/SKILL.md` per-dir layout). Has never been run successfully (judging by stale dates everywhere). |

### Skill directories deleted

| Path | Size | Reason |
|---|---|---|
| `.agents/skills/index-related-work/` | 2.18 KB | Vapor-skill operating on the empty `.agents/memory/related-work/` registry. Never used (the registry has zero entries despite ~6 weeks since skill creation). Re-add when the related-work catalog gains entries. |
| `.agents/skills/search-related-work/` | 1.91 KB | Same: vapor-skill against empty registry. The skill description literally requires "The related work index has entries" as a prerequisite, and there are none. |

**Total deleted: 5 items, ~10.9 KB.**

### Registry updates

| File | Change |
|---|---|
| `.agents/skills/index.jsonl` | Removed entries for `index-related-work` and `search-related-work`. Was 9 entries; now 7. |

### Symlink hygiene

`.agents/scripts/sync-skills.sh` was run to prune now-stale symlinks under
`.claude/skills/` that pointed at the deleted skill directories. Output captured
in the run log.

---

## What was KEPT (despite being candidates)

| Path | Why kept |
|---|---|
| `.agents/scripts/sync-skills.sh` | User explicitly requested keep. **Verified**: this script is INDEPENDENT of STATUS.md / sync-dashboard.md. It mirrors `.agents/skills/` → `.claude/skills/` via symlinks for Claude Code skill discovery. Self-contained, useful, prunes its own stale symlinks. |
| `.agents/memory/related-work/README.md` | Empty placeholder, but the schema/template is reusable. Kept for when first related-work entry is added. |
| `.agents/memory/experiment-journal/README.md` | Same: empty placeholder with template; kept for when journaling begins. |
| `.agents/lessons/README.md` | Same: empty placeholder, reusable schema. |
| `.agents/exploration/README.md` | Active template for new exploration logs. Kept. |

---

## Remaining broken cross-references (FOLLOW-UP NEEDED)

These files still reference deleted content. **NOT fixed in Phase 1** — track for
the next pass (Phase 2: rewrites/dedupe).

### References to deleted `STATUS.md`

| Referencing file | Action needed |
|---|---|
| `.agents/onboarding/README.md` | Quick-reference tree (line ~65) lists `STATUS.md ← dashboard: completeness & trust of all components`. Remove that line + the `ONBOARDING.md` typo (file is `README.md`). |

### References to deleted `pr-link-review.md`

| Referencing file | Action needed |
|---|---|
| `.agents/memory/dreamverse-integration/state.md` | "Untracked but present" / "Source docs (archived)" sections still mention `pr-link-review.md` as kept. Update to reflect deletion. |
| `.agents/memory/dreamverse-integration/README.md` | Same — table row for `pr-link-review.md` says "kept in exploration dir". Update or remove the row. |

### References to deleted skills (`index-related-work`, `search-related-work`)

| Referencing file | Action needed |
|---|---|
| `.agents/memory/related-work/README.md` | Says "Use the `index-related-work` skill". Either remove that hint or note "skill removed; re-add when registry has entries". |
| `.agents/workflows/evaluation-development.md` | Step 1 says "Search `.agents/memory/related-work/` for existing evaluation approaches" — that's still valid (manual search). No change needed. |

### References to deleted `sync-dashboard.md`

| Referencing file | Action needed |
|---|---|
| `.agents/memory/evaluation-registry/README.md` | Doesn't reference sync-dashboard directly. No change. |
| `.agents/STATUS.md` | Already being deleted. |

---

## Other registry inconsistencies discovered (NOT FIXED in Phase 1)

While editing `.agents/skills/index.jsonl`, two skill directories were found
that exist on disk but **are not registered** in `index.jsonl`:

| Skill dir | Status | Why missing from index |
|---|---|---|
| `.agents/skills/diagnose-ssim-failure/` | Untracked locally; NOT on `origin/main`. 12.3 KB SKILL.md + `scripts/compare_latent_pt.py`. Recent mtime (2026-05-01). | Created in a prior session but the registration step was skipped. |
| `.agents/skills/review-pr-link/` | Untracked locally; NOT on `origin/main`. 2.9 KB SKILL.md + `scripts/prepare_pr_review.py` + `agents/openai.yaml`. The promotion target of the deleted `pr-link-review.md` exploration log. | Skipped registration when promoted from exploration log. |

Both skills are functional and exposed via `sync-skills.sh` symlinks (just verified in
`.claude/skills/`), but agents reading `index.jsonl` to discover skills will miss them.

**Action for Phase 2**: Add entries to `.agents/skills/index.jsonl` for both,
likely with `trust: medium` since they have working scripts and recent use.

---

## Skill registry parity check

After Phase 1, `.agents/skills/` contains 9 directories but `index.jsonl` lists 7:

| In `index.jsonl` | On disk |
|---|---|
| ✓ launch-experiment | ✓ launch-experiment/ |
| ✓ monitor-experiment | ✓ monitor-experiment/ |
| ✓ summarize-run | ✓ summarize-run/ |
| ✓ log-experiment | ✓ log-experiment/ |
| ✓ evaluate-video-quality | ✓ evaluate-video-quality/ |
| ✓ seed-ssim-references | ✓ seed-ssim-references/ |
| ✓ reseed-ssim-references | ✓ reseed-ssim-references/ |
| ❌ (missing) | ⚠ diagnose-ssim-failure/ |
| ❌ (missing) | ⚠ review-pr-link/ |

`.claude/skills/` symlinks (the runtime-discoverable surface) include all 9 ✓.

---

## Phase 2+ items (NOT executed in this session)

For future cleanup sessions, the prior plan identified:

**Phase 2 (rewrites)**:
- Rewrite `.agents/onboarding/worldmodel-training/README.md` to drop ~50% structural duplication with `codebase-map/README.md`
- Refresh `.agents/memory/codebase-map/README.md` (last updated 2026-03-08; missing `fastvideo/api/`, `fastvideo/entrypoints/streaming/`, etc.)
- Refresh `.agents/memory/evaluation-registry/README.md` (last updated 2026-03-02; references old `evaluation_registry.md` filename)
- Merge `.agents/workflows/experiment-journaling.md` into `experiment-lifecycle.md` (one SOP per workflow)
- Fix the broken cross-references listed above

**Phase 3 (additions)**:
- `fastvideo/api/AGENTS.md`
- `fastvideo/entrypoints/AGENTS.md`
- `tests/AGENTS.md` (top-level, distinct from `fastvideo/tests/AGENTS.md`)
- `fastvideo/distributed/AGENTS.md`
- `examples/AGENTS.md`
- `docs/AGENTS.md`
- `benchmarks/AGENTS.md`

**Phase 4 (registry)**:
- Add `.agents/workflows/index.jsonl`
- Standardize all three index.jsonl schemas

**Phase 5 (skills quality)**:
- Promote tested skills (`seed-ssim-references`, `reseed-ssim-references`, `diagnose-ssim-failure`, `review-pr-link`) from `trust: low` to `trust: medium`
- Mark untested skills (`launch-experiment`, `monitor-experiment`, `summarize-run`, `log-experiment`, `evaluate-video-quality`) explicitly with their gating prerequisite (e.g. "operates on empty registry")

---

## Recovery

All deletions are local (`will/ltx2_sr_port`, not committed). To restore any
deleted file:

```bash
git restore --source=HEAD .agents/STATUS.md
git restore --source=HEAD .agents/exploration/pr-link-review.md
git restore --source=HEAD .agents/workflows/sync-dashboard.md
git restore --source=HEAD .agents/skills/index-related-work/SKILL.md
git restore --source=HEAD .agents/skills/search-related-work/SKILL.md
```

---

## When to delete THIS file

Once:
1. The Phase 1 deletions are committed (or merged), AND
2. Phase 2 (broken cross-reference cleanup) is also committed,

remove this file. Its purpose is transient bookkeeping for a multi-phase cleanup.
