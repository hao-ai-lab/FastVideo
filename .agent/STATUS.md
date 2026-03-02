# Agent Infrastructure — Status Dashboard

Developer-maintained overview of all agent components and their maturity.
Use this to understand what exists, how complete it is, and how much to trust it.

_Last synced: 2026-03-02_

> To resync this dashboard, use the workflow: `.agent/workflows/sync-dashboard.md`

---

## Summary

| Category | Total | ✅ Ready | 🟡 Draft | 🔴 Stub | Trust |
|----------|-------|---------|---------|---------|-------|
| Skills | 8 | 0 | 8 | 0 | Low — newly created, untested |
| Workflows (SOPs) | 4 | 0 | 4 | 0 | Low — newly created, untested |
| Memory files | 4 | 1 | 3 | 0 | Medium — codebase_map is solid |
| Lessons | 0 | — | — | — | N/A — empty |
| Exploration logs | 0 | — | — | — | N/A — empty |

---

## Skills (`.agent/skills/`)

| Skill | File | Status | Trust | Tested | Notes |
|-------|------|--------|-------|--------|-------|
| Launch Experiment | `launch-experiment.md` | 🟡 Draft | Low | ❌ | Needs dry-run validation |
| Monitor Experiment | `monitor-experiment.md` | 🟡 Draft | Low | ❌ | Requires W&B API access to test |
| Summarize Run | `summarize-run.md` | 🟡 Draft | Low | ❌ | Pattern from existing test infra |
| Log Experiment | `log-experiment.md` | 🟡 Draft | Low | ❌ | Journal formatting only |
| Evaluate Video Quality | `evaluate-video-quality.md` | 🟡 Draft | Low | ❌ | SSIM section most mature |
| Index Related Work | `index-related-work.md` | 🟡 Draft | Low | ❌ | Schema defined, no entries yet |
| Search Related Work | `search-related-work.md` | 🟡 Draft | Low | ❌ | Depends on indexed entries |
| Skill Template | `SKILL_TEMPLATE.md` | ✅ Ready | High | ✅ | Meta-template, stable |

### Trust Level Definitions
- **High**: Tested in production, validated against real experiments
- **Medium**: Logic is sound, partially tested or based on existing patterns
- **Low**: Newly written, not yet validated
- **None**: Placeholder only

---

## Workflows / SOPs (`.agent/workflows/`)

| Workflow | File | Status | Trust | Tested | Notes |
|----------|------|--------|-------|--------|-------|
| Experiment Lifecycle | `experiment-lifecycle.md` | 🟡 Draft | Low | ❌ | End-to-end flow, untested |
| Evaluation Development | `evaluation-development.md` | 🟡 Draft | Low | ❌ | Metric dev process |
| Experiment Journaling | `experiment-journaling.md` | 🟡 Draft | Low | ❌ | Journaling cadence |
| Lesson Capture | `lesson-capture.md` | 🟡 Draft | Low | ❌ | Post-experiment reflection |
| Sync Dashboard | `sync-dashboard.md` | 🟡 Draft | Low | ❌ | This dashboard's updater |

---

## Memory (`.agent/memory/`)

| File | Status | Trust | Notes |
|------|--------|-------|-------|
| `codebase_map.md` | ✅ Ready | High | Synthesized from full repo research |
| `experiment_journal.md` | 🟡 Draft | Medium | Schema defined, no entries yet |
| `evaluation_registry.md` | 🟡 Draft | Medium | SSIM/loss metrics documented |
| `related_work/README.md` | 🟡 Draft | Medium | Schema defined, no entries yet |

---

## Lessons (`.agent/lessons/`)

| File | Category | Severity | Notes |
|------|----------|----------|-------|

_No lessons captured yet._

---

## Exploration Logs (`.agent/exploration/`)

| File | Status | Topic | Notes |
|------|--------|-------|-------|

_No exploration logs yet._

---

## What to Do Next

1. **Validate skills**: Run a minimal training experiment using the
   `experiment-lifecycle` SOP to test `launch-experiment` → `monitor-experiment`
   → `summarize-run` end-to-end.
2. **Index first related work**: Use `index-related-work` to add at least one
   paper (e.g., the Self-Forcing paper used in the codebase).
3. **Capture first lesson**: After the validation run, capture any findings.
4. **Promote to Ready**: As each skill/SOP is tested, update its status here.
