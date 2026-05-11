# Agent Infrastructure — Status Dashboard

Developer-maintained overview of all agent components and their maturity.
Use this to understand what exists, how complete it is, and how much to trust it.

_Last synced: 2026-05-10_

> To resync this dashboard, use the workflow: `.agents/workflows/sync-dashboard.md`

---

## Summary

| Category | Total | ✅ Ready | ✅ Tested | 🟡 Draft | 🔴 Stub | Trust |
|----------|-------|---------|----------|---------|---------|-------|
| Skills | 21 | 0 | 1 | 20 | 0 | Low — most skills draft; one tested at medium trust |
| Workflows (SOPs) | 5 | 0 | 0 | 5 | 0 | Low — newly created, untested |
| Memory files | 4 | 1 | 0 | 3 | 0 | Medium — codebase_map is solid |
| Lessons | 0 | — | — | — | — | N/A — empty |
| Exploration logs | 0 | — | — | — | — | N/A — empty |

---

## Skills (`.agents/skills/`)

| Skill | File | Status | Trust | Tested | Notes |
|-------|------|--------|-------|--------|-------|
| Add Model | `add-model/SKILL.md` | 🟡 Draft | Low | ❌ | Manual model-port orchestrator |
| Add Model Prep | `add-model-01-prep/SKILL.md` | 🟡 Draft | Low | ❌ | Prep phase for model ports |
| Add Model Parity | `add-model-02-parity/SKILL.md` | 🟡 Draft | Low | ❌ | Component parity scaffolding |
| Add Model Port DiT | `add-model-03-port-dit/SKILL.md` | 🟡 Draft | Low | ❌ | DiT component port helper |
| Add Model Port VAE | `add-model-04-port-vae/SKILL.md` | 🟡 Draft | Low | ❌ | VAE component port helper |
| Add Model Port Encoder | `add-model-05-port-encoder/SKILL.md` | 🟡 Draft | Low | ❌ | Encoder component port helper |
| Add Model Port Generic | `add-model-06-port-generic/SKILL.md` | 🟡 Draft | Low | ❌ | Generic component port helper |
| Add Model Conversion | `add-model-07-conversion/SKILL.md` | 🟡 Draft | Low | ❌ | Checkpoint conversion helper |
| Add Model Trace | `add-model-08-trace/SKILL.md` | 🟡 Draft | Low | ❌ | Activation divergence tracing |
| Add Model Pipeline | `add-model-09-pipeline/SKILL.md` | 🟡 Draft | Low | ❌ | Pipeline wiring and parity helper |
| Add Model PR Review | `add-model-10-pr-review/SKILL.md` | 🟡 Draft | Low | ❌ | Model-port review rubric |
| Decompose Pipeline PR | `decompose-pipeline-pr/SKILL.md` | ✅ Tested | Medium | ✅ | Worked example: PR #1280 |
| Evaluate Video Quality | `evaluate-video-quality/SKILL.md` | 🟡 Draft | Low | ❌ | SSIM section most mature |
| Index Related Work | `index-related-work/SKILL.md` | 🟡 Draft | Low | ❌ | Schema defined, no entries yet |
| Launch Experiment | `launch-experiment/SKILL.md` | 🟡 Draft | Low | ❌ | Needs dry-run validation |
| Log Experiment | `log-experiment/SKILL.md` | 🟡 Draft | Low | ❌ | Journal formatting only |
| Monitor Experiment | `monitor-experiment/SKILL.md` | 🟡 Draft | Low | ❌ | Requires W&B API access to test |
| Re-seed SSIM References | `reseed-ssim-references/SKILL.md` | 🟡 Draft | Low | ❌ | Existing-reference refresh flow |
| Search Related Work | `search-related-work/SKILL.md` | 🟡 Draft | Low | ❌ | Depends on indexed entries |
| Seed SSIM References | `seed-ssim-references/SKILL.md` | 🟡 Draft | Low | ❌ | First-time SSIM reference flow |
| Summarize Run | `summarize-run/SKILL.md` | 🟡 Draft | Low | ❌ | Pattern from existing test infra |
| Skill Template | `SKILL_TEMPLATE.md` | ✅ Ready | High | ✅ | Meta-template, stable |

### Trust Level Definitions
- **High**: Tested in production, validated against real experiments
- **Medium**: Logic is sound, partially tested or based on existing patterns
- **Low**: Newly written, not yet validated
- **None**: Placeholder only

---

## Workflows / SOPs (`.agents/workflows/`)

| Workflow | File | Status | Trust | Tested | Notes |
|----------|------|--------|-------|--------|-------|
| Experiment Lifecycle | `experiment-lifecycle.md` | 🟡 Draft | Low | ❌ | End-to-end flow, untested |
| Evaluation Development | `evaluation-development.md` | 🟡 Draft | Low | ❌ | Metric dev process |
| Experiment Journaling | `experiment-journaling.md` | 🟡 Draft | Low | ❌ | Journaling cadence |
| Lesson Capture | `lesson-capture.md` | 🟡 Draft | Low | ❌ | Post-experiment reflection |
| Sync Dashboard | `sync-dashboard.md` | 🟡 Draft | Low | ❌ | This dashboard's updater |

---

## Memory (`.agents/memory/`)

| File | Status | Trust | Notes |
|------|--------|-------|-------|
| `codebase-map/README.md` | ✅ Ready | High | Synthesized from full repo research |
| `experiment-journal/README.md` | 🟡 Draft | Medium | Schema defined, no entries yet |
| `evaluation-registry/README.md` | 🟡 Draft | Medium | SSIM/loss metrics documented |
| `related-work/README.md` | 🟡 Draft | Medium | Schema defined, no entries yet |

---

## Lessons (`.agents/lessons/`)

| File | Category | Severity | Notes |
|------|----------|----------|-------|

_No lessons captured yet._

---

## Exploration Logs (`.agents/exploration/`)

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
