# Source Archive

These are the original unsynthesized design and integration docs that
predate the consolidation in
[`../`](../). They are **NOT** the source of truth — the synthesized
sibling files in the parent directory are.

Archived 2026-05-03. All previously untracked.

## Contents

| File | Original location | Date | Synthesized into |
|---|---|---|---|
| `apirefactor.md` | `FastVideo/` (repo root) | 2026-04-21 | [`../design.md`](../design.md) |
| `PR-plan.md` (was `PR plan.md` at repo root) | `FastVideo/` (repo root) | 2026-04-25 | [`../pr-roadmap.md`](../pr-roadmap.md) |
| `dreamverse_review.md` | `FastVideo/` (repo root) | 2026-04-26 | [`../decisions-log.md`](../decisions-log.md) + [`../state.md`](../state.md) |
| `handoff-nvfp4-launch-demo.md` | `.agents/exploration/` | 2026-05-02 | [`../state.md`](../state.md) + [`../quantization.md`](../quantization.md) + [`../open-threads.md`](../open-threads.md) |
| `streaming-server-upstream-plan.md` | `.agents/exploration/` | 2026-04-17 | [`../streaming-server.md`](../streaming-server.md) + [`../decisions-log.md`](../decisions-log.md) |
| `dreamverse_integration.md` | `.agents/exploration/` | 2026-04-23 | [`../cross-repo-surfaces.md`](../cross-repo-surfaces.md) |
| `video-generator-config-api-design.md` | `.agents/exploration/` | 2026-04-02 | [`../design.md`](../design.md) (early-draft material) |

## Why archived (not deleted)

- Future agents may want the **full unsynthesized rationale** for a
  decision the synthesis abbreviated.
- The originals remain useful as a **time machine** for understanding
  how the design evolved.
- These docs were never committed to git, so leaving them on disk costs
  nothing.

## When to read the archive vs. the synthesis

- **Read the synthesis (`../*.md`)** for: current state, decision
  status, action items, design rationale at the conceptual level.
- **Read the archive (here)** for: deep historical context, exact wording
  of design decisions, full PR plan with all sub-PR commit details,
  the original Q-1..Q-9 / D-1..D-11 prose.

## Maintenance rule

Do NOT edit files in this archive. They are point-in-time snapshots.
If new design material appears that supersedes an entry here, update the
synthesis (the parent dir) and append a note to that synthesis file —
do not mutate this archive.
