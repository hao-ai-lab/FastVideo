# PR Reviewers

Repo-specific automatic PR reviewers for FastVideo. Each reviewer is a scoped
agent definition (role + checklist + references) tailored to a concrete slice
of the codebase. A dispatcher routes an incoming PR to one or more reviewers
based on the changed paths and labels.

> Status: **Draft / Skeleton.** None of these reviewers have been exercised on
> real PRs yet. Treat their output as a first pass, not a replacement for human
> review. See [STATUS.md](../STATUS.md) for the trust-level ladder.

## Layout

```
.agents/reviewers/
├── README.md              ← this file
├── ROUTING.md             ← label/path → reviewer routing table
├── index.jsonl            ← machine-readable reviewer index
├── dispatcher/
│   └── DISPATCHER.md      ← instructions for the orchestrator
├── shared/
│   ├── pr-context.md      ← how to fetch PR diff/files/labels via gh
│   ├── review-output.md   ← required output format + severity levels
│   └── repo-conventions.md ← conventions every reviewer should know
├── model/                 ← model / pipeline / config reviewer
│   ├── REVIEWER.md
│   ├── checklist.md
│   └── references.md
├── kernel/                ← fastvideo-kernel + attention backend reviewer
│   ├── REVIEWER.md
│   ├── checklist.md
│   └── references.md
├── training/              ← training / distillation / dataset reviewer
│   ├── REVIEWER.md
│   ├── checklist.md
│   └── references.md
└── general/               ← inference serving / CI / docs / fallback
    ├── REVIEWER.md
    ├── checklist.md
    └── references.md
```

## Reviewers at a glance

| Reviewer | Covers | Primary labels | Primary paths |
|----------|--------|----------------|---------------|
| [model](model/REVIEWER.md) | New models, DiT/VAE/encoder edits, pipeline wiring, arch configs, SSIM coverage | `scope: model`, `type: new-model` | `fastvideo/models/`, `fastvideo/layers/`, `fastvideo/configs/models/`, `fastvideo/pipelines/basic/`, `fastvideo/tests/ssim/` |
| [kernel](kernel/REVIEWER.md) | CUDA/C++/Triton kernels, attention backend fusions, kernel↔layer wiring | `scope: kernel`, `scope: attention` | `fastvideo-kernel/`, `csrc/`, `fastvideo/attention/` |
| [training](training/REVIEWER.md) | Training methods, distillation, datasets, SP/TP/FSDP wiring, checkpointing | `scope: training`, `scope: data`, `scope: distributed` | `fastvideo/train/`, `fastvideo/training/`, `fastvideo/dataset/`, `fastvideo/distributed/`, `examples/train*/`, `examples/distill/` |
| [general](general/REVIEWER.md) | Inference serving, public API, CLI, CI, docs, UI, dependencies, fallback | `scope: inference`, `scope: infra`, `scope: docs`, `scope: ui`, `type: ci`, `type: docs`, `type: misc` | `fastvideo/entrypoints/`, `fastvideo/api/`, `fastvideo/worker/`, `.github/`, `docs/`, `ui/`, `pyproject.toml` |

Full rules live in [ROUTING.md](ROUTING.md). When a PR matches multiple
reviewers (common — a new model PR often hits model + training + general), the
dispatcher fans out and each reviewer reports independently.

## Invocation

Today, invocation is manual. The intended entry points are:

1. **Human invokes reviewer directly.** Paste the contents of a `REVIEWER.md`
   into a Claude Code prompt along with PR number / diff.
2. **Dispatcher-driven.** Run the dispatcher (see `dispatcher/DISPATCHER.md`)
   with a PR number; it fetches context, picks reviewers, and runs each.
3. **Future: CI hook.** A GitHub Action can call the dispatcher on
   `pull_request` events. Not wired up yet.

## How to add a new reviewer

1. Create `.agents/reviewers/<name>/` with `REVIEWER.md`, `checklist.md`,
   `references.md`.
2. Add its triggers to [ROUTING.md](ROUTING.md).
3. Register it in [index.jsonl](index.jsonl).
4. Update the table above.

## Conventions

- All reviewers use the **common output format** in
  [shared/review-output.md](shared/review-output.md) so their output is
  composable.
- All reviewers **MUST cite file paths with line numbers** (`file:line`) when
  flagging issues, so the user can jump straight to the location.
- Severity levels: **BLOCKER** → **MAJOR** → **MINOR** → **NIT**. Defined in
  [shared/review-output.md](shared/review-output.md).
- Reviewers must **not** block on style issues — `pre-commit` handles that.
  Focus on things humans care about: correctness, design, test coverage,
  regressions, perf, GPU-memory, API stability.
