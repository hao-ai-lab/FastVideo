# Agent Onboarding — FastVideo-WorldModel

Welcome, agent. Follow these steps **in order** before doing any work.

## Step 1: Understand the Codebase

Read these files to build your context:

| Priority | File | What you learn |
|----------|------|----------------|
| 1 | `AGENTS.md` | Coding guidelines, build/test commands, PR conventions |
| 2 | `docs/design/overview.md` | Architecture: models, pipelines, configs, registry |
| 3 | `docs/training/overview.md` | Training data flow and preprocessing |
| 4 | `docs/training/finetune.md` | Training arguments, parallelism, LoRA, validation |
| 5 | `docs/contributing/coding_agents.md` | How to add model pipelines with agent assistance |

## Step 2: Load Agent Memory

| File | Purpose |
|------|---------|
| `.agents/memory/codebase_map.md` | Structural index of the entire repo |
| `.agents/memory/experiment_journal.md` | Log of all past experiments and insights |
| `.agents/memory/evaluation_registry.md` | Available metrics and their status |

## Step 3: Check for Existing Skills & SOPs

Before writing new code or procedures:

1. **Skills**: Browse `.agents/skills/` — each `.md` file is a self-contained skill with instructions.
2. **Workflows/SOPs**: Browse `.agents/workflows/` — step-by-step procedures for common tasks.
3. **Lessons**: Browse `.agents/lessons/` — known pitfalls and their fixes.

If a skill or SOP exists for your task, **use it**. If not, you are in **exploration mode** — see Step 5.

## Step 4: Check Related Work

If your task involves evaluation metrics, training techniques, or architecture decisions:
- Browse `.agents/memory/related_work/` for indexed papers and repos.
- Use the `search-related-work` skill if available.

## Step 5: Exploration Mode

If no existing skill/SOP covers your task:

1. Document your progress in `.agents/exploration/<topic>.md` using the template in `.agents/exploration/README.md`.
2. At the end of your session, reflect:
   - **What worked** → propose a new skill or SOP in the exploration log.
   - **What failed** → create a lesson in `.agents/lessons/`.
3. Flag the exploration log for human review.

## Quick Reference

```
.agents/
├── ONBOARDING.md          ← you are here
├── STATUS.md              ← dashboard: completeness & trust of all components
├── skills/                ← reusable agent skills
├── workflows/             ← SOPs and procedures
├── memory/                ← persistent context
│   ├── codebase_map.md
│   ├── experiment_journal.md
│   ├── evaluation_registry.md
│   └── related_work/
├── lessons/               ← mistakes and fixes
└── exploration/           ← draft procedures
```
