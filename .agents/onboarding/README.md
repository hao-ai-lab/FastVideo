# Agent Onboarding — FastVideo-WorldModel

Welcome, agent. This is the **master onboarding** guide. Follow the steps below,
then check if a **domain-specific onboarding** exists for your task.

## Domain-Specific Onboarding

If your task falls into one of these areas, read the specialized guide **after**
completing the general steps below:

| Domain | Guide | When to Use |
|--------|-------|-------------|
| **WorldModel Training** | `worldmodel-training/README.md` | Training, finetuning, distillation, experiment management |

---

## Step 1: Understand the Codebase

Read these files to build your context:

| Priority | File | What you learn |
|----------|------|----------------|
| 1 | `AGENTS.md` | Coding guidelines, build/test commands, PR conventions |
| 2 | `docs/design/overview.md` | Architecture: models, pipelines, configs, registry |
| 3 | `docs/training/overview.md` | Training data flow and preprocessing |
| 4 | `docs/training/finetune.md` | Training arguments, parallelism, LoRA, validation |
| 5 | `docs/contributing/coding_agents.md` | How to add model pipelines with agent assistance |

## Step 2: Discover Available Resources

Read these two index files to see what skills and memory modules exist:

- **`.agents/skills/index.jsonl`** — catalog of all agent skills (name + description)
- **`.agents/memory/index.jsonl`** — catalog of all memory modules (name + description)

Each entry has a `path` field pointing to the full content. Only load the
full README.md for modules relevant to your current task.

## Step 3: Check for Existing Skills & SOPs

Before writing new code or procedures:

1. **Skills**: Read `.agents/skills/index.jsonl` — find a matching skill by description.
2. **Workflows/SOPs**: Browse `.agents/workflows/` — step-by-step procedures for common tasks.
3. **Lessons**: Browse `.agents/lessons/` — known pitfalls and their fixes.

If a skill or SOP exists for your task, **use it**. If not, you are in **exploration mode** — see Step 4.

## Step 4: Exploration Mode

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
├── memory/                ← persistent context (folder per topic + index.jsonl)
│   ├── index.jsonl
│   ├── codebase-map/
│   ├── experiment-journal/
│   ├── evaluation-registry/
│   └── related-work/
├── lessons/               ← mistakes and fixes
└── exploration/           ← draft procedures
```
