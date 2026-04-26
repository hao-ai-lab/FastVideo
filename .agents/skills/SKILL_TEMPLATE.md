---
name: <skill-name>
description: <one-line description — Codex uses this for implicit invocation matching>
---

# <Skill Name>

## Purpose
<Why this skill exists and when to use it.>

## Prerequisites
- <What must be true before using this skill>

## Inputs
| Parameter | Required | Description |
|-----------|----------|-------------|
| `param1` | Yes | ... |

## Steps

1. **Step 1 title**
   - Detail...

2. **Step 2 title**
   - Detail...

## Outputs
- <What this skill produces>

## Example Usage
```
<Example invocation or prompt snippet>
```

## References
- <Links to relevant files in the codebase>

---

## Folder Structure

Each skill lives in its own directory under `.agents/skills/`:

```
.agents/skills/<skill-name>/
├── SKILL.md          # Required: instructions + metadata (this file)
├── scripts/          # Optional: executable helper scripts
├── references/       # Optional: documentation, papers
└── assets/           # Optional: templates, resources
```

After creating a new skill, add an entry to `.agents/skills/index.jsonl`:
```json
{"name": "<skill-name>", "description": "<description>", "path": "<skill-name>/SKILL.md", "status": "draft", "trust": "low"}
```
