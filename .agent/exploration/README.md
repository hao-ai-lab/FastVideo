# Exploration Logs

This directory holds draft procedures and investigation notes for tasks that
don't yet have a standardized skill or SOP. Each exploration should follow this
template.

## When to Create an Exploration Log

- You are working on a task with no existing skill or workflow.
- You are experimenting with a new metric, training technique, or tool.
- You want to document findings before they are promoted to a standard.

## File Naming

`<topic-slug>.md` — e.g., `fvd-metric-investigation.md`

## Template

```markdown
# Exploration Log: <Topic>

## Status: draft | under_review | promoted | abandoned

## Context
<Why this exploration is needed — link to experiment or task if applicable.>

## Progress
- [ ] Step 1: ...
- [ ] Step 2: ...

## Findings
<What you have learned so far.>

## Mistakes / Dead Ends
<What didn't work and why — these become lessons.>

## Proposed Standardization
<If this works, describe the skill/SOP/workflow to create.>
```

## Lifecycle

1. **Create** during exploration mode.
2. **Update** as you make progress.
3. **Promote**: If findings are solid, create a skill in `.agent/skills/` or an SOP in `.agent/workflows/`.
4. **Archive mistakes**: Move failures into `.agent/lessons/`.
