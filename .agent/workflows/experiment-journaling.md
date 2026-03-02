---
description: When and how to log experiments in the experiment journal
---

# Experiment Journaling SOP

Ensures every experiment is properly recorded with context and outcomes.

## When to Log

**Always.** Every experiment — even quick tests — should be journaled.

## Steps

### 1. Before Launch — Create Draft Entry

Use the `log-experiment` skill with `status: running`:
- Include hypothesis and config.
- Leave metrics, duration, and insight blank.

### 2. After 30-Minute Check — Update with Initial Metrics

Update the entry with:
- Current loss and its trajectory direction.
- Step time.
- Number of validation videos generated.
- Preliminary go/no-go assessment.

### 3. On Completion — Fill Final Entry

Update the entry with `status: completed`:
- Final loss, grad norm, avg step time.
- Total duration and steps.
- Checkpoint path.
- Key insight.

### 4. On Failure — Document Failure Mode

Update the entry with `status: failed`:
- What went wrong (OOM, NaN, crash, etc.).
- At what step the failure occurred.
- Create a lesson in `.agent/lessons/` for non-trivial failures.

### 5. Cross-Reference

- Link related lessons: `**Related lessons**: .agent/lessons/<filename>.md`
- Link related experiments: if this is a follow-up, reference the prior entry.
