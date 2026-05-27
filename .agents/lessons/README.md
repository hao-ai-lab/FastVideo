# Lessons Learned Database

This directory stores documented mistakes, unexpected behaviors, and their fixes.
Each lesson is a permanent record that helps agents and humans avoid repeating
past errors.

## When to Create a Lesson

- An experiment failed for a non-obvious reason.
- A configuration or hyperparameter choice led to wasted compute.
- A porting, data, or infrastructure issue was discovered and resolved.
- A workaround was needed for a known framework/library bug.

## File Naming

`<YYYY-MM-DD>_<short-slug>.md` — e.g., `2026-03-02_lr-too-high-for-lora.md`

## Template

```markdown
---
date: <ISO-8601>
experiment: <reference to experiment_journal.md entry, if applicable>
category: hyperparameter | data | infrastructure | evaluation | porting | other
severity: critical | important | minor
---

# <Short Descriptive Title>

## What Happened
<Description of the problem and its symptoms.>

## Root Cause
<Analysis of why it happened.>

## Fix / Workaround
<What resolved the issue.>

## Prevention
<How to avoid this in the future — updated skills, SOPs, or checks.>
```

## Usage

- Before starting a task, **search this directory** for relevant lessons.
- After completing or failing a task, **check if a new lesson should be created**.
- Periodically review lessons for **patterns** — recurring themes may warrant
  a new skill, SOP, or codebase fix.
