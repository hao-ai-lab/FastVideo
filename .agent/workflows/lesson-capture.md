---
description: Post-experiment reflection to capture lessons learned
---

# Lesson Capture SOP

Systematic procedure for turning experiment outcomes into persistent knowledge.

## When to Use

After **every** completed or failed experiment. Even successful experiments
can yield lessons (e.g., "LR 5e-5 works better than 1e-5 for LoRA").

## Steps

### 1. Review the Experiment

Read the experiment journal entry. Ask:
- Did anything go wrong?
- Was anything surprising?
- Did anything take longer than expected?
- Was a workaround needed?

### 2. Decide: Lesson or Not?

| Situation | Action |
|-----------|--------|
| Something broke | Create a lesson (category: `infrastructure` or `data`) |
| Hyperparameter choice mattered | Create a lesson (category: `hyperparameter`) |
| Porting issue found | Create a lesson (category: `porting`) |
| Evaluation metric was misleading | Create a lesson (category: `evaluation`) |
| Everything went smoothly | No lesson needed, but note in the journal insight |

### 3. Create the Lesson File

In `.agent/lessons/`, create `<YYYY-MM-DD>_<short-slug>.md`:

```markdown
---
date: <ISO-8601>
experiment: <journal entry reference>
category: hyperparameter | data | infrastructure | evaluation | porting
severity: critical | important | minor
---

# <Short Descriptive Title>

## What Happened
<description>

## Root Cause
<analysis>

## Fix / Workaround
<resolution>

## Prevention
<how to avoid in future>
```

### 4. Cross-Reference

- Update the experiment journal entry with a link to the lesson file.
- If a similar lesson already exists, add a reference or update it.

### 5. Periodic Pattern Review

Every ~10 lessons, scan for patterns:
- Multiple lessons in the same category → consider a new skill or SOP.
- Repeated mistakes → strengthen the relevant SOP with a checklist item.
- Infrastructure issues → propose a codebase fix.
