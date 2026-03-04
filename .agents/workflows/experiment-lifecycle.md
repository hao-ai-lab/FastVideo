---
description: End-to-end experiment lifecycle from hypothesis to lessons learned
---

# Experiment Lifecycle SOP

Standard operating procedure for running ML training experiments on
FastVideo-WorldModel. Every experiment should follow this flow.

## Overview

```
Plan → Launch → Monitor → Summarize → Journal → Reflect
```

## Steps

### 1. Plan the Experiment

Before launching:
- [ ] Define a clear **hypothesis** (what you expect to learn).
- [ ] Select the **model** and **pipeline** type (finetune, distill, lora, etc.).
- [ ] Prepare the **dataset** (preprocessed into parquet format).
- [ ] Review existing experiments in `.agents/memory/experiment-journal/README.md` for related work.
- [ ] Check `.agents/lessons/` for known pitfalls with this configuration.
- [ ] Document the plan in the experiment journal as a draft entry.

### 2. Launch the Experiment

Use the `launch-experiment` skill:
- Provide: pipeline, model, data_path, num_gpus, and any hyperparameter overrides.
- The skill generates the `torchrun` command and creates a journal entry.
- Verify the command looks correct before executing.

Reference: `.agents/skills/launch-experiment.md`

### 3. Monitor the Experiment

Use the `monitor-experiment` skill:
- Provide the W&B run ID (or output_dir for offline).
- Monitor alerts: loss spikes, NaN gradients, step time regressions.
- At the **30-minute mark**: perform the quality check.
  - Is loss decreasing?
  - Are validation videos reasonable?
  - Is step time consistent?
- **Decision point**: Continue or abort based on the 30-min check.

Reference: `.agents/skills/monitor-experiment.md`

### 4. Summarize the Run

After completion (or at any checkpoint), use the `summarize-run` skill:
- Extract final metrics from W&B summary.
- Compare against reference runs if available.
- Generate a structured report.

Reference: `.agents/skills/summarize-run.md`

### 5. Update the Experiment Journal

Use the `log-experiment` skill to update the journal entry:
- Fill in final metrics, duration, checkpoint paths.
- Record the key insight learned.
- Set status to `completed`, `failed`, or `abandoned`.

Reference: `.agents/skills/log-experiment.md`

### 6. Reflect and Capture Lessons

After every experiment:
- **What went right?** → Note in the journal insight field.
- **What went wrong?** → Create a lesson in `.agents/lessons/`:
  - Use the template in `.agents/lessons/README.md`.
  - Cross-reference the experiment journal entry.
- **What was surprising?** → Consider creating an exploration log if this
  warrants further investigation.

Reference: `.agents/workflows/lesson-capture.md`

## Validation Criteria

This SOP is validated when an agent can:
1. Follow steps 1–6 end-to-end for a minimal training run
   (e.g., `examples/training/finetune/wan_t2v_1.3B/crush_smol/finetune_t2v.sh`
   with `--max_train_steps 5`).
2. Produce a complete experiment journal entry.
3. Generate a run summary report.
