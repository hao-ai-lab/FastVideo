---
name: log-experiment
description: Append or update an experiment entry in the experiment journal
---

# Log Experiment

## Purpose
Create or update an entry in `.agent/memory/experiment_journal.md` to maintain
a living record of all experiments and their outcomes.

## Prerequisites
- `.agent/memory/experiment_journal.md` exists.

## Inputs

| Parameter | Required | Description |
|-----------|----------|-------------|
| `name` | Yes | Experiment name / identifier |
| `hypothesis` | No | What you expected to learn |
| `config` | Yes | Key config: model, lr, sp_size, gpus, script |
| `wandb_run` | No | W&B run ID or URL |
| `duration` | No | Total wall time |
| `metrics` | No | Key metrics dict (loss, step_time, grad_norm) |
| `checkpoint` | No | Path to checkpoint |
| `insight` | No | What was learned |
| `status` | Yes | `running`, `completed`, `failed`, `abandoned` |
| `lessons` | No | Paths to related lesson files |

## Steps

### 1. Check for existing entry

Search `.agent/memory/experiment_journal.md` for an entry with the same name.
If found, update it instead of creating a duplicate.

### 2. Format the entry

```markdown
## [YYYY-MM-DD] Experiment: <name>
- **Hypothesis**: <hypothesis or "N/A">
- **Config**: model=<model>, lr=<lr>, sp_size=<sp>, gpus=<n>, script=<script>
- **W&B run**: <wandb_run or "pending">
- **Duration**: <duration or "in progress">
- **Key metrics**: loss=<loss>, step_time=<step_time>, grad_norm=<grad_norm>
- **Checkpoint**: <checkpoint or "N/A">
- **Insight**: <insight or "pending">
- **Status**: <status>
- **Related lessons**: <lessons or "none">
```

### 3. Insert at the top of the journal

New entries go at the top of the file (after the header), so the most recent
experiments are always visible first.

### 4. Warn on duplicates

If a similar experiment name exists with `status: completed`, warn that this
may be a repeat. If it's `status: running`, assume this is an update.

## Outputs
- Updated `.agent/memory/experiment_journal.md`.

## Example Usage
```
Log a completed experiment:

  name: wan-t2v-finetune-lr5e5-sp4
  config: model=wan-t2v-1.3B, lr=5e-5, sp_size=4, gpus=4
  wandb_run: fastvideo/training/run_abc123
  duration: 2h 15m
  metrics: {loss: 0.065, step_time: 2.3, grad_norm: 0.35}
  checkpoint: outputs/wan_finetune/checkpoint-1000
  insight: LR 5e-5 converges 30% faster than 1e-5 with no quality loss
  status: completed
```

## References
- `.agent/memory/experiment_journal.md` — journal file
- `.agent/workflows/experiment-lifecycle.md` — when to log

## Changelog
| Date | Change |
|------|--------|
| 2026-03-02 | Initial version |
