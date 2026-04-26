---
name: monitor-experiment
description: Poll a running W&B training run for progress and emit structured alerts
---

# Monitor Experiment

## Purpose
Continuously (or on-demand) check a running experiment's W&B metrics and emit
alerts for anomalies. Supports the "30-minute quality check" paradigm: after
the first 30 minutes of a long training run, produce a checkpoint quality
report before committing more resources.

## Prerequisites
- `WANDB_API_KEY` is set in the environment.
- The experiment is actively logging to W&B (not in `WANDB_MODE=offline`).
- For offline mode: read from local `wandb-summary.json` instead.

## Inputs

| Parameter | Required | Description |
|-----------|----------|-------------|
| `run_id` | Yes* | W&B run ID (e.g., `entity/project/run_id`) |
| `output_dir` | Yes* | Local output directory (for offline mode fallback) |
| `poll_interval` | No | Seconds between polls (default: 60) |
| `alert_on` | No | List of alert conditions to enable (default: all) |

\* One of `run_id` or `output_dir` is required.

## Steps

### 1. Connect to the run

**Online mode** (preferred):
```python
import wandb
api = wandb.Api()
run = api.run("<run_id>")
```

**Offline fallback**:
```python
import json
summary_path = f"{output_dir}/tracker/wandb/latest-run/files/wandb-summary.json"
with open(summary_path) as f:
    summary = json.load(f)
```

### 2. Track key metrics

| Metric | W&B Key | Description |
|--------|---------|-------------|
| Training loss | `train_loss` | Primary training loss |
| Gradient norm | `grad_norm` | Gradient magnitude |
| Step time | `step_time` | Wall-clock seconds per step |
| Learning rate | `learning_rate` | Current LR |
| Avg step time | `avg_step_time` | Running average step time |
| Validation videos | `validation_videos_*` | Generated validation samples |

### 3. Evaluate alert conditions

| Alert | Condition | Severity |
|-------|-----------|----------|
| **Loss spike** | `current_loss > 3 × rolling_avg_loss` | 🔴 Critical |
| **NaN/Inf gradient** | `grad_norm` is NaN or Inf | 🔴 Critical |
| **Step time regression** | `step_time > 2 × baseline_step_time` | 🟡 Warning |
| **No progress** | No new W&B logs for > 10 minutes | 🟡 Warning |
| **Loss plateau** | Loss change < 1% over last 100 steps | 🟢 Info |

### 4. Emit structured status

Output format (agent-consumable):
```json
{
  "run_id": "...",
  "step": 500,
  "metrics": {
    "train_loss": 0.078,
    "grad_norm": 0.41,
    "step_time": 2.5,
    "learning_rate": 1e-6
  },
  "alerts": [
    {"type": "loss_spike", "severity": "critical", "message": "Loss jumped to 0.45 (avg: 0.08)"}
  ],
  "status": "running"
}
```

### 5. 30-Minute Quality Check

After the first 30 minutes of wall-clock time:
1. Summarize the loss curve shape (decreasing? at what rate?).
2. Check if validation videos have been generated.
3. Report step count, loss at start vs. current, and estimated time to completion.
4. Produce a go/no-go recommendation.

```markdown
## 30-Minute Check: <run_name>
- **Steps completed**: 150
- **Loss**: 0.12 → 0.08 (↓ 33%)
- **Grad norm**: stable at ~0.4
- **Step time**: 2.5s/step (consistent)
- **Validation videos**: 5 generated at step 100
- **Recommendation**: ✅ Continue — loss is decreasing normally
```

## Outputs
- Structured JSON status updates.
- Alert messages for anomalous conditions.
- 30-minute checkpoint quality report.

## Example Usage
```
Monitor W&B run "fastvideo/Wan_distillation/abc123":

  run_id: fastvideo/Wan_distillation/abc123
  poll_interval: 120
  alert_on: [loss_spike, nan_gradient, step_time_regression]
```

## References
- `fastvideo/training/trackers.py` — `WandbTracker` implementation
- `fastvideo/tests/training/Vanilla/test_training_loss.py` — how summaries are compared
- `fastvideo/tests/training/Vanilla/a40_reference_wandb_summary.json` — reference summary format

## Changelog
| Date | Change |
|------|--------|
| 2026-03-02 | Initial version |
