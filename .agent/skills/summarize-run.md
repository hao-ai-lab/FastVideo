---
name: summarize-run
description: Extract a W&B run summary into a structured experiment report
---

# Summarize Run

## Purpose
After a training run completes (or at any checkpoint), extract key metrics from
the W&B run summary and produce a structured markdown report. Supports both
online (W&B API) and offline (local `wandb-summary.json`) modes.

## Prerequisites
- Run has completed or reached a checkpoint with a saved summary.
- For online: `WANDB_API_KEY` set in environment.
- For offline: access to `<output_dir>/tracker/wandb/latest-run/files/wandb-summary.json`.

## Inputs

| Parameter | Required | Description |
|-----------|----------|-------------|
| `run_id` | Yes* | W&B run ID for online access |
| `output_dir` | Yes* | Local output dir for offline access |
| `reference_run` | No | Path to reference `wandb-summary.json` for comparison |
| `experiment_name` | No | Name for the journal entry (default: from W&B) |

\* One of `run_id` or `output_dir` is required.

## Steps

### 1. Load run summary

**Online**:
```python
import wandb
api = wandb.Api()
run = api.run("<run_id>")
summary = dict(run.summary)
config = dict(run.config)
```

**Offline** (existing codebase pattern from `fastvideo/tests/training/`):
```python
import json
summary_path = f"{output_dir}/tracker/wandb/latest-run/files/wandb-summary.json"
with open(summary_path) as f:
    summary = json.load(f)
```

### 2. Extract key fields

| Field | Source | Description |
|-------|--------|-------------|
| `train_loss` | `summary["train_loss"]` | Final training loss |
| `avg_step_time` | `summary["avg_step_time"]` | Average seconds per step |
| `step_time` | `summary["step_time"]` | Last step time |
| `grad_norm` | `summary["grad_norm"]` | Final gradient norm |
| `learning_rate` | `summary["learning_rate"]` | Final LR |
| `_step` | `summary["_step"]` | Total steps completed |
| `_runtime` | `summary["_runtime"]` | Total wall-clock seconds |
| `validation_videos_*` | `summary[key]` | Validation video artifacts |

### 3. Compare against reference (optional)

Follow the pattern in `fastvideo/tests/training/Vanilla/test_training_loss.py`:

```python
# Fields to compare
compare_fields = ["train_loss", "grad_norm", "avg_step_time"]
tolerance = 0.05  # 5% relative tolerance

for field in compare_fields:
    ref_val = reference_summary[field]
    cur_val = summary[field]
    diff_pct = abs(cur_val - ref_val) / abs(ref_val) * 100
    status = "✅" if diff_pct < tolerance * 100 else "⚠️"
    print(f"{status} {field}: {cur_val:.4f} (ref: {ref_val:.4f}, diff: {diff_pct:.1f}%)")
```

### 4. Generate report

```markdown
# Run Summary: <experiment_name>

| Metric | Value | Reference | Diff |
|--------|-------|-----------|------|
| Train Loss | 0.0788 | 0.0800 | -1.5% ✅ |
| Avg Step Time | 2.81s | 2.80s | +0.4% ✅ |
| Grad Norm | 0.408 | 0.410 | -0.5% ✅ |
| Total Steps | 500 | — | — |
| Wall Time | 23m 30s | — | — |

## Configuration
- Model: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
- Learning Rate: 1e-6
- Batch Size: 1
- GPUs: 8 × (SP=1, TP=1)
- Mixed Precision: bf16

## Validation Videos
<list of validation video paths if available>

## Notes
<any observations or anomalies>
```

### 5. Update experiment journal

Append or update the experiment's entry in `.agent/memory/experiment_journal.md`
with the final metrics and status.

## Outputs
- Structured markdown report.
- Updated experiment journal entry.

## Example Usage
```
Summarize the run in output directory "outputs/wan_finetune":

  output_dir: outputs/wan_finetune
  reference_run: fastvideo/tests/training/Vanilla/a40_reference_wandb_summary.json
  experiment_name: wan-t2v-finetune-lr1e6
```

## References
- `fastvideo/tests/training/Vanilla/test_training_loss.py` — reference comparison pattern
- `fastvideo/tests/training/Vanilla/a40_reference_wandb_summary.json` — example summary
- `fastvideo/tests/training/lora/test_lora_training.py` — LoRA summary comparison
- `fastvideo/training/trackers.py` — tracker summary generation

## Changelog
| Date | Change |
|------|--------|
| 2026-03-02 | Initial version |
