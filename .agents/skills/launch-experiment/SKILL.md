---
name: launch-experiment
description: Generate and execute a training launch command for FastVideo models
---

# Launch Experiment

## Purpose
Construct a fully-specified `torchrun` training command for a FastVideo model
given a target pipeline, dataset, and hyperparameter overrides. This skill
automates the boilerplate of setting environment variables, picking the right
entrypoint, and applying defaults from the closest example script.

## Prerequisites
- The repo is cloned and `fastvideo` is installed (`uv pip install -e .[dev]`).
- Dataset is preprocessed (see `docs/training/data_preprocess.md`).
- `WANDB_API_KEY` is set in the environment (or `WANDB_MODE=offline` for local).
- GPU resources are available (multi-GPU requires NCCL).

## Inputs

| Parameter | Required | Description |
|-----------|----------|-------------|
| `pipeline` | Yes | Training pipeline type: `finetune`, `distill-dmd`, `self-forcing`, `lora`, `consistency` |
| `model` | Yes | Model family: `wan-t2v-1.3B`, `wan-i2v-14B`, `ltx2`, `matrixgame` |
| `data_path` | Yes | Path to preprocessed dataset (parquet) |
| `num_gpus` | Yes | Number of GPUs |
| `overrides` | No | Dict of hyperparameter overrides (any CLI arg) |
| `output_dir` | No | Output directory (default: `outputs/<model>_<pipeline>`) |
| `run_name` | No | W&B run name (default: auto-generated) |

## Steps

### 1. Identify the training entrypoint

| Pipeline | Entrypoint |
|----------|-----------|
| `finetune` (Wan T2V) | `fastvideo/training/wan_training_pipeline.py` |
| `finetune` (Wan I2V) | `fastvideo/training/wan_i2v_training_pipeline.py` |
| `finetune` (LTX-2) | `fastvideo/training/ltx2_training_pipeline.py` |
| `finetune` (MatrixGame) | `fastvideo/training/matrixgame_training_pipeline.py` |
| `distill-dmd` | `fastvideo/training/wan_distillation_pipeline.py` |
| `self-forcing` | `fastvideo/training/wan_self_forcing_distillation_pipeline.py` |

### 2. Resolve default hyperparameters

Find the closest example script in `examples/training/` for the model:

| Model | Example Script Directory |
|-------|-------------------------|
| `wan-t2v-1.3B` | `examples/training/finetune/wan_t2v_1.3B/crush_smol/` |
| `wan-i2v-14B` | `examples/training/finetune/wan_i2v_14B_480p/crush_smol/` |
| `ltx2` | `examples/training/finetune/ltx2/` |
| `matrixgame` | `examples/training/finetune/MatrixGame2.0/` |
| `distill-dmd` | `scripts/distill/v1_distill_dmd_wan.sh` |

Read the script to extract default values for:
- `--learning_rate`, `--train_batch_size`, `--sp_size`, `--tp_size`
- `--num_latent_t`, `--num_height`, `--num_width`, `--num_frames`
- `--gradient_accumulation_steps`, `--max_train_steps`
- `--mixed_precision`, `--weight_decay`, `--max_grad_norm`
- `--validation_steps`, `--validation_sampling_steps`

### 3. Set environment variables

```bash
export WANDB_API_KEY="${WANDB_API_KEY}"
export WANDB_BASE_URL="https://api.wandb.ai"
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache
```

### 4. Construct the torchrun command

```bash
torchrun --nnodes 1 --nproc_per_node <num_gpus> \
    <entrypoint> \
    --pretrained_model_name_or_path <model_hf_id> \
    --data_path "<data_path>" \
    --output_dir "<output_dir>" \
    --wandb_run_name "<run_name>" \
    --tracker_project_name "<project_name>" \
    --log_validation \
    <...all hyperparameters...>
```

### 5. Log to experiment journal

After launching, append an entry to `.agents/memory/experiment_journal.md`:
```markdown
## [YYYY-MM-DD] Experiment: <run_name>
- **Hypothesis**: <user-provided or auto-generated>
- **Config**: model=<model>, lr=<lr>, sp_size=<sp>, gpus=<n>, script=<entrypoint>
- **W&B run**: <pending — will be updated by monitor skill>
- **Status**: running
```

## Outputs
- A ready-to-execute shell command.
- An experiment journal entry.

## Example Usage
```
Launch a Wan T2V 1.3B finetune on 4 GPUs with lr=5e-5 and max_train_steps=1000:

  pipeline: finetune
  model: wan-t2v-1.3B
  data_path: data/crush_smol_preprocessed/
  num_gpus: 4
  overrides:
    learning_rate: 5e-5
    max_train_steps: 1000
```

## References
- `examples/training/finetune/wan_t2v_1.3B/crush_smol/finetune_t2v.sh`
- `scripts/distill/v1_distill_dmd_wan.sh`
- `docs/training/finetune.md` (training arguments table)
- `fastvideo/training/trackers.py` (tracker initialization)

## Changelog
| Date | Change |
|------|--------|
| 2026-03-02 | Initial version |
