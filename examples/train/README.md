# Training

## Quick Start

### Single-node

```bash
bash examples/train/run.sh <config.yaml> [--dotted.key value ...]
```

The script auto-detects available GPUs. Override with `NUM_GPUS`:

```bash
NUM_GPUS=4 bash examples/train/run.sh \
    examples/train/configs/fine_tuning/wan/t2v.yaml
```

### Multi-node (Slurm)

```bash
bash examples/train/run_slurm.sh <config.yaml> <num_nodes> [--dotted.key value ...]
```

```bash
bash examples/train/run_slurm.sh \
    examples/train/configs/fine_tuning/wan/t2v.yaml 4 \
    --training.distributed.hsdp_shard_dim 8 \
    --training.distributed.hsdp_replicate_dim 4
```

Slurm environment variables:

| Variable | Default | Description |
|---|---|---|
| `PARTITION` | `main` | Slurm partition |
| `NUM_GPUS` | `8` | GPUs per node |
| `CPUS_PER_TASK` | `128` | CPUs per task |
| `MEM` | `1440G` | Memory per node |
| `EXCLUDE` | `""` | Nodes to exclude |

### Overriding config values

Any config field can be overridden from the command line using dotted keys:

```bash
bash examples/train/run.sh examples/train/configs/fine_tuning/wan/t2v.yaml \
    --training.optimizer.learning_rate 1e-5 \
    --training.loop.max_train_steps 1000 \
    --training.checkpoint.output_dir outputs/my_experiment
```

### Resuming from a checkpoint

```bash
bash examples/train/run.sh examples/train/configs/fine_tuning/wan/t2v.yaml \
    --training.checkpoint.resume_from_checkpoint outputs/my_experiment/checkpoint-500
```

## W&B Logging

Training metrics and validation videos are logged to
[Weights & Biases](https://wandb.ai). Set your API key before launching:

```bash
export WANDB_API_KEY=your_key_here
```

To disable W&B (e.g. for local debugging):

```bash
export WANDB_MODE=offline
```

The project name and run name are set in the config under `training.tracker`:

```yaml
training:
  tracker:
    project_name: my_project
    run_name: my_run
```

## Directory Layout

```
examples/train/
├── configs/      # Single-step training configs (by method and model)
├── scenario/     # Multi-step end-to-end training pipelines
├── run.sh        # Single-node launcher
└── run_slurm.sh  # Multi-node Slurm launcher
```

See `configs/README.md` and `scenario/README.md` for details.
